from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import nbformat
import numpy as np

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parents[0]
DATA = SITE / "data"


def _json(path: Path):
    return json.loads(path.read_text())


def test_every_notebook_cell_is_accounted_for():
    coverage = _json(SITE / "content_map.json")
    for name in ("evolution_simulators.ipynb", "SBI_tutorial.ipynb"):
        nb = nbformat.read(ROOT / name, as_version=4)
        rows = coverage[name]
        assert len(rows) == len(nb.cells)
        assert [r["cell_id"] for r in rows] == [c.get("id") or f"cell-{i}" for i, c in enumerate(nb.cells)]
        assert all(r["status"] in {"included", "deliberately_collapsed", "excluded"} for r in rows)
        assert all(r["reason"] for r in rows if r["status"] == "excluded")


def test_interaction_manifest_ids_exist_and_all_nine_stations_are_built():
    manifest = _json(SITE / "interaction_manifest.json")
    found = []
    for name, placements in manifest.items():
        ids = {c.get("id") for c in nbformat.read(ROOT / name, as_version=4).cells}
        assert set(placements) <= ids
        for stations in placements.values():
            found.extend(stations)
    assert set(found) == {
        "evolution-playground", "dfe-example", "chuong-parameter-challenge",
        "zhou-model-playground", "training-viewer", "collective-outlier-lab",
        "zhou-schedule-designer", "guess-parameter", "ppc-detective",
    }
    pages = (SITE / "evolution.html").read_text() + (SITE / "sbi.html").read_text()
    assert all(f'id="{name}"' in pages for name in found)


def test_zhou_hashes_and_asset_contract():
    manifest = _json(DATA / "zhou_manifest.json")
    assert manifest["n_masks"] == 4096
    assert manifest["odd_passages"] == [0, 1, 3, 5, 7, 9, 11]
    assert manifest["even_passages"] == [0, 2, 4, 6, 8, 10, 12]
    assert all(row["passed"] for row in manifest["notebook_odd_even_validation"].values())
    for i, expected in enumerate(manifest["model_hashes"]):
        model = ROOT / "zhou_npe_models" / f"robust_npe_seed_{i}.pkl"
        if model.exists():
            assert hashlib.sha256(model.read_bytes()).hexdigest() == expected
    assert [row["n_sims"] for row in manifest["training_summaries"]] == [800_000] * 3
    assert [row["n_independent_trajectories"] for row in manifest["training_summaries"]] == [200_000] * 3
    draws = np.fromfile(DATA / "zhou_draws.f32", dtype="<f4").reshape(manifest["draw_shape"])
    qs = np.fromfile(DATA / "zhou_quantiles.f32", dtype="<f4").reshape(manifest["quantile_shape"])
    assert np.isfinite(draws).all() and np.isfinite(qs).all()
    assert np.all(qs[:, :, 0] <= qs[:, :, 1]) and np.all(qs[:, :, 1] <= qs[:, :, 2])


def _python_zhou(theta, p0, generations=120):
    p = np.asarray(p0, float); out = [p.copy()]
    mu_wt, mu_loh = 10 ** theta[0], 10 ** theta[3]
    matrix = np.array([[1 - mu_wt - mu_loh, 0, 0], [mu_wt, 1, 0], [mu_loh, 0, 1]])
    growth = np.diag([theta[1], 1, theta[2]]) @ matrix
    for _ in range(generations):
        p = growth @ p; p /= p.sum(); out.append(p.copy())
    return np.asarray(out)


def _python_avecilla(theta, generations=120):
    delta_c, delta_b = 10 ** theta[0], 10 ** theta[1]
    fitness = np.array([1, 1 + theta[2], 1 + theta[3]], float)
    p = np.array([1, 0, 0], float); out = [p.copy()]
    for _ in range(generations):
        selected = p * fitness
        p = np.array([
            selected[0] * (1 - delta_c - delta_b),
            selected[1] + selected[0] * delta_c,
            selected[2] + selected[0] * delta_b,
        ])
        p /= p.sum(); out.append(p.copy())
    return np.asarray(out)


def _python_chuong(theta, generations=(8, 21, 29, 37, 50, 58, 66, 79, 87, 95, 108, 116)):
    log_s, log_m, log_p0 = theta
    s, m, p0 = 10 ** np.array([log_s, log_m, log_p0])
    fitness = np.array([1, 1 + s, 1 + s, 1.001], float)
    p = np.array([1 - p0, 0, p0, 0], float); out = []
    for g in range(max(generations) + 1):
        if g in generations: out.append(p[1])
        selected = p * fitness
        p = np.array([
            selected[0] * (1 - m - 1e-5),
            selected[1] + selected[0] * m,
            selected[2],
            selected[3] + selected[0] * 1e-5,
        ])
        p /= p.sum()
    return np.asarray(out)


def test_javascript_scientific_kernel_matches_python_and_seed_is_reproducible():
    result = subprocess.run(["node", str(SITE / "tests/science_checks.cjs")], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    cases = [([-4, .96, .99, -4.4], [.99, .0075, .0025]), ([-3.2, 1.01, .94, -5.1], [.8, .15, .05]), ([-6, .9, 1.04, -3], [1, 0, 0])]
    for actual, (theta, p0) in zip(payload["output"], cases):
        np.testing.assert_allclose(actual, _python_zhou(theta, p0), rtol=0, atol=2e-15)
    np.testing.assert_allclose(payload["avecilla"], _python_avecilla([-4.2, -5, .07, .001]), rtol=0, atol=2e-15)
    np.testing.assert_allclose(payload["chuong"], _python_chuong([-.74, -4.84, -4.32]), rtol=0, atol=2e-15)
    expected_rmse = np.sqrt(np.mean((np.array([-.8, -4.7, -4.4]) - np.array([-.74, -4.84, -4.32])) ** 2))
    assert abs(payload["score"]["rmse"] - expected_rmse) < 1e-14
    assert abs(payload["score"]["score"] - 100 / (1 + expected_rmse)) < 1e-12
    assert payload["reproducible"]
    assert payload["odd"] == [0, 1, 3, 5, 7, 9, 11]
    assert payload["even"] == [0, 2, 4, 6, 8, 10, 12]
    lab = _json(DATA / "collective_lab.json")
    expected = np.sum(np.asarray(lab["replicate_log_posteriors"])[[0, 1, 2, 3]], axis=0) - 3 * np.asarray(lab["prior_log"])
    np.testing.assert_allclose(payload["collective"], expected, atol=1e-12)
    np.testing.assert_allclose(payload["robustLoose"]["standard"], expected, atol=1e-12)
    np.testing.assert_allclose(payload["robustTight"]["standard"], expected, atol=1e-12)
    raw = np.asarray(lab["replicate_log_posteriors"])[[0, 1, 2, 3]]
    expected_robust = np.maximum(raw, -10).sum(axis=0) - 3 * np.asarray(lab["prior_log"])
    np.testing.assert_allclose(payload["robustLoose"]["robust"], expected_robust, atol=1e-12)
    assert -4 < payload["estimatedLogEpsilon"] < -1
    assert abs(payload["estimatedLogEpsilon"] - payload["fineGridLogEpsilon"]) < .015
    assert payload["epsilonReproducible"]
    np.testing.assert_allclose(payload["neutralContaminantMean"], lab["truth_theta"], atol=1e-12)
    np.testing.assert_allclose(payload["fullContaminantMean"], lab["posterior_means"][lab["contaminated_index"]], atol=1e-12)
    np.testing.assert_allclose(payload["jointEstimated"]["standard"], payload["jointFixed"]["standard"], atol=1e-12)
    assert not np.allclose(payload["jointEstimated"]["standard"], payload["jointEstimated"]["robust"])
    grid = np.asarray(lab["grid"])
    def marginal_mean(log_values):
        values = np.exp(np.asarray(log_values) - np.max(log_values))
        return np.trapz(grid * values, grid) / np.trapz(values, grid)
    standard_mean = marginal_mean(payload["jointEstimated"]["standard"])
    robust_mean = marginal_mean(payload["jointEstimated"]["robust"])
    fine_robust_mean = marginal_mean(payload["jointFineGrid"]["robust"])
    clean_mean = marginal_mean(payload["jointClean"]["standard"])
    assert abs(standard_mean - lab["truth"]) > .06
    assert abs(robust_mean - lab["truth"]) < .035
    assert abs(robust_mean - fine_robust_mean) < .003
    assert abs(clean_mean - lab["truth"]) < .025
    low_standard = marginal_mean(payload["jointLowContamination"]["standard"])
    high_standard = marginal_mean(payload["jointHighContamination"]["standard"])
    low_robust = marginal_mean(payload["jointLowContamination"]["robust"])
    high_robust = marginal_mean(payload["jointHighContamination"]["robust"])
    assert abs(high_standard - low_standard) > .08
    assert abs(high_robust - low_robust) < .015


def test_training_snapshots_and_ppc_quantiles():
    data = _json(DATA / "training_viewer.json")
    assert data["pedagogical_not_production"] is True
    assert data["epochs"] == 100 and data["snapshot_epochs"] == sorted(data["snapshot_epochs"])
    assert data["snapshot_epochs"][0] == 0 and data["snapshot_epochs"][-1] == 100
    for snapshot in data["snapshots"]:
        lo, med, hi = map(np.asarray, (snapshot["ppc_q05"], snapshot["ppc_median"], snapshot["ppc_q95"]))
        assert np.all(lo <= med) and np.all(med <= hi)


def test_static_pages_are_subpath_safe_and_assets_exist():
    for page in (SITE / "index.html", SITE / "evolution.html", SITE / "sbi.html"):
        text = page.read_text()
        assert not re.search(r'''(?:src|href)=["']/''', text)
        assert "<noscript>" in text or page.name == "index.html"
        for ref in re.findall(r'''(?:src|href)="([^"#]+)"''', text):
            if "://" not in ref and not ref.startswith("mailto:"):
                asset_path = ref.split("?", 1)[0]
                assert (SITE / asset_path).exists(), f"broken link {ref} in {page.name}"
    assert not list(SITE.rglob("*.pkl"))
    assert not list(SITE.rglob("*.pickle"))


def test_assessment_is_versioned_private_by_design_and_recomputable():
    assessment = SITE / "assessment"
    bank_path = assessment / "questions" / "v1.0.0.json"
    bank = _json(bank_path)
    meta = _json(assessment / "build-meta.json")
    assert bank["schema_version"] == "1.0" and bank["assessment_version"] == "1.0.0"
    assert len(bank["knowledge_questions"]) == 6
    assert len(bank["confidence_questions"]) == 2
    assert len(bank["post_evaluation_questions"]) == 2
    assert len({question["question_id"] for question in bank["knowledge_questions"]}) == 6
    assert meta["assessment_version"] == bank["assessment_version"]
    assert meta["question_bank_sha256"] == hashlib.sha256(bank_path.read_bytes()).hexdigest()
    assert meta["workshop_git_sha"] == "development" or re.fullmatch(r"[0-9a-f]{40}", meta["workshop_git_sha"])

    index = (SITE / "index.html").read_text()
    sbi = (SITE / "sbi.html").read_text()
    page = (assessment / "index.html").read_text()
    script = (assessment / "app.js").read_text()
    config = (assessment / "config.js").read_text()
    backend = (assessment / "backend.js").read_text()
    migration = (assessment / "supabase" / "001_assessment_events.sql").read_text().lower()
    assert 'href="assessment/?phase=pre"' in index
    assert "assessment-pre-qr.svg" in index and "workshop-qr.svg" in index
    assert 'href="assessment/?phase=post"' in sbi and "assessment-post-qr.svg" in sbi
    assert "One last experiment" in sbi and "Let’s see what changed." in sbi
    assert 'href="../evolution.html"' in page and "<noscript>" in page
    assert all((assessment / "assets" / name).exists() for name in (
        "nyu-official-seal.svg", "umn-official-logo.svg", "tau-official-logo.png", "lone-wolf.svg"
    ))
    assert all(token in script for token in (
        "question_id", "question_revision", "variant_id", "displayed_order", "response", "duration_ms",
        "assessment_schema_version", "question_bank_hash", "workshop_git_sha", "consent_version",
        "crypto.randomUUID"
    ))
    assert "evoSbiWorkshopParticipantId" in config
    assert all(forbidden not in script for forbidden in (
        "navigator.userAgent", "screen.width", "document.referrer", "Intl.DateTimeFormat", "geolocation.getCurrentPosition"
    ))
    assert "localstorage" in backend.lower() and "return=minimal" in backend.lower()
    assert "grant insert" in migration and "to anon" in migration
    assert "grant select" not in migration and "grant update" not in migration and "grant delete" not in migration
    assert "enable row level security" in migration and "force row level security" in migration
    subprocess.run(["node", "--check", str(assessment / "app.js")], check=True)
    subprocess.run(["node", "--check", str(assessment / "backend.js")], check=True)

    module_path = assessment / "analysis" / "score_assessment.py"
    spec = importlib.util.spec_from_file_location("score_assessment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    correct = [{
        "question_id": question["question_id"], "response": question["correct_response"]
    } for question in bank["knowledge_questions"]]
    confidence = [{"question_id": question["question_id"], "response": 3} for question in bank["confidence_questions"]]
    records = []
    for phase, suffix in (("pre", "1"), ("post", "2")):
        records.append({
            "id": "event-" + suffix, "participant_id": "anonymous", "received_at": f"2026-09-10T08:00:0{suffix}Z",
            "phase": phase, "event_type": "completed", "venue": "NYU", "assessment_version": "1.0.0",
            "payload": {"pairing_code": "WOLF-TEST-CODE", "answers": {"knowledge": correct, "confidence": confidence}}
        })
    summary = module.summarize(records, bank, True)
    assert summary["matched_n"] == 1
    assert summary["knowledge_score_0_to_6"]["pre_mean"] == 6
    assert summary["knowledge_score_0_to_6"]["post_mean"] == 6
    assert summary["by_venue"]["NYU"]["post"]["n"] == 1


def test_dedicated_interactive_poster_route_and_components():
    page = SITE / "poster" / "index.html"
    text = page.read_text()
    script = (page.parent / "poster.js").read_text()
    assert page.exists()
    assert all(f'id="{section}"' in text for section in ("sbi", "abc", "extensions", "cases"))
    assert all(f'id="{control}"' in text for control in (
        "story-play", "abc-run", "abc-progress", "abc-trajectories",
        "abc-posterior", "case-experience", "case-play",
    ))
    assert all(f'data-case="{case}"' in text for case in ("candida", "ms2"))
    assert "Run rejection ABC" in text and "Choose a microbial case study" in text
    assert "why-sbi-argument" in text and "SBI changes the question" in text
    assert "The forward story is clear. The inverse is not." in text
    assert 'href="#sbi-story"' in text and "Play the three steps" in text
    assert text.count('data-story-step="') == 3
    assert "Draw + simulate" in text and "Infer posterior" in text
    assert "collective_extension.png" in text and "flexible_extension.png" in text
    assert "Combine replicate evidence without letting one outlier dominate" in text
    assert "Reuse one estimator across supported sampling schedules" in text
    assert "Write the biological rules" not in script and "Play the four steps" not in script
    assert "requestAnimationFrame" in script and "caseStudies" in script
    assert not re.search(r'''(?:src|href)=["']/''', text)
    for ref in re.findall(r'''(?:src|href)="([^"#]+)"''', text):
        if "://" not in ref and not ref.startswith("mailto:"):
            assert (page.parent / ref.split("?", 1)[0]).resolve().exists(), f"broken poster link {ref}"


def test_scripts_and_stylesheets_are_content_versioned():
    for page in (SITE / "index.html", SITE / "evolution.html", SITE / "sbi.html"):
        text = page.read_text()
        refs = re.findall(r'''(?:src|href)="((?:css|js)/[^"?]+)\?v=([0-9a-f]{12})"''', text)
        assert refs, f"no versioned assets in {page.name}"
        for relative_path, version in refs:
            assert hashlib.sha256((SITE / relative_path).read_bytes()).hexdigest().startswith(version)


def test_equations_fallbacks_accessibility_and_no_obsolete_20k_claim():
    text = (SITE / "evolution.html").read_text() + (SITE / "sbi.html").read_text()
    assert "<math" in text and "aria-label=" in text
    assert text.count("class=\"static-fallback\"") == 9
    assert ".static-fallback{display:block}" in text
    assert "20k" not in text.lower() and "20,000" not in text
    css = (SITE / "css/workshop.css").read_text()
    assert "prefers-reduced-motion" in css and "focus-visible" in css


def test_chapter_one_revision_contract():
    text = (SITE / "evolution.html").read_text()
    sbi = (SITE / "sbi.html").read_text()
    script = (SITE / "js" / "evolution.js").read_text()
    notebook = (ROOT / "evolution_simulators.ipynb").read_text()
    assert "Predict a three-genotype chemostat trajectory" in text
    assert 'id="evo-delta-c"' in text and 'id="evo-order-canvas"' in text
    assert "Fitness effects differ among newly formed CNVs" in text
    assert "100 / (1 + RMSE)" in sbi
    assert 'id="chuong-parameter-challenge"' not in text
    assert sbi.index('id="chuong-parameter-challenge"') < sbi.index('class="chapter-walkthrough"')
    assert text.index('id="zhou-model-playground"') > text.index('data-cell-id="2e99f96f"')
    assert text.index('data-cell-id="21748f7d"') < text.index('id="avecilla-model-builder"')
    assert text.index('id="avecilla-model-builder"') < text.index('id="effective-population-size"') < text.index('id="evolution-playground"')
    assert text.index('id="evolution-playground"') < text.index('data-cell-id="6ec896e6"')
    assert text.index('id="model-equivalence"') < text.index('data-cell-id="629428f4"')
    assert text.index('data-cell-id="104a1da1"') < text.index('data-cell-id="f9cb77d4"') < text.index('data-cell-id="97c3a42c"')
    assert text.index('id="chuong-equation-exercise"') < text.index('code-fill-exercise') < text.index('id="chuong-standing-variation"')
    assert 'data-model-force="mutation"' in text and 'data-model-force="selection"' in text and 'data-model-force="drift"' in text
    assert 'class="active" type="button" role="tab"' not in text
    assert 'data-force-panel="mutation" hidden' in text and 'show("mutation")' not in script
    assert "Mutation moves probability between states" in text and "Selection reweights reproductive contribution" in text
    assert 'id="chuong-standing-variation"' in text and "φ = 10⁻¹²" in text and "φ = 10⁻⁴" in text
    assert 'id="chuong-phi-play"' in text and 'id="chuong-phi" type="range"' in text
    assert "baseline = simulate(-12)" in script and '"φ=0 baseline"' in script and "observed (dashed)" in script
    assert 'id="chuong-equation-exercise"' in text and 'class="chuong-matrix" hidden' in text
    assert 'class="code-fill-list"' in text and text.count('data-code-answer=') == 4 and text.count('data-check-code-line') == 4
    assert "n = ____  # (1)" in text and "mutated = ____  # (2)" in text and "weighted = ____  # (3)" in text and "n = ____  # (4)" in text
    assert text.count('class="annotated-code-layout"') == 3 and "equivalent NumPy expressions are accepted" in text
    assert "Chuong chemostat data" in text and "simpler model" in text
    assert "incoming fresh" in text and "mutation flow" in text and "summation notation is unnecessary" in text
    assert "Zhou et al. model (Selmecki lab, UMN)" in text and 'id="model-equivalence"' in text
    assert "Lauer et al. (2018)" in text and "journal.pbio.3000069" in text
    assert "Motivation: repeated" in text and "finite populations <strong>sample</strong>" not in text
    assert 'class="paper-figure experimental-primer-figure"' in text
    assert "lauer-experimental-primer.jpg" in text and "fluorescent reporter linked to <em>GAP1</em>" in text
    assert text.index("Motivation: repeated") < text.index("lauer-experimental-primer.jpg") < text.index("The resulting trajectories")
    assert "Published fit</button>" in text and "Reference values</button>" in text
    assert 'id="zhou-model-play"' in text and "The plot starts empty" in text
    assert "The three-state Avecilla model misses the LTRΔ structure!" in text and "misses the early LTRΔ structure" not in text
    assert "Chuong&nbsp;WF" in text
    assert "Black-box stress test" not in text and "Can one selection coefficient describe the whole sweep?" in text
    assert "Let <math" in text and "fraction of cells carrying one label" in text and "t</mi><mo>&#x0003D;</mo><mn>900" in text
    assert 'id="chemostat-ne-simulator"' in text and 'id="chemostat-ne-run"' in text and 'id="chemostat-ne-calculation"' in text
    assert 'id="serial-ne-simulator"' in text and 'id="serial-ne-run"' in text and 'id="serial-ne-calculation"' in text
    assert "Published scale" in text and "Visible drift" in text and "bottleneck contribution" in text
    assert "function effectivePopulationSimulators()" in script and "p * (1 - p) / variance" in script and "generations / reciprocalSum" in script
    assert "more common among new CNVs" not in script
    assert "The two solid curves should not coincide" in text
    assert 'id="evolution-references"' in text
    assert all(label in text for label in ("Lauer et al. (2018)", "Avecilla et al. (2022)", "Chuong et al. (2025)", "De et al. (2025)"))
    assert "What does s mean?" in text and "fraction of newly formed CNVs" in text
    assert "recurring grammar" not in text and "The executable mechanism" not in text
    assert text.count("Model-fit preview") == 5
    assert "color=C['avecilla_wf']" in notebook
    assert re.search(r'src="assets/chapter/chuong-fit-orange\.png\?v=[0-9a-f]{12}"', text)
    assert "orange predictions, blue observations" in text
    assert "color=C['chuong']" in notebook
    assert "Error loading sheet" not in text and "NoneType" not in text
    assert "evo-presets" in text and "<span>CNV formation log₁₀(δ<sub>C</sub>)</span>" in text


def test_sbi_revision_contract():
    text = (SITE / "sbi.html").read_text()
    script = (SITE / "js/sbi.js").read_text()
    assert "Run rejection ABC" in text
    assert 'id="abc-quantile"' in text and 'id="abc-sims"' in text
    assert 'id="guess-s"' not in text and 'id="guess-m"' not in text
    assert all(f'<option value="{epsilon}"' in text for epsilon in ("0", "-10", "-100", "-1000"))
    assert 'id="zhou-summary"' not in text
    assert "zhou-flex-data-out" not in text and "zhou-flex-inference-out" not in text
    assert "10.1371/journal.pcbi.1014534" in text and "under review" not in text
    assert "P.legend" in script
    assert "Check my diagnosis" in text and "observed data" in script
    assert "stackrel" not in text and "stackrel" not in (ROOT / "SBI_tutorial.ipynb").read_text()
    assert "Standard collective" in text and "Robust collective" in text
    assert "What is evaluated—and what is sampled?" not in text
    assert 'value="auto:0.95" selected' in text
    assert 'id="abc-progress"' in text and "requestAnimationFrame" in script
    assert "In this chapter:" in text
    assert "Tavaré et al. (1997)" in text and "10.1093/genetics/145.2.505" in text
    assert 'class="bayes-components"' in text and "The simulator samples from the likelihood" in text
    assert 'class="amortization-note"' in text and 'class="intermediate-summary"' in text
    assert "Flexibility has a contract" not in text
    assert "Using Bayes' rule we can get:" in text and "removes the additional copies of the shared prior" in text
    assert "CC BY 4.0" not in text
    assert "Accepted simulations turn a distance threshold into parameter uncertainty" not in text
    assert "NPE returns a joint posterior after one conditioning step" not in text
    assert 'class="paper-figure abc-framework-figure"' in text and 'class="abc-framework-viewport"' in text and "abc-framework.png" in text
    assert ".abc-framework-viewport{overflow:hidden" in (SITE / "css" / "workshop.css").read_text()
    assert ".abc-framework-viewport img{position:static;display:block;width:100%;max-width:100%;height:auto" in (SITE / "css" / "workshop.css").read_text()
    assert '<details class="code-panel" open' not in text
    assert "Run ABC progressively" not in text and ">Run ABC</button>" in text
    assert 'const displayOrder = [1, 0, 2, 3]' in script
    assert '["w_Het", "μ_Het→WT", "w_LOH", "μ_Het→LOH"]' in script
    assert '$$(\'[data-schedule="zero"]\')[0].click()' in script
    assert "addEventListener(\"resize\", drawAbc); clearAbc();" in script


def test_landing_page_is_focused_on_the_two_lessons():
    home = (SITE / "index.html").read_text()
    favicon = (SITE / "assets/favicon.svg").read_text()
    assert "Simulation-based inference for experimental evolution" in home
    assert '<span>θ</span> SBI for experimental evolution' in home
    assert ">θ</text>" in favicon and "Drift &amp; Design" not in home
    assert 'id="workshop-mode"' not in home and "Workshop mode" not in home
    assert 'class="primer-journey"' not in home and "Foundations" not in home
    assert "75 min" not in home and "15 min" not in home and "60 min" not in home
    assert home.count("Click to open lesson") == 2
    assert "Scan once" not in home
    evolution = (SITE / "evolution.html").read_text()
    sbi = (SITE / "sbi.html").read_text()
    assert 'class="chapter-walkthrough"' not in evolution
    assert sbi.count('class="story-slide') == 4
    assert 'class="lesson-timing"' not in evolution + sbi


def test_presentation_revision_contract():
    home = (SITE / "index.html").read_text()
    evolution = (SITE / "evolution.html").read_text()
    sbi = (SITE / "sbi.html").read_text()
    script = (SITE / "js/sbi.js").read_text()

    assert '<a class="secondary-link" href="evolution.html">Chapter 01</a>' in home
    assert '<a class="secondary-link" href="sbi.html">Chapter 02</a>' in home
    assert "Open lesson →" not in home
    assert 'src="assets/workshop-qr.svg"' in home
    assert "https://nadavbennun1.github.io/evo-simulators-tutorial/" in home
    assert "Running on Google Colab" not in evolution
    assert len(re.findall(r'<code class="language-python">', evolution)) == 4
    assert len(re.findall(r'<code class="language-python">', sbi)) == 3
    assert evolution.count('class="force-code-grid"') == 0
    assert len(re.findall(r'<pre class="annotated-code(?: |")', evolution)) == 4
    assert evolution.count('class="code-fill-list"') == 1
    assert "Effective population size belongs to the life cycle" in evolution
    assert "Serial dilution: bottlenecks dominate the harmonic mean" in evolution
    assert "WF = ODE?" not in evolution and "Infer mutation rate via SBI?" not in evolution
    assert "Appendix: why the quick fits use perturbed parameters" in evolution
    assert "Results for ALLΔ" not in sbi and "Flexible Zhou NPE" not in sbi
    assert "collective-figure-1.png" in sbi and "collective-figure-4.png" in sbi
    assert "Fig. 1." in sbi and "Fig. 4." in sbi and "CC BY 4.0" not in sbi
    assert "paper-figure-wide" not in sbi
    assert "log₁₀ ε" in sbi and "Math.LN10" in script
    assert "const caseOrder = [3, 0, 4, 2, 1]" in script
    assert "Posterior prediction of CNV-lineage diversity" in sbi
    assert "Effective diversity" in sbi and "Shannon entropy" in sbi and "3.2" in sbi and "ARSΔ" in sbi
    assert "<h3>References</h3>" in sbi and "Short references" not in sbi
    assert "assets/chapter/chuong-diversity-figure-3b.jpg" in sbi
    assert 'loading="eager"' in sbi and 'class="diversity-rank"' in sbi
    assert 'class="inverse-process-diagram"' in sbi and 'id="inverse-example"' in sbi
    assert 'class="mini-posterior-density"' in sbi
    assert "typeset-story-equation" in sbi
    assert "Several mechanisms can match the sampled trajectory" in sbi
    assert "ten equally abundant lineages" in sbi and 'class="diversity-calculation"' in sbi
    assert "<strong>Important:</strong> inference is not finished" in sbi
    assert "∝" not in evolution + sbi
    assert "MathJax" in sbi and "mml-chtml.js" in sbi
    assert '<p><div class="math-scroll"' not in evolution + sbi
    for removed in (
        "The mystery cultures are deliberately shuffled",
        "Monte Carlo stabilization.",
        "This is the Avecilla three-genotype mechanism:",
        "This is an illustrative flexibility demonstration, not a coverage study.",
        "Chuong parameters",
        "Design a Zhou passage schedule",
        "PPC mismatch detective",
    ):
        assert removed not in evolution + sbi
    for doi in (
        "10.1371/journal.pbio.3001633", "10.7554/eLife.98934",
        "10.1101/2025.07.21.665951", "10.1371/journal.pcbi.1014534",
        "10.21105/joss.02505",
    ):
        assert doi in evolution + sbi


def test_javascript_syntax():
    scripts = list((SITE / "js").glob("*.js")) + [SITE / "poster" / "poster.js"]
    for script in scripts:
        subprocess.run(["node", "--check", str(script)], check=True, capture_output=True)
