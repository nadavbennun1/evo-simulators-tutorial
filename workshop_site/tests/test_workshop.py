from __future__ import annotations

import hashlib
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


def test_interaction_manifest_ids_exist_and_all_six_stations_are_built():
    manifest = _json(SITE / "interaction_manifest.json")
    found = []
    for name, placements in manifest.items():
        ids = {c.get("id") for c in nbformat.read(ROOT / name, as_version=4).cells}
        assert set(placements) <= ids
        for stations in placements.values():
            found.extend(stations)
    assert set(found) == {"evolution-playground", "training-viewer", "collective-outlier-lab", "zhou-schedule-designer", "guess-parameter", "ppc-detective"}
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


def test_javascript_scientific_kernel_matches_python_and_seed_is_reproducible():
    result = subprocess.run(["node", str(SITE / "tests/science_checks.cjs")], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    cases = [([-4, .96, .99, -4.4], [.99, .0075, .0025]), ([-3.2, 1.01, .94, -5.1], [.8, .15, .05]), ([-6, .9, 1.04, -3], [1, 0, 0])]
    for actual, (theta, p0) in zip(payload["output"], cases):
        np.testing.assert_allclose(actual, _python_zhou(theta, p0), rtol=0, atol=2e-15)
    assert payload["reproducible"]
    assert payload["odd"] == [0, 1, 3, 5, 7, 9, 11]
    assert payload["even"] == [0, 2, 4, 6, 8, 10, 12]
    lab = _json(DATA / "collective_lab.json")
    expected = np.sum(np.asarray(lab["replicate_log_posteriors"])[[0, 1, 2, 3]], axis=0) - 3 * np.asarray(lab["prior_log"])
    np.testing.assert_allclose(payload["collective"], expected, atol=1e-12)


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
                assert (SITE / ref).exists(), f"broken link {ref} in {page.name}"
    assert not list(SITE.rglob("*.pkl"))
    assert not list(SITE.rglob("*.pickle"))


def test_equations_fallbacks_accessibility_and_no_obsolete_20k_claim():
    text = (SITE / "evolution.html").read_text() + (SITE / "sbi.html").read_text()
    assert "<math" in text and "aria-label=" in text
    assert text.count("class=\"static-fallback\"") == 6
    assert ".static-fallback{display:block}" in text
    assert "20k" not in text.lower() and "20,000" not in text
    css = (SITE / "css/workshop.css").read_text()
    assert "prefers-reduced-motion" in css and "focus-visible" in css


def test_javascript_syntax():
    for script in (SITE / "js").glob("*.js"):
        subprocess.run(["node", "--check", str(script)], check=True, capture_output=True)
