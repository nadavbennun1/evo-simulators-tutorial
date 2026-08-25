#!/usr/bin/env python3
"""Build the static evolution/SBI workshop and its deterministic derived assets.

The browser never receives an NPE checkpoint. ``--scientific-assets`` performs the
offline model work; ordinary builds verify and reuse those versioned assets.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import os
import pickle
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown
import nbformat
import numpy as np
from latex2mathml.converter import convert as latex_to_mathml

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(__file__).resolve().parent
DATA = SITE / "data"
ASSETS = SITE / "assets"
NOTEBOOK_ASSETS = ASSETS / "notebook"
FALLBACK = ASSETS / "fallback"
CHAPTER_ASSETS = ASSETS / "chapter"
EVODESIGN_SRC = ROOT.parent / "evodesign" / "src"
NOTEBOOKS = {
    "evolution": ROOT / "evolution_simulators.ipynb",
    "sbi": ROOT / "SBI_tutorial.ipynb",
}
SEED = 20260825


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versioned_asset(relative_path: str) -> str:
    """Return a subpath-safe asset URL whose query changes with its contents."""
    return f"{relative_path}?v={sha256(SITE / relative_path)[:12]}"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def math_to_html(source: str) -> str:
    """Convert TeX spans to MathML before Markdown consumes punctuation."""
    placeholders: list[str] = []

    def repl(match: re.Match[str]) -> str:
        tex = match.group(2).strip()
        display = bool(match.group(1))
        try:
            rendered = latex_to_mathml(tex, display="block" if display else "inline")
            if display:
                rendered = f'<div class="math-scroll" role="math">{rendered}</div>'
        except Exception:
            rendered = f'<code class="math-source">{html.escape(tex)}</code>'
        token = f"MATHPLACEHOLDER{len(placeholders)}X"
        placeholders.append(rendered)
        return token

    # Display first, then conservative inline spans (currency does not occur here).
    source = re.sub(r"(\$\$)(.*?)(\$\$)", lambda m: repl(_MathMatch(True, m.group(2))), source, flags=re.S)
    source = re.sub(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", lambda m: repl(_MathMatch(False, m.group(1))), source)
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    for idx, value in enumerate(placeholders):
        rendered = rendered.replace(f"MATHPLACEHOLDER{idx}X", value)
    return rendered


class _MathMatch:
    def __init__(self, display: bool, body: str):
        self.display, self.body = display, body
    def group(self, index: int) -> str:
        if index == 1:
            return "$$" if self.display else ""
        return self.body


def output_html(output: dict, stem: str, output_index: int) -> str:
    data = output.get("data", {})
    pieces: list[str] = []
    if "image/png" in data:
        if stem == "66cce2fa":
            name = "chuong-fit-orange.png"
            if not (CHAPTER_ASSETS / name).exists():
                raise RuntimeError(f"missing revised Chuong fit figure: assets/chapter/{name}")
            pieces.append(f'<figure class="notebook-figure"><img loading="lazy" src="assets/chapter/{name}" alt="Chuong Wright-Fisher simulations and LTR data, both shown in the Avecilla orange for comparison"><figcaption>Chuong WF fit · orange matches the Avecilla fit</figcaption></figure>')
        else:
            name = f"{stem}-out-{output_index}.png"
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            (NOTEBOOK_ASSETS / name).write_bytes(base64.b64decode(raw))
            pieces.append(f'<figure class="notebook-figure"><img loading="lazy" src="assets/notebook/{name}" alt="Stored notebook figure from cell {stem}"><figcaption>Stored notebook output · cell {stem}</figcaption></figure>')
    text = output.get("text")
    if text:
        text = "".join(text) if isinstance(text, list) else str(text)
        if text.strip():
            pieces.append(f'<pre class="cell-output" aria-label="Stored notebook output">{html.escape(text.rstrip())}</pre>')
    if output.get("output_type") == "error":
        value = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        pieces.append(f'<pre class="cell-output error-output">{html.escape(value)}</pre>')
    # text/plain from rich display is useful unless it is just a Figure/widget repr.
    plain = data.get("text/plain")
    if plain:
        plain = "".join(plain) if isinstance(plain, list) else str(plain)
        if plain.strip() and not plain.lstrip().startswith(("<Figure", "<IPython.core.display.Image", "Drawing ")):
            pieces.append(f'<pre class="cell-output">{html.escape(plain.rstrip())}</pre>')
    return "".join(pieces)


def render_notebook(key: str, interactions: dict[str, list[str]]) -> tuple[str, list[dict]]:
    nb = nbformat.read(NOTEBOOKS[key], as_version=4)
    blocks, coverage = [], []
    for index, cell in enumerate(nb.cells):
        cid = cell.get("id") or f"cell-{index}"
        source = cell.source or ""
        if not source.strip():
            coverage.append({"index": index, "cell_id": cid, "type": cell.cell_type,
                             "status": "excluded", "reason": "empty cell"})
            continue
        anchor = f'cell-{cid}'
        if cell.cell_type == "markdown":
            body = math_to_html(source)
            blocks.append(f'<section class="lesson-cell prose-cell" id="{anchor}" data-cell-id="{cid}">{body}</section>')
            status = "included"
        else:
            lang = "python"
            code = html.escape(source)
            output = "".join(output_html(dict(out), cid, j) for j, out in enumerate(cell.get("outputs", [])))
            short = len(source.splitlines()) <= 9
            if short:
                code_block = f'<div class="code-panel visible-code"><div class="code-toolbar"><span>Python · notebook cell {index}</span><button class="copy-code" type="button">Copy</button></div><pre><code class="language-python">{code}</code></pre></div>'
                status = "included"
            else:
                code_block = f'<details class="code-panel"><summary>Show the code <span>Python · cell {index}</span></summary><div class="code-toolbar"><span>Reproducible source</span><button class="copy-code" type="button">Copy</button></div><pre><code class="language-python">{code}</code></pre></details>'
                status = "deliberately_collapsed"
            blocks.append(f'<section class="lesson-cell code-cell" id="{anchor}" data-cell-id="{cid}">{code_block}{output}</section>')
        coverage.append({"index": index, "cell_id": cid, "type": cell.cell_type,
                         "status": status, "reason": "long code is progressively disclosed" if status == "deliberately_collapsed" else ""})
        for interaction in interactions.get(cid, []):
            blocks.append(station_markup(interaction))
    return "\n".join(blocks), coverage


def station_markup(name: str) -> str:
    fallback = f'assets/fallback/{name}.png'
    common_start = f'''<section class="station answer-gated" id="{name}" data-station="{name}">
      <div class="station-kicker">Interactive station</div>'''
    common_end = f'''<div class="static-fallback"><img src="{fallback}" alt="Representative static result for {name.replace('-', ' ')}"><p>This representative result remains available when scripting is unavailable.</p></div>
      <noscript><p class="noscript">JavaScript is off; use the static result and conclusion above.</p></noscript></section>'''
    if name == "evolution-playground":
        body = '''<h2>Predict the Avecilla evolutionary trajectory</h2><p class="prediction">Question: will the GAP1 CNV, another beneficial lineage, or drift dominate the chemostat population?</p>
        <div class="preset-row"><button data-evo-preset="fit">Avecilla fit</button><button data-evo-preset="cnv">CNV sweep</button><button data-evo-preset="competing">Competing beneficial</button><button data-evo-preset="drift">Small population</button></div>
        <div class="interactive-grid"><form class="controls" id="evo-controls">
          <label>CNV formation log₁₀(δ<sub>C</sub>) <output id="evo-delta-c-label"></output><input id="evo-delta-c" type="range" min="-7" max="-2" step="0.05" value="-4.2"></label>
          <label>Other-beneficial log₁₀(δ<sub>B</sub>) <output id="evo-delta-b-label"></output><input id="evo-delta-b" type="range" min="-7" max="-2" step="0.05" value="-5"></label>
          <label>CNV advantage s<sub>C</sub> <output id="evo-s-c-label"></output><input id="evo-s-c" type="range" min="0" max="0.14" step="0.002" value="0.07"></label>
          <label>Other-beneficial advantage s<sub>B</sub> <output id="evo-s-b-label"></output><input id="evo-s-b" type="range" min="0" max="0.14" step="0.002" value="0.001"></label>
          <label>Generations <output id="evo-duration-label"></output><input id="evo-duration" type="range" min="20" max="140" step="5" value="120"></label>
          <label>Effective population Nₑ <select id="evo-ne"><option>1000</option><option>10000</option><option>100000</option><option>1000000</option><option selected>330000000</option></select></label>
          <label>Replicate trajectories <input id="evo-reps" type="number" min="1" max="24" value="8"></label>
          <label>Seed <input id="evo-seed" type="number" min="0" value="20260825"></label>
          <div class="button-row"><button id="evo-play" type="button">Play</button><button class="reset" type="reset">Reset</button></div></form>
          <div class="viz"><canvas id="evo-canvas" width="760" height="440" aria-label="Population frequency trajectories"></canvas><p class="plot-summary" id="evo-summary" aria-live="polite"></p><div class="composition" id="evo-composition"></div></div></div>
        <div class="what-changed"><strong>What changed?</strong> <span id="evo-driver"></span></div>'''
    elif name == "dfe-example":
        body = '''<h2>An <em>s</em>-DFE at a glance</h2><p class="prediction">The x-axis is the selection coefficient carried by a newly formed CNV; height is its relative probability under a gamma-shaped distribution of fitness effects.</p>
        <div class="interactive-grid"><form class="controls" id="dfe-controls">
          <label>Mean effect s̄ <output id="dfe-mean-label"></output><input id="dfe-mean" type="range" min="0.01" max="0.09" step="0.0025" value="0.045"></label>
          <label>Gamma shape <output id="dfe-shape-label"></output><input id="dfe-shape" type="range" min="0.7" max="5" step="0.1" value="2"></label>
          <button type="reset">Reset</button></form>
          <div class="viz"><canvas id="dfe-canvas" width="760" height="400" aria-label="Gamma-shaped distribution of selection coefficients"></canvas><p id="dfe-summary" class="plot-summary" aria-live="polite"></p></div></div>
        <div class="what-changed"><strong>Read the plot.</strong> <span id="dfe-change"></span></div>'''
    elif name == "chuong-parameter-challenge":
        body = '''<h2>Infer the hidden Chuong parameters</h2><p class="prediction">A new noisy CNV-frequency observation is generated each round. Guess the three log₁₀ parameters, then score your parameter RMSE.</p>
        <div class="interactive-grid"><form class="controls" id="chuong-challenge-controls">
          <label>log₁₀(s) <output id="chuong-guess-s-label"></output><input id="chuong-guess-s" type="range" min="-1.3" max="-0.45" step="0.01" value="-0.8"></label>
          <label>log₁₀(δ) <output id="chuong-guess-m-label"></output><input id="chuong-guess-m" type="range" min="-6" max="-3.8" step="0.02" value="-4.8"></label>
          <label>log₁₀(φ) <output id="chuong-guess-p0-label"></output><input id="chuong-guess-p0" type="range" min="-7" max="-3" step="0.02" value="-4.5"></label>
          <div class="button-row"><button id="chuong-score" type="button">Score guess</button><button id="chuong-new" type="button">New observation</button><button type="reset">Reset guess</button></div></form>
          <div class="viz"><canvas id="chuong-challenge-canvas" width="760" height="430" aria-label="Noisy Chuong observation and trajectory implied by the current parameter guess"></canvas><p id="chuong-challenge-summary" class="plot-summary" aria-live="polite"></p><div id="chuong-score-card" class="score-card" aria-live="polite"></div></div></div>
        <div class="what-changed"><strong>Score.</strong> RMSE is computed directly across the three log₁₀ parameters; points = 100 / (1 + RMSE). A perfect guess earns 100.</div>'''
    elif name == "zhou-model-playground":
        body = '''<h2>Explore the Zhou chromosome-loss model</h2><p class="prediction">Prediction: does the trisomic population resolve mainly through euploid recovery, LOH, or a fitness-driven mixture?</p>
        <div class="preset-row"><button data-zhou-model-preset="fit">Zhou fit</button><button data-zhou-model-preset="wt">WT route</button><button data-zhou-model-preset="loh">LOH route</button><button data-zhou-model-preset="fitness">Fitness reversal</button></div>
        <div class="interactive-grid"><form class="controls" id="zhou-model-controls">
          <label>Tri → WT log₁₀ rate <output id="zhou-model-mu-wt-label"></output><input id="zhou-model-mu-wt" type="range" min="-6" max="-2.5" step="0.05" value="-3.47"></label>
          <label>Tri → LOH log₁₀ rate <output id="zhou-model-mu-loh-label"></output><input id="zhou-model-mu-loh" type="range" min="-6" max="-2.5" step="0.05" value="-3.28"></label>
          <label>Trisomic fitness <output id="zhou-model-w-tri-label"></output><input id="zhou-model-w-tri" type="range" min="0.85" max="1.05" step="0.002" value="0.92"></label>
          <label>LOH fitness <output id="zhou-model-w-loh-label"></output><input id="zhou-model-w-loh" type="range" min="0.85" max="1.05" step="0.002" value="0.986"></label>
          <label>Initial trisomic fraction <output id="zhou-model-p0-label"></output><input id="zhou-model-p0" type="range" min="0.7" max="1" step="0.01" value="0.99"></label>
          <button type="reset">Reset</button></form>
          <div class="viz"><canvas id="zhou-model-canvas" width="760" height="430" aria-label="Interactive Zhou trisomic, wild-type, and LOH trajectories"></canvas><p id="zhou-model-summary" class="plot-summary" aria-live="polite"></p><div class="composition" id="zhou-model-composition"></div></div></div>
        <div class="what-changed"><strong>What changed?</strong> <span id="zhou-model-change"></span></div>'''
    elif name == "training-viewer":
        body = '''<h2>100 epochs of learning</h2><p class="prediction">Prediction: when does lower validation loss begin to produce useful posterior predictions?</p>
        <div class="preset-row"><button data-epoch="0">Untrained</button><button data-epoch="5">Early</button><button data-epoch="50">Mid-training</button><button data-epoch="best">Best validation</button><button data-epoch="100">Final</button></div>
        <label class="wide-control">Epoch <output id="epoch-label">0</output><input id="epoch-slider" type="range" min="0" max="100" value="0"></label>
        <div class="button-row"><button id="epoch-play" type="button">Play</button><button id="epoch-reset" type="button">Reset</button></div>
        <p class="snap-note" id="epoch-snap">Posterior snapshots snap to genuine stored checkpoints.</p>
        <div class="plot-pair training-plots"><canvas id="loss-canvas" width="660" height="340" aria-label="Training and validation loss across 100 epochs"></canvas><canvas id="training-posterior-canvas" width="660" height="340" aria-label="Posterior marginals at selected checkpoint"></canvas><canvas id="training-ppc-canvas" width="660" height="340" aria-label="Posterior predictive check at selected checkpoint"></canvas></div>
        <p class="plot-summary" id="training-summary" aria-live="polite"></p>
        <div class="what-changed"><strong>What changed?</strong> Lower loss improves the learned conditional density on average; a useful PPC is related evidence, not a calibration guarantee.</div>'''
    elif name == "collective-outlier-lab":
        body = '''<h2>Collective posterior outlier laboratory</h2><p class="prediction">Question: which replicate has the most leverage on the shared estimate?</p>
        <div class="preset-row"><button data-coll-select="all">Select all</button><button data-coll-select="clean">Clean only</button><button data-coll-select="outliers">Outliers only</button><button id="coll-loo">Leave one out</button></div>
        <div class="interactive-grid"><div class="controls"><fieldset id="replicate-checks"><legend>Replicates entering sensitivity analysis</legend></fieldset><label>Investigate <select id="coll-investigate"></select></label><label>Contamination strength <output id="contam-label">1.0×</output><input id="contam-strength" type="range" min="0" max="1.5" step="0.1" value="1"></label><button id="coll-reset" type="button">Reset</button></div>
        <div class="viz"><canvas id="collective-trajectory-canvas" width="760" height="310" aria-label="Selected replicate trajectories"></canvas><canvas id="collective-posterior-canvas" width="760" height="310" aria-label="Individual, naive product, and prior-adjusted collective posterior"></canvas><p id="collective-summary" class="plot-summary" aria-live="polite"></p></div></div>
        <div class="what-changed"><strong>Scientific caveat.</strong> Exclusion here is sensitivity analysis, not automatic justification for discarding data. The browser uses Σ log pᵢ − (r−1) log p(prior), exactly as derived above.</div>'''
    elif name == "zhou-schedule-designer":
        body = '''<h2>Design a Zhou passage schedule</h2><p class="prediction">Prediction: which passages constrain rates, and which constrain relative fitness?</p>
        <div class="preset-row"><button data-schedule="odd">Odd passages</button><button data-schedule="even">Even passages</button><button data-schedule="early">Early only</button><button data-schedule="late">Late only</button><button data-schedule="sparse">Sparse</button><button data-schedule="full">Full schedule</button><button data-schedule="zero">Passage 0 only</button></div>
        <div class="passage-grid" id="passage-grid"></div><label class="inline-toggle"><input id="reveal-withheld" type="checkbox" checked> Reveal withheld observations</label>
        <div class="plot-pair"><canvas id="zhou-trajectory-canvas" width="720" height="380" aria-label="Latent trajectory and selected or withheld passage observations"></canvas><canvas id="zhou-posterior-canvas" width="720" height="380" aria-label="Four Zhou posterior marginals for the selected schedule"></canvas></div>
        <div id="zhou-summary" class="parameter-summary" aria-live="polite"></div><canvas id="zhou-ppc-canvas" width="1100" height="340" aria-label="Posterior predictive intervals at observed and withheld passages"></canvas>
        <div class="button-row"><button id="zhou-reset" type="button">Reset</button></div>
        <div class="what-changed"><strong>What changed?</strong> <span id="zhou-change"></span> This is an illustrative flexibility demonstration, not a coverage study.</div>'''
    elif name == "guess-parameter":
        body = '''<h2>Guess the parameters—then reveal uncertainty</h2><p class="prediction">Inspect the frozen trajectory. Choose plausible values; several combinations can look alike.</p>
        <div class="interactive-grid"><form class="controls" id="guess-controls"><label>log₁₀(s) <output id="guess-s-label"></output><input id="guess-s" type="range" min="-2" max="0" step="0.05" value="-1"></label><label>log₁₀(δ) <output id="guess-m-label"></output><input id="guess-m" type="range" min="-7" max="-2" step="0.05" value="-4.5"></label><label>log₁₀(φ) <output id="guess-p0-label"></output><input id="guess-p0" type="range" min="-8" max="-2" step="0.05" value="-5"></label><button id="guess-reveal" type="button">Reveal posterior</button><button id="guess-new" type="button">Next fixed example</button><button type="reset">Reset</button></form><div class="viz"><canvas id="guess-canvas" width="760" height="430" aria-label="Hidden-parameter trajectory and posterior reveal"></canvas><p id="guess-summary" class="plot-summary"></p></div></div>'''
    else:
        body = '''<h2>PPC mismatch detective</h2><p class="prediction">Choose a diagnosis, then inspect what posterior predictive checks can—and cannot—tell you.</p><div class="preset-row" id="ppc-cases"></div><div class="diagnosis-row"><label><input type="radio" name="diagnosis" value="well-specified"> Well specified</label><label><input type="radio" name="diagnosis" value="noise"> Noise mismatch</label><label><input type="radio" name="diagnosis" value="outlier"> Contamination</label><label><input type="radio" name="diagnosis" value="support"> Support issue</label><label><input type="radio" name="diagnosis" value="structure"> Structural mismatch</label></div><canvas id="ppc-canvas" width="1100" height="430" aria-label="Observation, posterior predictive band, and residuals"></canvas><div class="button-row"><button id="ppc-reveal" type="button">Reveal diagnosis</button><button id="ppc-reset" type="button">Reset</button></div><p id="ppc-summary" class="plot-summary"></p><div class="what-changed"><strong>Interpret carefully.</strong> PPC tension can flag mismatch, but it rarely identifies a unique cause by itself.</div>'''
    return common_start + body + common_end


def deterministic_zhou(theta: np.ndarray, p0: np.ndarray, generations: int = 120) -> np.ndarray:
    theta = np.asarray(theta, float).reshape(-1, 4)
    p0 = np.asarray(p0, float)
    if p0.ndim == 1:
        p0 = np.repeat(p0[None, :], len(theta), axis=0)
    out = np.zeros((len(theta), generations + 1, 3), float)
    out[:, 0] = p0
    for b, (log_wt, w_tri, w_loh, log_loh) in enumerate(theta):
        M = np.array([[1 - 10**log_wt - 10**log_loh, 0, 0], [10**log_wt, 1, 0], [10**log_loh, 0, 1]])
        G = np.diag([w_tri, 1, w_loh]) @ M
        for g in range(generations):
            nxt = G @ out[b, g]
            out[b, g + 1] = nxt / nxt.sum()
    return out


def wf_deterministic(theta: np.ndarray) -> np.ndarray:
    generations = np.array([8, 21, 29, 37, 50, 58, 66, 79, 87, 95, 108, 116])
    log_s, log_m, log_p0 = map(float, theta)
    s, m, p0 = 10 ** np.array([log_s, log_m, log_p0])
    w = np.array([1, 1+s, 1+s, 1.001])
    M = np.array([[1-m-1e-5,0,0,0],[m,1,0,0],[0,0,1,0],[1e-5,0,0,1]],float)
    E = M @ np.diag(w)
    p = np.array([1-p0,0,p0,0],float); out=[]
    for g in range(117):
        if g in generations: out.append(p[1])
        p = E @ p; p /= p.sum()
    return np.array(out)


def generate_zhou_assets() -> dict:
    sys.path.insert(0, str(EVODESIGN_SRC))
    import torch
    from evodesign.simulators.observation import ObservationModel
    model_dir = ROOT / "zhou_npe_models"
    payloads, hashes, metadata = [], [], []
    for idx in range(3):
        model = model_dir / f"robust_npe_seed_{idx}.pkl"
        meta = json.loads((model_dir / f"robust_npe_seed_{idx}.json").read_text())
        digest = sha256(model)
        if digest != meta["model"]["sha256"]:
            raise RuntimeError(f"Zhou checkpoint {idx} hash mismatch")
        with model.open("rb") as handle:
            payloads.append(pickle.load(handle))
        hashes.append(digest); metadata.append(meta)
    bank = json.loads((model_dir / "grouped_validation_examples.json").read_text())
    position = bank["indices"].index(35)
    truth = np.asarray(bank["theta"][position], np.float32)
    p0 = np.asarray(bank["p0"][position], np.float32)
    latent = np.asarray(bank["latent_passages"][position], np.float32)
    space = payloads[0]["space"]
    latent_full = torch.zeros(1, 121, 3)
    latent_full[:, ::10, :] = torch.tensor(latent)
    full_design = space.full(depth=100_000)
    observation = ObservationModel(noise_model="multinomial_gaussian", gaussian_noise=.02, fp_rate=1e-4).observe(latent_full, full_design, seed=SEED)
    observed_passages = observation[0, ::10, :].cpu().numpy().astype(np.float32)
    draw_count = 48
    draws = np.zeros((4096, draw_count * 3, 4), np.float32)
    quantiles = np.zeros((4096, 4, 3), np.float32)
    seed_quantiles = np.zeros((4096, 3, 4, 3), np.float32)
    odd = sum(1 << (p-1) for p in [1,3,5,7,9,11])
    even = sum(1 << (p-1) for p in [2,4,6,8,10,12])
    for mask in range(4096):
        points = [i + 1 for i in range(12) if mask & (1 << i)]
        design = space.make(points, depths=100_000)
        masked = torch.full_like(observation, float("nan"))
        gens = list(design.sampled_generations)
        masked[:, gens, :] = observation[:, gens, :]
        batches = []
        for seed_idx, payload in enumerate(payloads):
            condition = payload["encoder"].encode(masked, design, context=torch.tensor([p0]))
            # Preserve the notebook's exact odd/even sampling streams. Other masks use
            # deterministic non-overlapping mask-derived streams.
            sample_seed = (20261000 + seed_idx) if mask == odd else ((20261100 + seed_idx) if mask == even else 20300000 + mask * 3 + seed_idx)
            torch.manual_seed(sample_seed)
            sample = payload["estimator"].sample((draw_count,), condition=condition).reshape(draw_count, 4).detach().cpu().numpy()
            if not np.isfinite(sample).all():
                raise RuntimeError(f"non-finite Zhou draws for mask {mask}, seed {seed_idx}")
            batches.append(sample)
            seed_quantiles[mask, seed_idx] = np.quantile(sample, [.05,.5,.95], axis=0).T
        ensemble = np.concatenate(batches)
        draws[mask] = ensemble
        quantiles[mask] = np.quantile(ensemble, [.05,.5,.95], axis=0).T
        if mask % 512 == 0:
            print(f"Zhou schedules: {mask}/4096", flush=True)
    # npz is compact and browser-unfriendly, so store typed arrays as base64-free binary.
    draws.astype("<f4").tofile(DATA / "zhou_draws.f32")
    quantiles.astype("<f4").tofile(DATA / "zhou_quantiles.f32")
    seed_quantiles.astype("<f4").tofile(DATA / "zhou_seed_quantiles.f32")
    notebook_validation = {}
    tolerances = np.array([.12, .01, .01, .12])
    for schedule_idx, (name, points, mask) in enumerate([
        ("odd", [1,3,5,7,9,11], odd), ("even", [2,4,6,8,10,12], even)
    ]):
        design = space.make(points, depths=100_000)
        masked = torch.full_like(observation, float("nan"))
        gens = list(design.sampled_generations)
        masked[:, gens, :] = observation[:, gens, :]
        reference_batches = []
        for seed_idx, payload in enumerate(payloads):
            condition = payload["encoder"].encode(masked, design, context=torch.tensor([p0]))
            torch.manual_seed(20261000 + 100 * schedule_idx + seed_idx)
            reference_batches.append(payload["estimator"].sample((800,), condition=condition).reshape(800, 4).detach().cpu().numpy())
        reference_q = np.quantile(np.concatenate(reference_batches), [.05,.5,.95], axis=0).T
        difference = np.abs(reference_q - quantiles[mask])
        if np.any(difference.max(axis=1) > tolerances):
            raise RuntimeError(f"{name} site summaries exceed notebook Monte Carlo tolerance")
        notebook_validation[name] = {
            "notebook_draws_per_checkpoint": 800,
            "site_draws_per_checkpoint": draw_count,
            "sampling_seeds": [20261000 + 100 * schedule_idx + i for i in range(3)],
            "max_abs_quantile_difference_by_parameter": difference.max(axis=1).tolist(),
            "absolute_tolerance_by_parameter": tolerances.tolist(),
            "passed": True,
        }
    manifest = {
        "kind": "real-three-checkpoint-Zhou-NPE-derived-assets",
        "model_hashes": hashes, "bank_example_index": 35, "seed": SEED,
        "schedule_encoding": "12-bit little-endian mask for passages 1..12; passage 0 implicit and forced",
        "n_masks": 4096, "draws_per_checkpoint": draw_count, "ensemble_draws_per_mask": draw_count*3,
        "draw_shape": list(draws.shape), "quantile_shape": list(quantiles.shape),
        "truth": truth.tolist(), "p0": p0.tolist(), "latent_passages": latent.tolist(),
        "observed_passages": observed_passages.tolist(), "odd_mask": odd, "even_mask": even,
        "odd_passages": [0,1,3,5,7,9,11], "even_passages": [0,2,4,6,8,10,12],
        "notebook_odd_even_validation": notebook_validation,
        "parameter_names": ["log10_mu_tri_to_wt","w_tri","w_loh","log10_mu_tri_to_loh"],
        "parameter_bounds": [[-7,-2],[.85,1.05],[.85,1.05],[-7,-2]],
        "training_summaries": [m["training_summary"] for m in metadata],
        "caveat": "Illustrative design-conditioned inference demonstration, not a coverage study."
    }
    write_json(DATA / "zhou_manifest.json", manifest)
    return manifest


def generate_teaching_training() -> dict:
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    n, n_val = 3200, 640
    lo=np.array([-2,-7,-8],np.float32); hi=np.array([0,-2,-2],np.float32)
    theta=np.random.uniform(lo,hi,(n+n_val,3)).astype(np.float32)
    curves=np.stack([wf_deterministic(t) for t in theta]).astype(np.float32)
    curves=np.clip(curves+np.random.default_rng(SEED).normal(0,.02,curves.shape),0,1).astype(np.float32)
    xmean=curves[:n].mean(0); xstd=curves[:n].std(0)+1e-5
    tmid=(lo+hi)/2; tscale=(hi-lo)/2
    X=torch.tensor((curves-xmean)/xstd); Y=torch.tensor((theta-tmid)/tscale)
    model=torch.nn.Sequential(torch.nn.Linear(12,48),torch.nn.GELU(),torch.nn.Linear(48,48),torch.nn.GELU(),torch.nn.Linear(48,6))
    opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5)
    snapshots=[0,1,2,3,5,8,12,16,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    truth=np.array([-.9,-5,-5.5],np.float32); obs=wf_deterministic(truth).astype(np.float32)
    records=[]; train_loss=[]; val_loss=[]
    def snap(epoch:int):
        with torch.no_grad():
            raw=model(torch.tensor(((obs-xmean)/xstd)[None,:],dtype=torch.float32))[0]
            mean=(raw[:3].numpy()*tscale+tmid); sd=(np.exp(np.clip(raw[3:].numpy(),-3,1))*tscale)
            mean=np.clip(mean,lo,hi); sd=np.clip(sd,.02,(hi-lo)/2)
        rng=np.random.default_rng(SEED+epoch)
        draws=np.clip(rng.normal(mean,sd,(160,3)),lo,hi)
        ppc=np.stack([wf_deterministic(d) for d in draws[:96]])
        records.append({"epoch":epoch,"mean":mean.tolist(),"sd":sd.tolist(),"q05":np.quantile(draws,.05,axis=0).tolist(),"q95":np.quantile(draws,.95,axis=0).tolist(),"ppc_q05":np.quantile(ppc,.05,axis=0).tolist(),"ppc_median":np.quantile(ppc,.5,axis=0).tolist(),"ppc_q95":np.quantile(ppc,.95,axis=0).tolist()})
    snap(0)
    batch=128
    for epoch in range(1,101):
        model.train(); perm=torch.randperm(n); total=0.0
        for start in range(0,n,batch):
            idx=perm[start:start+batch]; raw=model(X[idx]); mean,logsd=raw[:,:3],raw[:,3:].clamp(-5,2)
            loss=(.5*((Y[idx]-mean)/logsd.exp()).square()+logsd).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss)*len(idx)
        train_loss.append(total/n)
        model.eval()
        with torch.no_grad():
            raw=model(X[n:]); mean,logsd=raw[:,:3],raw[:,3:].clamp(-5,2)
            loss=(.5*((Y[n:]-mean)/logsd.exp()).square()+logsd).sum(1).mean()
            val_loss.append(float(loss))
        if epoch in snapshots: snap(epoch)
    best=int(np.argmin(val_loss)+1)
    payload={"kind":"genuine-teaching-scale-diagonal-Gaussian-NPE","pedagogical_not_production":True,"seed":SEED,"n_simulations":n+n_val,"train_rows":n,"validation_rows":n_val,"architecture":"12 → 48 GELU → 48 GELU → 6 (diagonal Gaussian mean/log-scale)","optimizer":"Adam lr=0.002, weight_decay=1e-5","epochs":100,"snapshot_epochs":snapshots,"best_validation_epoch":best,"validation_criterion":"held-out conditional Gaussian negative log density","software":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},"truth":truth.tolist(),"observation":obs.tolist(),"generations":[8,21,29,37,50,58,66,79,87,95,108,116],"train_loss":[None]+train_loss,"validation_loss":[None]+val_loss,"snapshots":records,"caveat":"Short pedagogical estimator; it is not the large Zhou ensemble and is not production-grade."}
    write_json(DATA/"training_viewer.json",payload); return payload


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return np.exp(-.5*((x-mean)/sd)**2)/(sd*np.sqrt(2*np.pi))


def generate_collective_and_exercises() -> None:
    rng=np.random.default_rng(SEED); grid=np.linspace(-1.8,-.35,320); truth=-.9
    labels=["R1","R2","R3","R4","R5","R6 subtle","R7 outlier"]
    shifts=np.array([-.02,.03,-.01,.05,-.04,.13,.42]); sds=np.array([.22,.2,.23,.19,.21,.23,.2])
    prior=normal_pdf(grid,-1.05,.48); factors=[]; trajectories=[]
    for i,(shift,sd) in enumerate(zip(shifts,sds)):
        likelihood=normal_pdf(grid,truth+shift,sd)
        post=likelihood*prior; post/=np.trapz(post,grid)
        factors.append(np.log(post+1e-30))
        trajectories.append(np.clip(wf_deterministic([truth+shift,-5,-5.5])+rng.normal(0,.012,12),0,1).tolist())
    write_json(DATA/"collective_lab.json",{"grid":grid.tolist(),"truth":truth,"prior_log":np.log(prior+1e-30).tolist(),"replicate_log_posteriors":np.array(factors).tolist(),"labels":labels,"types":["clean"]*5+["subtle","outlier"],"trajectories":trajectories,"generations":[8,21,29,37,50,58,66,79,87,95,108,116],"formula":"sum(log posterior_i) - (r-1) * log(prior)"})
    examples=[]
    for idx,t in enumerate([[-.9,-5,-5.5],[-1.18,-4.2,-6.2],[-.68,-5.7,-4.7]]):
        curve=wf_deterministic(t); draws=rng.normal(t,[.09,.45,.7],(220,3)); draws=np.clip(draws,[-2,-7,-8],[0,-2,-2])
        examples.append({"id":idx+1,"trajectory":curve.tolist(),"truth":t,"draws":draws.round(5).tolist(),"interpretation":"Selection is usually constrained by sweep shape; mutation rate and initial frequency remain confounded because both seed early CNV abundance."})
    cases=[]
    base=np.array(wf_deterministic([-.9,-5,-5.5]));
    specs=[("Correctly specified","well-specified",base), ("Observation noise","noise",np.clip(base+rng.normal(0,.075,12),0,1)), ("One contaminated passage","outlier",np.where(np.arange(12)==6,np.clip(base+.28,0,1),base)), ("Outside training support","support",wf_deterministic([.15,-5,-5.5])), ("Structural mismatch","structure",np.clip(base+.08*np.sin(np.linspace(0,3*np.pi,12)),0,1))]
    qlo=np.clip(base-.05,0,1); qhi=np.clip(base+.05,0,1)
    reasons={"well-specified":"Observed deviations are compatible with the assumed simulator and noise.","noise":"Residuals are too dispersed for the assumed observation noise.","outlier":"One passage is contaminated; this local discrepancy does not imply the remaining model is correct.","support":"The generating selection value lies outside the training prior, so the posterior cannot represent it.","structure":"A systematic oscillation is absent from the fitted model family."}
    for title,kind,obs in specs: cases.append({"title":title,"kind":kind,"observation":np.asarray(obs).tolist(),"q05":qlo.tolist(),"median":base.tolist(),"q95":qhi.tolist(),"reason":reasons[kind]})
    write_json(DATA/"exercises.json",{"generations":[8,21,29,37,50,58,66,79,87,95,108,116],"guess_examples":examples,"ppc_cases":cases})


def generate_fallbacks() -> None:
    import matplotlib.pyplot as plt
    FALLBACK.mkdir(parents=True,exist_ok=True); CHAPTER_ASSETS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"figure.facecolor":"#fffdf8","axes.facecolor":"#fffdf8","axes.spines.top":False,"axes.spines.right":False})
    x=np.arange(13); colors=["#335f52","#4e7890","#b4684f"]
    z=json.loads((DATA/"zhou_manifest.json").read_text()); latent=np.array(z["latent_passages"])
    ave_theta=np.array([-4.2,-5,.07,.001]); ave=np.array([[1.,0.,0.]])
    for _ in range(120):
        p=ave[-1]; selected=p*np.array([1,1+ave_theta[2],1+ave_theta[3]])
        nxt=np.array([selected[0]*(1-10**ave_theta[0]-10**ave_theta[1]),selected[1]+selected[0]*10**ave_theta[0],selected[2]+selected[0]*10**ave_theta[1]])
        ave=np.vstack([ave,nxt/nxt.sum()])
    dfe_x=np.linspace(.0001,.18,240); dfe_shape=2.; dfe_scale=.045/dfe_shape
    dfe_y=dfe_x**(dfe_shape-1)*np.exp(-dfe_x/dfe_scale); dfe_y/=dfe_y.max()
    chu_g=np.array([8,21,29,37,50,58,66,79,87,95,108,116]); chu=wf_deterministic([-.74,-4.84,-4.32])
    figs={
      "evolution-playground":lambda ax:[ax.plot(np.arange(121),ave[:,i],color=["#607069","#e67e22","#8e44ad"][i],lw=2.5) for i in range(3)],
      "dfe-example":lambda ax:ax.fill_between(dfe_x,0,dfe_y,color="#e67e22",alpha=.32),
      "chuong-parameter-challenge":lambda ax:[ax.plot(chu_g,chu,color="#577d91",lw=2.3),ax.scatter(chu_g,np.clip(chu+np.array([.01,-.02,.015,-.01,.02,-.012,.01,-.008,.004,.006,-.003,.002]),0,1),color="#e67e22",s=28)],
      "zhou-model-playground":lambda ax:[ax.plot(x,latent[:,i],color=colors[i],lw=2.5) for i in range(3)],
      "zhou-schedule-designer":lambda ax:[ax.plot(x,latent[:,i],color=colors[i],lw=2) for i in range(3)],
      "training-viewer":lambda ax:ax.plot(range(101),json.loads((DATA/"training_viewer.json").read_text())["validation_loss"],color=colors[0]),
      "collective-outlier-lab":lambda ax:[ax.plot(json.loads((DATA/"collective_lab.json").read_text())["generations"],t,alpha=.7) for t in json.loads((DATA/"collective_lab.json").read_text())["trajectories"]],
      "guess-parameter":lambda ax:ax.plot(json.loads((DATA/"exercises.json").read_text())["generations"],json.loads((DATA/"exercises.json").read_text())["guess_examples"][0]["trajectory"],color=colors[0],marker="o"),
      "ppc-detective":lambda ax:ax.plot(json.loads((DATA/"exercises.json").read_text())["generations"],json.loads((DATA/"exercises.json").read_text())["ppc_cases"][2]["observation"],color=colors[2],marker="o")}
    for name,draw in figs.items():
        fig,ax=plt.subplots(figsize=(7,3)); draw(ax)
        ax.set_xlabel("Selection coefficient s" if name=="dfe-example" else "Passage / generation")
        ax.set_ylabel("Relative density" if name=="dfe-example" else ("Validation loss" if name=="training-viewer" else "Frequency"))
        fig.tight_layout(); fig.savefig(FALLBACK/f"{name}.png",dpi=140); plt.close(fig)

    # The notebook source now deliberately uses the Avecilla orange for the Chuong fit.
    # Render that focused output independently so an ordinary static build does not keep
    # serving the stale blue PNG embedded in an older notebook execution.
    theta=np.array([-.74,-4.84,-4.32]); sigma=np.array([.005,.1,.1]); chuong_trajs=[]
    legacy_state=np.random.get_state()
    try:
        for seed in range(50):
            th=theta+np.random.default_rng(seed).normal(0,sigma); s,m,p0=10**th
            fitness=np.array([1,1+s,1+s,1.001]); transition=np.array([[1-m-1e-5,0,0,0],[m,1,0,0],[0,0,1,0],[1e-5,0,0,1]],float)
            evolution=transition@np.diag(fitness); n=np.array([3.3e8*(1-p0),0,3.3e8*p0,0]); values=[]
            np.random.seed(seed)
            for generation in range(chu_g[-1]+1):
                if generation in chu_g: values.append(n[1]/3.3e8)
                probability=evolution@(n/3.3e8); probability/=probability.sum()
                n=np.random.multinomial(int(3.3e8),probability).astype(float)
            chuong_trajs.append(values)
    finally:
        np.random.set_state(legacy_state)
    ltr=np.genfromtxt(ROOT/"data/ltr.csv",delimiter=",",skip_header=1,usecols=range(1,13))
    fig,ax=plt.subplots(figsize=(11,4.4))
    for i,trajectory in enumerate(chuong_trajs): ax.plot(chu_g,trajectory,lw=1.15,alpha=.22,color="#e67e22",label="Simulations" if i==0 else None)
    for i,row in enumerate(ltr): ax.plot(chu_g,row,"o-",color="#e67e22",ms=4.8,lw=1,label="Data" if i==0 else None,zorder=5)
    ax.legend(); ax.set_xlabel("Generation"); ax.set_ylabel("GAP1 CNV⁺ frequency"); ax.set_title("Chuong WF fit vs. LTR data"); ax.set_ylim(-.02,1.02)
    fig.suptitle("Chuong WF — Model Fit"); fig.tight_layout(); fig.savefig(CHAPTER_ASSETS/"chuong-fit-orange.png",dpi=150); plt.close(fig)


def page_shell(title: str, eyebrow: str, active: str, content: str, scripts: list[str]) -> str:
    nav=''.join(f'<a class="{"active" if active==key else ""}" href="{href}">{label}</a>' for key,href,label in [("home","index.html","Start"),("evolution","evolution.html","Evolution"),("sbi","sbi.html","SBI")])
    script_tags=''.join(f'<script src="{versioned_asset("js/" + s + ".js")}" defer></script>' for s in scripts)
    stylesheet = versioned_asset("css/workshop.css")
    shared_scripts = ''.join(
        f'<script src="{versioned_asset("js/" + name + ".js")}" defer></script>'
        for name in ("science", "core", "plots")
    )
    chapter_header = "" if active == "home" else f'<header class="chapter-hero"><p class="eyebrow">{eyebrow}</p><h1>{html.escape(title)}</h1><p class="lede">Mechanistic evolution, uncertainty, and experimental design—kept close enough to inspect.</p></header>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{html.escape(title)}</title><link rel="icon" href="assets/favicon.svg"><link rel="stylesheet" href="{stylesheet}"><noscript><style>.static-fallback{{display:block}}</style></noscript></head><body data-page="{active}"><a class="skip-link" href="#main">Skip to lesson</a><header class="site-header"><a class="wordmark" href="index.html"><span>Δ</span> Drift &amp; Design</a><nav aria-label="Workshop chapters">{nav}</nav><div class="header-actions"><button id="workshop-mode" type="button" aria-pressed="false">Workshop mode</button><button id="reset-all" type="button">Reset all</button></div></header><div class="progress-track" aria-hidden="true"><span id="reading-progress"></span></div><main id="main">{chapter_header}{content}</main><footer><p>Drift &amp; Design · Static workshop edition</p><nav><a href="index.html">Start</a><a href="evolution.html">Evolution</a><a href="sbi.html">SBI</a></nav></footer>{shared_scripts}{script_tags}</body></html>'''


def landing_page() -> str:
    return page_shell("Evolution × inference", "An interactive scientific workshop", "home", '''<section class="landing-hero"><div><span class="method-status"><i></i>Static · reproducible · model-free at runtime</span><h2>Follow a population.<br>Then infer what moved it.</h2><p>Two authored chapters connect evolutionary simulators to simulation-based inference, with nine prediction-first laboratories.</p><div class="button-row"><a class="primary-link" href="evolution.html">Begin chapter 01</a><a class="secondary-link" href="sbi.html">Jump to SBI</a></div></div><div class="hero-orbit" aria-hidden="true"><span></span><span></span><span></span><b>θ</b></div></section><section class="objectives"><p class="section-kicker">Workshop outcomes</p><h2>What you will be able to do</h2><div class="objective-grid"><article><b>01</b><h3>Read the mechanism</h3><p>Translate mutation, selection, and drift into population-frequency trajectories.</p></article><article><b>02</b><h3>Reason with uncertainty</h3><p>Compare ABC, NPE, collective evidence, and posterior predictive checks.</p></article><article><b>03</b><h3>Design observations</h3><p>See how a passage mask changes what the same experiment can identify.</p></article></div></section><section class="chapter-cards"><a href="evolution.html"><span>Chapter 01 · 50–65 min</span><h2>Evolutionary simulators</h2><p>Five models, from chemostat kinetics to chromosome stability.</p><strong>Enter the simulator chapter →</strong></a><a href="sbi.html"><span>Chapter 02 · 70–90 min</span><h2>Simulation-based inference</h2><p>ABC, NPE, schedule-aware inference, collective posteriors, and diagnostics.</p><strong>Enter the inference chapter →</strong></a></section><section class="glossary"><p class="section-kicker">Pocket glossary</p><div class="glossary-grid"><details><summary>Drift</summary><p>Random frequency change caused by finite population sampling.</p></details><details><summary>Posterior</summary><p>A probability distribution over parameters after conditioning on observations.</p></details><details><summary>Amortization</summary><p>Paying simulation and training cost once so new observations can be inferred quickly.</p></details><details><summary>PPC</summary><p>A posterior predictive check: simulate from inferred parameters and compare with observed data.</p></details></div></section>''', [])


def build_content() -> None:
    NOTEBOOK_ASSETS.mkdir(parents=True,exist_ok=True)
    interactions=json.loads((SITE/"interaction_manifest.json").read_text())
    all_coverage={}
    evo,cov=render_notebook("evolution",interactions["evolution_simulators.ipynb"]); all_coverage["evolution_simulators.ipynb"]=cov
    sbi,cov=render_notebook("sbi",interactions["SBI_tutorial.ipynb"]); all_coverage["SBI_tutorial.ipynb"]=cov
    write_json(SITE/"content_map.json",all_coverage)
    (SITE/"index.html").write_text(landing_page())
    evo_nav='<nav class="chapter-nav" aria-label="Chapter navigation"><a href="index.html">← Workshop home</a><a href="sbi.html">Next: simulation-based inference →</a></nav>'
    sbi_nav='<nav class="chapter-nav" aria-label="Chapter navigation"><a href="evolution.html">← Evolutionary simulators</a><a href="index.html">Workshop home →</a></nav>'
    (SITE/"evolution.html").write_text(page_shell("Evolutionary simulators","Chapter 01 · Models before methods","evolution",'<div class="lesson-layout"><aside class="toc" aria-label="On this page"><button class="toc-toggle" type="button">On this page</button><div class="toc-links"></div></aside><article class="notebook-lesson">'+evo+evo_nav+'</article></div>',["evolution"]))
    (SITE/"sbi.html").write_text(page_shell("Simulation-based inference","Chapter 02 · Learning from simulations","sbi",'<div class="lesson-layout"><aside class="toc" aria-label="On this page"><button class="toc-toggle" type="button">On this page</button><div class="toc-links"></div></aside><article class="notebook-lesson">'+sbi+sbi_nav+'</article></div>',["sbi"]))


def provenance() -> None:
    artifacts={}
    for p in sorted([*DATA.glob("*"),*FALLBACK.glob("*"),*CHAPTER_ASSETS.glob("*"),SITE/"index.html",SITE/"evolution.html",SITE/"sbi.html",SITE/"content_map.json"]):
        if p.is_file(): artifacts[str(p.relative_to(SITE))]=sha256(p)
    zhou_manifest = json.loads((DATA / "zhou_manifest.json").read_text())
    models = {}
    for i, expected in enumerate(zhou_manifest["model_hashes"]):
        name = f"robust_npe_seed_{i}.pkl"
        local_model = ROOT / "zhou_npe_models" / name
        if local_model.exists() and sha256(local_model) != expected:
            raise RuntimeError(f"local Zhou checkpoint {i} no longer matches the derived asset manifest")
        models[name] = expected
    simulator_path = EVODESIGN_SRC / "evodesign/simulators/zhou2026.py"
    if simulator_path.exists():
        simulator_hash = sha256(simulator_path)
    elif (SITE / "provenance.json").exists():
        simulator_hash = json.loads((SITE / "provenance.json").read_text())["simulator_source"]["sha256"]
    else:
        raise RuntimeError("simulator source is unavailable and no prior provenance hash exists")
    write_json(SITE/"provenance.json",{"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"source_notebooks":{p.name:sha256(p) for p in NOTEBOOKS.values()},"zhou_models":models,"simulator_source":{"path":"../evodesign/src/evodesign/simulators/zhou2026.py","sha256":simulator_hash},"seeds":[SEED,20261000],"asset_generation_command":"PYTHONPATH=../evodesign/src python workshop_site/build_site.py --scientific-assets","artifacts":artifacts,"scientific_caveats":["Zhou assets are an illustrative inference-flexibility demonstration, not a coverage study.","Training viewer is a genuine teaching-scale diagonal-Gaussian NPE, not the production Zhou ensemble.","Interactive replicate exclusion is sensitivity analysis, not a data-discarding rule."]})


def verify_existing_assets() -> None:
    required=[DATA/"zhou_manifest.json",DATA/"zhou_draws.f32",DATA/"zhou_quantiles.f32",DATA/"zhou_seed_quantiles.f32",DATA/"training_viewer.json",DATA/"collective_lab.json",DATA/"exercises.json"]
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise RuntimeError("Missing derived scientific assets; run with --scientific-assets: "+", ".join(missing))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--scientific-assets",action="store_true"); parser.add_argument("--verify-only",action="store_true"); args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True); ASSETS.mkdir(parents=True,exist_ok=True)
    if args.scientific_assets:
        generate_zhou_assets(); generate_teaching_training(); generate_collective_and_exercises(); generate_fallbacks()
    else: verify_existing_assets()
    if not args.verify_only: build_content(); provenance()
    print("Workshop build complete")


if __name__ == "__main__": main()
