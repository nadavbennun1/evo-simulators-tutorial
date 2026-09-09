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
PUBLIC_URL = "https://nadavbennun1.github.io/evo-simulators-tutorial/"

PAPERS = {
    "avecilla": ("Avecilla et al. (2022)", "https://doi.org/10.1371/journal.pbio.3001633"),
    "chuong": ("Chuong et al. (2025)", "https://doi.org/10.7554/eLife.98934"),
    "de": ("De et al. (2025)", "https://doi.org/10.1101/2025.07.21.665951"),
    "collective": ("Ben Nun et al. (2026)", "https://doi.org/10.1371/journal.pcbi.1014534"),
    "sbi": ("Tejero-Cantero et al. (2020)", "https://doi.org/10.21105/joss.02505"),
}

EVOLUTION_SIMULATOR_CELLS = {"efcdf8fa", "6cfee4f5", "2539f7c5", "5f9d90ff"}
EVOLUTION_OUTPUT_ONLY = {
    "21748f7d", "b09b721a", "54e9f8de", "da8788a6", "6355cf57", "7b0a27f1",
    "d29cac1e", "f8f106e5", "104a1da1", "97c3a42c", "66cce2fa", "8db7a696",
    "35d77585", "65a843c6", "7fd6e35c", "2e99f96f",
}
EVOLUTION_CELL_ORDER = [
    "9bb0927d", "7f319e7c", "cd6f3ea8", "21748f7d", "b09b721a", "3282b174",
    "efcdf8fa", "54e9f8de", "6ec896e6", "252e23bf", "4ff32a3e", "d29cac1e",
    "4cd32c54", "f8f106e5", "f9cb77d4", "104a1da1", "97c3a42c", "5de3ea7b",
    "6cfee4f5", "66cce2fa", "2973e9a8", "e7044463", "8db7a696", "35d77585",
    "3d886968", "2539f7c5", "65a843c6", "1466cfd7", "705d171d", "7fd6e35c",
    "b4a01787", "5f9d90ff", "2e99f96f", "629428f4", "da8788a6", "6355cf57",
    "7b0a27f1", "f37292e6", "49b50682", "40f8e547", "6b93477c",
]
SBI_VISIBLE_CODE = {"67d19e3c", "88e4194b", "90cc8760"}
SBI_OUTPUT_ONLY = {"cdda66b7", "0ba5658f", "da54003d", "098a16bd"}

OUTPUT_CAPTIONS = {
    "21748f7d": ("GAP1 CNV frequency across chemostat populations", "Replicate GAP1 CNV trajectories from the Avecilla study"),
    "b09b721a": ("The Avecilla model connects mutation, selection, drift, and continuous culture", "Avecilla evolutionary-model diagram"),
    "54e9f8de": ("A compact three-genotype simulator reproduces the main sweep dynamics", "Avecilla model fit"),
    "da8788a6": ("A distribution of effects creates many competing CNV lineages", "DFE trajectory comparison"),
    "6355cf57": ("Sliding windows ask whether one constant selection coefficient is enough", "Windowed selection diagnostic"),
    "7b0a27f1": ("Selection enriches the upper tail of the DFE as the sweep progresses", "DFE and fitted selection through time"),
    "d29cac1e": ("Continuous chemostat dynamics and discrete Wright–Fisher dynamics can tell the same frequency story", "Chemostat ODE and Wright-Fisher comparison"),
    "f8f106e5": ("LTRΔ populations begin with an early plateau before the CNV sweep", "Chuong experimental trajectories"),
    "104a1da1": ("The three-state Avecilla model misses the early LTRΔ structure", "Avecilla model applied to Chuong data"),
    "97c3a42c": ("A pre-existing CNV pool supplies the missing biological state", "Chuong four-genotype model diagram"),
    "66cce2fa": ("The four-genotype model captures the observed chemostat trajectories", "Chuong model fit"),
    "8db7a696": ("CNV reporters reveal whether amplification persists after selection is removed", "De et al. reporter trajectories"),
    "35d77585": ("The reversion model follows movement from CNV to single copy", "De et al. two-genotype model diagram"),
    "65a843c6": ("Different reporter loci can imply different reversion dynamics", "De et al. model fit"),
    "7fd6e35c": ("Chromosome loss can resolve through euploid recovery or LOH", "Zhou three-state model diagram"),
    "2e99f96f": ("Three measured states constrain two loss routes and their fitnesses", "Zhou model fit"),
    "cdda66b7": ("The same parameters produce a family of stochastic observations", "Repeated Wright-Fisher simulations"),
    "0ba5658f": ("A synthetic observation gives us a known answer for checking ABC", "Synthetic trajectory for ABC"),
    "da54003d": ("Accepted simulations turn a distance threshold into parameter uncertainty", "ABC posterior result"),
    "098a16bd": ("NPE returns a joint posterior after one conditioning step", "Neural posterior estimate"),
}

OUTPUT_STORIES = {
    "da8788a6": "A single mutation rate can feed many CNV classes. Their effects are drawn once from the DFE; selection then changes which classes remain visible.",
    "6355cf57": "Now fit a constant-s model to short windows. If the biological effect really is constant, the local estimate should stay flat.",
    "7b0a27f1": "Under a DFE, weak lineages disappear and strong lineages dominate. The rising local estimate is therefore a population-level signature of sorting within the DFE.",
}


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
    placeholders: list[tuple[str, bool]] = []

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
        placeholders.append((rendered, display))
        return token

    # Display first, then conservative inline spans (currency does not occur here).
    source = re.sub(r"(\$\$)(.*?)(\$\$)", lambda m: repl(_MathMatch(True, m.group(2))), source, flags=re.S)
    source = re.sub(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", lambda m: repl(_MathMatch(False, m.group(1))), source)
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    # Markdown may wrap one or several adjacent display placeholders in a single
    # paragraph. Remove that wrapper before substituting block-level containers.
    rendered = re.sub(r"<p>\s*((?:MATHPLACEHOLDER\d+X\s*)+)</p>", r"\1", rendered)
    for idx, (value, display) in enumerate(placeholders):
        if display:
            rendered = rendered.replace(f"<p>MATHPLACEHOLDER{idx}X</p>", value)
        rendered = rendered.replace(f"MATHPLACEHOLDER{idx}X", value)
    return rendered


class _MathMatch:
    def __init__(self, display: bool, body: str):
        self.display, self.body = display, body
    def group(self, index: int) -> str:
        if index == 1:
            return "$$" if self.display else ""
        return self.body


def output_html(output: dict, stem: str, output_index: int, figures_only: bool = False) -> str:
    # This notebook cell records a missing-optional-dependency failure rather than
    # a scientific result. Keep its reproducible source, but do not publish the
    # stale "Error loading sheet" / cascading NoneType output as lesson content.
    if stem == "705d171d":
        return ""
    data = output.get("data", {})
    pieces: list[str] = []
    if "image/png" in data:
        if stem == "66cce2fa":
            name = "chuong-fit-orange.png"
            if not (CHAPTER_ASSETS / name).exists():
                raise RuntimeError(f"missing revised Chuong fit figure: assets/chapter/{name}")
            source = versioned_asset(f"assets/chapter/{name}")
            pieces.append(f'<figure class="notebook-figure"><img loading="lazy" src="{source}" alt="Four-state Wright-Fisher predictions in orange and LTR deletion observations in blue"><figcaption>Four-state Wright–Fisher fit · orange predictions, blue observations</figcaption></figure>')
        else:
            name = f"{stem}-out-{output_index}.png"
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            (NOTEBOOK_ASSETS / name).write_bytes(base64.b64decode(raw))
            source = versioned_asset(f"assets/notebook/{name}")
            caption, alt = OUTPUT_CAPTIONS.get(stem, ("Scientific notebook result", f"Scientific result from notebook cell {stem}"))
            pieces.append(f'<figure class="notebook-figure"><img loading="lazy" src="{source}" alt="{html.escape(alt)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    if figures_only:
        return "".join(pieces)
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


def paper(key: str) -> str:
    label, url = PAPERS[key]
    return f"[{label}]({url})"


def paper_html(key: str) -> str:
    """Return the same compact citation for use inside raw HTML blocks."""
    label, url = PAPERS[key]
    return f'<a href="{html.escape(url)}">{html.escape(label)}</a>'


def curated_markdown(key: str, cid: str, source: str) -> str | None:
    """Replace notebook prose with a presentation-first scientific narrative."""
    if cid in {"9bb0927d", "46c3d4e4", "4f69d96e"}:
        return None
    replacements = {
        "cd6f3ea8": fr'''## Start with the data: repeated *GAP1* CNV sweeps

In the glutamine-limited populations of {paper("avecilla")}, *GAP1* copy-number
variants repeatedly rise from rarity. That movie contains three evolutionary forces:
new variants **appear**, fitter lineages **expand**, and finite populations **sample**
their next generation.

Nine replicate trajectories motivate the model-building question: which biological assumptions
are sufficient to reproduce the timing and shape of these sweeps? We begin with observable states
and build one generation from mutation, selection, and drift.''',
        "3282b174": r'''## Define the biological states

The model partitions every cell into exactly one of three genotypes: ancestral ($A$), *GAP1*
CNV ($C$), or another beneficial genotype ($B$). Their frequencies form
$x=(x_A,x_C,x_B)^\mathsf{T}$ with $x_A+x_C+x_B=1$.

This state vector is the model's vocabulary. Mutation moves frequency between states, selection
changes their expected contributions, and drift samples a finite next generation.''',
        "629428f4": r'''## Black-box stress test: can one effect describe the whole sweep?

For now, treat fitting as a black box: parameters enter the simulator, and an optimizer searches
for values whose trajectories resemble the data. Chapter 2 will replace that vague step with a
full account of posterior inference and uncertainty.

A **distribution of fitness effects (DFE)** assigns a selection coefficient $s$ to each new
mutation. Its mean describes a typical new variant, while its shape controls how often rare,
large-effect mutations occur.

For CNVs this matters because breakpoint, copy number, and amplicon size differ among lineages.
If new CNVs draw $s_k$ from a gamma DFE, their contribution is reweighted by selection:

$$\Pr(k\mid\mathrm{{CNV\ at\ }}t)=
\frac{{\Pr(k\mid\mathrm{{new\ CNV}})(1+s_k)^t}}
{{\sum_j\Pr(j\mid\mathrm{{new\ CNV}})(1+s_j)^t}}$$

The DFE itself stays fixed, but the CNV-bearing population becomes enriched for its upper tail.
That creates a testable prediction: a constant-$s$ fit applied to successive windows should
rise through time for DFE-generated data, but remain flat for truly constant-$s$ data.''',
        "f37292e6": r'''The full trajectory alone cannot reliably separate one large constant effect from a DFE
with a lower mean and a compensating upper tail. Windowed fits add a temporal diagnostic.

Constant-$s$ simulations recover a flat $\hat{s}$. DFE simulations instead produce a rising
$\hat{s}$ as selection enriches high-effect CNV lineages. The Lauer trajectories also rise,
but this is evidence against a single constant effect—not proof of a DFE. Detection limits,
clonal interference, and an incorrectly fixed $\delta_C$ can create related patterns.

For inference, fit DFE parameters directly or use the windowed-$\hat{s}$ trend as a posterior
predictive diagnostic of the simpler model.''',
        "6ec896e6": r'''## The continuous chemostat model

A chemostat has overlapping generations and nutrient-limited growth. The corresponding ODE
tracks ancestral, CNV, and other-beneficial cells together with the residual limiting-substrate
concentration $S(t)$ inside the vessel:

$$\frac{dX_i}{dt}=X_i\bigl(\mu_i(S)-D\bigr)
+\sum_{j\ne i}q_{ij}\mu_j(S)X_j
-\sum_{k\ne i}q_{ki}\mu_i(S)X_i,\qquad
\mu_i(S)=r_i\frac{S}{S+k}$$
$$\frac{dS}{dt}=D(S_0-S)-\frac{1}{Y}\sum_iX_i\mu_i(S)$$

Here $D$ is the dilution rate, $S_0$ is the substrate concentration in the **incoming fresh
medium**, and $S(t)$ is the generally lower concentration remaining in the vessel. At startup the
model may set $S(0)=S_0$; after equilibration, the steady-state value is defined by $dS/dt=0$ and
need not equal $S_0$. The two sums are mutation flow: births of genotype $j$ that produce genotype
$i$ at probability $q_{ij}$, minus births of genotype $i$ that mutate into another state.

The ODE exposes reactor biology that the Wright–Fisher approximation compresses into a generation
clock and an effective population size.''',
        "4ff32a3e": r'''### Connecting hours to generations

The two descriptions share interpretable parameters:

| Chemostat quantity | Wright–Fisher quantity | Conversion |
|---|---|---|
| growth-rate difference | selection coefficient | $s_C=(r_C-r_A)/(r_A\ln2)$ |
| hourly formation rate | per-generation formation rate | $\delta_C^{gen}=\delta_C^{hr}\ln2/D$ |
| time in hours | generations | $g=tD/\ln2$ |

For frequency-only inference, the discrete simulator is usually sufficient. The ODE remains
valuable when nutrient concentration and reactor transients are themselves part of the question.''',
        "4cd32c54": fr'''## Chuong chemostat data require one more hidden state

In {paper("chuong")}, seven LTRΔ populations were observed at 12 generations through 116.
Their early plateau cannot emerge from a model that starts with no CNV-bearing cells. The model
therefore separates newly formed CNV⁺ cells from a small, pre-existing CNV⁻ pool.''',
        "f9cb77d4": r'''### Why the simpler model fails

The mismatch is biological, not merely cosmetic. These are also glutamine-limited chemostats, but
the reporter distinguishes newly formed CNV⁺ cells from CNV-bearing lineages that lack reporter
amplification. Adding the CNV⁻ state lets $\varphi$ represent that initially rare hidden component;
$\delta$ supplies new reported CNVs and $s$ controls their subsequent rise.''',
        "5de3ea7b": None,
        "2973e9a8": fr'''## Copy-number reversion after selection is removed

{paper("de")} moved CNV strains from the environment that selected the amplification into rich
medium and followed fluorescent reporters through 1:64 serial transfers. Here the CNV begins
common; fitter single-copy revertants may arise and replace it.''',
        "3d886968": r'''## A two-state model of CNV reversion

The transition direction is now CNV $\rightarrow$ non-CNV at rate $\delta$. If revertants have
fitness $1+s$, their frequency can rise through formation and selection together:

$$M=\begin{pmatrix}1-\delta&0\\\delta&1\end{pmatrix},\qquad
E=M\,\mathrm{diag}(1,1+s)$$

Because both parameters accelerate loss of the CNV, trajectory shape and repeated measurements
are essential for separating reversion rate from fitness advantage.''',
        "1466cfd7": r'''## Zhou et al. model: two routes out of aneuploidy

The Zhou et al. model (Selmecki lab, UMN) follows trisomic cells as they resolve either to wild type or to loss of
heterozygosity (LOH). The observation is now a three-part composition, so every passage reports
which route gained population share.''',
        "b4a01787": r'''## Zhou et al. model of competing chromosome-loss routes

Trisomic cells move to WT at rate $\mu_{WT}$ or LOH at rate $\mu_{LOH}$; the three states then
compete with relative fitnesses $(w_{Tri},1,w_{LOH})$.

$$M=\begin{pmatrix}1-\mu_{WT}-\mu_{LOH}&0&0\\\mu_{WT}&1&0\\\mu_{LOH}&0&1\end{pmatrix},
\qquad G=\mathrm{diag}(w_{Tri},1,w_{LOH})M$$

This is the same mathematical model with different genotype labels: define allowed transitions,
apply fitness, sample drift, and observe the passages the experiment actually measured.''',
        "49b50682": r'''## Summary

| | Avecilla WF | Avecilla ODE | Chuong WF | De WF | Zhou WF |
|---|---|---|---|---|---|
| **Genotypes** | 3 | 3 | 4 | 2 | 3 |
| **Time axis** | generations | hours | generations | generations | passages |
| **Population model** | fixed $N_e$ | continuous reactor | fixed $N_e$ | serial-dilution $N_e$ | fixed $N_e$ |
| **Transitions** | ancestor → CNV/beneficial | continuous mutation flow | WT → CNV⁺ | CNV → single copy | Tri → WT/LOH |

> **Take home:** choose states and transitions from the biology, choose $N_e$ from the experimental
> life cycle, and keep observation noise separate from evolutionary drift.

<details class="appendix-note"><summary>Appendix: why the quick fits use perturbed parameters</summary>
The displayed fits draw replicate parameters around a central value to illustrate between-replicate
variation, while multinomial sampling produces within-replicate drift. These are fast visual
justifications of each simulator—not the inference target of this lesson. Chapter 2 develops the
systematic inference workflow.
</details>''',
        "fa1ab176": fr'''## Inferring parameters from stochastic trajectories

The *GAP1* frequencies from {paper("chuong")} are discrete observations of an evolving population.
The targets are the selection coefficient $s$, formation rate $\delta$, and initial CNV fraction
$\varphi$. Distinct parameter combinations can produce similar frequency trajectories.

$$p(\theta\mid x_{{obs}})=
\frac{{p(x_{{obs}}\mid\theta)p(\theta)}}{{p(x_{{obs}})}}$$

The simulator can generate $x$ for any $\theta$, but its likelihood cannot be evaluated in closed
form. SBI approximates the posterior from simulated parameter–data pairs. Posterior-predictive
simulations then evaluate whether the fitted model reproduces relevant features of the data.''',
        "f992e16d": r'''## Rejection ABC approximates the posterior

Approximate Bayesian computation draws $\theta$ from the prior, simulates a trajectory, and computes
its distance from the observation:

$$\theta\ \text{is accepted when}\ d(x_{sim},x_{obs})\leq\varepsilon_{ABC}$$

The threshold $\varepsilon_{ABC}$ is a tolerance in **data space**. Smaller values make accepted
simulations more observation-like but demand a larger simulation budget. The progressive station
shows that trade-off rather than hiding it behind one final posterior.''',
        "8d7d8c01": r'''## Neural posterior estimation learns a conditional density

Neural posterior estimation first creates simulated pairs
$(\theta_i,x_i)\sim p(\theta)p(x\mid\theta)$. A conditional density estimator learns
$q_\phi(\theta\mid x)$ by minimizing

$$\mathcal L(\phi)=-\mathbb E_{p(\theta,x)}[\log q_\phi(\theta\mid x)].$$

Training is expensive once; conditioning and sampling are fast for every supported observation
afterward. The next code sections retain only the scientifically meaningful operations: training
the density estimator and sampling parameter draws after conditioning on $x_{obs}$.''',
        "zhou-flex-intro": r'''## One trained NPE can accept different passage schedules

The observation schedule is part of the data. A design-conditioned estimator learns

$$q_\phi(\theta\mid y_{observed},m,d,p_0),$$

where $m$ marks measured passages, $d$ records measurement depth, and $p_0$ supplies the initial
three-state composition. Missing is therefore different from a measured frequency of zero.''',
        "zhou-flex-contract": r'''### Flexibility has a contract

Training exposes the estimator to supported masks while keeping all views of one biological
trajectory in the same data split. It can then reuse one learned posterior across subsets of
passages 0–12 without retraining. It does not promise extrapolation to a new assay or passages
outside that horizon. Passage 0 remains required because it defines the experiment's starting state.''',
        "zhou-flex-takeaway": r'''### What changes when the schedule changes?

Odd and even passages expose different stochastic snapshots, so their posteriors need not be
identical. Agreement means the conclusion is stable for this example; disagreement identifies
where another measurement could be valuable. Flexibility preserves the information the experiment
collected—it does not manufacture information that was never observed.''',
        "928bf2bf": fr'''## Combining replicate-specific posteriors

Each independent replicate gives an individual posterior $p_i(\theta\mid x_i)$. Multiplying them
directly counts the shared prior $r$ times. The standard collective removes those extra copies:

$$p(\theta\mid x_{{1:r}})=
\frac{{\dfrac{{\prod_{{i=1}}^r p_i(\theta\mid x_i)}}{{p(\theta)^{{r-1}}}}}}
{{\displaystyle\int_\Theta
\dfrac{{\prod_{{i=1}}^r p_i(\vartheta\mid x_i)}}{{p(\vartheta)^{{r-1}}}}\,d\vartheta}}$$

<figure class="paper-figure">
  <img loading="lazy" src="{versioned_asset('assets/chapter/collective-figure-1.png')}" alt="Five-stage collective posterior workflow from empirical trajectories through individual and robust collective posteriors to posterior predictive checks">
  <figcaption>From replicate trajectories to individual posteriors, a robust collective posterior, and predictive checks. {paper_html("collective")} · Fig. 1, CC BY 4.0.</figcaption>
</figure>

### Role of the density floor ε

An outlying replicate can assign vanishing density to the region supported by all others. In a
product, that one near-zero factor can overwhelm the consensus. The robust method replaces each
individual density with a floor,

$$p_\epsilon(\theta\mid x_i)=\max\!\left[p_i(\theta\mid x_i),\epsilon\right],$$
$$p_\epsilon(\theta\mid x_{{1:r}})=
\frac{{\dfrac{{\prod_{{i=1}}^r p_\epsilon(\theta\mid x_i)}}{{p(\theta)^{{r-1}}}}}}
{{\displaystyle\int_\Theta
\dfrac{{\prod_{{i=1}}^r p_\epsilon(\vartheta\mid x_i)}}{{p(\vartheta)^{{r-1}}}}\,d\vartheta}}$$

This ε is a **minimum posterior density**, not the ABC distance tolerance. As $\epsilon\to0$ the
robust result approaches the standard collective. Raising ε limits how strongly unsupported tails
can veto the overlap among replicates; raising it too far discards real information. The workshop
estimates ε for the selected replicate set on a deterministic prior grid and exposes fixed values
for sensitivity analysis.

<figure class="paper-figure">
  <img loading="lazy" src="{versioned_asset('assets/chapter/collective-figure-4.png')}" alt="Individual and collective posteriors and predictive trajectories with weak versus stronger epsilon flooring">
  <figcaption>A near-zero floor lets an outlier pull the collective away from the replicate consensus; a stronger floor restores posterior and predictive agreement. {paper_html("collective")} · Fig. 4, CC BY 4.0.</figcaption>
</figure>''',
        "d74c479a": None,
        "3fce18ac": fr'''## Summary

| Method | Core move | Best role |
|---|---|---|
| **ABC** | retain simulations close to the observation | transparent baseline and prototyping |
| **NPE** | learn $q_\phi(\theta\mid x)$ from simulations | repeated, fast posterior inference |
| **Flexible NPE** | condition on observations and their design mask | supported passage schedules without retraining |
| **Collective posterior** | combine replicate posteriors with prior correction and ε flooring | shared parameters with outlier resistance |

> **Important:** inference is not finished when a posterior appears. Check simulation coverage,
> prior support, replicate sensitivity, and posterior predictions before making a biological claim.

### References

- {paper("sbi")} — the `sbi` software toolkit
- {paper("chuong")} — the experimental-evolution case study
- {paper("collective")} — collective and robust replicate inference
- {paper("avecilla")} — SBI for chemostat adaptation dynamics
- {paper("de")} — CNV stability under serial dilution

*Contact: Nadav Ben Nun · nadavbennun1@mail.tau.ac.il*''',
    }
    return replacements.get(cid, source)


def mechanism_code(stem: str, source: str, cell_index: int) -> str:
    if stem == "6cfee4f5":
        return chuong_code_exercise(cell_index)
    titles = {
        "efcdf8fa": "Three competing genotypes",
        "6cfee4f5": "Standing variation and de novo CNV formation",
        "2539f7c5": "CNV reversion",
        "5f9d90ff": "Two chromosome-loss routes",
    }
    snippets = {
        "efcdf8fa": [("Parameters become rates", "delta_C = 10 ** log_delta_C\ndelta_B = 10 ** log_delta_B"), ("Mutation becomes a matrix", "M = [[1-delta_C-delta_B, 0, 0], ...]"), ("Selection follows mutation", "E = np.diag(w) @ M"), ("Drift becomes a draw", "n = np.random.multinomial(N, p)")],
        "6cfee4f5": [("Log parameters enter biology", "s, m, p0 = 10 ** np.array([...])"), ("Standing variation sets the start", "n[0] = N * (1 - p0)\nn[2] = N * p0"), ("Mutation and selection compose", "E = M @ np.diag(w)"), ("Drift creates replicate histories", "n = np.random.multinomial(N, p)")],
        "2539f7c5": [("Reversion points CNV → single copy", "M = [[1-m, 0], [m, 1]]"), ("Fitness favors the revertant", "diag([1.0, 1.0 + s])"), ("Initial rarity is explicit", "n = [N*(1-p0), N*p0]"), ("Drift samples the next generation", "n = np.random.multinomial(N, p)")],
        "5f9d90ff": [("Two routes leave trisomy", "M[1,0] = mu_tri\nM[2,0] = mu_loh"), ("Each state has its fitness", "S = diag([w_tri, 1, w_loh])"), ("Order is declared", "G = S @ M  # mutate, then select"), ("Passages choose observations", "ret = p[ret_gens]")],
    }
    if stem == "efcdf8fa":
        source = source.replace("E = M @ np.diag(w)", "E = np.diag(w) @ M  # mutate, then select")
    cards = "".join(f'<article><span>{i:02d}</span><h3>{html.escape(label)}</h3><pre><code>{html.escape(code)}</code></pre></article>' for i,(label,code) in enumerate(snippets[stem],1))
    return f'''<section class="mechanism-code lesson-cell" data-cell-id="{stem}"><p class="section-kicker">Now in code</p><h2>{titles[stem]}</h2><p class="mechanism-intro">Follow one generation from biological assumption to code.</p><div class="force-code-grid">{cards}</div><details class="code-panel full-simulator"><summary>Open the complete simulator <span>Python · cell {cell_index}</span></summary><div class="code-toolbar"><span>Reproducible source</span><button class="copy-code" type="button">Copy</button></div><pre><code class="language-python">{html.escape(source)}</code></pre></details></section>'''


def chapter_walkthrough(chapter: str) -> str:
    if chapter == "evolution":
        slides = '''
        <article class="story-slide"><div class="story-copy"><span>01</span><h2>From genotypes to trajectories</h2><p>Represent an evolving population as frequencies of heritable states.</p></div><div class="culture-cartoon" role="img" aria-label="Three laboratory populations with different colored cell mixtures"><i></i><i></i><i></i><div><b></b><b></b><b></b><em></em><em></em><strong></strong></div></div></article>
        <article class="story-slide"><div class="story-copy"><span>02</span><h2>Mutation, selection, and drift</h2><p>Three operators determine each stochastic generation.</p><div class="story-equation">pₜ → M pₜ → fitness × M pₜ → Multinomial(Nₑ, p*)</div></div><div class="force-cartoon"><div>mutation<small>new states</small></div><b>→</b><div>selection<small>unequal growth</small></div><b>→</b><div>drift<small>finite sampling</small></div></div></article>
        <article class="story-slide"><div class="story-copy"><span>03</span><h2>Experimental life cycles</h2><p>Chemostats and serial dilution impose different clocks and drift scales.</p></div><div class="vessel-cartoon"><div class="chemostat-vessel"><i></i><span>continuous flow</span></div><div class="batch-vessels"><i></i><b>→</b><i></i><b>→</b><i></i><span>grow · dilute · repeat</span></div></div></article>
        <article class="story-slide story-goal"><div class="story-copy"><span>04</span><h2>Mechanistic predictions</h2><p>Perturb rates, fitnesses, and effective population size; compare trajectories.</p></div><div class="trajectory-cartoon" aria-hidden="true"><i></i><b></b><em></em><span>frequency</span><small>time</small></div></article>'''
    else:
        abc_equation = ''.join([
            '<div><small>Draw</small>' + math_to_html(r'$\theta\thicksim p(\theta)$') + '</div>',
            '<div><small>Simulate</small>' + math_to_html(r'$x_{\mathrm{sim}}\thicksim p(x\mid\theta)$') + '</div>',
            '<div><small>Accept</small>' + math_to_html(r'$d(x_{\mathrm{sim}},x_{\mathrm{obs}})\leq\varepsilon_{\mathrm{ABC}}$') + '</div>',
        ])
        npe_equation = '<div class="npe-equation"><small>Learned conditional density</small>' + math_to_html(r'$q_\phi(\theta\mid x)\approx p(\theta\mid x)$') + '</div>'
        slides = f'''
        <article class="story-slide sbi-inverse-slide"><div class="story-copy"><span>01</span><h2>The inverse problem</h2><p>Use an observed trajectory to infer a distribution over the parameters that could have generated it.</p></div><svg class="inverse-process-diagram" viewBox="0 0 620 300" role="img" aria-label="Forward simulation maps parameters to synthetic data; inference maps observed data to a posterior distribution"><defs><marker id="inverse-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs><text class="row-title" x="12" y="28">FORWARD MODEL</text><rect class="parameter-node" x="18" y="48" width="126" height="66" rx="12"/><text x="81" y="87" text-anchor="middle">parameters θ</text><line class="diagram-arrow" x1="154" y1="81" x2="196" y2="81"/><rect class="simulator-node" x="207" y="48" width="112" height="66" rx="12"/><text x="263" y="87" text-anchor="middle">simulator</text><line class="diagram-arrow" x1="329" y1="81" x2="373" y2="81"/><g class="trajectory-node"><line x1="385" y1="114" x2="605" y2="114"/><line x1="385" y1="114" x2="385" y2="43"/><path d="M392 107 C440 106 451 95 478 74 S540 50 596 48"/></g><text class="row-title" x="12" y="169">INVERSE PROBLEM</text><g class="data-node"><line x1="18" y1="270" x2="184" y2="270"/><line x1="18" y1="270" x2="18" y2="190"/><circle cx="40" cy="262" r="7"/><circle cx="73" cy="254" r="7"/><circle cx="106" cy="232" r="7"/><circle cx="140" cy="207" r="7"/><circle cx="174" cy="199" r="7"/></g><line class="diagram-arrow" x1="194" y1="230" x2="230" y2="230"/><rect class="inference-node" x="241" y="197" width="92" height="66" rx="12"/><text x="287" y="237" text-anchor="middle">SBI</text><line class="diagram-arrow" x1="343" y1="230" x2="383" y2="230"/><g class="posterior-node"><line x1="395" y1="270" x2="605" y2="270"/><path d="M402 269 C435 268 445 203 488 203 C531 203 541 268 598 269 Z"/></g><text x="500" y="188" text-anchor="middle">posterior p(θ | x<tspan baseline-shift="sub" font-size="14">obs</tspan>)</text></svg></article>
        <article class="story-slide"><div class="story-copy"><span>02</span><h2>Rejection ABC</h2><p>Retain parameters whose simulations are closest to the observation.</p><div class="story-equation typeset-story-equation story-math-stack">{abc_equation}</div></div><div class="abc-cartoon"><span>prior</span><i>simulate</i><b>compare</b><em>keep</em></div></article>
        <article class="story-slide"><div class="story-copy"><span>03</span><h2>Neural posterior estimation</h2><p>Learn a conditional density from simulated parameter–trajectory pairs.</p><div class="story-equation typeset-story-equation">{npe_equation}</div></div><div class="network-cartoon" aria-hidden="true"><div><i></i><i></i><i></i></div><b></b><div><i></i><i></i><i></i><i></i></div><b></b><div><i></i><i></i></div></div></article>
        <article class="story-slide story-goal"><div class="story-copy"><span>04</span><h2>Prediction and model checking</h2><p>Propagate posterior uncertainty into trajectories and derived biological quantities.</p></div><div class="predictive-cartoon" role="img" aria-label="Posterior density flows through a simulator into a predictive trajectory band"><svg class="mini-posterior-density" viewBox="0 0 150 150" aria-hidden="true"><line x1="8" y1="137" x2="144" y2="137"/><path d="M10 136 C39 135 45 38 77 38 C109 38 116 135 142 136 Z"/></svg><b>→</b><div class="mini-simulator">simulate</div><b>→</b><div class="mini-predictive"><i></i><span></span><em></em><strong></strong></div></div></article>'''
    return f'<section class="chapter-walkthrough" aria-label="Chapter outline">{slides}</section>'


def avecilla_model_builder_section() -> str:
    mutation = math_to_html(r'''<h3>Mutation moves probability between states</h3>
<p>In a general discrete-state model, column $j$ of $M$ describes where offspring of genotype
$j$ go. Thus $M_{ij}=\Pr(i\leftarrow j)$ and mutation alone gives</p>
$$x^{(m)}=Mx_t.$$
<p>For the Avecilla states $A,C,B$, only the ancestor creates the two derived genotypes:</p>
$$M_A=\begin{pmatrix}1-\delta_C-\delta_B&0&0\\
\delta_C&1&0\\\delta_B&0&1\end{pmatrix},\qquad
x^{(m)}=M_Ax_t.$$
<p>$\delta_C$ is the per-generation *GAP1* CNV formation probability and $\delta_B$ is the
formation probability of another beneficial genotype.</p>''')
    selection = math_to_html(r'''<h3>Selection reweights reproductive contribution</h3>
<p>Give each genotype a non-negative relative fitness $w_i$. Selection multiplies its post-mutation
frequency and then renormalizes:</p>
$$x_i^{(s)}=\frac{w_i x_i^{(m)}}{\sum_jw_jx_j^{(m)}}.$$
<p>Avecilla uses the ancestor as reference, so</p>
$$w_A=1,\qquad w_C=1+s_C,\qquad w_B=1+s_B.$$
<p>A positive $s_C$ does not guarantee a CNV sweep: the lineage must first be supplied by mutation,
and it competes with the other beneficial state.</p>''')
    drift = math_to_html(r'''<h3>Drift turns an expectation into one realized population</h3>
<p>After mutation and selection define expected frequencies, a Wright–Fisher generation samples
$N_e$ cells:</p>
$$n_{t+1}\sim\operatorname{Multinomial}(N_e,x^{(s)}),\qquad
x_{t+1}=\frac{n_{t+1}}{N_e}.$$
<p>The expectation remains $x^{(s)}$, but replicate trajectories differ. Smaller $N_e$ produces
larger sampling variance; $N_e$ is an effective drift scale, not automatically a cell count.</p>''')
    return f'''<section class="lesson-cell model-builder" id="avecilla-model-builder">
      <p class="section-kicker">Build one generation</p><h2>From a biological picture to three operators</h2>
      <p class="model-builder-intro">Select an evolutionary force. The highlighted part of the population model becomes a general equation and then the corresponding Avecilla equation.</p>
      <div class="model-force-tabs" role="tablist" aria-label="Evolutionary force">
        <button class="active" type="button" role="tab" aria-selected="true" data-model-force="mutation">1 · Mutation</button>
        <button type="button" role="tab" aria-selected="false" data-model-force="selection">2 · Selection</button>
        <button type="button" role="tab" aria-selected="false" data-model-force="drift">3 · Drift</button>
      </div>
      <div class="model-builder-grid">
        <div class="model-illustration" aria-label="Ancestral cells produce CNV and other-beneficial cells, which reproduce unequally before finite sampling">
          <div class="genotype-stage"><span class="state-a">A<small>ancestral</small></span><i>→</i><span class="state-c">C<small>GAP1 CNV</small></span><span class="state-b">B<small>other beneficial</small></span></div>
          <div class="operator-stage"><b data-force-node="mutation">mutation<small>new states</small></b><i>→</i><b data-force-node="selection">selection<small>unequal growth</small></b><i>→</i><b data-force-node="drift">drift<small>finite sample</small></b></div>
          <div class="state-vector">x = (x<sub>A</sub>, x<sub>C</sub>, x<sub>B</sub>)<sup>T</sup></div>
        </div>
        <div class="force-equations" aria-live="polite">
          <article data-force-panel="mutation">{mutation}</article>
          <article data-force-panel="selection" hidden>{selection}</article>
          <article data-force-panel="drift" hidden>{drift}</article>
        </div>
      </div>
    </section>'''


def fit_scope_note(model_name: str) -> str:
    return f'''<aside class="fit-scope-note"><strong>Model-fit preview · {html.escape(model_name)}</strong><p>This comparison asks whether the simulator can reproduce the main pattern in the data. Parameter inference and uncertainty are introduced systematically in Chapter 2.</p></aside>'''


def chuong_standing_variation_section() -> str:
    return '''<section class="station" id="chuong-standing-variation">
      <div class="station-kicker">Interactive mechanism</div><h2>What does the hidden CNV⁻ population change?</h2>
      <p class="prediction">Hold the LTRΔ selection and formation parameters fixed, then compare an almost absent CNV⁻ population (φ = 10⁻⁸) with standing CNV⁻ variation (φ = 10⁻⁴).</p>
      <div class="preset-row" role="group" aria-label="Initial CNV-negative frequency"><button type="button" data-phi-view="low">φ = 10⁻⁸</button><button type="button" data-phi-view="high">φ = 10⁻⁴</button><button class="active" type="button" data-phi-view="both">Compare both</button></div>
      <canvas id="chuong-phi-canvas" width="980" height="430" aria-label="Total GAP1 CNV and reporter-positive CNV trajectories for two initial CNV-negative frequencies"></canvas>
      <p id="chuong-phi-summary" class="plot-summary" aria-live="polite"></p>
      <div class="what-changed"><strong>Biological interpretation.</strong> CNV⁻ cells already carry a <em>GAP1</em> amplification but are invisible to the reporter-defined CNV⁺ curve. Their initial frequency can therefore change total CNV abundance and early competition without looking like de novo reporter amplification.</div>
      <noscript><p class="noscript">Enable JavaScript to compare the two fixed φ values.</p></noscript>
    </section>'''


def chuong_equation_exercise() -> str:
    matrix = math_to_html(r'''<h3>Matrix form</h3>
$$M=\begin{pmatrix}1-\delta-\mu_{SNV}&0&0&0\\
\delta&1&0&0\\0&0&1&0\\\mu_{SNV}&0&0&1\end{pmatrix},\qquad
w=(1,1+s,1+s,1+s_{SNV})^\mathsf{T}$$
$$x^{(m)}=Mx_t,\qquad x^{(s)}=\frac{w\odot x^{(m)}}{\sum_jw_jx_j^{(m)}},\qquad
n_{t+1}\sim\operatorname{Multinomial}(N_e,x^{(s)}).$$''')
    body = f'''<section class="lesson-cell equation-exercise" id="chuong-equation-exercise">
      <p class="section-kicker">Exercise pause</p><h2>Build the Chuong update one force at a time</h2>
      <p>Begin with $x_t=(x_A,x_{{C+}},x_{{C-}},x_B)^\mathsf{{T}}$. Reveal mutation, then selection, then drift; the compact matrix form appears only after the three biological steps are assembled.</p>
      <div class="equation-reveal-controls"><button type="button" data-chuong-step="1">Reveal mutation</button><button type="button" data-chuong-step="2" disabled>Reveal selection</button><button type="button" data-chuong-step="3" disabled>Reveal drift</button><button type="button" id="chuong-equation-reset">Reset</button></div>
      <div class="equation-reveal-sequence" aria-live="polite">
        <article data-chuong-card="1" hidden><b>1 · Mutation</b><p>WT supplies reporter-positive CNVs at $\delta$ and the other-beneficial state at $\mu_{{SNV}}$; CNV⁻ is standing variation.</p></article>
        <article data-chuong-card="2" hidden><b>2 · Selection</b><p>CNV⁺ and CNV⁻ share fitness $1+s$ because both amplify <em>GAP1</em>; the competitor has fitness $1+s_{{SNV}}$.</p></article>
        <article data-chuong-card="3" hidden><b>3 · Drift</b><p>A multinomial draw of size $N_e$ creates one stochastic replicate from the expected frequencies.</p></article>
      </div>
      <div class="chuong-matrix" hidden>{matrix}</div>
    </section>'''
    return math_to_html(body)


def chuong_code_exercise(cell_index: int) -> str:
    blanks = [
        ("Initial CNV⁻ state", "n = np.array([N*(1-phi), 0, N*phi, 0])"),
        ("Mutation", "mutated = M @ p"),
        ("Selection", "weighted = w * mutated"),
        ("Drift", "n = np.random.multinomial(N, weighted / weighted.sum())"),
    ]
    rows = ''.join(f'<label><span>{html.escape(label)}</span><input type="text" autocomplete="off" spellcheck="false" data-code-answer="{html.escape(answer, quote=True)}" aria-label="Complete the {html.escape(label)} line"><small></small></label>' for label, answer in blanks)
    return f'''<section class="mechanism-code lesson-cell code-fill-exercise" data-cell-id="6cfee4f5"><p class="section-kicker">Now in code</p><h2>Complete one Chuong generation</h2><p class="mechanism-intro">The function structure is visible. Fill the four biological lines before revealing the answers.</p><pre class="code-frame"><code class="language-python">def WF_Chuong(log_s, log_delta, log_phi, N, generations):
    s, delta, phi = 10 ** np.array([log_s, log_delta, log_phi])
    w = np.array([1, 1+s, 1+s, 1+S_SNV])
    M = chuong_mutation_matrix(delta, M_SNV)
    # complete the initial state and generation loop</code></pre><div class="code-fill-list">{rows}</div><pre class="code-frame code-frame-tail"><code>    return np.asarray(cnv_frequency)</code></pre><div class="button-row"><button type="button" id="check-chuong-code">Check lines</button><button type="button" id="reveal-chuong-code">Reveal answers</button><button type="button" id="reset-chuong-code">Reset</button></div><p id="chuong-code-summary" class="plot-summary" aria-live="polite"></p></section>'''


def model_equivalence_section() -> str:
    source = r'''## The Avecilla and Zhou models are the same three-state operator

The biological labels change, but the transition graph does not: one source genotype can produce
either of two descendants, the three states have relative fitnesses, and drift samples the next
population.

| Mathematical role | Avecilla et al. | Zhou et al. |
|---|---|---|
| source state | ancestor $A$ | trisomic |
| first descendant | *GAP1* CNV $C$ | wild type |
| second descendant | other beneficial $B$ | LOH |
| two transition rates | $\delta_C,\delta_B$ | $\mu_{WT},\mu_{LOH}$ |
| relative fitnesses | $1,1+s_C,1+s_B$ | $w_{Tri},1,w_{LOH}$ |

The difference is parameterization, not model topology. In the displayed Avecilla fit, the
other-beneficial lineage is represented by fixed competitor parameters. Zhou assigns distinct
route and fitness parameters to the two descendant outcomes rather than using fixed constants for
a generic second mutant.'''
    return f'<section class="lesson-cell prose-cell equivalence-box" id="model-equivalence">{math_to_html(source)}</section>'


def effective_population_section() -> str:
    source = fr'''## Effective population size belongs to the life cycle

$N_e$ is the size of an ideal Wright–Fisher population with the same drift variance as the
experiment. It is not automatically the largest cell count—or even the census average.

### Chemostat: match the variance of neutral frequency change

{paper("avecilla")} simulated two neutral alleles at chemostat steady state and matched their
one-generation conditional variance:

$$\mathrm{{Var}}(p'\mid p)=\frac{{p(1-p)}}{{N_e}},\qquad
\widehat N_e=\frac{{p(1-p)}}{{\frac1t\sum_{{j=1}}^t\mathrm{{Var}}(p'_j\mid p_j)}}.$$

Their chemostat conditions gave $N_e=3.3\times10^8$, about two-thirds of the steady-state census.

### Serial dilution: bottlenecks dominate the harmonic mean

{paper("de")} used 1:64 transfers, corresponding to six doublings per cycle. For generation-level
sizes $N_0,\ldots,N_5$, the appropriate cycle summary is

$$N_e^{{cycle}}=\frac6{{\sum_{{g=0}}^5 1/N_g}}.$$

If $N_g=N_0 2^g$ and $N_0=6.25\times10^5$ cells, the culture reaches $4\times10^7$ cells before
transfer, yet $N_e^{{cycle}}\approx1.90\times10^6$. The harmonic mean stays close to the bottleneck
because drift is strongest when the culture is smallest.

<div class="ne-contrast"><span><b>Chemostat</b> infer $N_e$ from neutral variance at steady state</span><i>vs.</i><span><b>Serial dilution</b> harmonically average the changing population sizes</span></div>'''
    return f'<section class="lesson-cell prose-cell ne-section" id="effective-population-size">{math_to_html(source)}</section>'


def inverse_problem_figure() -> str:
    """A focused identifiability figure for the opening of the SBI lesson."""
    return '''<section class="lesson-cell inverse-example" id="inverse-example" data-cell-id="cdda66b7">
      <div class="inverse-example-copy"><p class="section-kicker">Identifiability</p><h2>Several mechanisms can match the sampled trajectory</h2><p>The colored curves come from different combinations of selection, formation rate, and initial CNV frequency. At the measured generations, all three remain plausible. Inference must therefore preserve their joint uncertainty rather than select one curve by eye.</p><div class="hypothesis-key"><span class="hypothesis-a"><b>A</b> faster formation, weaker selection</span><span class="hypothesis-b"><b>B</b> intermediate rates</span><span class="hypothesis-c"><b>C</b> slower formation, stronger selection</span><span class="observed-key"><b></b> observed frequencies</span></div></div>
      <svg class="overlap-figure" viewBox="0 0 720 390" role="img" aria-label="Three parameter combinations produce similar CNV frequency trajectories at the observed generations"><line class="axis" x1="72" y1="320" x2="682" y2="320"/><line class="axis" x1="72" y1="320" x2="72" y2="38"/><text class="axis-label" x="380" y="375" text-anchor="middle">Generation</text><text class="axis-label" x="22" y="180" text-anchor="middle" transform="rotate(-90 22 180)">CNV frequency</text><path class="curve-a" d="M75 312 C170 311 247 294 316 239 S408 95 520 65 S628 58 676 57"/><path class="curve-b" d="M75 313 C176 312 250 299 318 243 S410 104 520 69 S628 60 676 59"/><path class="curve-c" d="M75 314 C182 313 256 302 322 247 S415 112 522 73 S630 62 676 61"/><g class="observed-points"><circle cx="112" cy="311" r="7"/><circle cx="182" cy="306" r="7"/><circle cx="252" cy="288" r="7"/><circle cx="322" cy="242" r="7"/><circle cx="392" cy="155" r="7"/><circle cx="462" cy="91" r="7"/><circle cx="542" cy="67" r="7"/><circle cx="622" cy="59" r="7"/></g><text class="tick-label" x="72" y="342">0</text><text class="tick-label" x="676" y="342" text-anchor="end">120</text><text class="tick-label" x="56" y="320" text-anchor="end">0</text><text class="tick-label" x="56" y="48" text-anchor="end">1</text></svg>
      <p class="concept-caption"><strong>Same observations, different parameters.</strong> This overlap is the inferential problem that ABC and NPE must represent.</p>
    </section>'''


def posterior_prediction_section() -> str:
    """Connect parameter inference to a derived biological prediction."""
    figure_url = versioned_asset("assets/chapter/chuong-diversity-figure-3b.jpg")
    source = fr'''## Posterior prediction of CNV-lineage diversity

Parameter estimation is often only an intermediate step. Posterior prediction propagates every
plausible parameter value through the simulator and then calculates a quantity that was not used
as the inference target.

### Purpose

In {paper("chuong")}, CNV-frequency trajectories constrain formation rate and selection. The fitted
model is then used to ask a different biological question: **how many effectively distinct CNV
lineages should coexist through time?**

<figure class="diversity-prediction-figure">
  <img loading="eager" fetchpriority="high" width="1000" height="961" src="{figure_url}" alt="Chuong Figure 3B showing posterior predictions of CNV Shannon diversity through time for wild type, LTR deletion, ARS deletion, and ALL deletion strains">
  <figcaption>The vertical axis is logarithmic and reports the effective number of CNV lineages; curves show the posterior mean. {paper_html("chuong")} · Fig. 3B.</figcaption>
</figure>

<div class="diversity-rank" aria-label="Predicted final diversity rank"><span class="diversity-wt">WT <b>highest</b></span><i>›</i><span class="diversity-ltr">LTRΔ</span><i>›</i><span class="diversity-all">ALLΔ</span><i>›</i><span class="diversity-ars">ARSΔ <b>lowest</b></span></div>

### Calculation

At one time point, the simulator gives the fraction $f_i(t)$ carried by each CNV lineage. A simple
lineage count would treat a lineage at 40% frequency exactly like one at 0.001%. Shannon entropy
instead uses the complete frequency distribution, so abundant lineages contribute more than rare
ones.

<div class="diversity-calculation" aria-label="Three steps from lineage frequencies to effective diversity"><span><b>1</b><strong>Lineage fractions</strong><small>$f_1(t),f_2(t),\ldots$ sum to one</small></span><i>→</i><span><b>2</b><strong>Shannon entropy</strong><small>summarizes how evenly frequency is distributed</small></span><i>→</i><span><b>3</b><strong>Effective diversity</strong><small>converts entropy back to a lineage count</small></span></div>

$$H^{{(m)}}(t)=-\sum_i f_i^{{(m)}}(t)\log f_i^{{(m)}}(t),\qquad
D^{{(m)}}(t)=\exp\!\left[H^{{(m)}}(t)\right]$$

The interpretation is direct: ten equally abundant lineages give $D=10$. If a few lineages
dominate the same population, $D$ is smaller. The calculation is repeated for every posterior draw
$m$, producing uncertainty in diversity at each time point.

### Result

Predicted diversity rises rapidly during selection and then saturates. At the final time point it
ranges from about $1.6\times10^4$ lineages for ARSΔ to $3.2\times10^5$ for wild type, with the rank
order **WT > LTRΔ > ALLΔ > ARSΔ**, mirroring inferred CNV formation rates. Because this model omits
competition, clonal interference, and recurrent formation, the absolute values are likely
overestimates; the between-strain comparison is the more defensible prediction.
'''
    return f'<section class="lesson-cell prose-cell prediction-box" id="posterior-predictions">{math_to_html(source)}</section>'


def render_notebook(key: str, interactions: dict[str, list[str]]) -> tuple[str, list[dict]]:
    nb = nbformat.read(NOTEBOOKS[key], as_version=4)
    blocks, coverage = [], []
    indexed_cells = list(enumerate(nb.cells))
    if key == "evolution":
        rank = {cid: i for i, cid in enumerate(EVOLUTION_CELL_ORDER)}
        indexed_cells.sort(key=lambda row: rank.get(row[1].get("id"), len(rank) + row[0]))
    for index, cell in indexed_cells:
        cid = cell.get("id") or f"cell-{index}"
        source = cell.source or ""
        if not source.strip():
            coverage.append({"index": index, "cell_id": cid, "type": cell.cell_type,
                             "status": "excluded", "reason": "empty cell"})
            continue
        anchor = f'cell-{cid}'
        if cell.cell_type == "markdown":
            curated = curated_markdown(key, cid, source)
            if curated is None:
                status, reason = "excluded", "replaced by the visual chapter walkthrough or deliberately removed from the lesson"
            else:
                body = math_to_html(curated)
                blocks.append(f'<section class="lesson-cell prose-cell" id="{anchor}" data-cell-id="{cid}">{body}</section>')
                status, reason = "included", ""
        else:
            if key == "evolution" and cid in EVOLUTION_SIMULATOR_CELLS:
                blocks.append(mechanism_code(cid, source, index))
                status, reason = "deliberately_collapsed", "simulator force map is visible and the complete implementation is progressively disclosed"
            elif key == "sbi" and cid in SBI_VISIBLE_CODE:
                labels = {"67d19e3c": "The ABC loop", "88e4194b": "Train the neural posterior", "90cc8760": "Condition and sample from NPE"}
                code = html.escape(source)
                open_attr = " open" if cid != "88e4194b" else ""
                blocks.append(f'<section class="purposeful-code lesson-cell" id="{anchor}" data-cell-id="{cid}"><p class="section-kicker">Code worth keeping</p><h2>{labels[cid]}</h2><details class="code-panel"{open_attr}><summary>Python implementation <span>notebook cell {index}</span></summary><div class="code-toolbar"><span>Reproducible source</span><button class="copy-code" type="button">Copy</button></div><pre><code class="language-python">{code}</code></pre></details></section>')
                status, reason = ("included", "") if open_attr else ("deliberately_collapsed", "training implementation is available on demand")
            elif key == "sbi" and cid == "cdda66b7":
                blocks.append(inverse_problem_figure())
                status, reason = "included", "notebook output replaced by a focused identifiability diagram"
            elif (key == "evolution" and cid in EVOLUTION_OUTPUT_ONLY) or (key == "sbi" and cid in SBI_OUTPUT_ONLY):
                output = "".join(output_html(dict(out), cid, j, figures_only=True) for j, out in enumerate(cell.get("outputs", [])))
                story = OUTPUT_STORIES.get(cid, "")
                if key == "evolution" and cid in {"54e9f8de", "104a1da1", "66cce2fa", "65a843c6", "2e99f96f"}:
                    model_labels = {"54e9f8de": "Avecilla Wright–Fisher", "104a1da1": "three-state diagnostic", "66cce2fa": "Chuong Wright–Fisher", "65a843c6": "De Wright–Fisher", "2e99f96f": "Zhou et al."}
                    blocks.append(fit_scope_note(model_labels[cid]))
                if story:
                    blocks.append(f'<section class="figure-story lesson-cell" id="{anchor}" data-cell-id="{cid}"><p>{html.escape(story)}</p>{output}</section>')
                elif output:
                    blocks.append(f'<section class="lesson-cell output-only-cell" id="{anchor}" data-cell-id="{cid}">{output}</section>')
                if story or output:
                    status, reason = "included", "notebook code hidden; only its scientific figure or scientific interpretation is used in the presentation"
                else:
                    status, reason = "excluded", "notebook code hidden; the interactive station replaces this stored implementation"
            else:
                status, reason = "excluded", "implementation detail is outside this presentation's learning goals"
        coverage.append({"index": index, "cell_id": cid, "type": cell.cell_type,
                         "status": status, "reason": reason})
        if key == "evolution" and cid == "3282b174":
            blocks.append(avecilla_model_builder_section())
            blocks.append(effective_population_section())
        for interaction in interactions.get(cid, []):
            if not (key == "evolution" and interaction == "chuong-parameter-challenge"):
                blocks.append(station_markup(interaction))
        if key == "evolution" and cid == "104a1da1":
            blocks.append(chuong_standing_variation_section())
        if key == "evolution" and cid == "97c3a42c":
            blocks.append(chuong_equation_exercise())
        if key == "evolution" and cid == "b4a01787":
            blocks.append(model_equivalence_section())
        if key == "sbi" and cid == "098a16bd":
            blocks.append(posterior_prediction_section())
    coverage.sort(key=lambda row: row["index"])
    return "\n".join(blocks), coverage


def station_markup(name: str) -> str:
    fallback = versioned_asset(f'assets/fallback/{name}.png')
    common_start = f'''<section class="station answer-gated" id="{name}" data-station="{name}">
      <div class="station-kicker">Interactive station</div>'''
    common_end = f'''<div class="static-fallback"><img src="{fallback}" alt="Representative static result for {name.replace('-', ' ')}"><p>This representative result remains available when scripting is unavailable.</p></div>
      <noscript><p class="noscript">JavaScript is off; use the static result and conclusion above.</p></noscript></section>'''
    if name == "evolution-playground":
        body = '''<h2>Predict a three-genotype chemostat trajectory</h2><p class="prediction">Question: will the GAP1 CNV, another beneficial lineage, or drift dominate the population?</p>
        <div class="preset-row evo-presets"><button data-evo-preset="fit">Published fit</button><button data-evo-preset="cnv">CNV sweep</button><button data-evo-preset="competing">Competing beneficial</button><button data-evo-preset="drift">Small population</button><button data-evo-preset="order">Order effect</button></div>
        <div class="interactive-grid"><form class="controls" id="evo-controls">
          <label><span>CNV formation log₁₀(δ<sub>C</sub>)</span><output id="evo-delta-c-label"></output><input id="evo-delta-c" type="range" min="-7" max="-2" step="0.05" value="-4.2"></label>
          <label><span>Other-beneficial log₁₀(δ<sub>B</sub>)</span><output id="evo-delta-b-label"></output><input id="evo-delta-b" type="range" min="-7" max="-2" step="0.05" value="-5"></label>
          <label><span>CNV advantage s<sub>C</sub></span><output id="evo-s-c-label"></output><input id="evo-s-c" type="range" min="0" max="0.14" step="0.002" value="0.07"></label>
          <label><span>Other-beneficial advantage s<sub>B</sub></span><output id="evo-s-b-label"></output><input id="evo-s-b" type="range" min="0" max="0.14" step="0.002" value="0.001"></label>
          <label><span>Generations</span><output id="evo-duration-label"></output><input id="evo-duration" type="range" min="20" max="140" step="5" value="120"></label>
          <label><span>Effective population Nₑ</span><select id="evo-ne"><option>1000</option><option>10000</option><option>100000</option><option>1000000</option><option selected>330000000</option></select></label>
          <label><span>Replicate trajectories</span><input id="evo-reps" type="number" min="1" max="24" value="8"></label>
          <label><span>Seed</span><input id="evo-seed" type="number" min="0" value="20260825"></label>
          <div class="button-row"><button id="evo-play" type="button">Play</button><button class="reset" type="reset">Reset</button></div></form>
          <div class="viz"><div class="plot-pair order-comparison"><figure><figcaption>Mutation → selection → drift</figcaption><canvas id="evo-canvas" width="760" height="440" aria-label="Population-frequency trajectories when mutation occurs before selection"></canvas></figure><figure><figcaption>Selection → mutation → drift</figcaption><canvas id="evo-order-canvas" width="760" height="440" aria-label="Population-frequency trajectories when selection occurs before mutation"></canvas></figure></div><p class="plot-summary" id="evo-summary" aria-live="polite"></p><div class="composition" id="evo-composition"></div><div class="what-changed"><strong>What if selection happens first?</strong> <span id="evo-order-summary">At the published mutation rates the two conventions are nearly indistinguishable; the Order effect preset makes their non-commutativity visible.</span></div></div></div>'''
    elif name == "dfe-example":
        body = '''<h2>An <em>s</em>-DFE at a glance</h2><p class="prediction">The x-axis is the selection coefficient carried by a newly formed CNV; height is its relative probability under a gamma-shaped distribution of fitness effects.</p>
        <div class="interactive-grid"><form class="controls" id="dfe-controls">
          <label>Mean effect s̄ <output id="dfe-mean-label"></output><input id="dfe-mean" type="range" min="0.01" max="0.09" step="0.0025" value="0.045"></label>
          <label>Gamma shape <output id="dfe-shape-label"></output><input id="dfe-shape" type="range" min="0.7" max="5" step="0.1" value="2"></label>
          <button type="reset">Reset</button></form>
          <div class="viz"><canvas id="dfe-canvas" width="760" height="400" aria-label="Gamma-shaped distribution of selection coefficients"></canvas><p id="dfe-summary" class="plot-summary" aria-live="polite"></p></div></div>
        <div class="what-changed"><strong>Evolutionary consequence.</strong> <span id="dfe-change"></span></div>'''
    elif name == "chuong-parameter-challenge":
        body = '''<h2>Infer selection, formation, and initial frequency</h2><p class="prediction">Each round generates a synthetic CNV-frequency dataset. Estimate log₁₀(s), log₁₀(δ), and log₁₀(φ); the score is determined by parameter RMSE.</p>
        <div class="interactive-grid"><form class="controls" id="chuong-challenge-controls">
          <label>log₁₀(s) <output id="chuong-guess-s-label"></output><input id="chuong-guess-s" type="range" min="-1.3" max="-0.45" step="0.01" value="-0.8"></label>
          <label>log₁₀(δ) <output id="chuong-guess-m-label"></output><input id="chuong-guess-m" type="range" min="-6" max="-3.8" step="0.02" value="-4.8"></label>
          <label>log₁₀(φ) <output id="chuong-guess-p0-label"></output><input id="chuong-guess-p0" type="range" min="-7" max="-3" step="0.02" value="-4.5"></label>
          <div class="button-row"><button id="chuong-score" type="button">Score guess</button><button id="chuong-new" type="button">New observation</button><button type="reset">Reset guess</button></div></form>
          <div class="viz"><canvas id="chuong-challenge-canvas" width="760" height="430" aria-label="Synthetic CNV-frequency observations and trajectory implied by the current parameter estimate"></canvas><p id="chuong-challenge-summary" class="plot-summary" aria-live="polite"></p><div id="chuong-score-card" class="score-card" aria-live="polite"></div></div></div>
        <div class="what-changed"><strong>Score.</strong> RMSE is computed directly across the three log₁₀ parameters; points = 100 / (1 + RMSE). A perfect guess earns 100.</div>'''
    elif name == "zhou-model-playground":
        body = '''<h2>Explore competing chromosome-loss routes</h2><p class="prediction">Prediction: does the trisomic population resolve mainly through euploid recovery, LOH, or a fitness-driven mixture?</p>
        <div class="preset-row"><button data-zhou-model-preset="fit">Published fit</button><button data-zhou-model-preset="wt">WT route</button><button data-zhou-model-preset="loh">LOH route</button><button data-zhou-model-preset="fitness">Fitness reversal</button></div>
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
        body = '''<h2>Sensitivity to an outlying replicate</h2><p class="prediction">Question: which replicate has the most leverage on the collective posterior?</p>
        <div class="preset-row"><button data-coll-select="all">Select all</button><button data-coll-select="clean">Clean only</button><button data-coll-select="outliers">Outliers only</button><button id="coll-loo">Leave one out</button></div>
        <div class="interactive-grid"><div class="controls"><fieldset id="replicate-checks"><legend>Replicates entering sensitivity analysis</legend></fieldset><label>Investigate <select id="coll-investigate"></select></label><label>Robustness floor <select id="coll-epsilon"><option value="auto:0.80">Estimate from 80th percentile</option><option value="auto:0.90">Estimate from 90th percentile</option><option value="auto:0.95" selected>Estimate from 95th percentile</option><option value="auto:0.99">Estimate from 99th percentile</option><option value="0">Fixed log₁₀ ε = 0</option><option value="-10">Fixed log₁₀ ε = −10</option><option value="-100">Fixed log₁₀ ε = −100</option><option value="-1000">Fixed log₁₀ ε = −1000</option></select><output id="coll-epsilon-value">Estimating…</output></label><label>R7 displacement from consensus <output id="contam-label">1.0×</output><input id="contam-strength" type="range" min="0" max="1.5" step="0.1" value="1"></label><button id="coll-reset" type="button">Reset</button></div>
        <div class="viz"><canvas id="collective-trajectory-canvas" width="760" height="310" aria-label="Selected replicate trajectories"></canvas><canvas id="collective-posterior-canvas" width="760" height="310" aria-label="Individual, standard collective, and robust collective posterior densities"></canvas><p id="collective-summary" class="plot-summary" aria-live="polite"></p></div></div>
        <details class="method-note"><summary>What is evaluated—and what is sampled?</summary><p>The browser evaluates a normalized three-parameter joint posterior grid, applies the ε floor to each full joint density, aggregates, and only then marginalizes to the displayed selection axis. It does not draw posterior samples. To estimate ε deterministically, a uniform midpoint grid discretizes the prior, each selected replicate posterior is evaluated at every grid point, and the chosen density percentile is used. This is a grid approximation to the published prior-draw heuristic; the published production implementation samples the high-dimensional collective target with Sampling-importance-resampling (SIR). Fixed controls report log₁₀ ε, matching the paper; calculations convert these values to natural-log density internally.</p></details>
        <div class="what-changed"><strong>Move R7, then compare.</strong> At 0×, R7 is centered on the shared truth; increasing displacement moves its trajectory and its posterior center in all three parameters. The gold Standard collective should follow R7, while the green Robust collective should resist it. Fixed log₁₀ ε = −1000 intentionally removes that resistance. Exclusion remains sensitivity analysis, not a data-discarding rule.</div>'''
    elif name == "zhou-schedule-designer":
        body = '''<h2>Design a passage schedule</h2><p class="prediction">Prediction: which passages constrain transition rates, and which constrain relative fitness?</p>
        <div class="preset-row"><button data-schedule="odd">Odd passages</button><button data-schedule="even">Even passages</button><button data-schedule="early">Early only</button><button data-schedule="late">Late only</button><button data-schedule="sparse">Sparse</button><button data-schedule="full">Full schedule</button><button data-schedule="zero">Passage 0 only</button></div>
        <div class="passage-grid" id="passage-grid"></div><label class="inline-toggle"><input id="reveal-withheld" type="checkbox" checked> Reveal withheld observations</label>
        <div class="plot-pair"><canvas id="zhou-trajectory-canvas" width="720" height="380" aria-label="Latent trajectory and selected or withheld passage observations"></canvas><canvas id="zhou-posterior-canvas" width="720" height="380" aria-label="Four Zhou posterior marginals for the selected schedule"></canvas></div>
        <canvas id="zhou-ppc-canvas" width="1100" height="340" aria-label="Posterior predictive intervals at observed and withheld passages"></canvas>
        <div class="button-row"><button id="zhou-reset" type="button">Reset</button></div>'''
    elif name == "guess-parameter":
        body = '''<h2>Run rejection ABC</h2><p class="prediction">Choose a simulation budget and acceptance quantile. ABC keeps the closest simulated trajectories; watch the accepted parameter cloud tighten as ε decreases.</p>
        <div class="interactive-grid"><form class="controls" id="abc-controls"><label>Acceptance quantile <output id="abc-quantile-label">5%</output><input id="abc-quantile" type="range" min="1" max="25" step="1" value="5"></label><label>Simulation budget <select id="abc-sims"><option>250</option><option selected>1000</option><option>3000</option><option>10000</option></select></label><label>Seed <input id="abc-seed" type="number" min="0" value="20260825"></label><button id="abc-run" type="button">Run ABC progressively</button><button type="reset">Reset</button><label class="progress-label" for="abc-progress">Simulation progress <output id="abc-progress-label">0 / 1000</output></label><progress id="abc-progress" max="1000" value="0"></progress><div id="abc-milestones" class="milestone-row" aria-label="ABC simulation milestones"></div></form><div class="viz"><canvas id="abc-trajectory-canvas" width="760" height="350" aria-label="Observed trajectory and accepted ABC simulations"></canvas><canvas id="guess-canvas" width="760" height="350" aria-label="ABC posterior marginals for selection, mutation, and initial frequency"></canvas><p id="abc-summary" class="plot-summary" aria-live="polite"></p></div></div>'''
    else:
        body = '''<h2>Diagnose posterior-predictive mismatch</h2><p class="prediction">Compare the orange observations with the blue posterior-predictive distribution and select the most plausible biological or measurement explanation.</p><div class="preset-row" id="ppc-cases"></div><div class="diagnosis-row"><label><input type="radio" name="diagnosis" value="well-specified"> This culture looks plausible</label><label><input type="radio" name="diagnosis" value="noise"> Measurements are noisier than assumed</label><label><input type="radio" name="diagnosis" value="outlier"> One time point may be contaminated</label><label><input type="radio" name="diagnosis" value="support"> Biology lies outside the training range</label><label><input type="radio" name="diagnosis" value="structure"> The simulator misses a biological process</label></div><canvas id="ppc-canvas" width="1100" height="430" aria-label="Observed trajectory and posterior predictive band"></canvas><div class="button-row"><button id="ppc-reveal" type="button">Check my diagnosis</button><button id="ppc-reset" type="button">Reset</button></div><p id="ppc-summary" class="plot-summary" aria-live="polite"></p><div class="what-changed"><strong>Interpret carefully.</strong> A PPC localizes tension between observation and prediction. It can suggest a failure mode, but the pattern rarely proves one unique cause.</div>'''
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
    rng=np.random.default_rng(SEED); grid=np.linspace(-1.8,-.35,180); truth_theta=np.array([-.9,-5,-5.5])
    labels=["R1","R2","R3","R4","R5","R6 subtle","R7 outlier"]
    bounds=np.array([[-2,0],[-7,-2],[-8,-2]],float)
    means=np.array([[-.92,-5.05,-5.45],[-.87,-4.9,-5.65],[-.91,-5.1,-5.35],
                    [-.85,-4.95,-5.55],[-.94,-5,-5.6],[-.77,-4.65,-5],
                    [-.48,-3.7,-6.9]])
    sds=np.array([[.22,.55,.7],[.2,.5,.65],[.23,.58,.75],[.19,.48,.65],
                  [.21,.52,.7],[.23,.55,.7],[.2,.45,.55]])
    normalizers=[]; factors=[]; trajectories=[]
    for mean,sd in zip(means,sds):
        masses=[.5*(math.erf((hi-m)/(s*math.sqrt(2)))-math.erf((lo-m)/(s*math.sqrt(2))))
                for (lo,hi),m,s in zip(bounds,mean,sd)]
        normalizers.append(float(np.log(masses).sum()))
        post=normal_pdf(grid,mean[0],sd[0]); post/=np.trapz(post,grid)
        factors.append(np.log(post+1e-30))
        trajectories.append(np.clip(wf_deterministic(mean)+rng.normal(0,.012,12),0,1).tolist())
    selection_prior_log=np.full(len(grid),-math.log(bounds[0,1]-bounds[0,0]))
    write_json(DATA/"collective_lab.json",{
        "grid":grid.tolist(),"truth":float(truth_theta[0]),"truth_theta":truth_theta.tolist(),
        "parameter_names":["log10_s","log10_delta","log10_phi"],"parameter_bounds":bounds.tolist(),
        "posterior_means":means.tolist(),"posterior_sds":sds.tolist(),
        "posterior_log_normalizers":normalizers,
        "prior_log_density":float(-np.log(bounds[:,1]-bounds[:,0]).sum()),
        "prior_log":selection_prior_log.tolist(),
        "replicate_log_posteriors":np.array(factors).tolist(),
        "joint_grid_shape":[len(grid),25,25],
        "epsilon_calibration":{"method":"replicate-set midpoint prior-grid density percentile","grid_points_per_axis":17,"default_quantile":.95},
        "labels":labels,"types":["clean"]*5+["subtle","outlier"],"contaminated_index":6,"trajectories":trajectories,
        "generations":[8,21,29,37,50,58,66,79,87,95,108,116],
        "formula":"aggregate full joint log posterior_i, subtract (r-1) log prior, then marginalize"
    })
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
      "zhou-model-playground":lambda ax:[ax.plot(x,latent[:,i],color=["#607069","#e67e22","#8e44ad"][i],lw=2.5) for i in range(3)],
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

    # Render this focused output independently of the stale notebook PNG:
    # predictions use Avecilla orange and observations use blue.
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
    for i,row in enumerate(ltr): ax.plot(chu_g,row,"o-",color="#577d91",ms=4.8,lw=1,label="Data" if i==0 else None,zorder=5)
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
    chapter_header = "" if active == "home" else f'<header class="chapter-hero"><p class="eyebrow">{eyebrow}</p><h1>{html.escape(title)}</h1></header>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{html.escape(title)}</title><link rel="icon" href="assets/favicon.svg"><link rel="stylesheet" href="{stylesheet}"><script>window.MathJax={{options:{{enableMenu:false}},chtml:{{scale:1.04,matchFontHeight:false}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/mml-chtml.js"></script><noscript><style>.static-fallback{{display:block}}</style></noscript></head><body data-page="{active}"><a class="skip-link" href="#main">Skip to lesson</a><header class="site-header"><a class="wordmark" href="index.html"><span>θ</span> SBI for experimental evolution</a><nav aria-label="Workshop chapters">{nav}</nav><div class="header-actions"><button id="reset-all" type="button">Reset all</button></div></header><div class="progress-track" aria-hidden="true"><span id="reading-progress"></span></div><main id="main">{chapter_header}{content}</main><footer><p>SBI for experimental evolution</p><nav><a href="index.html">Start</a><a href="evolution.html">Evolution</a><a href="sbi.html">SBI</a></nav></footer>{shared_scripts}{script_tags}</body></html>'''


def landing_page() -> str:
    content = '''<section class="landing-hero"><div><h1>Simulation-based inference for experimental evolution</h1><div class="button-row lesson-choice"><a class="secondary-link" href="evolution.html">Chapter 01</a><a class="secondary-link" href="sbi.html">Chapter 02</a></div></div><div class="hero-orbit" aria-hidden="true"><span></span><span></span><span></span><b>θ</b></div></section>
    <section class="objectives"><p class="section-kicker">Workshop outcomes</p><h2>What you will be able to do</h2><div class="objective-grid"><article><b>01</b><h3>Read the mechanism</h3><p>Translate mutation, selection, and drift into population-frequency trajectories.</p></article><article><b>02</b><h3>Reason with uncertainty</h3><p>Compare ABC, NPE, collective evidence, and posterior predictive checks.</p></article><article><b>03</b><h3>Design observations</h3><p>See how a passage mask changes what the same experiment can identify.</p></article></div></section>
    <section class="chapter-cards" aria-label="Choose a lesson"><a href="evolution.html"><span>Chapter 01</span><h2>Evolutionary simulators</h2><p>Mechanistic models of allele-frequency change.</p><strong>Click to open lesson</strong></a><a href="sbi.html"><span>Chapter 02</span><h2>Simulation-based inference</h2><p>Posterior inference from stochastic simulators.</p><strong>Click to open lesson</strong></a></section>
    <section class="phone-entry"><div><p class="section-kicker">Bring the workshop to your bench</p><h2>Open this page on your phone</h2><code>nadavbennun1.github.io/evo-simulators-tutorial</code></div><a href="https://nadavbennun1.github.io/evo-simulators-tutorial/" aria-label="Open the workshop landing page"><img src="assets/workshop-qr.svg" alt="QR code for this workshop landing page" width="220" height="220"></a></section>
    <section class="primer-journey" id="foundations-primer"><header class="primer-heading"><div><p class="section-kicker">Foundations</p><h2>One population, one continuous inference story</h2><p>Follow a CNV from its first appearance to a testable posterior prediction.</p></div><span class="time-badge">15 minutes</span></header><div class="primer-story">
      <article class="primer-step"><div class="step-marker">01</div><div class="step-copy"><p class="scene-label">Begin with a culture</p><h3>A population is a changing composition</h3><p>Each colored cell represents a <strong>genotype</strong>, a heritable state. Its <strong>frequency</strong> is its fraction of the population, and all state frequencies sum to one.</p><dl class="concept-notes"><div><dt>Mutation rate</dt><dd>Probability per generation that one state produces another.</dd></div><div><dt>Fitness and <em>s</em></dt><dd>Expected reproductive success; <em>s</em> measures advantage relative to a reference.</dd></div></dl></div><div class="primer-illustration population-sketch" role="img" aria-label="A mostly ancestral population changes into a mixture of three genotypes"><div class="dot-row"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><b></b></div><span>generations</span><div class="dot-row late"><i></i><i></i><b></b><b></b><em></em><em></em><b></b><em></em></div></div></article>
      <article class="primer-step"><div class="step-marker">02</div><div class="step-copy"><p class="scene-label">Advance one generation</p><h3>Mechanism sets the expectation; drift selects one future</h3><p>A Wright–Fisher simulator applies mutation and selection, normalizes the expected frequencies, and samples a finite next generation. <strong>Genetic drift</strong> is this sampling variation.</p><div class="primer-equation">p<sub>t</sub> → mutation → selection → Multinomial(N<sub>e</sub>, p*)</div><dl class="concept-notes"><div><dt>Effective population size, N<sub>e</sub></dt><dd>The idealized size producing the experiment’s drift variance—not necessarily its census.</dd></div><div><dt>Simulator</dt><dd>An executable generative model from parameters and random draws to synthetic data.</dd></div></dl></div><div class="force-cartoon primer-force" aria-hidden="true"><div>mutation<small>states appear</small></div><b>→</b><div>selection<small>growth differs</small></div><b>→</b><div>drift<small>sample N<sub>e</sub></small></div></div></article>
      <article class="primer-step"><div class="step-marker">03</div><div class="step-copy"><p class="scene-label">Measure the experiment</p><h3>The population changes between measurements</h3><p>The green curve is the complete genotype-frequency history generated by the model. The experiment observes only selected passages. Orange points can deviate from the curve because sequencing and sampling add measurement error.</p><div class="primer-flow" role="img" aria-label="Parameters pass through an evolutionary simulator to generate a complete frequency history, from which selected passages are measured"><span><b>θ</b><small>parameters</small></span><i>→</i><span><b>Simulator</b><small>evolution</small></span><i>→</i><span><b>x(t)</b><small>full history</small></span><i>→</i><span><b>x<sub>obs</sub></b><small>measured passages</small></span></div><dl class="concept-notes"><div><dt>Unobserved state</dt><dd>The full population history between the measured passages.</dd></div><div><dt>Observation model</dt><dd>How biological frequencies become noisy measurements.</dd></div></dl></div><svg class="sampling-diagram" viewBox="0 0 560 300" role="img" aria-label="A continuous population-frequency curve sampled at five passages with noisy observed points"><line class="axis" x1="68" y1="238" x2="535" y2="238"/><line class="axis" x1="68" y1="238" x2="68" y2="50"/><path class="true-history" d="M72 228 C148 227 190 214 237 176 S309 82 380 60 S476 51 530 50"/><g class="sampling-guides"><line x1="125" y1="238" x2="125" y2="221"/><line x1="220" y1="238" x2="220" y2="188"/><line x1="315" y1="238" x2="315" y2="98"/><line x1="410" y1="238" x2="410" y2="56"/><line x1="500" y1="238" x2="500" y2="50"/></g><g class="measured-points"><circle cx="125" cy="223" r="7"/><circle cx="220" cy="196" r="7"/><circle cx="315" cy="109" r="7"/><circle cx="410" cy="65" r="7"/><circle cx="500" cy="44" r="7"/></g><text class="axis-label" x="302" y="286" text-anchor="middle">Generation</text><text class="axis-label" x="22" y="144" text-anchor="middle" transform="rotate(-90 22 144)">Genotype frequency</text><g class="sampling-legend"><line class="legend-history" x1="88" y1="70" x2="126" y2="70"/><text x="136" y="74">complete history</text><circle class="legend-point" cx="94" cy="92" r="6"/><text x="108" y="96">measured passages</text></g></svg></article>
      <article class="primer-step"><div class="step-marker">04</div><div class="step-copy"><p class="scene-label">Run the inverse problem</p><h3>Bayes updates uncertainty about parameters</h3><p>The <strong>prior</strong> assigns plausibility before observing the data. The <strong>likelihood</strong> evaluates the data under each parameter value. The evidence p(x<sub>obs</sub>) normalizes their product into the <strong>posterior</strong>.</p><div class="primer-equation bayes-equation" role="math" aria-label="Posterior equals likelihood times prior divided by the probability of the observed data"><span>p(θ | x<sub>obs</sub>) =</span><span class="equation-fraction"><span>p(x<sub>obs</sub> | θ) p(θ)</span><span>p(x<sub>obs</sub>)</span></span></div><dl class="concept-notes"><div><dt>Identifiability</dt><dd>Whether distinct parameter values make distinguishable predictions.</dd></div><div><dt>Credible interval</dt><dd>An interval containing a stated fraction of posterior probability, conditional on model and prior.</dd></div></dl></div><svg class="density-sketch" viewBox="0 0 620 150" role="img" aria-label="A broad prior and a likelihood combine into a narrower posterior"><path d="M20 126 C100 126 125 35 210 35 C295 35 315 126 390 126" class="prior"/><path d="M180 126 C245 126 270 54 330 54 C390 54 410 126 470 126" class="likelihood"/><path d="M205 126 C255 126 276 25 322 25 C368 25 382 126 430 126" class="posterior"/><text x="76" y="28">prior</text><text x="390" y="50">likelihood</text><text x="285" y="18">posterior</text><line x1="20" y1="127" x2="590" y2="127"/></svg></article>
      <article class="primer-step"><div class="step-marker">05</div><div class="step-copy"><p class="scene-label">Approximate the intractable likelihood</p><h3>SBI learns from simulations</h3><p><strong>ABC</strong> retains simulations within distance tolerance ε. <strong>NPE</strong> learns a normalized conditional density from many parameter–trajectory pairs; <strong>amortization</strong> makes later observations cheap to analyze.</p><div class="primer-flow sbi-flow" role="img" aria-label="Prior draws generate simulations used by ABC or NPE to infer a posterior"><span><b>Prior draws</b><small>θ₁ … θₙ</small></span><i>→</i><span><b>Simulations</b><small>x₁ … xₙ</small></span><i>→</i><span><b>ABC / NPE</b><small>compare or learn</small></span><i>→</i><span><b>Posterior</b><small>p(θ | x<sub>obs</sub>)</small></span></div><dl class="concept-notes"><div><dt>Calibration</dt><dd>Across repeated datasets, posterior coverage matches its stated probability.</dd></div><div><dt>SIR and ESS</dt><dd>Importance resampling approximates a target; effective sample size reports weight concentration.</dd></div></dl></div><div class="abc-cartoon primer-abc" aria-hidden="true"><span>draw</span><i>simulate</i><b>compare</b><em>infer</em></div></article>
      <article class="primer-step primer-finale"><div class="step-marker">06</div><div class="step-copy"><p class="scene-label">Return to biology</p><h3>A posterior is useful when it predicts and survives checks</h3><p>A <strong>posterior predictive distribution</strong> simulates trajectories—or new derived quantities—from posterior draws. A <strong>PPC</strong> compares these predictions with observations. A <strong>collective posterior</strong> combines independent replicate evidence while correcting repeated prior factors.</p><dl class="concept-notes"><div><dt>Prediction</dt><dd>Propagate posterior uncertainty through the simulator to a biological quantity.</dd></div><div><dt>Model check</dt><dd>Locate systematic disagreements between posterior simulations and data.</dd></div></dl></div><div class="predictive-cartoon primer-predictive" role="img" aria-label="Posterior uncertainty passes through a simulator to prediction and model checking"><svg class="mini-posterior-density" viewBox="0 0 150 150" aria-hidden="true"><line x1="8" y1="137" x2="144" y2="137"/><path d="M10 136 C39 135 45 38 77 38 C109 38 116 135 142 136 Z"/></svg><b>→</b><div class="mini-simulator">simulate</div><b>→</b><div class="mini-predictive"><i></i><span></span><em></em><strong></strong></div></div></article>
      </div><div class="primer-check"><strong>The complete loop:</strong> define states → simulate evolution → observe passages → infer parameters → predict new biology → check the model.</div>
    </section>'''
    # The former foundations walkthrough will be redistributed between the two lessons.
    content = content.split('<section class="primer-journey"', 1)[0]
    return page_shell("Simulation-based inference for experimental evolution", "", "home", content, [])


def implementation_notebook(chapter: str) -> str:
    if chapter == "evolution":
        steps = [("1", "Choose state", "Represent mutually exclusive genotype frequencies as a vector that sums to one."),
                 ("2", "Transform parameters", "Work in log₁₀ space for small formation rates, then exponentiate inside the simulator."),
                 ("3", "Update mechanism", "Apply mutation and relative fitness in a declared order; matrix orientation must match the state vector."),
                 ("4", "Sample drift", "For Wright–Fisher dynamics, draw the next finite population rather than only normalizing expectations."),
                 ("5", "Observe", "Record only scheduled generations and apply a separate measurement model.")]
        note = "Implementation invariant: after every generation, frequencies are finite, non-negative, and sum to one."
    else:
        steps = [("1", "Specify prior", "Set biologically defensible bounds; a posterior cannot recover values outside prior support."),
                 ("2", "Simulate pairs", "Draw θ from the prior, run the simulator, and store (θ, x) with fixed seeds and metadata."),
                 ("3", "Approximate", "ABC ranks a discrepancy; NPE learns a normalized conditional density q(θ|x)."),
                 ("4", "Combine or design", "Correct duplicated priors across replicates and track which observation schedule produced x."),
                 ("5", "Validate", "Use simulation-based calibration, held-out simulations, sensitivity analyses, and PPCs.")]
        note = "Implementation invariant: training, inference, and validation must use the same parameter transforms, state ordering, and observation definition."
    cards = "".join(f'<li><b>{number}</b><div><strong>{title}</strong><p>{description}</p></div></li>' for number,title,description in steps)
    return f'''<section class="implementation-notebook"><p class="section-kicker">Implementation notebook</p><h2>From equation to trustworthy code</h2><ol>{cards}</ol><p class="implementation-invariant">{note}</p></section>'''


def build_content() -> None:
    NOTEBOOK_ASSETS.mkdir(parents=True,exist_ok=True)
    interactions=json.loads((SITE/"interaction_manifest.json").read_text())
    all_coverage={}
    evo,cov=render_notebook("evolution",interactions["evolution_simulators.ipynb"]); all_coverage["evolution_simulators.ipynb"]=cov
    sbi,cov=render_notebook("sbi",interactions["SBI_tutorial.ipynb"]); all_coverage["SBI_tutorial.ipynb"]=cov
    write_json(SITE/"content_map.json",all_coverage)
    (SITE/"index.html").write_text(landing_page())
    evo_nav='<nav class="chapter-nav" aria-label="Chapter navigation"><a href="index.html">Workshop home</a><a href="sbi.html">Next: simulation-based inference</a></nav>'
    sbi_nav='<nav class="chapter-nav" aria-label="Chapter navigation"><a href="evolution.html">Evolutionary simulators</a><a href="index.html">Workshop home</a></nav>'
    (SITE/"evolution.html").write_text(page_shell("Evolutionary simulators","Chapter 01","evolution",'<div class="lesson-layout"><aside class="toc" aria-label="On this page"><button class="toc-toggle" type="button">On this page</button><div class="toc-links"></div></aside><article class="notebook-lesson">'+evo+evo_nav+'</article></div>',["evolution"]))
    sbi_opening = station_markup("chuong-parameter-challenge")
    (SITE/"sbi.html").write_text(page_shell("Simulation-based inference","Chapter 02","sbi",'<div class="lesson-layout"><aside class="toc" aria-label="On this page"><button class="toc-toggle" type="button">On this page</button><div class="toc-links"></div></aside><article class="notebook-lesson">'+sbi_opening+chapter_walkthrough("sbi")+sbi+sbi_nav+'</article></div>',["evolution", "sbi"]))


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
