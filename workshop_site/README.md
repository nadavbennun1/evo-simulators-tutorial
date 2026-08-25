# Drift & Design workshop site

This directory is a statically hosted, project-subpath-safe workshop generated from
`evolution_simulators.ipynb` and `SBI_tutorial.ipynb`. It needs no Python server or
inference API at runtime, and no checkpoint or pickle is copied into the published site.

## Architecture

- `build_site.py` converts notebook Markdown to HTML/MathML, extracts stored outputs,
  inserts interactions at stable cell IDs, and writes the coverage/provenance manifests.
- `interaction_manifest.json` is the cell-ID insertion contract. A missing referenced ID
  fails the tests.
- `content_map.json` accounts for every Markdown and code cell. Long code is included but
  deliberately collapsed; only empty cells are excluded, with a reason.
- `js/science.js` holds the small testable scientific kernel. `evolution.js` and `sbi.js`
  own the interactive stations; `plots.js` provides dependency-free canvas primitives.
- `data/zhou_*.f32` contains compact, little-endian float32 arrays derived offline from
  the three real Zhou models. `data/zhou_manifest.json` defines all shapes and encoding.

## Scientific provenance

The Zhou designer uses checkpoint seeds 0–2 from `../zhou_npe_models`. Every checkpoint
hash is verified against its metadata. All 4,096 masks over passages 1–12 are evaluated;
passage 0 is implicit and always supplied. Each mask stores 48 draws per checkpoint (144
balanced ensemble draws), posterior quantiles, and per-checkpoint quantiles. The shared
latent trajectory, shared noisy observation, seeds, training summaries, and schedule
encoding are recorded in `data/zhou_manifest.json` and `provenance.json`.

The training viewer is a genuine 100-epoch teaching-scale conditional diagonal-Gaussian
NPE trained on 3,840 deterministic Wright–Fisher simulations with observation noise. It
is explicitly pedagogical and is not the production Zhou ensemble. It stores the full
loss history and 25 real checkpoint snapshots.

## Build and test

From `tutorials/`:

```bash
python -m pip install -r workshop_site/requirements-build.txt
python workshop_site/build_site.py
python -m pytest workshop_site/tests -q
# If pytest is unavailable locally, the same plain-assert suite has a zero-dependency runner:
python workshop_site/tests/run_checks.py
python serve_workshop.py
```

The launcher binds fixed forwarded port 8765, matching the `sbi_for_growth` IDE workflow.
Use the IDE's **Open in Browser** notification or open port 8765 from the Ports panel.
The process stays in the foreground; the IDE's red stop button or Ctrl-C closes the socket,
removes its PID record, and confirms the port is released.

For an optional detached preview, use:

```bash
python workshop_site/scripts/preview.py start
```

To test project-page behavior, serve `tutorials/` and open `/workshop_site/`; all links
are relative.

The managed preview records only its own PID, refuses to kill unrelated listeners, and
verifies that the port is reusable when stopped:

```bash
python workshop_site/scripts/preview.py status
python workshop_site/scripts/preview.py stop
```

Use `--port 8766` with any of these commands if another application legitimately owns
port 8765. Running `serve` instead of `start` keeps the server in the foreground; Ctrl-C
closes the socket and removes the PID record.

Rebuild the expensive derived scientific assets only when checkpoints, scientific seeds,
or the source simulator change:

```bash
PYTHONPATH=../evodesign/src MPLCONFIGDIR=/tmp/workshop-mpl \
  python workshop_site/build_site.py --scientific-assets
```

After either notebook changes, run the ordinary build and tests. The generated content
and `content_map.json` will synchronize to the new cells. Update
`interaction_manifest.json` only if an insertion cell ID intentionally changes.

## Deployment

Public GitHub Pages is configured to use GitHub Actions. The workflow in
`.github/workflows/workshop-pages.yml` verifies the checked-in derived assets, builds/tests
the site, and deploys `workshop_site/` from `main` or a manual dispatch. It does not
regenerate the three-model Zhou asset bank in CI. The checkpoint pickle files remain local
and are not committed or uploaded; CI verifies their compact, hash-recorded derivatives
instead.

## Bundle size and limitations

Run `du -sh workshop_site` and `find workshop_site -type f -printf '%s %p\n' | sort -nr`
for an exact current summary. The dominant asset is `zhou_draws.f32`; model weights remain
outside the deploy directory. Canvas stations include textual live summaries, keyboard
controls, reduced-motion behavior, and representative static fallbacks. Automated checks
cover content, hashes, formulas, numerical parity, links, syntax, and a static-server smoke
test; visual browser inspection remains part of the documented manual checklist when a
browser runner is unavailable.

## Add an interaction

Add a station ID beneath a stable notebook cell in `interaction_manifest.json`, add its
markup in `station_markup()`, implement behavior in the appropriate focused JavaScript
module, generate a fallback figure, and extend the station/content tests.
