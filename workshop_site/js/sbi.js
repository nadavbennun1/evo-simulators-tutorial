(() => {
  "use strict";
  const {$, $$, fetchJSON, fetchF32, normal, softmax, quantile} = Workshop;
  const P = WorkshopPlots, S = WorkshopScience;

  function gaussianCurve(mean, sd, lo, hi, n = 100) {
    const xs = Array.from({length: n}, (_, i) => lo + (hi - lo) * i / (n - 1));
    const ys = xs.map(x => normal(x, mean, Math.max(sd, 1e-4)));
    const peak = Math.max(...ys);
    return {xs, ys: ys.map(y => y / peak)};
  }

  function kde(values, lo, hi, n = 100) {
    const xs = Array.from({length: n}, (_, i) => lo + (hi - lo) * i / (n - 1));
    const spread = Math.max(quantile(values, .95) - quantile(values, .05), (hi - lo) * .04);
    const bandwidth = Math.max(spread * Math.pow(values.length, -.2) * .45, (hi - lo) * .012);
    const ys = xs.map(x => values.reduce((sum, value) => sum + Math.exp(-.5 * ((x - value) / bandwidth) ** 2), 0));
    const peak = Math.max(...ys, 1e-12);
    return {xs, ys: ys.map(y => y / peak)};
  }

  async function trainingViewer() {
    const canvas = $("#loss-canvas");
    if (!canvas) return;
    const data = await fetchJSON("data/training_viewer.json"), slider = $("#epoch-slider");
    let timer = null;
    const selected = () => data.snapshots.reduce((a, b) => Math.abs(b.epoch - +slider.value) < Math.abs(a.epoch - +slider.value) ? b : a);

    function drawTrainingPosterior(snapshot) {
      const posterior = $("#training-posterior-canvas"), f = P.frame(posterior, 0, 1.08, 0, 3);
      const bounds = [[-2, 0], [-7, -2], [-8, -2]], names = ["log s", "log δ", "log φ"], colors = [P.C.tri, P.C.blue, P.C.clay];
      bounds.forEach(([lo, hi], i) => {
        const curve = gaussianCurve(snapshot.mean[i], snapshot.sd[i], lo, hi);
        P.line(f, curve.xs.map(x => i + (x - lo) / (hi - lo) * .82 + .09), curve.ys, colors[i], 2.3);
        const truthX = i + (data.truth[i] - lo) / (hi - lo) * .82 + .09;
        P.line(f, [truthX, truthX], [0, 1], P.C.ink, 1, 1, [4, 3]);
        P.text(f, names[i], i + .5, 1.04, {align: "center", font: "bold 11px system-ui"});
      });
      P.legend(f, [{label: "posterior density", color: P.C.tri}, {label: "truth", color: P.C.ink}]);

      const ppc = $("#training-ppc-canvas"), pf = P.frame(ppc, 0, 1, 8, 116);
      P.band(pf, data.generations, snapshot.ppc_q05, snapshot.ppc_q95, P.C.blue, .2);
      P.line(pf, data.generations, snapshot.ppc_median, P.C.blue, 2.2);
      P.points(pf, data.generations, data.observation, P.C.clay, 3.5);
      P.legend(pf, [{label: "90% PPC + median", color: P.C.blue}, {label: "fixed observation", color: P.C.clay}]);
    }

    function draw() {
      const epoch = +slider.value, snapshot = selected();
      const losses = [...data.train_loss.slice(1), ...data.validation_loss.slice(1)].filter(Number.isFinite);
      const y0 = Math.min(...losses) - .25, y1 = Math.max(...losses) + .25;
      const f = P.frame(canvas, y0, y1, 1, 100), xs = Array.from({length: 100}, (_, i) => i + 1);
      P.line(f, xs, data.train_loss.slice(1), P.C.blue, 2);
      P.line(f, xs, data.validation_loss.slice(1), P.C.clay, 2);
      P.line(f, [epoch, epoch], [y0, y1], P.C.ink, 1, 1, [4, 4]);
      P.legend(f, [{label: "training", color: P.C.blue}, {label: "validation", color: P.C.clay}, {label: "selected epoch", color: P.C.ink}]);
      drawTrainingPosterior(snapshot);
      $("#epoch-label").textContent = String(epoch);
      $("#epoch-snap").textContent = epoch === snapshot.epoch ? `Showing genuine checkpoint epoch ${snapshot.epoch}.` : `Epoch ${epoch} snaps to genuine checkpoint ${snapshot.epoch}.`;
      $("#training-summary").textContent = `Checkpoint ${snapshot.epoch}: posterior means log₁₀(s) ${snapshot.mean[0].toFixed(2)}, log₁₀(δ) ${snapshot.mean[1].toFixed(2)}, log₁₀(φ) ${snapshot.mean[2].toFixed(2)}. Truth: ${data.truth.join(", ")}. The PPC band is generated from 96 posterior draws.`;
    }

    slider.addEventListener("input", draw);
    $$('[data-epoch]').forEach(button => button.addEventListener("click", () => { slider.value = button.dataset.epoch === "best" ? data.best_validation_epoch : button.dataset.epoch; draw(); }));
    $("#epoch-play").addEventListener("click", event => {
      if (timer) { clearInterval(timer); timer = null; event.target.textContent = "Play"; return; }
      event.target.textContent = "Pause";
      timer = setInterval(() => {
        slider.value = (+slider.value + 1) % 101; draw();
        if (+slider.value === 100) { clearInterval(timer); timer = null; event.target.textContent = "Play"; }
      }, 120);
    });
    $("#epoch-reset").addEventListener("click", () => { slider.value = 0; draw(); });
    addEventListener("resize", draw); draw();
  }

  async function collectiveLab() {
    const posteriorCanvas = $("#collective-posterior-canvas");
    if (!posteriorCanvas) return;
    const data = await fetchJSON("data/collective_lab.json"), box = $("#replicate-checks"), investigate = $("#coll-investigate"), epsilonControl = $("#coll-epsilon");
    data.labels.forEach((label, i) => {
      box.insertAdjacentHTML("beforeend", `<label><input type="checkbox" value="${i}" checked> ${label}</label>`);
      investigate.insertAdjacentHTML("beforeend", `<option value="${i}">${label}</option>`);
    });
    let leave = 0;
    const chosen = () => $$("input[type=checkbox]", box).filter(x => x.checked).map(x => +x.value);
    function density(logs) {
      const values = softmax(logs), dx = data.grid[1] - data.grid[0], z = values.reduce((a, b) => a + b, 0) * dx;
      return values.map(x => x / z);
    }
    function adjustedLogPosterior(replicate, j, strength) {
      const base = data.replicate_log_posteriors[replicate][j];
      return replicate === 6 ? data.prior_log[j] + strength * (base - data.prior_log[j]) : base;
    }
    function drawTrajectories(ids) {
      const c = $("#collective-trajectory-canvas"), f = P.frame(c, 0, 1, 8, 116), focus = +investigate.value;
      data.trajectories.forEach((trajectory, i) => P.line(f, data.generations, trajectory, i === focus ? P.C.clay : P.C.blue, i === focus ? 3 : 1, ids.includes(i) ? .75 : .12));
      P.line(f, data.generations, data.trajectories[focus], P.C.clay, 2.5);
      P.legend(f, [{label: "selected replicates", color: P.C.blue}, {label: "investigated replicate", color: P.C.clay}]);
    }
    function draw() {
      const ids = chosen(), strength = +$("#contam-strength").value, epsilon = +epsilonControl.value;
      $("#contam-label").textContent = `${strength.toFixed(1)}×`;
      if (!ids.length) {
        P.frame(posteriorCanvas, 0, 1, data.grid[0], data.grid.at(-1));
        $("#collective-summary").textContent = "Select at least one replicate."; drawTrajectories(ids); return;
      }
      const individualLogs = ids.map(i => data.grid.map((_, j) => Math.max(adjustedLogPosterior(i, j, strength), epsilon)));
      const collective = data.grid.map((_, j) => individualLogs.reduce((sum, row) => sum + row[j], 0) - Math.max(0, ids.length - 1) * data.prior_log[j]);
      const naive = data.grid.map((_, j) => individualLogs.reduce((sum, row) => sum + row[j], 0));
      const dens = density(collective), naiveDensity = density(naive), individuals = individualLogs.map(density);
      const maxDensity = Math.max(...dens, ...naiveDensity, ...individuals.flat()) * 1.08;
      const f = P.frame(posteriorCanvas, 0, maxDensity, data.grid[0], data.grid.at(-1));
      individuals.forEach((values, k) => P.line(f, data.grid, values, data.types[ids[k]] === "outlier" ? P.C.clay : P.C.blue, 1, .3));
      P.line(f, data.grid, naiveDensity, P.C.gold, 2, 1, [5, 3]); P.line(f, data.grid, dens, P.C.tri, 3);
      P.line(f, [data.truth, data.truth], [0, maxDensity], P.C.ink, 1.5, 1, [4, 4]);
      P.legend(f, [{label: "individual", color: P.C.blue}, {label: "naïve product", color: P.C.gold}, {label: "collective", color: P.C.tri}, {label: "truth", color: P.C.ink}]);
      const cdf = []; dens.reduce((sum, value) => { cdf.push(sum + value); return sum + value; }, 0);
      const total = cdf.at(-1), at = q => data.grid[cdf.findIndex(x => x >= q * total)];
      const mean = data.grid.reduce((sum, x, j) => sum + x * dens[j], 0) / dens.reduce((a, b) => a + b, 0);
      $("#collective-summary").textContent = `ε = ${epsilon}; ${ids.length} replicate${ids.length === 1 ? "" : "s"}. Prior-adjusted mean ${mean.toFixed(3)}; 90% interval [${at(.05).toFixed(3)}, ${at(.95).toFixed(3)}]; truth ${data.truth}. The floor is applied to the frozen per-replicate log-posterior grid.`;
      drawTrajectories(ids);
    }
    box.addEventListener("change", draw); investigate.addEventListener("change", draw); epsilonControl.addEventListener("change", draw); $("#contam-strength").addEventListener("input", draw);
    $$('[data-coll-select]').forEach(button => button.addEventListener("click", () => {
      $$("input", box).forEach((input, i) => { input.checked = button.dataset.collSelect === "all" || (button.dataset.collSelect === "clean" && data.types[i] === "clean") || (button.dataset.collSelect === "outliers" && data.types[i] !== "clean"); }); draw();
    }));
    $("#coll-loo").addEventListener("click", () => { $$("input", box).forEach((input, i) => { input.checked = i !== leave; }); investigate.value = leave; leave = (leave + 1) % data.labels.length; draw(); });
    $("#coll-reset").addEventListener("click", () => { $$("input", box).forEach(input => { input.checked = true; }); $("#contam-strength").value = 1; epsilonControl.value = -1000; draw(); });
    addEventListener("resize", draw); draw();
  }

  const zhouDet = (theta, p0) => S.zhouDeterministic(theta, p0, 120).filter((_, i) => i % 10 === 0);

  async function zhouDesigner() {
    const trajectoryCanvas = $("#zhou-trajectory-canvas");
    if (!trajectoryCanvas) return;
    const [manifest, draws, quantiles, seeds] = await Promise.all([fetchJSON("data/zhou_manifest.json"), fetchF32("data/zhou_draws.f32"), fetchF32("data/zhou_quantiles.f32"), fetchF32("data/zhou_seed_quantiles.f32")]);
    const grid = $("#passage-grid");
    for (let passage = 0; passage <= 12; passage++) grid.insertAdjacentHTML("beforeend", `<label class="${passage === 0 ? "locked" : ""}"><input type="checkbox" value="${passage}" ${passage === 0 ? 'checked disabled aria-locked="true"' : ""}> P${passage}${passage === 0 ? "<small>required</small>" : ""}</label>`);
    const checks = $$("input", grid), presets = {odd: [1, 3, 5, 7, 9, 11], even: [2, 4, 6, 8, 10, 12], early: [1, 2, 3, 4], late: [9, 10, 11, 12], sparse: [3, 7, 11], full: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], zero: []};
    const mask = () => checks.slice(1).reduce((sum, input) => sum + (input.checked ? (1 << (+input.value - 1)) : 0), 0);
    const q = (scheduleMask, parameter, k) => quantiles[(scheduleMask * 4 + parameter) * 3 + k];
    const sq = (scheduleMask, seed, parameter, k) => seeds[((scheduleMask * 3 + seed) * 4 + parameter) * 3 + k];

    function drawPosterior(scheduleMask) {
      const c = $("#zhou-posterior-canvas"), f = P.frame(c, 0, 1.1, 0, 4), full = 4095;
      for (let parameter = 0; parameter < 4; parameter++) {
        const [lo, hi] = manifest.parameter_bounds[parameter], median = q(scheduleMask, parameter, 1);
        const sd = Math.max((q(scheduleMask, parameter, 2) - q(scheduleMask, parameter, 0)) / 3.29, 1e-3), curve = gaussianCurve(median, sd, lo, hi);
        P.line(f, curve.xs.map(x => parameter + .08 + .84 * (x - lo) / (hi - lo)), curve.ys, P.C.tri, 2.7);
        for (let seed = 0; seed < 3; seed++) {
          const seedMedian = sq(scheduleMask, seed, parameter, 1), seedSd = Math.max((sq(scheduleMask, seed, parameter, 2) - sq(scheduleMask, seed, parameter, 0)) / 3.29, 1e-3), seedCurve = gaussianCurve(seedMedian, seedSd, lo, hi);
          P.line(f, seedCurve.xs.map(x => parameter + .08 + .84 * (x - lo) / (hi - lo)), seedCurve.ys, P.C.blue, .8, .35);
        }
        const fullX = parameter + .08 + .84 * (q(full, parameter, 1) - lo) / (hi - lo), truthX = parameter + .08 + .84 * (manifest.truth[parameter] - lo) / (hi - lo);
        P.line(f, [fullX, fullX], [0, .9], P.C.gold, 1, 1, [3, 3]); P.line(f, [truthX, truthX], [0, 1], P.C.clay, 1.5, 1, [5, 3]);
        P.text(f, ["Tri→WT", "Tri fitness", "LOH fitness", "Tri→LOH"][parameter], parameter + .5, 1.05, {align: "center", font: "bold 10px system-ui"});
      }
      P.legend(f, [{label: "schedule posterior", color: P.C.tri}, {label: "seed runs", color: P.C.blue}, {label: "full schedule", color: P.C.gold}, {label: "truth", color: P.C.clay}]);
    }

    function drawPpc(scheduleMask, selected) {
      const count = manifest.ensemble_draws_per_mask, start = scheduleMask * count * 4, trajectories = [];
      for (let i = 0; i < count; i++) trajectories.push(zhouDet(Array.from(draws.slice(start + i * 4, start + i * 4 + 4)), manifest.p0));
      const xs = Array.from({length: 13}, (_, i) => i), c = $("#zhou-ppc-canvas"), f = P.frame(c, 0, 1, 0, 12), colors = [P.C.tri, P.C.blue, P.C.clay];
      for (let state = 0; state < 3; state++) {
        const lo = xs.map(passage => quantile(trajectories.map(t => t[passage][state]), .05)), median = xs.map(passage => quantile(trajectories.map(t => t[passage][state]), .5)), hi = xs.map(passage => quantile(trajectories.map(t => t[passage][state]), .95));
        P.band(f, xs, lo, hi, colors[state], .12); P.line(f, xs, median, colors[state], 2);
        const observed = xs.filter(x => selected.has(x)); P.points(f, observed, observed.map(x => manifest.observed_passages[x][state]), colors[state], 3);
      }
      P.legend(f, [{label: "Tri PPC", color: colors[0]}, {label: "WT PPC", color: colors[1]}, {label: "LOH PPC", color: colors[2]}]);
    }

    function draw() {
      const scheduleMask = mask(), selected = new Set([0, ...checks.slice(1).filter(x => x.checked).map(x => +x.value)]);
      const f = P.frame(trajectoryCanvas, 0, 1, 0, 12), xs = Array.from({length: 13}, (_, i) => i), colors = [P.C.tri, P.C.blue, P.C.clay];
      colors.forEach((color, state) => {
        P.line(f, xs, manifest.latent_passages.map(x => x[state]), color, 2, .38);
        const observed = xs.filter(x => selected.has(x)), withheld = xs.filter(x => !selected.has(x));
        P.points(f, observed, observed.map(x => manifest.observed_passages[x][state]), color, 4);
        if ($("#reveal-withheld").checked) P.points(f, withheld, withheld.map(x => manifest.observed_passages[x][state]), color, 3, true);
      });
      P.legend(f, [{label: "Trisomic", color: colors[0]}, {label: "Wild type", color: colors[1]}, {label: "LOH", color: colors[2]}]);
      drawPosterior(scheduleMask); drawPpc(scheduleMask, selected);
      const widths = Array.from({length: 4}, (_, parameter) => q(scheduleMask, parameter, 2) - q(scheduleMask, parameter, 0));
      $("#zhou-change").textContent = `Schedule mask ${scheduleMask}, ${selected.size} measured passages including locked passage 0. Mean 90% interval width ${(widths.reduce((a, b) => a + b, 0) / 4).toFixed(3)}.`;
    }
    checks.forEach(x => x.addEventListener("change", draw)); $("#reveal-withheld").addEventListener("change", draw);
    $$('[data-schedule]').forEach(button => button.addEventListener("click", () => { const selected = new Set(presets[button.dataset.schedule]); checks.slice(1).forEach(x => { x.checked = selected.has(+x.value); }); draw(); }));
    $("#zhou-reset").addEventListener("click", () => $$('[data-schedule="odd"]')[0].click()); addEventListener("resize", draw); $$('[data-schedule="odd"]')[0].click();
  }

  async function abcAndPpcExercises() {
    const abcPosterior = $("#guess-canvas"), ppc = $("#ppc-canvas");
    if (!abcPosterior && !ppc) return;
    const data = await fetchJSON("data/exercises.json");

    if (abcPosterior) {
      const trajectoryCanvas = $("#abc-trajectory-canvas"), generations = data.generations;
      let lastRun = null;
      function runAbc() {
        const budget = +$("#abc-sims").value, acceptedQuantile = +$("#abc-quantile").value / 100;
        const R = S.mulberry32(+$("#abc-seed").value || 0), observed = data.guess_examples[0].trajectory;
        const candidates = Array.from({length: budget}, () => {
          const theta = [-2 + 2 * R(), -7 + 5 * R(), -8 + 6 * R()], trajectory = S.chuongDeterministic(theta, generations);
          const distance = Math.sqrt(trajectory.reduce((sum, value, i) => sum + (value - observed[i]) ** 2, 0) / trajectory.length);
          return {theta, trajectory, distance};
        }).sort((a, b) => a.distance - b.distance);
        const nAccepted = Math.max(3, Math.floor(budget * acceptedQuantile));
        lastRun = {budget, acceptedQuantile, observed, accepted: candidates.slice(0, nAccepted), epsilon: candidates[nAccepted - 1].distance}; drawAbc();
      }
      function drawAbc() {
        if (!lastRun) return;
        const {budget, acceptedQuantile, observed, accepted, epsilon} = lastRun;
        const tf = P.frame(trajectoryCanvas, 0, 1, generations[0], generations.at(-1));
        accepted.slice(0, 20).forEach(candidate => P.line(tf, generations, candidate.trajectory, P.C.orange, 1, .16));
        P.line(tf, generations, accepted[0].trajectory, P.C.orange, 2.6); P.line(tf, generations, observed, P.C.blue, 1.5); P.points(tf, generations, observed, P.C.blue, 4);
        P.legend(tf, [{label: "observed", color: P.C.blue}, {label: "accepted simulations", color: P.C.orange}]);
        const pf = P.frame(abcPosterior, 0, 1.08, 0, 3), bounds = [[-2, 0], [-7, -2], [-8, -2]], names = ["log₁₀(s)", "log₁₀(δ)", "log₁₀(φ)"], truth = data.guess_examples[0].truth, summaries = [];
        bounds.forEach(([lo, hi], parameter) => {
          const values = accepted.map(candidate => candidate.theta[parameter]), curve = kde(values, lo, hi);
          P.line(pf, [parameter + .08, parameter + .92], [.08, .08], P.C.muted, 1, .5);
          P.line(pf, curve.xs.map(x => parameter + .08 + .84 * (x - lo) / (hi - lo)), curve.ys, P.C.orange, 2.6);
          const truthX = parameter + .08 + .84 * (truth[parameter] - lo) / (hi - lo); P.line(pf, [truthX, truthX], [0, 1], P.C.blue, 1.4, 1, [4, 3]);
          P.text(pf, names[parameter], parameter + .5, 1.04, {align: "center", font: "bold 10px system-ui"});
          summaries.push(`${names[parameter]} ${quantile(values, .5).toFixed(2)} [${quantile(values, .05).toFixed(2)}, ${quantile(values, .95).toFixed(2)}]`);
        });
        P.legend(pf, [{label: "ABC posterior", color: P.C.orange}, {label: "uniform prior", color: P.C.muted}, {label: "truth", color: P.C.blue}]);
        $("#abc-summary").textContent = `${accepted.length}/${budget} simulations accepted at the ${(acceptedQuantile * 100).toFixed(0)}% quantile; ε = ${epsilon.toFixed(4)}. Posterior medians and 90% intervals: ${summaries.join("; ")}.`;
      }
      $("#abc-quantile").addEventListener("input", event => { $("#abc-quantile-label").textContent = `${event.target.value}%`; });
      $("#abc-run").addEventListener("click", runAbc); $("#abc-controls").addEventListener("reset", () => setTimeout(() => { $("#abc-quantile-label").textContent = "5%"; runAbc(); }));
      addEventListener("resize", drawAbc); runAbc();
    }

    if (ppc) {
      let index = 0, result = "";
      const row = $("#ppc-cases");
      data.ppc_cases.forEach((_, i) => row.insertAdjacentHTML("beforeend", `<button type="button" data-case="${i}">${i + 1}. Dataset ${String.fromCharCode(65 + i)}</button>`));
      function drawPpc() {
        const item = data.ppc_cases[index], f = P.frame(ppc, 0, 1.05, 8, 116);
        P.band(f, data.generations, item.q05, item.q95, P.C.blue, .18); P.line(f, data.generations, item.median, P.C.blue, 2);
        P.line(f, data.generations, item.observation, P.C.orange, 1.8); P.points(f, data.generations, item.observation, P.C.orange, 4);
        P.legend(f, [{label: "90% PPC + median", color: P.C.blue}, {label: "observed data", color: P.C.orange}]);
        $("#ppc-summary").textContent = result || `Dataset ${String.fromCharCode(65 + index)}: choose the mismatch pattern that best explains where the orange observation departs from the blue predictive band.`;
      }
      $$('[data-case]', row).forEach(button => button.addEventListener("click", () => { index = +button.dataset.case; result = ""; $$('input[name="diagnosis"]').forEach(input => { input.checked = false; }); drawPpc(); }));
      $("#ppc-reveal").addEventListener("click", () => {
        const choice = $('input[name="diagnosis"]:checked'), item = data.ppc_cases[index];
        if (!choice) { result = "Choose one diagnosis first, then check your answer."; drawPpc(); return; }
        result = `${choice.value === item.kind ? "Correct." : "Not quite."} Known generating case: ${item.title}. ${item.reason} The PPC pattern flags tension, but does not by itself prove that cause.`; drawPpc();
      });
      $("#ppc-reset").addEventListener("click", () => { index = 0; result = ""; $$('input[name="diagnosis"]').forEach(input => { input.checked = false; }); drawPpc(); });
      addEventListener("resize", drawPpc); drawPpc();
    }
  }

  trainingViewer().catch(console.error);
  collectiveLab().catch(console.error);
  zhouDesigner().catch(console.error);
  abcAndPpcExercises().catch(console.error);
})();
