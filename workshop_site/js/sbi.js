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
      if (replicate !== data.contaminated_index) return base;
      const mean = S.effectivePosteriorMean(data, replicate, strength)[0], sd = data.posterior_sds[replicate][0];
      return -.5 * ((data.grid[j] - mean) / sd) ** 2 - Math.log(sd * Math.sqrt(2 * Math.PI));
    }
    function drawTrajectories(ids, strength) {
      const c = $("#collective-trajectory-canvas"), f = P.frame(c, 0, 1, 8, 116), focus = +investigate.value;
      const trajectoryAt = i => {
        if (i !== data.contaminated_index) return data.trajectories[i];
        const original = S.chuongDeterministic(data.posterior_means[i], data.generations);
        const shifted = S.chuongDeterministic(S.effectivePosteriorMean(data, i, strength), data.generations);
        return shifted.map((value, j) => Math.max(0, Math.min(1, value + data.trajectories[i][j] - original[j])));
      };
      data.trajectories.forEach((_, i) => P.line(f, data.generations, trajectoryAt(i), i === focus ? P.C.clay : P.C.blue, i === focus ? 3 : 1, ids.includes(i) ? .75 : .12));
      P.line(f, data.generations, trajectoryAt(focus), P.C.clay, 2.5);
      P.legend(f, [{label: "selected replicates", color: P.C.blue}, {label: "investigated replicate", color: P.C.clay}]);
    }
    function draw() {
      const ids = chosen(), strength = +$("#contam-strength").value, epsilonSetting = epsilonControl.value;
      $("#contam-label").textContent = `${strength.toFixed(1)}×`;
      if (!ids.length) {
        P.frame(posteriorCanvas, 0, 1, data.grid[0], data.grid.at(-1));
        $("#collective-summary").textContent = "Select at least one replicate."; drawTrajectories(ids, strength); return;
      }
      const rawIndividualLogs = ids.map(i => data.grid.map((_, j) => adjustedLogPosterior(i, j, strength)));
      const estimated = epsilonSetting.startsWith("auto:");
      const epsilonQuantile = estimated ? +epsilonSetting.split(":")[1] : null;
      // Posterior calculations use natural logs internally. Fixed controls follow
      // the paper's log10(epsilon) convention, so convert them before flooring.
      const logEpsilon = estimated ? S.estimateCollectiveLogEpsilon(data, ids, strength, epsilonQuantile) : +epsilonSetting * Math.LN10;
      const result = S.collectiveJointSelectionMarginals(data, ids, strength, logEpsilon);
      const standardDensity = density(result.standard), robustDensity = density(result.robust), individuals = rawIndividualLogs.map(density);
      const maxDensity = Math.max(...standardDensity, ...robustDensity, ...individuals.flat()) * 1.08;
      const f = P.frame(posteriorCanvas, 0, maxDensity, data.grid[0], data.grid.at(-1));
      individuals.forEach((values, k) => P.line(f, data.grid, values, data.types[ids[k]] === "outlier" ? P.C.clay : P.C.blue, 1, .3));
      P.line(f, data.grid, standardDensity, P.C.gold, 2.4, 1, [5, 3]); P.line(f, data.grid, robustDensity, P.C.tri, 3);
      P.line(f, [data.truth, data.truth], [0, maxDensity], P.C.ink, 1.5, 1, [4, 4]);
      P.legend(f, [{label: "individual", color: P.C.blue}, {label: "Standard collective", color: P.C.gold}, {label: "Robust collective", color: P.C.tri}, {label: "truth", color: P.C.ink}]);
      const summarize = values => {
        const cdf=[]; values.reduce((sum,value)=>{cdf.push(sum+value);return sum+value},0);
        const total=cdf.at(-1),at=q=>data.grid[cdf.findIndex(value=>value>=q*total)];
        return{mean:data.grid.reduce((sum,x,j)=>sum+x*values[j],0)/values.reduce((a,b)=>a+b,0),lo:at(.05),hi:at(.95)};
      };
      const standardSummary=summarize(standardDensity),robustSummary=summarize(robustDensity);
      const calibrationCount=ids.length*data.epsilon_calibration.grid_points_per_axis**3;
      const log10Epsilon = logEpsilon / Math.LN10;
      $("#coll-epsilon-value").textContent = (estimated ? "Estimated " + (epsilonQuantile*100).toFixed(0) + "th percentile: " : "Using ") + "log₁₀ ε = " + log10Epsilon.toFixed(3);
      const shiftedMean=S.effectivePosteriorMean(data,data.contaminated_index,strength);
      $("#collective-summary").textContent = "R7 center (" + shiftedMean.map(value=>value.toFixed(2)).join(", ") + "). " + (estimated ? "Set-specific ε from " + calibrationCount.toLocaleString() + " deterministic prior-grid evaluations. " : "Fixed floor. ") + "Standard mean " + standardSummary.mean.toFixed(3) + "; robust mean " + robustSummary.mean.toFixed(3) + " with 90% interval [" + robustSummary.lo.toFixed(3) + ", " + robustSummary.hi.toFixed(3) + "]; truth " + data.truth + ".";
      drawTrajectories(ids, strength);
    }
    box.addEventListener("change", draw); investigate.addEventListener("change", draw); epsilonControl.addEventListener("change", draw);
    $("#contam-strength").addEventListener("input", () => { investigate.value = String(data.contaminated_index); draw(); });
    $$('[data-coll-select]').forEach(button => button.addEventListener("click", () => {
      $$("input", box).forEach((input, i) => { input.checked = button.dataset.collSelect === "all" || (button.dataset.collSelect === "clean" && data.types[i] === "clean") || (button.dataset.collSelect === "outliers" && data.types[i] !== "clean"); }); draw();
    }));
    $("#coll-loo").addEventListener("click", () => { $$("input", box).forEach((input, i) => { input.checked = i !== leave; }); investigate.value = leave; leave = (leave + 1) % data.labels.length; draw(); });
    $("#coll-reset").addEventListener("click", () => { $$("input", box).forEach(input => { input.checked = true; }); $("#contam-strength").value = 1; epsilonControl.value = "auto:0.95"; draw(); });
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
    const displayOrder = [1, 0, 2, 3], displayLabels = ["w_Het", "μ_Het→WT", "w_LOH", "μ_Het→LOH"];
    const mask = () => checks.slice(1).reduce((sum, input) => sum + (input.checked ? (1 << (+input.value - 1)) : 0), 0);
    const q = (scheduleMask, parameter, k) => quantiles[(scheduleMask * 4 + parameter) * 3 + k];
    const sq = (scheduleMask, seed, parameter, k) => seeds[((scheduleMask * 3 + seed) * 4 + parameter) * 3 + k];

    function drawPosterior(scheduleMask) {
      const c = $("#zhou-posterior-canvas"), f = P.frame(c, 0, 1.1, 0, 4), full = 4095;
      for (let displayPosition = 0; displayPosition < 4; displayPosition++) {
        const parameter = displayOrder[displayPosition];
        const [lo, hi] = manifest.parameter_bounds[parameter], median = q(scheduleMask, parameter, 1);
        const sd = Math.max((q(scheduleMask, parameter, 2) - q(scheduleMask, parameter, 0)) / 3.29, 1e-3), curve = gaussianCurve(median, sd, lo, hi);
        P.line(f, curve.xs.map(x => displayPosition + .08 + .84 * (x - lo) / (hi - lo)), curve.ys, P.C.tri, 2.7);
        for (let seed = 0; seed < 3; seed++) {
          const seedMedian = sq(scheduleMask, seed, parameter, 1), seedSd = Math.max((sq(scheduleMask, seed, parameter, 2) - sq(scheduleMask, seed, parameter, 0)) / 3.29, 1e-3), seedCurve = gaussianCurve(seedMedian, seedSd, lo, hi);
          P.line(f, seedCurve.xs.map(x => displayPosition + .08 + .84 * (x - lo) / (hi - lo)), seedCurve.ys, P.C.blue, .8, .35);
        }
        const fullX = displayPosition + .08 + .84 * (q(full, parameter, 1) - lo) / (hi - lo), truthX = displayPosition + .08 + .84 * (manifest.truth[parameter] - lo) / (hi - lo);
        P.line(f, [fullX, fullX], [0, .9], P.C.gold, 1, 1, [3, 3]); P.line(f, [truthX, truthX], [0, 1], P.C.clay, 1.5, 1, [5, 3]);
        P.text(f, displayLabels[displayPosition], displayPosition + .5, 1.05, {align: "center", font: "bold 10px system-ui"});
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
      if ($("#zhou-change")) $("#zhou-change").textContent = `Schedule mask ${scheduleMask}, ${selected.size} measured passages including locked passage 0. Mean 90% interval width ${(widths.reduce((a, b) => a + b, 0) / 4).toFixed(3)}.`;
    }
    checks.forEach(x => x.addEventListener("change", draw)); $("#reveal-withheld").addEventListener("change", draw);
    $$('[data-schedule]').forEach(button => button.addEventListener("click", () => { const selected = new Set(presets[button.dataset.schedule]); checks.slice(1).forEach(x => { x.checked = selected.has(+x.value); }); draw(); }));
    $("#zhou-reset").addEventListener("click", () => $$('[data-schedule="zero"]')[0].click()); addEventListener("resize", draw); $$('[data-schedule="zero"]')[0].click();
  }

  async function abcAndPpcExercises() {
    const abcPosterior = $("#guess-canvas"), ppc = $("#ppc-canvas");
    if (!abcPosterior && !ppc) return;
    const data = await fetchJSON("data/exercises.json");

    if (abcPosterior) {
      const trajectoryCanvas = $("#abc-trajectory-canvas"), generations = data.generations;
      let lastRun = null, runToken = 0;
      const progress = $("#abc-progress"), progressLabel = $("#abc-progress-label"), milestoneRow = $("#abc-milestones"), runButton = $("#abc-run");
      function milestoneBudgets(budget) {
        const values = [50, 100, 250, 500, 1000, 3000, 10000].filter(value => value <= budget);
        if (!values.includes(budget)) values.push(budget);
        return values;
      }
      function updateMilestones(stages, current) {
        milestoneRow.innerHTML = stages.map(value => '<span class="' + (value < current ? "done" : value === current ? "active" : "") + '">' + value.toLocaleString() + "</span>").join("");
      }
      function runAbc() {
        const budget = +$("#abc-sims").value, acceptedQuantile = +$("#abc-quantile").value / 100;
        const R = S.mulberry32(+$("#abc-seed").value || 0), observed = data.guess_examples[0].trajectory;
        const stages = milestoneBudgets(budget), candidates = [], token = ++runToken;
        let nextStage = 0;
        progress.max = budget; progress.value = 0; progressLabel.textContent = "0 / " + budget.toLocaleString();
        runButton.textContent = "Restart ABC run"; updateMilestones(stages, 0);
        $("#abc-summary").textContent = "Drawing parameters from the prior and running the simulator…";
        function renderStage(count, complete = false) {
          const ranked = candidates.slice(0, count).sort((a, b) => a.distance - b.distance);
          const nAccepted = Math.max(3, Math.floor(count * acceptedQuantile));
          lastRun = {budget: count, targetBudget: budget, acceptedQuantile, observed, accepted: ranked.slice(0, nAccepted), epsilon: ranked[nAccepted - 1].distance, complete};
          updateMilestones(stages, count); drawAbc();
        }
        function advance() {
          if (token !== runToken) return;
          const stop = Math.min(budget, candidates.length + 64);
          while (candidates.length < stop) {
            const theta = [-2 + 2 * R(), -7 + 5 * R(), -8 + 6 * R()], trajectory = S.chuongDeterministic(theta, generations);
            const distance = Math.sqrt(trajectory.reduce((sum, value, i) => sum + (value - observed[i]) ** 2, 0) / trajectory.length);
            candidates.push({theta, trajectory, distance});
          }
          progress.value = candidates.length; progressLabel.textContent = candidates.length.toLocaleString() + " / " + budget.toLocaleString();
          let reachedMilestone = false;
          while (nextStage < stages.length && candidates.length >= stages[nextStage]) {
            renderStage(stages[nextStage], stages[nextStage] === budget); nextStage += 1;
            reachedMilestone = true;
          }
          if (candidates.length < budget) {
            if (reachedMilestone) setTimeout(() => requestAnimationFrame(advance), 850);
            else requestAnimationFrame(advance);
          }
          else { runButton.textContent = "Run ABC again"; updateMilestones(stages, budget); }
        }
        requestAnimationFrame(advance);
      }
      function drawAbc() {
        if (!lastRun) return;
        const {budget, targetBudget, acceptedQuantile, observed, accepted, epsilon, complete} = lastRun;
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
        $("#abc-summary").textContent = (complete ? "Complete" : "Milestone") + ": " + accepted.length + "/" + budget + " simulations accepted at the " + (acceptedQuantile * 100).toFixed(0) + "% quantile; ε = " + epsilon.toFixed(4) + (complete ? "" : " (target " + targetBudget.toLocaleString() + ")") + ". Posterior medians and 90% intervals: " + summaries.join("; ") + ".";
      }
      function clearAbc() {
        runToken += 1; lastRun = null;
        [trajectoryCanvas, abcPosterior].forEach(canvas => canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height));
        const budget = +$("#abc-sims").value;
        progress.max = budget; progress.value = 0; progressLabel.textContent = "0 / " + budget.toLocaleString();
        updateMilestones(milestoneBudgets(budget), 0); runButton.textContent = "Run ABC";
        $("#abc-summary").textContent = "Choose a simulation budget and acceptance quantile, then click Run ABC.";
      }
      $("#abc-quantile").addEventListener("input", event => { $("#abc-quantile-label").textContent = `${event.target.value}%`; });
      $("#abc-sims").addEventListener("change", clearAbc);
      $("#abc-run").addEventListener("click", runAbc); $("#abc-controls").addEventListener("reset", () => { runToken += 1; setTimeout(() => { $("#abc-quantile-label").textContent = "5%"; clearAbc(); }); });
      addEventListener("resize", drawAbc); clearAbc();
    }

    if (ppc) {
      const caseOrder = [3, 0, 4, 2, 1];
      let index = caseOrder[0], result = "";
      const row = $("#ppc-cases");
      caseOrder.forEach((caseIndex, position) => row.insertAdjacentHTML("beforeend", `<button type="button" data-case="${caseIndex}">Mystery culture ${position + 1}</button>`));
      function drawPpc() {
        const item = data.ppc_cases[index], f = P.frame(ppc, 0, 1.05, 8, 116);
        P.band(f, data.generations, item.q05, item.q95, P.C.blue, .18); P.line(f, data.generations, item.median, P.C.blue, 2);
        P.line(f, data.generations, item.observation, P.C.orange, 1.8); P.points(f, data.generations, item.observation, P.C.orange, 4);
        P.legend(f, [{label: "90% PPC + median", color: P.C.blue}, {label: "observed data", color: P.C.orange}]);
        $("#ppc-summary").textContent = result || `Mystery culture ${caseOrder.indexOf(index) + 1}: choose the biological or measurement explanation that best matches where the orange observation departs from the blue predictive band.`;
      }
      $$('[data-case]', row).forEach(button => button.addEventListener("click", () => { index = +button.dataset.case; result = ""; $$('input[name="diagnosis"]').forEach(input => { input.checked = false; }); drawPpc(); }));
      $("#ppc-reveal").addEventListener("click", () => {
        const choice = $('input[name="diagnosis"]:checked'), item = data.ppc_cases[index];
        if (!choice) { result = "Choose one diagnosis first, then check your answer."; drawPpc(); return; }
        result = `${choice.value === item.kind ? "Correct." : "Not quite."} Known generating case: ${item.title}. ${item.reason} The PPC pattern flags tension, but does not by itself prove that cause.`; drawPpc();
      });
      $("#ppc-reset").addEventListener("click", () => { index = caseOrder[0]; result = ""; $$('input[name="diagnosis"]').forEach(input => { input.checked = false; }); drawPpc(); });
      addEventListener("resize", drawPpc); drawPpc();
    }
  }

  function npeLossExplorer() {
    const canvas = $("#npe-loss-canvas");
    if (!canvas) return;
    const offset = $("#npe-loss-offset"), width = $("#npe-loss-width");
    function draw() {
      const center = +offset.value, sd = +width.value;
      const xs = Array.from({length:241},(_,i)=>-3+6*i/240);
      const density = x => Math.exp(-0.5*((x-center)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
      const ys = xs.map(density), atTruth = density(0), loss = -Math.log(Math.max(atTruth,1e-12));
      const f = P.frame(canvas,0,1,-3,3);
      P.band(f,xs,xs.map(()=>0),ys,P.C.blue,.22);
      P.line(f,xs,ys,P.C.blue,3);
      P.line(f,[0,0],[0,Math.min(1,atTruth)],P.C.orange,2.5,1,[6,4]);
      P.points(f,[0],[Math.min(1,atTruth)],P.C.orange,5);
      P.text(f,"known generating θᵢ",.08,.94,{color:P.C.orange,font:"bold 11px system-ui"});
      P.text(f,"learned density qϕ(θ | xᵢ)",-2.85,.86,{color:P.C.blue,font:"bold 11px system-ui"});
      P.text(f,"parameter θ",2.05,.06,{color:P.C.muted,font:"11px system-ui"});
      $("#npe-loss-offset-label").textContent = `${center>=0?"+":""}${center.toFixed(2)}`;
      $("#npe-loss-width-label").textContent = sd.toFixed(2);
      $("#npe-loss-density").innerHTML = `q<sub>ϕ</sub>(θ<sub>i</sub> | x<sub>i</sub>) = ${atTruth.toFixed(3)}`;
      $("#npe-loss-value").textContent = `−log q = ${loss.toFixed(2)}`;
      $("#npe-loss-meter-fill").style.width = `${Math.max(3,Math.min(100,100*Math.exp(-loss)))}%`;
      const distance = Math.abs(center);
      $("#npe-loss-summary").textContent = distance < .08 ? "The density is centered on the parameter that generated this trajectory, so this training example contributes little surprise." : distance < .55 ? "The generating parameter lies under substantial density. Training still nudges the prediction toward it." : "The network placed little density on the parameter that actually generated this simulation, so this example contributes a large loss.";
    }
    [offset,width].forEach(node=>node.addEventListener("input",draw));
    $("#npe-loss-improve").addEventListener("click",()=>{ const next=Math.abs(+offset.value)<.03?0:+offset.value*.55; offset.value=next.toFixed(2); draw(); });
    $("#npe-loss-controls").addEventListener("reset",()=>setTimeout(draw));
    addEventListener("resize",draw); draw();
  }

  function diversityPredictionLab() {
    const canvas = $("#diversity-lineage-canvas"), processCanvas = $("#diversity-process-canvas");
    if (!canvas || !processCanvas) return;
    const slider = $("#diversity-generation"), palette = ["#577d91", "#b56a50", "#c59b4f", "#315f52", "#8e6bbf", "#54a58a", "#df835e", "#79a9c5", "#b9a35d", "#765d8f", "#8aba6f", "#d06077"];
    const settings = {
      "WT": {lineages: 36, midpoint: 58, slope: .071, births: 84},
      "LTRΔ": {lineages: 28, midpoint: 63, slope: .074, births: 88},
      "ALLΔ": {lineages: 20, midpoint: 69, slope: .078, births: 92},
      "ARSΔ": {lineages: 13, midpoint: 75, slope: .083, births: 96}
    };
    let strain = "WT", timer = null, processStep = 0, processTimer = null;

    const processNotes = [
      "Begin with ancestral cells and three existing CNV lineages. A lineage color is inherited by every descendant.",
      "Formation: a new CNV event appears in one ancestral cell. It receives a new purple identity that its descendants retain.",
      "Growth: fitness changes descendant number. The orange lineage has the largest advantage here, so its family becomes the largest.",
      "Sampling: only a finite set enters the next generation. Some colors gain or lose share by chance; a rare lineage can disappear.",
      "Summarize: first pool all colors for the reporter frequency, then normalize within CNV cells. Here the surviving shares give an effective diversity of 3.5 lineages."
    ];

    function drawProcess() {
      const width = Math.max(700, Math.round(processCanvas.getBoundingClientRect().width || 940));
      const height = Math.round(width * 0.5), ratio = Math.min(devicePixelRatio || 1, 2);
      processCanvas.width = Math.round(width * ratio); processCanvas.height = Math.round(height * ratio);
      const ctx = processCanvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
      const ink = "#25332f", muted = "#66736d", border = "#d8d2c5", cream = "#fffdf8", ancestor = "#ead7a8";
      const colors = ["#577d91", "#54a58a", "#df835e", "#8e6bbf"], centers = [width*.09, width*.29, width*.49, width*.69, width*.89], top = height*.23;
      const titles = ["START", "FORM", "GROW", "SAMPLE", "SUMMARIZE"];
      ctx.fillStyle = cream; ctx.fillRect(0, 0, width, height);
      const rounded = (x,y,w,h,r,fill,stroke=border) => { ctx.beginPath(); ctx.roundRect(x,y,w,h,r); ctx.fillStyle=fill; ctx.fill(); ctx.strokeStyle=stroke; ctx.lineWidth=1; ctx.stroke(); };
      const arrow = (x1,x2,y,active) => { ctx.save(); ctx.globalAlpha=active?1:.16; ctx.strokeStyle="#b56a50"; ctx.fillStyle="#b56a50"; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(x1,y); ctx.lineTo(x2,y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x2,y); ctx.lineTo(x2-8,y-5); ctx.lineTo(x2-8,y+5); ctx.closePath(); ctx.fill(); ctx.restore(); };
      const cell = (x,y,color,r=10,alpha=1) => { ctx.save(); ctx.globalAlpha=alpha; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); ctx.strokeStyle="#31423c"; ctx.lineWidth=1; ctx.stroke(); ctx.beginPath(); ctx.arc(x-r*.3,y-r*.25,r*.18,0,Math.PI*2); ctx.fillStyle="rgba(255,255,255,.72)"; ctx.fill(); ctx.restore(); };
      const label = (text,x,y,size=12,color=muted,weight=600,align="center") => { ctx.fillStyle=color; ctx.font=`${weight} ${size}px system-ui`; ctx.textAlign=align; ctx.fillText(text,x,y); };

      centers.forEach((x,i) => {
        const active = i <= processStep;
        rounded(x-width*.08, height*.12, width*.16, height*.72, 14, active?"#ffffff":"#f5f4ef");
        label(titles[i],x,top-17,Math.max(11,width*.014),active?ink:"#a8aea9",800);
        if (i < 4) arrow(x+width*.082,centers[i+1]-width*.082,height*.48,i<processStep);
      });

      // Starting cells: beige ancestors plus three already formed CNV lineages.
      const startCells = [[-25,-42,ancestor], [5,-45,ancestor], [28,-20,ancestor], [-31,-6,ancestor], [0,-10,colors[0]], [27,13,colors[1]], [-18,26,colors[2]], [9,42,ancestor]];
      startCells.forEach(([dx,dy,c]) => cell(centers[0]+dx, height*.48+dy,c,9,processStep>=0?1:.15));
      label("3 CNV colors",centers[0],height*.75,11,muted,600);

      if (processStep >= 1) {
        startCells.forEach(([dx,dy,c]) => cell(centers[1]+dx*.82,height*.48+dy*.82,c,8));
        cell(centers[1]+34,height*.48-34,colors[3],9);
        label("✦ new lineage",centers[1]+2,height*.72,11,colors[3],800);
      }

      if (processStep >= 2) {
        const clusters = [
          {color:colors[0], y:height*.33, n:3}, {color:colors[1], y:height*.45, n:5},
          {color:colors[2], y:height*.59, n:8}, {color:colors[3], y:height*.72, n:2}
        ];
        clusters.forEach((group,g) => {
          ctx.save(); ctx.globalAlpha=.28; ctx.strokeStyle=group.color; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(centers[1]+(g===3?34:(g-1)*12),height*.48+(g-1)*8); ctx.quadraticCurveTo((centers[1]+centers[2])/2,group.y,centers[2]-25,group.y); ctx.stroke(); ctx.restore();
          for(let i=0;i<group.n;i++) cell(centers[2]-24+(i%4)*16,group.y+(Math.floor(i/4)-.35)*15,group.color,6.5);
        });
        label("different family sizes",centers[2],height*.79,11,muted,600);
      }

      if (processStep >= 3) {
        ctx.save(); ctx.strokeStyle="#7f9189"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(centers[3]-43,height*.31); ctx.lineTo(centers[3]-19,height*.51); ctx.lineTo(centers[3]-43,height*.70); ctx.stroke(); ctx.restore();
        label("finite draw",centers[3]-31,height*.26,9,muted,700);
        const sampled = [colors[0],colors[1],colors[2],colors[2],colors[2],colors[3]];
        sampled.forEach((color,i) => cell(centers[3]+2+(i%2)*20,height*.37+Math.floor(i/2)*28,color,8));
        label("6 sampled cells",centers[3]+7,height*.76,11,muted,600);
      }

      if (processStep >= 4) {
        const shares = [.167,.167,.5,.166]; let cursor = centers[4]-width*.06;
        shares.forEach((share,i) => { const w=width*.12*share; ctx.fillStyle=colors[i]; ctx.fillRect(cursor,height*.39,w,height*.075); cursor+=w; });
        ctx.strokeStyle=ink; ctx.strokeRect(centers[4]-width*.06,height*.39,width*.12,height*.075);
        label("lineage shares",centers[4],height*.35,11,ink,700);
        label("D = 3.5",centers[4],height*.59,Math.max(14,width*.019),"#315f52",800);
        label("effective lineages",centers[4],height*.66,10,muted,600);
      }
      $("#diversity-process-note").textContent = processNotes[processStep];
      $$('[data-diversity-process-step]').forEach(button => { const on=+button.dataset.diversityProcessStep===processStep; button.classList.toggle("active",on); button.setAttribute("aria-pressed",String(on)); });
      $("#diversity-next-step").textContent = processStep >= 4 ? "Start again" : "Next step";
    }

    function stopProcess(label="Play one generation") { if(processTimer) clearInterval(processTimer); processTimer=null; $("#diversity-play-process").textContent=label; }
    $$('[data-diversity-process-step]').forEach(button => button.addEventListener("click", () => { stopProcess(); processStep=+button.dataset.diversityProcessStep; drawProcess(); }));
    $("#diversity-next-step").addEventListener("click", () => { stopProcess(); processStep=processStep>=4?0:processStep+1; drawProcess(); });
    $("#diversity-play-process").addEventListener("click", () => {
      if(processTimer){ stopProcess("Resume generation"); return; }
      if(processStep>=4) processStep=0;
      $("#diversity-play-process").textContent="Pause"; drawProcess();
      const delay = matchMedia("(prefers-reduced-motion: reduce)").matches ? 80 : 850;
      processTimer=setInterval(()=>{ processStep++; drawProcess(); if(processStep>=4) stopProcess("Play again"); },delay);
    });
    $("#diversity-reset-process").addEventListener("click", () => { stopProcess(); processStep=0; drawProcess(); });
    const lineagesFor = name => Array.from({length: settings[name].lineages}, (_, i) => {
      const birth = 7 + Math.round(settings[name].births * ((i + .35) / settings[name].lineages) ** 1.18);
      return {birth, death: i % 5 === 2 ? Math.min(116, birth + 13 + (i * 7) % 24) : 117,
        rate: .024 + .007 * (.5 + .5 * Math.sin(i * 2.17 + name.length)),
        weight: .72 + .5 * (.5 + .5 * Math.cos(i * 1.73)), color: palette[i % palette.length]};
    });

    function stateAt(generation, members) {
      const total = generation < 7 ? 0 : .965 / (1 + Math.exp(-settings[strain].slope * (generation - settings[strain].midpoint)));
      const active = members.filter(item => item.birth <= generation && generation < item.death);
      const raw = active.map(item => item.weight * Math.exp(item.rate * (generation - item.birth)));
      const sum = raw.reduce((a, b) => a + b, 0) || 1;
      return {total, active, shares: raw.map(value => value / sum)};
    }

    function draw() {
      const generation = +slider.value, members = lineagesFor(strain), xs = Array.from({length: 117}, (_, i) => i);
      const f = P.frame(canvas, 0, 1, 0, 116), histories = xs.map(g => stateAt(g, members));
      const {ctx, X, Y} = f;
      members.forEach((member, i) => {
        const top = [], bottom = [];
        for (let g = 0; g <= generation; g++) {
          const state = histories[g], position = state.active.indexOf(member);
          const before = position < 0 ? 0 : state.shares.slice(0, position).reduce((a, b) => a + b, 0);
          const own = position < 0 ? 0 : state.shares[position];
          bottom.push(state.total * before); top.push(state.total * (before + own));
        }
        ctx.save(); ctx.globalAlpha = .82; ctx.fillStyle = member.color; ctx.beginPath();
        top.forEach((value, g) => g ? ctx.lineTo(X(g), Y(value)) : ctx.moveTo(X(g), Y(value)));
        for (let g = generation; g >= 0; g--) ctx.lineTo(X(g), Y(bottom[g]));
        ctx.closePath(); ctx.fill(); ctx.restore();
      });
      const totals = histories.map(state => state.total);
      P.line(f, xs, totals, P.C.ink, 2.4, .28, [5, 4]);
      P.line(f, [generation, generation], [0, 1], P.C.ink, 1.2, .7, [4, 4]);
      P.text(f, "CNV frequency", 2, .96, {color:P.C.muted, font:"bold 11px system-ui"});
      P.text(f, "generation", 102, .05, {color:P.C.muted, font:"11px system-ui"});

      const selected = stateAt(generation, members), entropy = -selected.shares.reduce((sum, share) => sum + (share > 0 ? share * Math.log(share) : 0), 0), effective = selected.shares.length ? Math.exp(entropy) : 0;
      $("#diversity-generation-label").textContent = String(generation);
      $("#diversity-total").textContent = `${(100 * selected.total).toFixed(selected.total < .1 ? 1 : 0)}% CNV`;
      $("#diversity-lineages").textContent = `${selected.active.length} colored ${selected.active.length === 1 ? "lineage" : "lineages"} present`;
      $("#diversity-effective").textContent = `D = ${effective.toFixed(effective < 10 ? 1 : 0)}`;
      $("#diversity-explanation").textContent = selected.active.length ? `${selected.active.length} lineages survive, but their unequal shares behave like ${effective.toFixed(1)} equally abundant lineages.` : "Advance time to form CNV lineages.";
      const ranked = selected.shares.map((share, i) => ({share, color:selected.active[i].color})).sort((a,b) => b.share-a.share);
      const shown = ranked.slice(0, 10), other = ranked.slice(10).reduce((sum, item) => sum + item.share, 0);
      $("#diversity-share-bar").innerHTML = shown.map((item, i) => `<i style="width:${(100 * item.share).toFixed(2)}%;background:${item.color}" title="Lineage ${i + 1}: ${(100 * item.share).toFixed(1)}%"></i>`).join("") + (other ? `<i class="other-lineages" style="width:${(100 * other).toFixed(2)}%" title="Other lineages: ${(100 * other).toFixed(1)}%"></i>` : "");
    }

    function stop(label = "Play lineage history") { if (timer) clearInterval(timer); timer = null; $("#diversity-play").textContent = label; }
    slider.addEventListener("input", () => { stop(); draw(); });
    $$("[data-diversity-strain]").forEach(button => button.addEventListener("click", () => {
      strain = button.dataset.diversityStrain; $$("[data-diversity-strain]").forEach(node => { const on = node === button; node.classList.toggle("active", on); node.setAttribute("aria-pressed", String(on)); }); slider.value = 0; stop(); draw();
    }));
    $("#diversity-play").addEventListener("click", () => {
      if (timer) { stop("Resume"); return; }
      if (+slider.value >= 116) slider.value = 0;
      $("#diversity-play").textContent = "Pause";
      timer = setInterval(() => { slider.value = Math.min(116, +slider.value + 2); draw(); if (+slider.value >= 116) stop("Play again"); }, 85);
    });
    $("#diversity-reset").addEventListener("click", () => { slider.value = 0; stop(); draw(); });
    addEventListener("resize", () => { drawProcess(); draw(); }); drawProcess(); draw();
  }

  trainingViewer().catch(console.error);
  collectiveLab().catch(console.error);
  zhouDesigner().catch(console.error);
  abcAndPpcExercises().catch(console.error);
  npeLossExplorer();
  diversityPredictionLab();
})();
