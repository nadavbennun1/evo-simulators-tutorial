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
    const canvas = $("#diversity-lineage-canvas"), processCanvas = $("#diversity-process-canvas"), slider = $("#diversity-generation");
    if (!canvas || !processCanvas || !slider) return;
    const colors = ["#577d91", "#b56a50", "#c59b4f", "#315f52", "#8e6bbf", "#54a58a", "#df835e", "#79a9c5", "#b9a35d", "#765d8f"];
    const maps = {
      "WT": {logS: -.7381561, logDelta: -4.344892, logPhi: -3.7729044},
      "LTRΔ": {logS: -.73768467, logDelta: -4.835881, logPhi: -4.3162537},
      "ARSΔ": {logS: -.8402579, logDelta: -5.6174593, logPhi: -5.388713},
      "ALLΔ": {logS: -.89968836, logDelta: -5.049737, logPhi: -5.218887}
    };
    const N = 3.3e8, lastGeneration = 116;
    let strain = "WT", simulation = simulate(maps.WT), step = 0, historyTimer = null, stepTimer = null;

    function simulate(parameters) {
      const s = 10 ** parameters.logS, delta = 10 ** parameters.logDelta, phi = 10 ** parameters.logPhi;
      const snvSelection = 1e-3, snvFormation = 1e-5;
      const formed = Array(lastGeneration + 2).fill(0), descendants = Array(lastGeneration + 2).fill(0);
      formed[0] = phi * N; descendants[0] = formed[0];
      let frequencies = [1 - phi, 0, phi, 0];
      const history = [];
      for (let generation = 0; generation <= lastGeneration; generation++) {
        const newLineages = frequencies[0] * N * delta;
        formed[generation + 1] = newLineages;
        descendants[generation + 1] = newLineages;
        const selected = [frequencies[0], frequencies[1] * (1 + s), frequencies[2] * (1 + s), frequencies[3] * (1 + snvSelection)];
        const next = [selected[0] * (1 - delta - snvFormation), selected[1] + selected[0] * delta, selected[2], selected[3] + selected[0] * snvFormation];
        const meanFitness = next.reduce((total, value) => total + value, 0);
        for (let cohort = 0; cohort <= generation + 1; cohort++) descendants[cohort] *= (1 + s) / meanFitness;
        frequencies = next.map(value => value / meanFitness);
        const cohorts = []; let totalDescendants = 0, richness = 0;
        for (let cohort = 0; cohort <= generation + 1; cohort++) {
          if (formed[cohort] <= 1 || descendants[cohort] <= 0) continue;
          cohorts.push({birth: cohort === 0 ? -1 : cohort - 1, lineages: formed[cohort], descendants: descendants[cohort]});
          totalDescendants += descendants[cohort]; richness += formed[cohort];
        }
        let entropy = 0;
        if (totalDescendants) cohorts.forEach(cohort => {
          const oneLineageShare = (cohort.descendants / cohort.lineages) / totalDescendants;
          entropy -= cohort.lineages * oneLineageShare * Math.log(oneLineageShare);
        });
        history.push({generation, cohorts, newLineages, richness, totalDescendants, diversity: totalDescendants ? Math.exp(entropy) : 0, reported: frequencies[1], totalCNV: frequencies[1] + frequencies[2]});
      }
      return {history, s, delta, phi};
    }

    const concise = value => value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : value.toFixed(value < 10 ? 1 : 0);
    const words = value => value >= 1e6 ? `${(value / 1e6).toFixed(2)} million` : value >= 1e3 ? `${(value / 1e3).toFixed(1)} thousand` : value.toFixed(value < 10 ? 1 : 0);
    const superscript = exponent => String(exponent).split("").map(character => ({"-":"⁻","0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"})[character]).join("");
    const scientific = value => {
      const exponent = Math.floor(Math.log10(value)), coefficient = value / 10 ** exponent;
      return `${coefficient.toFixed(2).replace(/\.?0+$/, "")} × 10${superscript(exponent)}`;
    };
    const cohortGroup = birth => birth < 0 ? 0 : Math.min(colors.length - 1, 1 + Math.floor(birth / (lastGeneration / (colors.length - 1))));

    function drawHistory() {
      const generation = +slider.value, states = simulation.history, current = states[generation];
      const f = P.frame(canvas, 0, 1, 0, lastGeneration), {ctx, X, Y} = f;
      for (let group = 0; group < colors.length; group++) {
        const lower = [], upper = [];
        for (let g = 0; g <= generation; g++) {
          const shares = Array(colors.length).fill(0);
          // Cohorts supply the composition; the genotype state supplies the total CNV frequency.
          // Rescaling puts both on one frequency scale, so the colored areas close exactly at the line.
          const scale = states[g].totalDescendants > 0 ? states[g].totalCNV / states[g].totalDescendants : 0;
          states[g].cohorts.forEach(cohort => shares[cohortGroup(cohort.birth)] += cohort.descendants * scale);
          lower.push(shares.slice(0, group).reduce((a, b) => a + b, 0));
          upper.push(shares.slice(0, group + 1).reduce((a, b) => a + b, 0));
        }
        ctx.save(); ctx.globalAlpha = .78; ctx.fillStyle = colors[group]; ctx.beginPath();
        upper.forEach((value, g) => g ? ctx.lineTo(X(g), Y(value)) : ctx.moveTo(X(g), Y(value)));
        for (let g = generation; g >= 0; g--) ctx.lineTo(X(g), Y(lower[g]));
        ctx.closePath(); ctx.fill(); ctx.restore();
      }
      P.line(f, states.map(state => state.generation), states.map(state => state.totalCNV), P.C.tri, 2.5);
      P.line(f, states.map(state => state.generation), states.map(state => state.reported), P.C.ink, 2.2, 1, [5, 4]);
      P.line(f, [generation, generation], [0, 1], P.C.ink, 1.2, .65, [4, 4]);
      P.text(f, "CNV frequency", 2, .96, {color:P.C.muted, font:"bold 11px system-ui"});
      P.text(f, "generation", 102, .05, {color:P.C.muted, font:"11px system-ui"});
      P.legend(f, [{label:"total CNV in model",color:P.C.tri},{label:"reported CNV",color:P.C.ink}]);
      $("#diversity-generation-label").textContent = String(generation);
      $("#diversity-total").textContent = `${(100 * current.reported).toFixed(current.reported < .1 ? 1 : 0)}% reported CNV`;
      $("#diversity-lineages").textContent = `Q(${generation}) ≈ ${concise(current.newLineages)} new lineages`;
      $("#diversity-cohort-explanation").textContent = `${words(current.richness)} assumed-unique lineages have accumulated across the generations shown.`;
      $("#diversity-effective").textContent = `D(${generation}) = ${concise(current.diversity)}`;
      const latest = current.cohorts[current.cohorts.length - 1];
      $("#diversity-explanation").textContent = latest ? `The ${words(latest.lineages)} lineages born together are each assigned ${words(latest.descendants / latest.lineages)} descendants at this generation.` : "No generation has yet contributed more than one lineage.";
      const grouped = Array(colors.length).fill(0);
      current.cohorts.forEach(cohort => grouped[cohortGroup(cohort.birth)] += cohort.descendants);
      const total = grouped.reduce((a, b) => a + b, 0) || 1;
      $("#diversity-share-bar").innerHTML = grouped.map((value, index) => value ? `<i style="width:${(100 * value / total).toFixed(2)}%;background:${colors[index]}" title="Birth-time group ${index + 1}: ${(100 * value / total).toFixed(1)}% of CNV cells"></i>` : "").join("");
    }

    function drawProcess() {
      const width = Math.max(320, Math.round(processCanvas.getBoundingClientRect().width || 940)), height = Math.max(270, Math.round(width * .45)), ratio = Math.min(devicePixelRatio || 1, 2);
      processCanvas.width = width * ratio; processCanvas.height = height * ratio;
      const ctx = processCanvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const current = simulation.history[+slider.value], pad = Math.max(18, width * .045), ink = "#25332f", muted = "#66736d", forest = "#315f52", clay = "#b56a50";
      ctx.fillStyle = "#fffdf8"; ctx.fillRect(0, 0, width, height);
      const label = (text, x, y, size=13, color=ink, weight=650, align="left") => { ctx.fillStyle=color; ctx.font=`${weight} ${size}px system-ui`; ctx.textAlign=align; ctx.fillText(text,x,y); };
      const cell = (x,y,color,r=8) => { ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); ctx.strokeStyle=ink; ctx.lineWidth=1; ctx.stroke(); ctx.beginPath(); ctx.arc(x-r*.3,y-r*.3,r*.18,0,Math.PI*2); ctx.fillStyle="rgba(255,255,255,.75)"; ctx.fill(); };
      const headings = ["ONE INFERRED EXPLANATION", "RUN THE EVOLUTIONARY MODEL", "COUNT NEW FORMATIONS", "FOLLOW EACH GENERATION", "CALCULATE DIVERSITY"];
      label(`${step + 1} / 5`,pad,27,12,clay,850); label(headings[step],pad,54,Math.max(15,Math.min(22,width*.025)),ink,850);
      for(let i=0;i<5;i++){ctx.fillStyle=i<=step?forest:"#dfe3dd";ctx.fillRect(pad+i*(width-2*pad)/5,67,(width-2*pad)/5-5,5);}
      let note="";
      if(step===0){
        const items=[["s꜀",simulation.s.toFixed(3),"selection coefficient"],["δ꜀",scientific(simulation.delta),"formation probability"],["φ",scientific(simulation.phi),"initial hidden fraction"]], card=(width-2*pad-20)/3;
        items.forEach(([symbol,value,name],i)=>{const x=pad+i*(card+10);ctx.fillStyle="#fff";ctx.strokeStyle="#d8d2c5";ctx.beginPath();ctx.roundRect(x,92,card,height-118,13);ctx.fill();ctx.stroke();label(symbol,x+card/2,137,21,colors[i+1],850,"center");label(value,x+card/2,169,Math.max(11,Math.min(14,width*.015)),ink,750,"center");label(name,x+card/2,200,Math.max(9,Math.min(12,width*.014)),muted,650,"center");});
        note=`${strain}: one joint draw of s꜀, δ꜀ and φ learned from the fluorescence trajectories.`;
      }else if(step===1){
        const x0=pad+35,x1=width-pad,y0=height-35,y1=93;ctx.strokeStyle=ink;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x0,y0);ctx.lineTo(x1,y0);ctx.stroke();ctx.strokeStyle=forest;ctx.lineWidth=3;ctx.beginPath();simulation.history.slice(0,current.generation+1).forEach((state,i)=>{const x=x0+(x1-x0)*state.generation/lastGeneration,y=y0-(y0-y1)*state.reported;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();label("reported CNV frequency",x0+6,y1+15,12,forest,750);label(`${(100*current.reported).toFixed(1)}% at T = ${current.generation}`,x1-4,y1+15,14,ink,800,"right");
        note=`The Wright–Fisher simulator produces ancestral, reported-CNV and hidden-CNV counts through generation ${current.generation}.`;
      }else if(step===2){
        const y=height*.58,left=width*.23,right=width*.75;for(let i=0;i<9;i++)cell(left+(i%3)*22-22,y+Math.floor(i/3)*22-22,"#ead7a8");ctx.strokeStyle=clay;ctx.fillStyle=clay;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(width*.38,y);ctx.lineTo(width*.57,y);ctx.stroke();ctx.beginPath();ctx.moveTo(width*.57,y);ctx.lineTo(width*.57-10,y-6);ctx.lineTo(width*.57-10,y+6);ctx.fill();for(let i=0;i<9;i++)cell(right+(i%3)*22-22,y+Math.floor(i/3)*22-22,colors[i%colors.length]);label(`Qₜ = δ꜀ × nₐ(t) ≈ ${concise(current.newLineages)}`,width/2,113,Math.max(15,Math.min(22,width*.026)),ink,850,"center");label("ancestral",left,y+59,11,muted,700,"center");label("unique formations",right,y+59,11,muted,700,"center");
        note=`At generation ${current.generation}, the model estimates ${words(current.newLineages)} new formation events and treats each as a different lineage.`;
      }else if(step===3){
        const examples=current.cohorts.length?[current.cohorts[0],current.cohorts[Math.floor(current.cohorts.length/2)],current.cohorts.at(-1)].filter((v,i,a)=>a.indexOf(v)===i):[];examples.forEach((cohort,row)=>{const y=112+row*59,color=colors[cohortGroup(cohort.birth)],radius=Math.max(5,Math.min(10,5+Math.log10(Math.max(1,cohort.descendants/cohort.lineages))*.65));label(cohort.birth<0?"present at start":`born at t = ${cohort.birth}`,pad,y,11,ink,750);for(let i=0;i<5;i++)cell(width*.52+i*radius*2.35,y-5,color,radius);label(`each ≈ ${concise(cohort.descendants/cohort.lineages)} cells`,width-pad,y+20,10,muted,650,"right");});label("same birth time + same s꜀ → same abundance",width/2,height-20,12,forest,850,"center");
        note="The model retains the number born in each generation and their combined descendants. Same-age lineages share one selection coefficient and therefore have equal abundance.";
      }else{
        const grouped=Array(8).fill(0);current.cohorts.forEach(c=>grouped[Math.min(7,cohortGroup(c.birth))]+=c.descendants);const total=grouped.reduce((a,b)=>a+b,0)||1;let cursor=pad;grouped.forEach((value,i)=>{const w=(width-2*pad)*value/total;ctx.fillStyle=colors[i];ctx.fillRect(cursor,116,w,34);cursor+=w;});ctx.strokeStyle=ink;ctx.strokeRect(pad,116,width-2*pad,34);label("one lineage = (cohort descendants ÷ lineages born) ÷ all CNV cells",width/2,201,Math.max(10,Math.min(15,width*.017)),ink,700,"center");label(`D(${current.generation}) = ${concise(current.diversity)}`,width/2,249,Math.max(20,Math.min(29,width*.033)),forest,850,"center");
        note=`At generation ${current.generation}, these model-based shares give an effective diversity of ${words(current.diversity)}.`;
      }
      $("#diversity-process-note").textContent=note;
      $$('[data-diversity-process-step]').forEach(button=>{const active=+button.dataset.diversityProcessStep===step;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));});
      $("#diversity-next-step").textContent=step===4?"Start again":"Next step";
    }

    const stopHistory=(label="Play generations")=>{if(historyTimer)clearInterval(historyTimer);historyTimer=null;$("#diversity-play").textContent=label;};
    const stopSteps=(label="Play calculation")=>{if(stepTimer)clearInterval(stepTimer);stepTimer=null;$("#diversity-play-process").textContent=label;};
    slider.addEventListener("input",()=>{stopHistory();drawHistory();drawProcess();});
    $$('[data-diversity-strain]').forEach(button=>button.addEventListener("click",()=>{strain=button.dataset.diversityStrain;simulation=simulate(maps[strain]);$$('[data-diversity-strain]').forEach(node=>{const active=node===button;node.classList.toggle("active",active);node.setAttribute("aria-pressed",String(active));});stopHistory();drawHistory();drawProcess();}));
    $$('[data-diversity-process-step]').forEach(button=>button.addEventListener("click",()=>{stopSteps();step=+button.dataset.diversityProcessStep;drawProcess();}));
    $("#diversity-next-step").addEventListener("click",()=>{stopSteps();step=step===4?0:step+1;drawProcess();});
    $("#diversity-play-process").addEventListener("click",()=>{if(stepTimer){stopSteps("Resume calculation");return;}if(step===4)step=0;$("#diversity-play-process").textContent="Pause";drawProcess();stepTimer=setInterval(()=>{step++;drawProcess();if(step===4)stopSteps("Play again");},matchMedia("(prefers-reduced-motion: reduce)").matches?80:850);});
    $("#diversity-reset-process").addEventListener("click",()=>{stopSteps();step=0;drawProcess();});
    $("#diversity-play").addEventListener("click",()=>{if(historyTimer){stopHistory("Resume");return;}if(+slider.value>=lastGeneration)slider.value=0;$("#diversity-play").textContent="Pause";historyTimer=setInterval(()=>{slider.value=Math.min(lastGeneration,+slider.value+2);drawHistory();drawProcess();if(+slider.value>=lastGeneration)stopHistory("Play again");},85);});
    $("#diversity-reset").addEventListener("click",()=>{slider.value=25;stopHistory();drawHistory();drawProcess();});
    addEventListener("resize",()=>{drawHistory();drawProcess();});
    drawHistory(); drawProcess();
  }

  trainingViewer().catch(console.error);
  collectiveLab().catch(console.error);
  zhouDesigner().catch(console.error);
  abcAndPpcExercises().catch(console.error);
  npeLossExplorer();
  diversityPredictionLab();
})();
