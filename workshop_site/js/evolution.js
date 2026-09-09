(() => {
  "use strict";
  const {$, $$} = Workshop, P = WorkshopPlots, S = WorkshopScience;

  function normal(R) {
    const u = Math.max(R(), 1e-12), v = R();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function binomial(n, p, R) {
    if (n <= 0 || p <= 0) return 0;
    if (p >= 1) return n;
    if (n <= 10000) {
      let k = 0;
      for (let i = 0; i < n; i++) if (R() < p) k++;
      return k;
    }
    const mean = n * p, sd = Math.sqrt(n * p * (1 - p));
    return Math.max(0, Math.min(n, Math.round(mean + sd * normal(R))));
  }

  function multinomial3(n, q, R) {
    const n0 = binomial(n, q[0], R);
    const n1 = binomial(n - n0, q[1] / Math.max(1 - q[0], 1e-12), R);
    return [n0, n1, n - n0 - n1];
  }

  function avecillaPlayground() {
    const canvas = $("#evo-canvas"), orderCanvas = $("#evo-order-canvas");
    if (!canvas) return;
    const ids = ["delta-c", "delta-b", "s-c", "s-b", "duration", "ne", "reps", "seed"];
    const el = Object.fromEntries(ids.map(id => [id, `#evo-${id}`]).map(([id, selector]) => [id, $(selector)]));
    let timer = null, visible = 0, trajectories = [], started = false;

    function simulate(seed, mutationFirst) {
      const R = S.mulberry32(seed), nEff = +el.ne.value, duration = +el.duration.value;
      const deltaC = 10 ** (+el["delta-c"].value), deltaB = 10 ** (+el["delta-b"].value);
      const fitness = [1, 1 + +el["s-c"].value, 1 + +el["s-b"].value];
      let p = [1, 0, 0], out = [p.slice()];
      for (let g = 1; g <= duration; g++) {
        const mutate = values => [values[0] * (1 - deltaC - deltaB), values[1] + values[0] * deltaC, values[2] + values[0] * deltaB];
        const select = values => values.map((x, i) => x * fitness[i]);
        const expected = mutationFirst ? select(mutate(p)) : mutate(select(p));
        const total = expected.reduce((a, b) => a + b, 0), q = expected.map(x => x / total);
        p = multinomial3(nEff, q, R).map(x => x / nEff);
        out.push(p.slice());
      }
      return out;
    }

    function draw() {
      const reps = Math.max(1, Math.min(24, +el.reps.value || 1));
      const duration = +el.duration.value, show = Math.min(visible, duration);
      const xs = Array.from({length: duration + 1}, (_, i) => i);
      const colors = [P.C.muted, P.C.orange, P.C.purple];
      if (!started) {
        P.frame(canvas, 0, 1, 0, duration);
        P.frame(orderCanvas, 0, 1, 0, duration);
        $("#evo-composition").innerHTML = "";
        $("#evo-summary").textContent = "The plot starts empty. Choose a scenario, then press Play to reveal its evolutionary trajectory.";
        if ($("#evo-driver")) $("#evo-driver").textContent = "No trajectory has been simulated yet.";
        return;
      }
      trajectories = Array.from({length: reps}, (_, i) => simulate((+el.seed.value || 0) + i, true));
      const orderTrajectories = Array.from({length: reps}, (_, i) => simulate((+el.seed.value || 0) + i, false));
      [[canvas, trajectories], [orderCanvas, orderTrajectories]].forEach(([target, runs]) => {
        const f = P.frame(target, 0, 1, 0, duration);
        runs.forEach((traj, r) => colors.forEach((color, state) => {
          P.line(f, xs.slice(0, show + 1), traj.slice(0, show + 1).map(x => x[state]), color, r === 0 ? 2 : 0.9, r === 0 ? 0.9 : 0.16);
        }));
        ["Ancestral", "GAP1 CNV", "Other beneficial"].forEach((name, i) => P.text(f, name, duration * 0.60, 0.96 - i * 0.065, {color: colors[i], font: "bold 11px system-ui"}));
      });
      const final = trajectories.map(x => x[show]);
      const med = [0, 1, 2].map(state => Workshop.quantile(final.map(x => x[state]), 0.5));
      $("#evo-composition").innerHTML = ["Ancestral", "GAP1 CNV", "Other beneficial"].map((name, i) => `<span><b>${name}</b><br>${(med[i] * 100).toFixed(1)}% median</span>`).join("");
      const sC = +el["s-c"].value, sB = +el["s-b"].value;
      const dC = 10 ** (+el["delta-c"].value), dB = 10 ** (+el["delta-b"].value), nEff = +el.ne.value;
      let driver;
      if (nEff <= 10000) driver = "finite-population drift makes replicate-to-replicate outcomes visibly unstable";
      else if (Math.abs(sC - sB) > 0.02) driver = `${sC > sB ? "the CNV" : "the other beneficial lineage"} has the larger selection coefficient`;
      else if (Math.max(dC, dB) / Math.max(Math.min(dC, dB), 1e-12) > 5) driver = `${dC > dB ? "CNV" : "other-beneficial"} mutational supply is larger`;
      else driver = "selection and mutational supply are closely balanced";
      if ($("#evo-driver")) $("#evo-driver").textContent = `This is the Avecilla three-genotype mechanism: ${driver}.`;
      $("#evo-summary").textContent = `${reps} seeded Wright–Fisher replicate${reps > 1 ? "s" : ""}; showing generation ${show}. Median CNV frequency ${(med[1] * 100).toFixed(1)}%. For Nₑ > 10,000 the browser uses a seeded normal approximation to binomial drift.`;
      const altFinal = orderTrajectories.map(x => x[show]);
      const altMed = [0, 1, 2].map(state => Workshop.quantile(altFinal.map(x => x[state]), 0.5));
      const gap = Math.max(...med.map((value, i) => Math.abs(value - altMed[i])));
      $("#evo-order-summary").textContent = `Largest median frequency difference at generation ${show}: ${(100 * gap).toFixed(2)} percentage points. The order matters because M and selection generally do not commute; at the published small formation rates the difference is usually tiny.`;
    }

    function labels(resetView = true) {
      $("#evo-delta-c-label").textContent = `10^${(+el["delta-c"].value).toFixed(2)}`;
      $("#evo-delta-b-label").textContent = `10^${(+el["delta-b"].value).toFixed(2)}`;
      $("#evo-s-c-label").textContent = (+el["s-c"].value).toFixed(3);
      $("#evo-s-b-label").textContent = (+el["s-b"].value).toFixed(3);
      $("#evo-duration-label").textContent = `${el.duration.value}`;
      if (resetView) {
        if (timer) clearInterval(timer);
        timer = null; started = false; visible = 0;
        $("#evo-play").textContent = "Play";
      }
      draw();
    }

    let debounce;
    Object.values(el).forEach(node => node.addEventListener("input", () => {
      clearTimeout(debounce);
      debounce = setTimeout(labels, 70);
    }));
    const presets = {
      fit: [-4.2, -5, 0.07, 0.001, 120, 330000000, 8],
      cnv: [-3.8, -5.5, 0.09, 0.01, 120, 330000000, 8],
      competing: [-4.2, -4.1, 0.055, 0.075, 120, 330000000, 10],
      drift: [-4.2, -4.3, 0.055, 0.05, 120, 1000, 20],
      order: [-2, -2.15, 0.14, 0.01, 80, 100000, 10],
    };
    $$('[data-evo-preset]').forEach(button => button.addEventListener("click", () => {
      const values = presets[button.dataset.evoPreset];
      [el["delta-c"], el["delta-b"], el["s-c"], el["s-b"], el.duration, el.ne, el.reps].forEach((node, i) => { node.value = values[i]; });
      labels();
    }));
    $("#evo-play").addEventListener("click", event => {
      if (timer) {
        clearInterval(timer); timer = null; event.target.textContent = "Play"; return;
      }
      if (!started || visible >= +el.duration.value) visible = 0;
      started = true; event.target.textContent = "Pause";
      timer = setInterval(() => {
        visible = Math.min(+el.duration.value, visible + 3); draw();
        if (visible >= +el.duration.value) { clearInterval(timer); timer = null; event.target.textContent = "Replay"; }
      }, 120);
    });
    $("#evo-controls").addEventListener("reset", () => setTimeout(labels));
    addEventListener("resize", draw);
    labels();
  }

  function modelBuilder() {
    const root = $("#avecilla-model-builder");
    if (!root) return;
    const buttons = $$('[data-model-force]', root), panels = $$('[data-force-panel]', root), nodes = $$('[data-force-node]', root);
    function show(force) {
      buttons.forEach(button => { const active = button.dataset.modelForce === force; button.classList.toggle("active", active); button.setAttribute("aria-selected", String(active)); });
      panels.forEach(panel => { panel.hidden = panel.dataset.forcePanel !== force; });
      nodes.forEach(node => node.classList.toggle("active", node.dataset.forceNode === force));
    }
    buttons.forEach(button => button.addEventListener("click", () => show(button.dataset.modelForce)));
    show("mutation");
  }

  function chuongStandingVariation() {
    const canvas = $("#chuong-phi-canvas");
    if (!canvas) return;
    let view = "both";
    const generations = Array.from({length: 117}, (_, i) => i), logS = -0.74, logDelta = -4.84;
    function simulate(logPhi) {
      const s = 10 ** logS, delta = 10 ** logDelta, phi = 10 ** logPhi, fitness = [1, 1+s, 1+s, 1.001];
      let p = [1-phi, 0, phi, 0], reported = [], total = [];
      generations.forEach(() => {
        reported.push(p[1]); total.push(p[1] + p[2]);
        const selected = p.map((x, i) => x * fitness[i]);
        const mutated = [selected[0]*(1-delta-1e-5), selected[1]+selected[0]*delta, selected[2], selected[3]+selected[0]*1e-5];
        const z = mutated.reduce((a,b) => a+b, 0); p = mutated.map(x => x/z);
      });
      return {reported, total};
    }
    const low = simulate(-8), high = simulate(-4);
    function draw() {
      const f = P.frame(canvas, 0, 1, 0, 116), selected = view === "low" ? [["φ=10⁻⁸", low, P.C.blue]] : view === "high" ? [["φ=10⁻⁴", high, P.C.orange]] : [["φ=10⁻⁸", low, P.C.blue], ["φ=10⁻⁴", high, P.C.orange]];
      selected.forEach(([label, result, color], i) => {
        P.line(f, generations, result.total, color, 3);
        P.line(f, generations, result.reported, color, 2, .9, [7,5]);
        P.text(f, `${label} total GAP1 CNV`, 7, .96-i*.07, {color, font:"bold 11px system-ui"});
      });
      P.text(f, "solid: total GAP1 CNV · dashed: reporter-positive CNV⁺", 7, selected.length === 2 ? .80 : .88, {color:P.C.ink, font:"11px system-ui"});
      const deltaTotal = 100 * (high.total[50] - low.total[50]), deltaReported = 100 * (high.reported[50] - low.reported[50]);
      $("#chuong-phi-summary").textContent = `At generation 50, changing φ from 10⁻⁸ to 10⁻⁴ changes total GAP1 CNV abundance by ${deltaTotal.toFixed(2)} percentage points, but reporter-positive CNV⁺ abundance by only ${deltaReported.toFixed(2)} points.`;
      $$('[data-phi-view]').forEach(button => button.classList.toggle("active", button.dataset.phiView === view));
    }
    $$('[data-phi-view]').forEach(button => button.addEventListener("click", () => { view = button.dataset.phiView; draw(); }));
    addEventListener("resize", draw); draw();
  }

  function chuongEquationExercise() {
    const root = $("#chuong-equation-exercise");
    if (!root) return;
    const buttons = $$('[data-chuong-step]', root), cards = $$('[data-chuong-card]', root), matrix = $(".chuong-matrix", root);
    let revealed = 0;
    function render() {
      cards.forEach(card => { card.hidden = +card.dataset.chuongCard > revealed; });
      buttons.forEach(button => { const step = +button.dataset.chuongStep; button.disabled = step > revealed + 1 || step <= revealed; });
      matrix.hidden = revealed < 3;
    }
    buttons.forEach(button => button.addEventListener("click", () => { if (+button.dataset.chuongStep === revealed + 1) { revealed++; render(); } }));
    $("#chuong-equation-reset").addEventListener("click", () => { revealed = 0; render(); });
    render();
  }

  function chuongCodeExercise() {
    const root = $(".code-fill-exercise");
    if (!root) return;
    const inputs = $$('[data-code-answer]', root), summary = $("#chuong-code-summary");
    const normalize = value => value.replace(/\s+/g, "").replace(/'/g, '"');
    $("#check-chuong-code").addEventListener("click", () => {
      let correct = 0;
      inputs.forEach(input => { const okay = normalize(input.value) === normalize(input.dataset.codeAnswer); input.classList.toggle("correct", okay); input.classList.toggle("incorrect", !okay); input.nextElementSibling.textContent = okay ? "Correct" : "Try again"; if (okay) correct++; });
      summary.textContent = `${correct} of ${inputs.length} biological lines are correct.`;
    });
    $("#reveal-chuong-code").addEventListener("click", () => { inputs.forEach(input => { input.value = input.dataset.codeAnswer; input.classList.add("correct"); input.classList.remove("incorrect"); input.nextElementSibling.textContent = "Revealed"; }); summary.textContent = "Answers revealed. Trace mutation, selection, and drift in that order."; });
    $("#reset-chuong-code").addEventListener("click", () => { inputs.forEach(input => { input.value = ""; input.classList.remove("correct", "incorrect"); input.nextElementSibling.textContent = ""; }); summary.textContent = ""; });
  }

  function dfeExample() {
    const canvas = $("#dfe-canvas");
    if (!canvas) return;
    const mean = $("#dfe-mean"), shape = $("#dfe-shape");
    function draw() {
      const mu = +mean.value, k = +shape.value, scale = mu / k, xMax = 0.18;
      const xs = Array.from({length: 241}, (_, i) => xMax * i / 240);
      const raw = xs.map(x => {
        const safeX = Math.max(x, xMax / 240);
        return safeX ** (k - 1) * Math.exp(-safeX / scale);
      });
      const peak = Math.max(...raw.filter(Number.isFinite)), ys = raw.map(x => Math.min(1, x / peak));
      const f = P.frame(canvas, 0, 1.08, 0, xMax);
      P.band(f, xs, xs.map(() => 0), ys, P.C.orange, 0.2);
      P.line(f, xs, ys, P.C.orange, 3);
      const grid = Array.from({length: 25}, (_, i) => xMax * (i + 1) / 25), gy = grid.map(x => {
        const value = x ** (k - 1) * Math.exp(-x / scale);
        return Math.min(1, value / peak);
      });
      P.points(f, grid, gy, P.C.orange, 2.2);
      P.line(f, [mu, mu], [0, 1], P.C.ink, 1.5, 1, [5, 4]);
      P.text(f, `mean s̄ = ${mu.toFixed(3)}`, Math.min(mu + 0.004, 0.13), 0.96, {font: "bold 11px system-ui"});
      const total = raw.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
      let cumulative = 0, q90 = xs.at(-1);
      for (let i = 0; i < xs.length; i++) { cumulative += Number.isFinite(raw[i]) ? raw[i] : 0; if (cumulative >= 0.9 * total) { q90 = xs[i]; break; } }
      $("#dfe-mean-label").textContent = mu.toFixed(3);
      $("#dfe-shape-label").textContent = k.toFixed(1);
      $("#dfe-summary").textContent = `Example gamma s-DFE: mean ${mu.toFixed(3)}, shape ${k.toFixed(1)}, approximate 90th percentile ${q90.toFixed(3)}. Dots show the 25-class discretization used by the teaching simulator.`;
      $("#dfe-change").textContent = k < 1.2 ? "Most new CNVs have tiny effects, with a long beneficial tail." : k > 3.5 ? "Effects cluster tightly around the mean." : "The distribution is right-skewed: many modest effects and a smaller high-fitness tail.";
    }
    [mean, shape].forEach(node => node.addEventListener("input", draw));
    $("#dfe-controls").addEventListener("reset", () => setTimeout(draw));
    addEventListener("resize", draw);
    draw();
  }

  function chuongChallenge() {
    const canvas = $("#chuong-challenge-canvas");
    if (!canvas) return;
    const generations = [8, 21, 29, 37, 50, 58, 66, 79, 87, 95, 108, 116];
    const controls = [$("#chuong-guess-s"), $("#chuong-guess-m"), $("#chuong-guess-p0")];
    let round = 0, truth = [], observation = [], scored = false;

    function newObservation() {
      const R = S.mulberry32(31051987 + round * 7919);
      truth = [-1.08 + 0.5 * R(), -5.55 + 1.25 * R(), -6.2 + 2.25 * R()];
      const latent = S.chuongDeterministic(truth, generations);
      observation = latent.map(x => Math.max(0, Math.min(1, x + 0.025 * normal(R))));
      scored = false;
      $("#chuong-score-card").innerHTML = "<b>Unscored round.</b> Fit the orange observations, then submit your parameters.";
      draw();
    }

    function guess() { return controls.map(node => +node.value); }

    function draw() {
      const values = guess(), predicted = S.chuongDeterministic(values, generations), f = P.frame(canvas, 0, 1, 8, 116);
      P.line(f, generations, predicted, P.C.blue, 2.5);
      P.points(f, generations, observation, P.C.orange, 4.3);
      if (scored) P.line(f, generations, S.chuongDeterministic(truth, generations), P.C.gold, 1.7, 1, [5, 4]);
      P.text(f, "● noisy observation", 12, 0.96, {color: P.C.orange, font: "bold 11px system-ui"});
      P.text(f, "— current guess", 12, 0.90, {color: P.C.blue, font: "bold 11px system-ui"});
      if (scored) P.text(f, `-- truth  s=${truth[0].toFixed(2)}  δ=${truth[1].toFixed(2)}  φ=${truth[2].toFixed(2)}`, 12, 0.84, {color: P.C.gold, font: "bold 11px system-ui"});
      ["s", "m", "p0"].forEach((name, i) => { $(`#chuong-guess-${name}-label`).textContent = values[i].toFixed(2); });
      const trajectoryRmse = Math.sqrt(predicted.reduce((sum, x, i) => sum + (x - observation[i]) ** 2, 0) / predicted.length);
      $("#chuong-challenge-summary").textContent = `Current simulated trajectory versus observation: frequency RMSE ${trajectoryRmse.toFixed(4)}. The score itself uses parameter RMSE on the shared log₁₀ scale.`;
    }

    controls.forEach(node => node.addEventListener("input", () => {
      if (scored) $("#chuong-score-card").innerHTML = "<b>Guess changed.</b> Submit again for a new score.";
      scored = false; draw();
    }));
    $("#chuong-score").addEventListener("click", () => {
      scored = true;
      const result = S.inverseRmseScore(guess(), truth);
      const band = result.score >= 80 ? "Excellent" : result.score >= 60 ? "Strong" : result.score >= 45 ? "Getting close" : "Try another round";
      const values = guess();
      const names = ["log₁₀(s)", "log₁₀(δ)", "log₁₀(φ)"];
      const breakdown = names.map((name, i) => `<span><b>${name}</b>guess ${values[i].toFixed(2)}<br>truth ${truth[i].toFixed(2)}<br>|error| ${Math.abs(values[i] - truth[i]).toFixed(2)}</span>`).join("");
      $("#chuong-score-card").innerHTML = `<strong>${band} · ${result.score.toFixed(1)} points</strong><br>Parameter RMSE ${result.rmse.toFixed(3)}<div class="score-breakdown">${breakdown}</div>`;
      draw();
    });
    $("#chuong-new").addEventListener("click", () => { round++; newObservation(); });
    $("#chuong-challenge-controls").addEventListener("reset", () => setTimeout(() => { scored = false; $("#chuong-score-card").innerHTML = "<b>Guess reset.</b> This observation is unchanged."; draw(); }));
    addEventListener("resize", draw);
    newObservation();
  }

  function zhouModelPlayground() {
    const canvas = $("#zhou-model-canvas");
    if (!canvas) return;
    const ids = ["mu-wt", "mu-loh", "w-tri", "w-loh", "p0"];
    const el = Object.fromEntries(ids.map(id => [id, $(`#zhou-model-${id}`)]));

    function draw() {
      const theta = [+el["mu-wt"].value, +el["w-tri"].value, +el["w-loh"].value, +el["mu-loh"].value];
      const tri0 = +el.p0.value, p0 = [tri0, (1 - tri0) * 0.6, (1 - tri0) * 0.4];
      const trajectory = S.zhouDeterministic(theta, p0, 120).filter((_, i) => i % 10 === 0);
      const xs = Array.from({length: 13}, (_, i) => i), colors = [P.C.muted, P.C.orange, P.C.purple], f = P.frame(canvas, 0, 1, 0, 12);
      colors.forEach((color, state) => P.line(f, xs, trajectory.map(x => x[state]), color, 2.7));
      ["Trisomic", "Wild type", "LOH"].forEach((name, i) => P.text(f, name, 8.4, 0.96 - i * 0.065, {color: colors[i], font: "bold 12px system-ui"}));
      const final = trajectory.at(-1);
      $("#zhou-model-composition").innerHTML = ["Trisomic", "Wild type", "LOH"].map((name, i) => `<span><b>${name}</b><br>${(final[i] * 100).toFixed(1)}% at P12</span>`).join("");
      $("#zhou-model-mu-wt-label").textContent = (+el["mu-wt"].value).toFixed(2);
      $("#zhou-model-mu-loh-label").textContent = (+el["mu-loh"].value).toFixed(2);
      $("#zhou-model-w-tri-label").textContent = (+el["w-tri"].value).toFixed(3);
      $("#zhou-model-w-loh-label").textContent = (+el["w-loh"].value).toFixed(3);
      $("#zhou-model-p0-label").textContent = `${(tri0 * 100).toFixed(0)}%`;
      const route = final[1] > final[2] * 1.25 ? "euploid recovery dominates" : final[2] > final[1] * 1.25 ? "LOH dominates" : "both chromosome-loss routes remain important";
      $("#zhou-model-change").textContent = `At passage 12, ${route}; transition rates set mutational supply while relative fitness reshapes the descendants after they appear.`;
      $("#zhou-model-summary").textContent = `Deterministic Zhou trajectory sampled every 10 generations. Final composition: ${(final[0] * 100).toFixed(1)}% trisomic, ${(final[1] * 100).toFixed(1)}% wild type, ${(final[2] * 100).toFixed(1)}% LOH.`;
    }

    Object.values(el).forEach(node => node.addEventListener("input", draw));
    const presets = {
      fit: [-3.47, -3.28, 0.92, 0.986, 0.99],
      wt: [-2.9, -5.2, 0.92, 0.96, 0.99],
      loh: [-5.2, -2.9, 0.92, 1.01, 0.99],
      fitness: [-3.5, -3.5, 1.02, 0.9, 0.99],
    };
    $$('[data-zhou-model-preset]').forEach(button => button.addEventListener("click", () => {
      const values = presets[button.dataset.zhouModelPreset];
      [el["mu-wt"], el["mu-loh"], el["w-tri"], el["w-loh"], el.p0].forEach((node, i) => { node.value = values[i]; });
      draw();
    }));
    $("#zhou-model-controls").addEventListener("reset", () => setTimeout(draw));
    addEventListener("resize", draw);
    draw();
  }

  avecillaPlayground();
  modelBuilder();
  chuongStandingVariation();
  chuongEquationExercise();
  chuongCodeExercise();
  dfeExample();
  chuongChallenge();
  zhouModelPlayground();
})();
