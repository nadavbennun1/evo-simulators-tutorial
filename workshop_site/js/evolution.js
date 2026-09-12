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
      [[canvas, trajectories], [orderCanvas, orderTrajectories]].forEach(([target, runs], panelIndex) => {
        const f = P.frame(target, 0, 1, 0, duration);
        runs.forEach((traj, r) => colors.forEach((color, state) => {
          P.line(f, xs.slice(0, show + 1), traj.slice(0, show + 1).map(x => x[state]), color, r === 0 ? 2 : 0.9, r === 0 ? 0.9 : 0.16);
        }));
        if (panelIndex === 0) ["Ancestral", "GAP1 CNV", "Other beneficial"].forEach((name, i) => P.text(f, name, duration * 0.05, 0.58 - i * 0.075, {color: colors[i], font: "bold 11px system-ui"}));
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

  function chuongStandingVariation() {
    const canvas = $("#chuong-phi-canvas");
    if (!canvas) return;
    const phi = $("#chuong-phi"), play = $("#chuong-phi-play");
    let timer = null, started = false, visible = 0;
    const generations = Array.from({length: 117}, (_, i) => i), logS = -0.74, logDelta = -4.84;
    const superscript = value => String(value).replace("-", "⁻").replace("0", "⁰").replace("1", "¹").replace("2", "²").replace("3", "³").replace("4", "⁴").replace("5", "⁵").replace("6", "⁶").replace("7", "⁷").replace("8", "⁸").replace("9", "⁹");
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
    function draw() {
      const comparisonLogPhi = +phi.value, baseline = simulate(-12), comparison = simulate(comparisonLogPhi);
      const f = P.frame(canvas, 0, 1, 0, 116);
      $("#chuong-phi-label").textContent = comparisonLogPhi.toFixed(1).replace("-", "−");
      if (!started) {
        $("#chuong-phi-summary").textContent = "The plot starts empty. Choose φ, then play the comparison.";
        return;
      }
      const show = Math.min(visible, 116), shownGenerations = generations.slice(0, show + 1);
      P.line(f, shownGenerations, baseline.reported.slice(0, show + 1), P.C.blue, 3);
      P.line(f, shownGenerations, comparison.total.slice(0, show + 1), P.C.orange, 3);
      P.line(f, shownGenerations, comparison.reported.slice(0, show + 1), P.C.orange, 2, .95, [7,5]);
      const exp = comparisonLogPhi.toFixed(Number.isInteger(comparisonLogPhi) ? 0 : 1);
      P.text(f, "φ=0 baseline", 7, .96, {color:P.C.blue, font:"bold 11px system-ui"});
      P.text(f, `φ=10${superscript(exp)} total`, 7, .89, {color:P.C.orange, font:"bold 11px system-ui"});
      P.text(f, `φ=10${superscript(exp)} observed (dashed)`, 7, .82, {color:P.C.orange, font:"bold 11px system-ui"});
      const deltaTotal = 100 * (comparison.total[show] - baseline.total[show]), deltaReported = 100 * (comparison.reported[show] - baseline.reported[show]);
      $("#chuong-phi-summary").textContent = `Generation ${show}: relative to the φ≈0 baseline, φ=10${superscript(exp)} changes total GAP1 CNV abundance by ${deltaTotal.toFixed(2)} percentage points and observed CNV⁺ abundance by ${deltaReported.toFixed(2)} points.`;
    }
    function resetView() {
      if (timer) clearInterval(timer);
      timer = null; started = false; visible = 0; play.textContent = "Play comparison"; draw();
    }
    phi.addEventListener("input", resetView);
    $$('[data-phi-preset]').forEach(button => button.addEventListener("click", () => { phi.value = button.dataset.phiPreset; resetView(); }));
    play.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = "Resume"; return; }
      if (!started || visible >= 116) visible = 0;
      started = true; play.textContent = "Pause";
      timer = setInterval(() => {
        visible = Math.min(116, visible + 2); draw();
        if (visible >= 116) { clearInterval(timer); timer = null; play.textContent = "Replay"; }
      }, 100);
      draw();
    });
    $("#chuong-phi-controls").addEventListener("reset", () => setTimeout(resetView));
    addEventListener("resize", draw); resetView();
  }

  function chuongEquationExercise() {
    const root = $("#chuong-equation-exercise");
    if (!root) return;
    const buttons = $$('[data-chuong-step]', root), cards = $$('[data-chuong-card]', root), matrix = $(".chuong-matrix", root);
    let revealed = 0;
    function render() {
      cards.forEach(card => { card.hidden = +card.dataset.chuongCard > revealed; });
      buttons.forEach(button => { const step = +button.dataset.chuongStep; button.disabled = step > revealed + 1 || step <= revealed; button.classList.toggle("revealed", step <= revealed); });
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
    const normalize = value => value.replace(/\s+/g, "").replace(/'/g, '"').replace(/;$/, "");
    function statusNode(input) { return $("small", input.closest(".code-line-entry")); }
    function check(input) {
      let value = normalize(input.value), lhs = `${normalize(input.dataset.codeLhs)}=`;
      if (value.startsWith(lhs)) value = value.slice(lhs.length);
      const okay = input.dataset.codeAlternatives.split("|||").map(normalize).includes(value);
      input.classList.toggle("correct", okay); input.classList.toggle("incorrect", !okay);
      statusNode(input).textContent = okay ? "Correct" : "Try an equivalent expression using the variables above";
      const correct = inputs.filter(node => node.classList.contains("correct")).length;
      summary.textContent = `${correct} of ${inputs.length} lines checked correctly.`;
    }
    $$('[data-check-code-line]', root).forEach(button => button.addEventListener("click", () => check($("input", button.closest(".code-fill-row")))));
    $("#reveal-chuong-code").addEventListener("click", () => { inputs.forEach(input => { input.value = input.dataset.codeAnswer; input.classList.add("correct"); input.classList.remove("incorrect"); statusNode(input).textContent = "Revealed"; }); summary.textContent = "All four lines revealed. Each line now matches its displayed equation."; });
    $("#reset-chuong-code").addEventListener("click", () => { inputs.forEach(input => { input.value = ""; input.classList.remove("correct", "incorrect"); statusNode(input).textContent = ""; }); summary.textContent = ""; });
  }

  function effectivePopulationSimulators() {
    const chemostatCanvas = $("#chemostat-ne-canvas"), serialCanvas = $("#serial-ne-canvas");
    if (!chemostatCanvas || !serialCanvas) return;
    const sci = (value, digits = 2) => {
      if (!Number.isFinite(value) || value === 0) return value === 0 ? "0" : "—";
      const exponent = Math.floor(Math.log10(Math.abs(value))), coefficient = value / (10 ** exponent);
      return `${coefficient.toFixed(digits)} × 10<sup>${exponent}</sup>`;
    };
    const compact = value => value >= 1e6 ? `${(value / 1e6).toFixed(value >= 1e8 ? 0 : 2)} million` : Math.round(value).toLocaleString();

    const chemoNe = $("#chemostat-ne"), chemoDraws = $("#chemostat-ne-draws"), chemoSeed = $("#chemostat-ne-seed");
    const chemoCards = $$("#chemostat-ne-calculation article strong");
    let chemoTimer = null, chemoStarted = false, chemoVisible = 0, chemoChanges = [];

    function prepareChemostat() {
      const nEff = Math.round(10 ** (+chemoNe.value)), total = +chemoDraws.value;
      const R = S.mulberry32(+chemoSeed.value || 0), p = 0.5;
      chemoChanges = Array.from({length: total}, () => binomial(nEff, p, R) / nEff - p);
    }

    function drawChemostat() {
      const nEff = Math.round(10 ** (+chemoNe.value)), total = +chemoDraws.value, p = 0.5;
      const theoreticalSd = Math.sqrt(p * (1 - p) / nEff);
      const multiplier = theoreticalSd < 1e-4 ? 1e5 : theoreticalSd < 1e-3 ? 1e4 : theoreticalSd < 1e-2 ? 1e3 : 1e2;
      const limit = 4.25 * theoreticalSd * multiplier;
      const f = P.frame(chemostatCanvas, -limit, limit, 1, total);
      $("#chemostat-ne-label").textContent = (+chemoNe.value).toFixed(2);
      $("#chemostat-ne-axis").textContent = `Vertical axis: (p′ − p) × ${multiplier.toLocaleString()}. The horizontal line is no frequency change.`;
      P.line(f, [1, total], [0, 0], P.C.muted, 1.4, 1, [5, 4]);
      P.text(f, "independent neutral draws →", Math.max(2, total * 0.03), limit * 0.9, {color:P.C.muted, font:"11px system-ui"});
      if (!chemoStarted) {
        chemoCards.forEach(card => { card.textContent = "—"; });
        $("#chemostat-ne-summary").textContent = "The plot starts empty. Run the neutral simulation to build the variance estimate.";
        return;
      }
      const shown = chemoChanges.slice(0, chemoVisible), xs = shown.map((_, i) => i + 1);
      P.points(f, xs, shown.map(value => value * multiplier), P.C.blue, total > 300 ? 1.5 : 2.2);
      chemoCards[0].innerHTML = `0.500 × 0.500 = <em>0.250</em>`;
      if (shown.length < 2) {
        chemoCards[1].textContent = "waiting for repeated draws";
        chemoCards[2].textContent = "—";
        $("#chemostat-ne-summary").textContent = `Draw ${shown.length} of ${total}. Variance requires repeated next-generation draws.`;
        return;
      }
      const mean = shown.reduce((sum, value) => sum + value, 0) / shown.length;
      const variance = shown.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (shown.length - 1);
      const estimate = p * (1 - p) / variance;
      chemoCards[1].innerHTML = sci(variance, 2);
      chemoCards[2].innerHTML = `0.250 / ${sci(variance, 2)} = <em>${sci(estimate, 2)}</em>`;
      $("#chemostat-ne-summary").textContent = `${shown.length} of ${total} draws: the simulated variance is ${variance.toExponential(2)}, giving N̂ₑ = ${compact(estimate)} cells. The generating value is ${compact(nEff)}.`;
    }

    function resetChemostat() {
      if (chemoTimer) clearInterval(chemoTimer);
      chemoTimer = null; chemoStarted = false; chemoVisible = 0; chemoChanges = [];
      $("#chemostat-ne-run").textContent = "Run neutral simulation"; drawChemostat();
    }

    [chemoNe, chemoDraws, chemoSeed].forEach(node => node.addEventListener("input", resetChemostat));
    $$('[data-chemostat-ne-preset]').forEach(button => button.addEventListener("click", () => { chemoNe.value = button.dataset.chemostatNePreset; resetChemostat(); }));
    $("#chemostat-ne-run").addEventListener("click", event => {
      if (chemoTimer) { clearInterval(chemoTimer); chemoTimer = null; event.target.textContent = "Resume"; return; }
      if (!chemoStarted || chemoVisible >= +chemoDraws.value) { chemoVisible = 0; prepareChemostat(); }
      chemoStarted = true; event.target.textContent = "Pause";
      const chunk = Math.max(2, Math.ceil(+chemoDraws.value / 45));
      chemoTimer = setInterval(() => {
        chemoVisible = Math.min(+chemoDraws.value, chemoVisible + chunk); drawChemostat();
        if (chemoVisible >= +chemoDraws.value) { clearInterval(chemoTimer); chemoTimer = null; event.target.textContent = "Run again"; }
      }, 70);
      drawChemostat();
    });
    $("#chemostat-ne-controls").addEventListener("reset", () => setTimeout(resetChemostat));

    const serialN0 = $("#serial-ne-n0"), serialDilution = $("#serial-ne-dilution");
    const serialCards = $$("#serial-ne-calculation article strong");
    let serialTimer = null, serialStarted = false, serialVisible = 0;

    function serialValues() {
      const n0 = 10 ** (+serialN0.value), fold = +serialDilution.value, generations = Math.round(Math.log2(fold));
      const sizes = Array.from({length: generations + 1}, (_, g) => n0 * (2 ** g));
      const intervalSizes = sizes.slice(0, generations), reciprocalSum = intervalSizes.reduce((sum, value) => sum + 1 / value, 0);
      return {n0, fold, generations, sizes, intervalSizes, reciprocalSum, nEff: generations / reciprocalSum};
    }

    function drawSerial() {
      const values = serialValues(), xMax = values.generations, yMin = Math.log10(values.n0) - 0.15, yMax = Math.log10(values.sizes.at(-1)) + 0.2;
      const f = P.frame(serialCanvas, yMin, yMax, 0, xMax);
      $("#serial-ne-n0-label").textContent = (+serialN0.value).toFixed(2);
      P.text(f, "log₁₀ population size", 0.12, yMax - 0.08, {color:P.C.muted, font:"11px system-ui"});
      if (!serialStarted) {
        $("#serial-ne-generations").innerHTML = "";
        serialCards.forEach(card => { card.textContent = "—"; });
        $("#serial-ne-summary").textContent = "The plot starts empty. Simulate a cycle to reveal each generation's contribution.";
        return;
      }
      const show = Math.min(serialVisible, values.generations), xs = Array.from({length: show + 1}, (_, g) => g);
      const shownLogs = values.sizes.slice(0, show + 1).map(Math.log10);
      P.line(f, xs, shownLogs, P.C.blue, 3); P.points(f, xs, shownLogs, P.C.blue, 4);
      P.text(f, "bottleneck", 0.08, Math.log10(values.n0) + 0.08, {color:P.C.clay, font:"bold 11px system-ui"});
      if (show === values.generations) P.text(f, `1:${values.fold} transfer`, Math.max(0.2, values.generations - 1.25), Math.log10(values.sizes.at(-1)) - 0.08, {color:P.C.clay, font:"bold 11px system-ui"});
      $("#serial-ne-generations").innerHTML = values.intervalSizes.map((size, g) => {
        const contribution = (1 / size) / values.reciprocalSum;
        const revealed = g < Math.max(1, show) || show === values.generations;
        return `<article class="${revealed ? "revealed" : ""}"><small>generation ${g}</small><b>N<sub>${g}</sub> = ${compact(size)}</b><span>1/N<sub>${g}</sub> = ${sci(1 / size, 2)}</span><em>${(100 * contribution).toFixed(1)}% of drift weight</em></article>`;
      }).join("");
      if (show < values.generations) {
        serialCards.forEach(card => { card.textContent = "complete the cycle"; });
        $("#serial-ne-summary").textContent = `${show} of ${values.generations} doublings revealed. The earliest interval is the smallest population and therefore contributes the largest reciprocal weight.`;
        return;
      }
      const bottleneckShare = (1 / values.n0) / values.reciprocalSum;
      serialCards[0].innerHTML = values.intervalSizes.map(value => sci(1 / value, 1)).join(" + ") + ` = <em>${sci(values.reciprocalSum, 2)}</em>`;
      serialCards[1].innerHTML = `${values.generations} / ${sci(values.reciprocalSum, 2)} = <em>${sci(values.nEff, 2)} cells</em>`;
      serialCards[2].innerHTML = `${sci(1 / values.n0, 2)} / ${sci(values.reciprocalSum, 2)} = <em>${(100 * bottleneckShare).toFixed(1)}%</em>`;
      $("#serial-ne-summary").textContent = `After ${values.generations} doublings the culture reaches ${compact(values.sizes.at(-1))} cells before its 1:${values.fold} transfer. The cycle effective size is ${compact(values.nEff)}, only ${(values.nEff / values.n0).toFixed(2)} times the bottleneck size.`;
    }

    function resetSerial() {
      if (serialTimer) clearInterval(serialTimer);
      serialTimer = null; serialStarted = false; serialVisible = 0;
      $("#serial-ne-run").textContent = "Simulate one cycle"; drawSerial();
    }

    [serialN0, serialDilution].forEach(node => node.addEventListener("input", resetSerial));
    $("#serial-ne-run").addEventListener("click", event => {
      const generations = serialValues().generations;
      if (serialTimer) { clearInterval(serialTimer); serialTimer = null; event.target.textContent = "Resume"; return; }
      if (!serialStarted || serialVisible >= generations) serialVisible = 0;
      serialStarted = true; event.target.textContent = "Pause"; drawSerial();
      serialTimer = setInterval(() => {
        serialVisible = Math.min(generations, serialVisible + 1); drawSerial();
        if (serialVisible >= generations) { clearInterval(serialTimer); serialTimer = null; event.target.textContent = "Run again"; }
      }, 380);
    });
    $("#serial-ne-controls").addEventListener("reset", () => setTimeout(resetSerial));
    addEventListener("resize", () => { drawChemostat(); drawSerial(); });
    resetChemostat(); resetSerial();
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
      P.text(f, `average new-CNV effect s̄ = ${mu.toFixed(3)}`, Math.min(mu + 0.004, 0.105), 0.96, {font: "bold 11px system-ui"});
      const total = raw.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
      let cumulative = 0, q90 = xs.at(-1);
      for (let i = 0; i < xs.length; i++) { cumulative += Number.isFinite(raw[i]) ? raw[i] : 0; if (cumulative >= 0.9 * total) { q90 = xs[i]; break; } }
      $("#dfe-mean-label").textContent = mu.toFixed(3);
      $("#dfe-shape-label").textContent = k.toFixed(1);
      $("#dfe-summary").textContent = `The average newly formed CNV has s=${mu.toFixed(3)}. About 10% of new CNVs have effects above s=${q90.toFixed(3)}. The 25 dots are the fitness classes represented in the simulation.`;
      $("#dfe-change").textContent = k < 1.2 ? "Most new CNVs are nearly neutral, while a small minority confer much larger growth advantages. Those rare lineages can later dominate the population." : k > 3.5 ? "New CNVs have relatively similar growth advantages, so selection changes their relative abundance more slowly." : "Many new CNVs have modest advantages and a smaller group has substantially larger effects. Selection progressively enriches that high-fitness group.";
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
    const play = $("#zhou-model-play");
    let timer = null, started = false, visible = 0;

    function draw() {
      const theta = [+el["mu-wt"].value, +el["w-tri"].value, +el["w-loh"].value, +el["mu-loh"].value];
      const tri0 = +el.p0.value, p0 = [tri0, (1 - tri0) * 0.6, (1 - tri0) * 0.4];
      const trajectory = S.zhouDeterministic(theta, p0, 120).filter((_, i) => i % 10 === 0);
      const xs = Array.from({length: 13}, (_, i) => i), colors = [P.C.muted, P.C.orange, P.C.purple], f = P.frame(canvas, 0, 1, 0, 12);
      $("#zhou-model-mu-wt-label").textContent = (+el["mu-wt"].value).toFixed(2);
      $("#zhou-model-mu-loh-label").textContent = (+el["mu-loh"].value).toFixed(2);
      $("#zhou-model-w-tri-label").textContent = (+el["w-tri"].value).toFixed(3);
      $("#zhou-model-w-loh-label").textContent = (+el["w-loh"].value).toFixed(3);
      $("#zhou-model-p0-label").textContent = `${(tri0 * 100).toFixed(0)}%`;
      if (!started) {
        $("#zhou-model-composition").innerHTML = "";
        $("#zhou-model-summary").textContent = "The plot starts empty. Choose a scenario, then press Play.";
        $("#zhou-model-change").textContent = "The trajectory will reveal how transition rates and fitness jointly determine the route out of trisomy.";
        return;
      }
      const show = Math.min(visible, 12), current = trajectory[show];
      colors.forEach((color, state) => P.line(f, xs.slice(0, show + 1), trajectory.slice(0, show + 1).map(x => x[state]), color, 2.7));
      ["Trisomic", "Wild type", "LOH"].forEach((name, i) => P.text(f, name, 0.45, 0.58 - i * 0.075, {color: colors[i], font: "bold 12px system-ui"}));
      $("#zhou-model-composition").innerHTML = ["Trisomic", "Wild type", "LOH"].map((name, i) => `<span><b>${name}</b><br>${(current[i] * 100).toFixed(1)}% at P${show}</span>`).join("");
      const route = current[1] > current[2] * 1.25 ? "euploid recovery dominates" : current[2] > current[1] * 1.25 ? "LOH dominates" : "both chromosome-loss routes remain important";
      $("#zhou-model-change").textContent = `At passage ${show}, ${route}; transition rates set mutational supply while relative fitness reshapes the descendants after they appear.`;
      $("#zhou-model-summary").textContent = `Passage ${show} of 12: ${(current[0] * 100).toFixed(1)}% trisomic, ${(current[1] * 100).toFixed(1)}% wild type, ${(current[2] * 100).toFixed(1)}% LOH.`;
    }

    function resetView() {
      if (timer) clearInterval(timer);
      timer = null; started = false; visible = 0; play.textContent = "Play"; draw();
    }
    Object.values(el).forEach(node => node.addEventListener("input", resetView));
    const presets = {
      fit: [-3.47, -3.28, 0.92, 0.986, 0.99],
      wt: [-2.9, -5.2, 0.92, 0.96, 0.99],
      loh: [-5.2, -2.9, 0.92, 1.01, 0.99],
      fitness: [-3.5, -3.5, 1.02, 0.9, 0.99],
    };
    $$('[data-zhou-model-preset]').forEach(button => button.addEventListener("click", () => {
      const values = presets[button.dataset.zhouModelPreset];
      [el["mu-wt"], el["mu-loh"], el["w-tri"], el["w-loh"], el.p0].forEach((node, i) => { node.value = values[i]; });
      resetView();
    }));
    play.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = "Resume"; return; }
      if (!started || visible >= 12) visible = 0;
      started = true; play.textContent = "Pause";
      timer = setInterval(() => {
        visible = Math.min(12, visible + 1); draw();
        if (visible >= 12) { clearInterval(timer); timer = null; play.textContent = "Replay"; }
      }, 220);
      draw();
    });
    $("#zhou-model-controls").addEventListener("reset", () => setTimeout(resetView));
    addEventListener("resize", draw);
    resetView();
  }

  avecillaPlayground();
  chuongStandingVariation();
  chuongEquationExercise();
  chuongCodeExercise();
  effectivePopulationSimulators();
  dfeExample();
  chuongChallenge();
  zhouModelPlayground();
})();
