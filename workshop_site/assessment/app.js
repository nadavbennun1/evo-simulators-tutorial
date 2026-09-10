(function () {
  "use strict";

  const root = document.getElementById("assessment-main");
  const progressBar = document.getElementById("assessment-progress-bar");
  const config = window.ASSESSMENT_CONFIG;
  const phase = new URLSearchParams(location.search).get("phase") === "post" ? "post" : "pre";
  const venues = [
    {id: "NYU", label: "NYU", image: "assets/nyu-official-seal.svg", alt: "New York University seal"},
    {id: "UMN", label: "UMN", image: "assets/umn-official-logo.svg", alt: "University of Minnesota logo"},
    {id: "TAU", label: "TAU", image: "assets/tau-official-logo.png", alt: "Tel Aviv University logo"},
    {id: "ONLINE", label: "Online", image: "assets/lone-wolf.svg", alt: "Cheerful lone wolf joining online", subtitle: "Self-guided · no live presentation"}
  ];
  const prohibitedPayloadKeys = new Set([
    "name", "email", "phone", "username", "department", "lab", "age", "gender", "career_stage",
    "geolocation", "latitude", "longitude", "ip", "user_agent", "screen", "referrer", "timezone",
    "advertising_id", "analytics_id"
  ]);
  const state = {
    bank: null,
    meta: null,
    venue: null,
    participantId: null,
    pairingCode: null,
    runId: null,
    runStartedAtClientMs: null,
    currentQuestion: 0,
    answers: {},
    orders: {},
    durations: {},
    screenStarted: null,
    selectedCard: null,
    confidence: {},
    evaluation: {},
    submitted: false
  };

  function uuid() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
  }

  function randomPairingCode() {
    const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    const bytes = crypto.getRandomValues(new Uint8Array(8));
    const characters = [...bytes].map((byte) => alphabet[byte % alphabet.length]);
    return `WOLF-${characters.slice(0, 4).join("")}-${characters.slice(4).join("")}`;
  }

  function shuffled(values) {
    const result = [...values];
    const random = crypto.getRandomValues(new Uint32Array(Math.max(1, result.length)));
    for (let index = result.length - 1; index > 0; index -= 1) {
      const other = random[index] % (index + 1);
      [result[index], result[other]] = [result[other], result[index]];
    }
    return result;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"]/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[character]));
  }

  function updateProgress(step, total) {
    progressBar.style.width = `${Math.max(0, Math.min(100, 100 * step / total))}%`;
  }

  function setScreen(markup, focusSelector) {
    root.innerHTML = `<section class="assessment-card">${markup}</section>`;
    requestAnimationFrame(() => {
      const focusTarget = root.querySelector(focusSelector || "h1, h2");
      if (focusTarget) {
        focusTarget.setAttribute("tabindex", "-1");
        focusTarget.focus({preventScroll: true});
      }
    });
  }

  function metaLine(label) {
    return `<div class="assessment-meta"><span>${escapeHtml(label)}</span><span>${phase === "pre" ? "Before" : "After"} workshop</span></div>`;
  }

  function cardIcon(icon) {
    const icons = {spark: "✦", growth: "↗", dice: "◇", draw: "θ", trajectory: "∿", compare: "≈", keep: "✓"};
    return icon ? `<span class="card-icon" aria-hidden="true">${icons[icon] || "•"}</span>` : "";
  }

  function venueScreen() {
    updateProgress(0, 9);
    const cards = venues.map((venue) => `<button class="venue-card" type="button" data-venue="${venue.id}" aria-pressed="${state.venue === venue.id}">
      <img src="${venue.image}" alt="${venue.alt}"><strong>${venue.label}</strong>${venue.subtitle ? `<small>${venue.subtitle}</small>` : ""}
    </button>`).join("");
    setScreen(`${metaLine("Start here · about 3 minutes")}
      <p class="assessment-kicker">Anonymous ${phase === "pre" ? "check-in" : "final check"}</p>
      <h1>Where did you attend this workshop?</h1>
      <div class="venue-grid">${cards}</div>
      <aside class="consent-box" aria-labelledby="consent-heading">
        <h2 id="consent-heading">${escapeHtml(config.consent.heading)}</h2>
        <p>${escapeHtml(config.consent.body)}</p><p>${escapeHtml(config.consent.exclusions)}</p>
        <p class="consent-version">Consent information version ${escapeHtml(config.consentVersion)}</p>
      </aside>
      <div class="assessment-actions"><a class="text-action" href="${phase === "pre" ? config.chapterOneUrl : config.workshopHomeUrl}">Skip and ${phase === "pre" ? "start" : "return to"} workshop</a><button id="participate" class="primary-action" type="button" ${state.venue ? "" : "disabled"}>Participate anonymously →</button></div>`);
    root.querySelectorAll("[data-venue]").forEach((button) => button.addEventListener("click", () => {
      state.venue = button.dataset.venue;
      root.querySelectorAll("[data-venue]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      root.querySelector("#participate").disabled = false;
    }));
    root.querySelector("#participate").addEventListener("click", consented);
  }

  function consented() {
    state.participantId = localStorage.getItem(config.participantIdKey);
    state.pairingCode = localStorage.getItem(config.pairingCodeKey);
    if (phase === "post" && !state.participantId) {
      pairingScreen();
      return;
    }
    if (!state.participantId) {
      state.participantId = uuid();
      localStorage.setItem(config.participantIdKey, state.participantId);
    }
    if (phase === "pre" && !state.pairingCode) {
      state.pairingCode = randomPairingCode();
      localStorage.setItem(config.pairingCodeKey, state.pairingCode);
    }
    startRun();
  }

  function pairingScreen() {
    updateProgress(0, 9);
    setScreen(`${metaLine("Anonymous pairing")}
      <p class="assessment-kicker">A different device?</p><h1>Pair with your check-in</h1>
      <p class="assessment-subtitle">If you saved your anonymous code, enter it here. It contains no name or venue information.</p>
      <div class="pairing-panel"><label for="pair-code-input">I have my anonymous code</label><div class="pairing-entry"><input id="pair-code-input" inputmode="text" autocomplete="off" maxlength="14" placeholder="WOLF-K7PM-4Q2X" aria-describedby="pair-error"><button id="use-pair-code" type="button">Use code</button></div><small id="pair-error" role="alert"></small></div>
      <div class="assessment-actions"><button id="pair-back" class="secondary-action" type="button">Back</button><button id="unpaired" class="primary-action" type="button">Continue without pairing</button></div>`);
    root.querySelector("#pair-back").addEventListener("click", venueScreen);
    root.querySelector("#use-pair-code").addEventListener("click", () => {
      const input = root.querySelector("#pair-code-input");
      const code = input.value.trim().toUpperCase();
      if (!/^WOLF-[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}$/.test(code)) {
        root.querySelector("#pair-error").textContent = "Enter the code in the form WOLF-XXXX-XXXX.";
        input.focus();
        return;
      }
      state.participantId = uuid();
      state.pairingCode = code;
      localStorage.setItem(config.participantIdKey, state.participantId);
      localStorage.setItem(config.pairingCodeKey, code);
      startRun();
    });
    root.querySelector("#unpaired").addEventListener("click", () => {
      state.participantId = uuid();
      state.pairingCode = null;
      localStorage.setItem(config.participantIdKey, state.participantId);
      localStorage.removeItem(config.pairingCodeKey);
      startRun();
    });
  }

  async function startRun() {
    state.runId = uuid();
    state.runStartedAtClientMs = Date.now();
    state.currentQuestion = 0;
    const event = eventEnvelope("started", {
      assessment_schema_version: config.assessmentSchemaVersion,
      question_bank_hash: state.meta.question_bank_sha256
    });
    await window.AssessmentBackend.enqueue(event);
    renderQuestion();
  }

  function eventEnvelope(eventType, payload) {
    const event = {
      id: uuid(),
      run_id: state.runId,
      participant_id: state.participantId,
      phase: phase,
      event_type: eventType,
      venue: state.venue,
      assessment_version: config.assessmentVersion,
      workshop_git_sha: state.meta.workshop_git_sha,
      consent_version: config.consentVersion,
      payload: payload
    };
    assertSafePayload(event);
    return event;
  }

  function assertSafePayload(value) {
    function visit(item) {
      if (!item || typeof item !== "object") return;
      for (const [key, nested] of Object.entries(item)) {
        if (prohibitedPayloadKeys.has(key.toLowerCase())) throw new Error(`Prohibited assessment field: ${key}`);
        visit(nested);
      }
    }
    visit(value);
  }

  function beginQuestionTiming(questionId) {
    state.screenStarted = {id: questionId, at: performance.now()};
  }

  function endQuestionTiming() {
    if (!state.screenStarted) return;
    const elapsed = Math.max(0, Math.round(performance.now() - state.screenStarted.at));
    state.durations[state.screenStarted.id] = (state.durations[state.screenStarted.id] || 0) + elapsed;
    state.screenStarted = null;
  }

  function questionHeader(question, number) {
    return `${metaLine(`Concept ${number} of ${state.bank.knowledge_questions.length}`)}<p class="assessment-kicker">Small experiment ${String(number).padStart(2, "0")}</p><h1>${escapeHtml(question.title)}</h1><p class="assessment-subtitle">${escapeHtml(question.prompt)}</p>`;
  }

  function answerReady(question, answer) {
    if (question.type === "matching" || question.type === "flow") return answer && Object.keys(answer).length === question.cards.length;
    if (question.type === "ordering") return answer && answer.length === question.cards.length && answer.every(Boolean);
    return Boolean(answer);
  }

  function renderQuestion() {
    const question = state.bank.knowledge_questions[state.currentQuestion];
    const number = state.currentQuestion + 1;
    updateProgress(number, 9);
    if (!state.orders[question.question_id]) {
      const choices = question.cards || question.choices;
      state.orders[question.question_id] = shuffled(choices.map((item) => item.id));
    }
    state.selectedCard = null;
    let interaction = "";
    if (question.type === "matching") interaction = matchingInteraction(question);
    if (question.type === "ordering") interaction = orderingInteraction(question);
    if (question.type === "flow") interaction = flowInteraction(question);
    if (question.type === "choice") interaction = choiceInteraction(question);
    if (question.type === "plot_choice") interaction = observationInteraction(question);
    const answer = state.answers[question.question_id];
    setScreen(`${questionHeader(question, number)}${interaction}<div class="assessment-actions"><button id="previous" class="secondary-action" type="button">${number === 1 ? "Back to venue" : "Previous"}</button><button id="question-next" class="primary-action" type="button" ${answerReady(question, answer) ? "" : "disabled"}>${number === state.bank.knowledge_questions.length ? "Confidence →" : "Next →"}</button></div>`);
    bindQuestion(question);
    root.querySelector("#previous").addEventListener("click", () => {
      endQuestionTiming();
      if (state.currentQuestion === 0) venueScreen();
      else {state.currentQuestion -= 1; renderQuestion();}
    });
    root.querySelector("#question-next").addEventListener("click", () => {
      endQuestionTiming();
      if (state.currentQuestion === state.bank.knowledge_questions.length - 1) confidenceScreen();
      else {state.currentQuestion += 1; renderQuestion();}
    });
    beginQuestionTiming(question.question_id);
  }

  function cardById(question, id) {
    return question.cards.find((card) => card.id === id);
  }

  function bankCards(question, used) {
    return state.orders[question.question_id].filter((id) => !used.includes(id)).map((id) => {
      const card = cardById(question, id);
      return `<button class="move-card" type="button" data-card="${id}" aria-pressed="false">${cardIcon(card.icon)}<span>${escapeHtml(card.label)}</span></button>`;
    }).join("");
  }

  function matchingInteraction(question) {
    const answer = state.answers[question.question_id] || {};
    const used = Object.keys(answer);
    const targets = question.targets.map((target) => {
      const cardId = Object.keys(answer).find((key) => answer[key] === target.id);
      const card = cardId ? cardById(question, cardId) : null;
      return `<button class="drop-target${card ? " filled" : ""}" type="button" data-drop="${target.id}" aria-label="${escapeHtml(target.label)}${card ? `, assigned ${card.label}` : ", empty"}"><span class="target-label">${escapeHtml(target.label)}</span>${card ? `<span class="placed-card">${escapeHtml(card.label)}</span>` : ""}</button>`;
    }).join("");
    return `<div class="choice-bank" aria-label="Cards to place">${bankCards(question, used)}</div><div class="drop-grid">${targets}</div>`;
  }

  function orderingInteraction(question) {
    const answer = state.answers[question.question_id] || Array(question.cards.length).fill(null);
    const used = answer.filter(Boolean);
    const slots = answer.map((cardId, index) => {
      const card = cardId ? cardById(question, cardId) : null;
      return `<button class="sequence-slot${card ? " filled" : ""}" type="button" data-drop="${index}" aria-label="Step ${index + 1}${card ? `, ${card.label}` : ", empty"}">${card ? `<span class="placed-card">${escapeHtml(card.label)}</span>` : ""}</button>`;
    }).join("");
    return `<div class="choice-bank" aria-label="Cards to place">${bankCards(question, used)}</div><div class="sequence-slots${question.cards.length === 4 ? " four" : ""}">${slots}</div>`;
  }

  function flowInteraction(question) {
    const answer = state.answers[question.question_id] || {};
    const used = Object.values(answer);
    const slotNames = {parameters: "Candidate input", simulator: "Forward mechanism", simulated: "Generated data", observed: "Experiment", compare: "Inference", posterior: "What we learn"};
    const slot = (slotId) => {
      const cardId = answer[slotId];
      const card = cardId ? cardById(question, cardId) : null;
      return `<button class="flow-slot${card ? " filled" : ""}" type="button" data-slot="${slotId}" data-drop="${slotId}" aria-label="${slotNames[slotId]}${card ? `, ${card.label}` : ", empty"}"><span class="slot-hint">${slotNames[slotId]}</span>${card ? `<span class="placed-card">${escapeHtml(card.label)}</span>` : ""}</button>`;
    };
    return `<div class="choice-bank" aria-label="Cards to place">${bankCards(question, used)}</div><div class="flow-board" aria-label="Simulation-based inference flow diagram">${slot("parameters")}<span class="flow-arrow a1" aria-hidden="true">→</span>${slot("simulator")}<span class="flow-arrow a2" aria-hidden="true">→</span>${slot("simulated")}${slot("observed")}<span class="flow-arrow a3" aria-hidden="true">+</span>${slot("compare")}<span class="flow-arrow a4" aria-hidden="true">→</span>${slot("posterior")}</div>`;
  }

  function identifiabilityPlot() {
    return `<svg class="science-plot" viewBox="0 0 620 255" role="img" aria-label="Three different smooth trajectories pass close to the same five observed timepoints"><line class="axis" x1="55" y1="215" x2="590" y2="215"/><line class="axis" x1="55" y1="215" x2="55" y2="25"/><path class="hypothesis h1" d="M60 204 C160 201 205 179 265 126 S380 52 585 43"/><path class="hypothesis h2" d="M60 205 C175 205 225 190 285 128 S410 48 585 45"/><path class="hypothesis h3" d="M60 202 C120 197 205 174 280 132 S390 66 585 42"/><g><circle class="observation" cx="100" cy="201" r="7"/><circle class="observation" cx="222" cy="175" r="7"/><circle class="observation" cx="300" cy="117" r="7"/><circle class="observation" cx="430" cy="57" r="7"/><circle class="observation" cx="560" cy="45" r="7"/></g><text x="515" y="242">Time</text><text x="72" y="45">Frequency</text></svg>`;
  }

  function choiceInteraction(question) {
    const selected = state.answers[question.question_id];
    const lookup = Object.fromEntries(question.choices.map((choice) => [choice.id, choice]));
    const choices = state.orders[question.question_id].map((id, index) => `<button class="large-choice" type="button" data-choice="${id}" aria-pressed="${selected === id}"><b>${String.fromCharCode(65 + index)}</b><span>${escapeHtml(lookup[id].label)}</span></button>`).join("");
    return `${identifiabilityPlot()}<div class="large-choice-grid">${choices}</div>`;
  }

  function observationPlot(selected) {
    const markerX = {early: 150, middle: 320, late: 505}[selected];
    return `<svg class="science-plot observation-design-plot" viewBox="0 0 620 275" role="img" aria-label="Three hypotheses are similar early and late but clearly separate in the middle"><line class="axis" x1="55" y1="225" x2="590" y2="225"/><line class="axis" x1="55" y1="225" x2="55" y2="28"/><rect class="region${selected === "early" ? " selected" : ""}" data-region-drop="early" x="58" y="28" width="175" height="197"/><rect class="region${selected === "middle" ? " selected" : ""}" data-region-drop="middle" x="233" y="28" width="175" height="197"/><rect class="region${selected === "late" ? " selected" : ""}" data-region-drop="late" x="408" y="28" width="178" height="197"/><path class="hypothesis h1" d="M60 210 C170 208 210 190 270 112 S410 45 585 42"/><path class="hypothesis h2" d="M60 211 C165 208 230 203 305 169 S430 64 585 43"/><path class="hypothesis h3" d="M60 209 C160 207 224 180 300 78 S430 41 585 42"/><g><circle class="observation" cx="105" cy="209" r="7"/><circle class="observation" cx="535" cy="43" r="7"/></g>${selected ? `<line class="sample-marker" x1="${markerX}" y1="38" x2="${markerX}" y2="217"/><path class="sample-head" d="M${markerX - 9} 38h18l-9 12Z"/>` : ""}<text x="120" y="251">Early</text><text x="300" y="251">Middle</text><text x="490" y="251">Late</text><text x="72" y="48">Frequency</text></svg>`;
  }

  function observationInteraction(question) {
    const selected = state.answers[question.question_id];
    const lookup = Object.fromEntries(question.choices.map((choice) => [choice.id, choice]));
    const buttons = state.orders[question.question_id].map((id) => `<button class="large-choice" type="button" data-choice="${id}" aria-pressed="${selected === id}"><b aria-hidden="true">+</b><span>${lookup[id].label}</span></button>`).join("");
    return `<div class="choice-bank"><button class="move-card" type="button" data-sampling-marker aria-pressed="false"><span class="card-icon" aria-hidden="true">↓</span><span>New sampling time</span></button></div>${observationPlot(selected)}<div class="large-choice-grid">${buttons}</div>`;
  }

  function bindQuestion(question) {
    root.querySelectorAll("[data-choice]").forEach((button) => button.addEventListener("click", () => {
      endQuestionTiming();
      state.answers[question.question_id] = button.dataset.choice;
      renderQuestion();
    }));
    root.querySelectorAll("[data-card]").forEach((button) => {
      button.addEventListener("click", () => selectCard(button.dataset.card, button));
      bindPointerDrag(button, button.dataset.card, (dropId) => placeCard(question, button.dataset.card, dropId));
    });
    root.querySelectorAll("[data-drop]").forEach((target) => target.addEventListener("click", () => {
      if (state.selectedCard) placeCard(question, state.selectedCard, target.dataset.drop);
      else removeFromSlot(question, target.dataset.drop);
    }));
    const marker = root.querySelector("[data-sampling-marker]");
    if (marker) bindPointerDrag(marker, "sampling-marker", (dropId) => {
      endQuestionTiming();
      state.answers[question.question_id] = dropId;
      renderQuestion();
    }, "[data-region-drop]");
  }

  function selectCard(cardId, button) {
    state.selectedCard = state.selectedCard === cardId ? null : cardId;
    root.querySelectorAll("[data-card]").forEach((item) => item.setAttribute("aria-pressed", String(item === button && state.selectedCard === cardId)));
    if (state.selectedCard) {
      const firstEmpty = [...root.querySelectorAll("[data-drop]")].find((target) => !target.classList.contains("filled"));
      if (firstEmpty) firstEmpty.focus();
    }
  }

  function placeCard(question, cardId, dropId) {
    if (question.type === "ordering") {
      const answer = state.answers[question.question_id] || Array(question.cards.length).fill(null);
      const prior = answer.indexOf(cardId);
      if (prior >= 0) answer[prior] = null;
      answer[Number(dropId)] = cardId;
      state.answers[question.question_id] = answer;
    } else if (question.type === "matching") {
      const answer = state.answers[question.question_id] || {};
      Object.keys(answer).forEach((key) => {if (answer[key] === dropId || key === cardId) delete answer[key];});
      answer[cardId] = dropId;
      state.answers[question.question_id] = answer;
    } else if (question.type === "flow") {
      const answer = state.answers[question.question_id] || {};
      Object.keys(answer).forEach((slot) => {if (answer[slot] === cardId) delete answer[slot];});
      answer[dropId] = cardId;
      state.answers[question.question_id] = answer;
    }
    endQuestionTiming();
    renderQuestion();
  }

  function removeFromSlot(question, dropId) {
    if (question.type === "ordering") {
      const answer = state.answers[question.question_id] || [];
      if (!answer[Number(dropId)]) return;
      answer[Number(dropId)] = null;
    } else if (question.type === "matching") {
      const answer = state.answers[question.question_id] || {};
      const key = Object.keys(answer).find((cardId) => answer[cardId] === dropId);
      if (!key) return;
      delete answer[key];
    } else if (question.type === "flow") {
      const answer = state.answers[question.question_id] || {};
      if (!answer[dropId]) return;
      delete answer[dropId];
    }
    endQuestionTiming();
    renderQuestion();
  }

  function bindPointerDrag(button, cardId, onDrop, targetSelector) {
    let start = null;
    let moved = false;
    button.addEventListener("pointerdown", (event) => {
      start = {x: event.clientX, y: event.clientY};
      moved = false;
      button.classList.add("dragging");
      button.setPointerCapture(event.pointerId);
    });
    button.addEventListener("pointermove", (event) => {
      if (!start) return;
      moved = moved || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 8;
      root.querySelectorAll(".drop-hover").forEach((item) => item.classList.remove("drop-hover"));
      const under = document.elementFromPoint(event.clientX, event.clientY);
      const target = under && under.closest(targetSelector || "[data-drop]");
      if (moved && target) target.classList.add("drop-hover");
    });
    button.addEventListener("pointerup", (event) => {
      button.classList.remove("dragging");
      root.querySelectorAll(".drop-hover").forEach((item) => item.classList.remove("drop-hover"));
      if (!start || !moved) {start = null; return;}
      const under = document.elementFromPoint(event.clientX, event.clientY);
      const target = under && under.closest(targetSelector || "[data-drop]");
      start = null;
      if (target) {
        event.preventDefault();
        onDrop(target.dataset.regionDrop || target.dataset.drop, cardId);
      }
    });
    button.addEventListener("pointercancel", () => {start = null; button.classList.remove("dragging");});
  }

  function confidenceScreen() {
    updateProgress(7, 9);
    const items = state.bank.confidence_questions.map((question) => ratingBlock(question, state.confidence, "Not yet", "Very confident")).join("");
    setScreen(`${metaLine("Two confidence statements")}<p class="assessment-kicker">Different from the concept score</p><h1>How confident do you feel?</h1><p class="assessment-subtitle">Choose the response that best reflects how you feel right now.</p><div class="scale-stack">${items}</div><div class="assessment-actions"><button id="confidence-back" class="secondary-action" type="button">Previous</button><button id="confidence-next" class="primary-action" type="button" ${ratingsComplete(state.bank.confidence_questions, state.confidence) ? "" : "disabled"}>${phase === "post" ? "Two final items →" : "Submit →"}</button></div>`);
    bindRatings(state.confidence, () => confidenceScreen());
    root.querySelector("#confidence-back").addEventListener("click", () => {endQuestionTiming(); state.currentQuestion = 5; renderQuestion();});
    root.querySelector("#confidence-next").addEventListener("click", () => phase === "post" ? evaluationScreen() : submitAssessment());
    beginQuestionTiming("confidence");
  }

  function ratingBlock(question, storage, low, high) {
    const buttons = [1, 2, 3, 4, 5].map((value) => `<button type="button" data-rating="${question.question_id}" data-value="${value}" aria-pressed="${storage[question.question_id] === value}" aria-label="${value} of 5">${value}</button>`).join("");
    return `<article class="scale-question"><p>${escapeHtml(question.statement)}</p><div class="rating-scale" role="group" aria-label="${escapeHtml(question.statement)}">${buttons}</div><div class="scale-ends"><span>${low}</span><span>${high}</span></div></article>`;
  }

  function bindRatings(storage, rerender) {
    root.querySelectorAll("[data-rating]").forEach((button) => button.addEventListener("click", () => {
      endQuestionTiming();
      storage[button.dataset.rating] = Number(button.dataset.value);
      rerender();
    }));
  }

  function ratingsComplete(definitions, storage) {
    return definitions.every((question) => Number.isInteger(storage[question.question_id]));
  }

  function evaluationScreen() {
    updateProgress(8, 9);
    const items = state.bank.post_evaluation_questions.map((question) => ratingBlock(question, state.evaluation, "Disagree", "Agree")).join("");
    setScreen(`${metaLine("Workshop reflection")}<p class="assessment-kicker">Two quick items</p><h1>Did the workshop connect?</h1><div class="scale-stack">${items}</div><div class="assessment-actions"><button id="evaluation-back" class="secondary-action" type="button">Previous</button><button id="evaluation-submit" class="primary-action" type="button" ${ratingsComplete(state.bank.post_evaluation_questions, state.evaluation) ? "" : "disabled"}>Submit →</button></div>`);
    bindRatings(state.evaluation, () => evaluationScreen());
    root.querySelector("#evaluation-back").addEventListener("click", () => {endQuestionTiming(); confidenceScreen();});
    root.querySelector("#evaluation-submit").addEventListener("click", submitAssessment);
    beginQuestionTiming("post_evaluation");
  }

  function rawAnswers() {
    const variant = state.bank.variants[phase];
    const knowledge = state.bank.knowledge_questions.map((question) => ({
      question_id: question.question_id,
      question_revision: question.question_revision,
      variant_id: `${question.question_id}-${variant}`,
      displayed_order: state.orders[question.question_id],
      response: state.answers[question.question_id],
      duration_ms: state.durations[question.question_id] || 0
    }));
    const confidence = state.bank.confidence_questions.map((question, index) => ({
      question_id: question.question_id,
      question_revision: question.question_revision,
      variant_id: `${question.question_id}-${variant}`,
      displayed_order: index + 1,
      response: state.confidence[question.question_id],
      duration_ms: state.durations.confidence || 0
    }));
    const evaluation = phase === "post" ? state.bank.post_evaluation_questions.map((question, index) => ({
      question_id: question.question_id,
      question_revision: question.question_revision,
      variant_id: `${question.question_id}-${variant}`,
      displayed_order: index + 1,
      response: state.evaluation[question.question_id],
      duration_ms: state.durations.post_evaluation || 0
    })) : [];
    return {knowledge: knowledge, confidence: confidence, post_evaluation: evaluation};
  }

  async function submitAssessment() {
    if (state.submitted) return;
    endQuestionTiming();
    state.submitted = true;
    updateProgress(9, 9);
    setScreen(`${metaLine("Saving anonymously")}<div class="assessment-loading"><div class="tiny-culture" aria-hidden="true"><i></i><i></i><i></i></div><h1>Finishing the run…</h1><p>Your raw answers are being saved.</p></div>`);
    const payload = {
      assessment_schema_version: config.assessmentSchemaVersion,
      assessment_version: config.assessmentVersion,
      question_bank_hash: state.meta.question_bank_sha256,
      workshop_git_sha: state.meta.workshop_git_sha,
      consent_version: config.consentVersion,
      phase: phase,
      venue: state.venue,
      pairing_code: state.pairingCode,
      client_duration_ms: Math.max(0, Date.now() - state.runStartedAtClientMs),
      answers: rawAnswers()
    };
    const status = await window.AssessmentBackend.enqueue(eventEnvelope("completed", payload));
    resultScreen(status);
  }

  function equalResponse(actual, expected) {
    if (Array.isArray(expected)) return Array.isArray(actual) && expected.length === actual.length && expected.every((value, index) => actual[index] === value);
    if (expected && typeof expected === "object") return actual && Object.keys(expected).every((key) => actual[key] === expected[key]) && Object.keys(actual).length === Object.keys(expected).length;
    return actual === expected;
  }

  function resultScreen(status) {
    const pending = status.pending > 0;
    const submission = pending ? `<p class="submission-status pending">Your answers are saved on this device and will be submitted when a connection is available.</p>` : `<p class="submission-status">Your anonymous response was submitted successfully.</p>`;
    if (phase === "pre") {
      setScreen(`${metaLine("Check-in complete")}<p class="assessment-kicker">Ready for the workshop</p><h1>Thanks — we’ll revisit these ideas at the end.</h1>${submission}<div class="pairing-panel"><strong>Your optional anonymous pairing code</strong><br><code class="pair-code">${escapeHtml(state.pairingCode)}</code><p>Save this only if you might complete the final check on another device.</p></div><div class="assessment-actions"><a class="primary-link primary-action" href="${config.chapterOneUrl}">Start the workshop →</a></div>`);
      return;
    }
    const results = state.bank.knowledge_questions.map((question) => ({question: question, correct: equalResponse(state.answers[question.question_id], question.correct_response)}));
    const score = results.filter((item) => item.correct).length;
    const feedback = results.map((item) => `<article class="${item.correct ? "" : "missed"}"><h3>${item.correct ? "✓" : "→"} ${escapeHtml(item.question.title)}</h3><p>${escapeHtml(item.question.feedback)}</p></article>`).join("");
    setScreen(`${metaLine("Final check complete")}<p class="assessment-kicker">A gentle reflection</p><h1>${score} / 6 concepts</h1><div class="result-score"><strong>${score}</strong><span>of 6 workshop concepts</span></div><p class="assessment-subtitle">You built the forward model, inverted it with simulations, and reasoned about uncertainty and informative sampling.</p>${submission}<div class="feedback-list">${feedback}</div><div class="assessment-actions"><a class="primary-link primary-action" href="${config.workshopHomeUrl}">Back to workshop →</a></div>`);
  }

  async function initialize() {
    try {
      const [bankResponse, metaResponse] = await Promise.all([fetch(config.questionBankPath), fetch(config.buildMetaPath)]);
      if (!bankResponse.ok || !metaResponse.ok) throw new Error("Assessment files unavailable");
      state.bank = await bankResponse.json();
      state.meta = await metaResponse.json();
      if (state.bank.assessment_version !== config.assessmentVersion || state.meta.assessment_version !== config.assessmentVersion) throw new Error("Assessment version mismatch");
      window.AssessmentBackend.flush();
      venueScreen();
    } catch (error) {
      setScreen(`<p class="assessment-kicker">Unable to start</p><h1>The assessment files did not load.</h1><p class="assessment-subtitle">The workshop is still available.</p><div class="assessment-actions"><a class="primary-link primary-action" href="${phase === "pre" ? config.chapterOneUrl : config.workshopHomeUrl}">${phase === "pre" ? "Start" : "Return to"} workshop →</a></div>`);
    }
  }

  initialize();
})();
