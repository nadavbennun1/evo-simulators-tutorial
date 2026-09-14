/*
 * Private Google Sheet receiver for the anonymous workshop assessment.
 * Bind this script to a private Sheet and deploy it as a Web app that executes
 * as you. Anonymous visitors can append validated events; doGet never exposes data.
 */

const ASSESSMENT_SHEET_NAME = 'assessment_events';
const ASSESSMENT_DASHBOARD_SHEET_NAME = 'assessment_dashboard';
const MAX_PAYLOAD_BYTES = 65536;
const HEADERS = [
  'id', 'received_at', 'run_id', 'participant_id', 'phase', 'event_type',
  'venue', 'assessment_version', 'workshop_git_sha', 'consent_version', 'payload',
  'forces_match', 'generation_order', 'inverse_process', 'identifiability',
  'abc_order', 'observation_design', 'confidence_simulator', 'confidence_inverse',
  'impact_models_data', 'research_relevance', 'client_duration_ms'
];
const PROHIBITED_KEYS = new Set([
  'name', 'email', 'phone', 'username', 'department', 'lab', 'age', 'gender',
  'career_stage', 'geolocation', 'latitude', 'longitude', 'ip', 'user_agent',
  'screen', 'referrer', 'timezone', 'advertising_id', 'analytics_id'
]);

/*
 * Dashboard scoring is deliberately versioned. Raw answers in assessment_events
 * remain authoritative; these rules only create an aggregate private view.
 */
const DASHBOARD_SCORING_RULES = {
  '1.0.0': {
    forces_match: {mutation: 'supplies_variants', selection: 'changes_contribution', drift: 'stochastic_trajectories'},
    generation_order: ['mutation', 'selection', 'drift'],
    inverse_process: {
      parameters: 'parameters', simulator: 'simulator', simulated: 'simulated',
      observed: 'observed', compare: 'compare', posterior: 'posterior'
    },
    identifiability: 'uncertainty',
    abc_order: ['prior', 'simulate', 'compare', 'keep'],
    observation_design: {pre: 'middle', post: 'middle'}
  },
  '1.1.0': {
    forces_match: {mutation: 'supplies_variants', selection: 'changes_contribution', drift: 'stochastic_trajectories'},
    generation_order: ['mutation', 'selection', 'drift'],
    inverse_process: {
      parameters: 'parameters', simulator: 'simulator', simulated: 'simulated',
      observed: 'observed', compare: 'compare', posterior: 'posterior'
    },
    identifiability: 'uncertainty',
    abc_order: ['prior', 'simulate', 'compare', 'keep'],
    observation_design: {pre: 'middle', post: 'late'}
  }
};

const DASHBOARD_ITEM_LABELS = {
  forces_match: 'Evolutionary forces',
  generation_order: 'One-generation order',
  inverse_process: 'Forward and inverse logic',
  identifiability: 'Identifiability',
  abc_order: 'Rejection ABC',
  observation_design: 'Observation design'
};

const DASHBOARD_CONFIDENCE_LABELS = {
  confidence_simulator: 'Build a stochastic simulator',
  confidence_inverse: 'Infer processes from trajectories'
};

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Assessment dashboard')
    .addItem('Refresh dashboard', 'refreshAssessmentDashboard')
    .addToUi();
}

function doGet(event) {
  const eventId = event && event.parameter && event.parameter.event_id;
  const callback = event && event.parameter && event.parameter.callback;
  if (eventId && callback) {
    if (!/^[0-9a-f-]{36}$/i.test(eventId) || !/^AssessmentAck_[0-9a-f]{32}$/i.test(callback)) {
      return javascriptResponse_('void 0');
    }
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(ASSESSMENT_SHEET_NAME);
    let found = false;
    if (sheet && sheet.getLastRow() > 1) {
      found = Boolean(sheet.getRange(2, 1, sheet.getLastRow() - 1, 1)
        .createTextFinder(eventId).matchEntireCell(true).findNext());
    }
    return javascriptResponse_(callback + '(' + JSON.stringify({ok: found}) + ')');
  }
  return jsonResponse_({ok: true, service: 'anonymous-workshop-assessment', data: 'not-readable'});
}

function doPost(event) {
  const lock = LockService.getScriptLock();
  try {
    if (!event || !event.postData || !event.postData.contents) throw new Error('Missing body');
    if (event.postData.contents.length > MAX_PAYLOAD_BYTES) throw new Error('Payload too large');
    const record = JSON.parse(event.postData.contents);
    validateEvent_(record);

    lock.waitLock(10000);
    const sheet = getEventSheet_();
    const lastRow = sheet.getLastRow();
    if (lastRow > 1) {
      const duplicate = sheet.getRange(2, 1, lastRow - 1, 1)
        .createTextFinder(record.id).matchEntireCell(true).findNext();
      if (duplicate) return jsonResponse_({ok: true, duplicate: true});
    }
    const answers = record.payload.answers || {};
    sheet.appendRow([
      record.id,
      new Date().toISOString(),
      record.run_id,
      record.participant_id,
      record.phase,
      record.event_type,
      record.venue,
      record.assessment_version,
      record.workshop_git_sha,
      record.consent_version,
      JSON.stringify(record.payload),
      response_(answers.knowledge, 'forces_match'),
      response_(answers.knowledge, 'generation_order'),
      response_(answers.knowledge, 'inverse_process'),
      response_(answers.knowledge, 'identifiability'),
      response_(answers.knowledge, 'abc_order'),
      response_(answers.knowledge, 'observation_design'),
      response_(answers.confidence, 'confidence_simulator'),
      response_(answers.confidence, 'confidence_inverse'),
      response_(answers.post_evaluation, 'impact_models_data'),
      response_(answers.post_evaluation, 'research_relevance'),
      record.payload.client_duration_ms || ''
    ]);
    return jsonResponse_({ok: true, duplicate: false});
  } catch (error) {
    return jsonResponse_({ok: false, error: String(error.message || error)});
  } finally {
    if (lock.hasLock()) lock.releaseLock();
  }
}

function response_(answers, questionId) {
  if (!Array.isArray(answers)) return '';
  const answer = answers.find(function (item) { return item.question_id === questionId; });
  if (!answer) return '';
  return typeof answer.response === 'object' ? JSON.stringify(answer.response) : answer.response;
}

function getEventSheet_() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = spreadsheet.getSheetByName(ASSESSMENT_SHEET_NAME);
  if (!sheet) sheet = spreadsheet.insertSheet(ASSESSMENT_SHEET_NAME);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADERS);
    sheet.setFrozenRows(1);
    sheet.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
  }
  const actual = sheet.getRange(1, 1, 1, HEADERS.length).getValues()[0];
  if (actual.join('|') !== HEADERS.join('|')) throw new Error('Unexpected sheet headers');
  return sheet;
}

function validateEvent_(record) {
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (!record || typeof record !== 'object' || Array.isArray(record)) throw new Error('Event must be an object');
  if (!uuid.test(record.id) || !uuid.test(record.run_id) || !uuid.test(record.participant_id)) throw new Error('Invalid UUID');
  if (!['pre', 'post'].includes(record.phase)) throw new Error('Invalid phase');
  if (!['started', 'completed'].includes(record.event_type)) throw new Error('Invalid event type');
  if (!['NYU', 'UMN', 'TAU', 'ONLINE'].includes(record.venue)) throw new Error('Invalid venue');
  if (!/^[0-9]+\.[0-9]+\.[0-9]+$/.test(record.assessment_version)) throw new Error('Invalid assessment version');
  if (!(record.workshop_git_sha === 'development' || /^[0-9a-f]{40}$/.test(record.workshop_git_sha))) throw new Error('Invalid workshop SHA');
  if (!/^[0-9]+\.[0-9]+$/.test(record.consent_version)) throw new Error('Invalid consent version');
  if (!record.payload || typeof record.payload !== 'object' || Array.isArray(record.payload)) throw new Error('Invalid payload');
  if (!record.payload.assessment_schema_version || !record.payload.question_bank_hash) throw new Error('Missing version metadata');
  if (record.event_type === 'completed' && !Array.isArray(record.payload.answers && record.payload.answers.knowledge)) throw new Error('Missing raw answers');
  rejectProhibitedKeys_(record);
}

function rejectProhibitedKeys_(value) {
  if (!value || typeof value !== 'object') return;
  Object.keys(value).forEach(function (key) {
    if (PROHIBITED_KEYS.has(key.toLowerCase())) throw new Error('Prohibited field');
    rejectProhibitedKeys_(value[key]);
  });
}

function jsonResponse_(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function javascriptResponse_(source) {
  return ContentService.createTextOutput(source)
    .setMimeType(ContentService.MimeType.JAVASCRIPT);
}

function refreshAssessmentDashboard() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const eventSheet = spreadsheet.getSheetByName(ASSESSMENT_SHEET_NAME);
  if (!eventSheet) {
    SpreadsheetApp.getUi().alert('No assessment_events tab was found yet. Submit one assessment, then refresh.');
    return;
  }

  const summary = summarizeAssessmentForDashboard_(eventSheet);
  let dashboard = spreadsheet.getSheetByName(ASSESSMENT_DASHBOARD_SHEET_NAME);
  if (!dashboard) dashboard = spreadsheet.insertSheet(ASSESSMENT_DASHBOARD_SHEET_NAME, 0);
  dashboard.getCharts().forEach(function (chart) { dashboard.removeChart(chart); });
  dashboard.getDataRange().breakApart();
  dashboard.clear();
  dashboard.setHiddenGridlines(true);
  dashboard.setFrozenRows(2);
  dashboard.setTabColor('#2f8f70');
  renderAssessmentDashboard_(dashboard, summary);
  spreadsheet.setActiveSheet(dashboard);
}

function summarizeAssessmentForDashboard_(sheet) {
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) return emptyDashboardSummary_();
  const headers = values[0].map(String);
  const rows = values.slice(1).map(function (row) {
    const record = {};
    headers.forEach(function (header, index) { record[header] = row[index]; });
    try {
      record.payload = typeof record.payload === 'string' ? JSON.parse(record.payload) : record.payload;
    } catch (error) {
      record.payload = {};
    }
    return record;
  });

  const unique = {};
  rows.forEach(function (record) {
    const id = String(record.id || '');
    if (id && !unique[id]) unique[id] = record;
  });
  const events = Object.keys(unique).map(function (id) { return unique[id]; });
  const startedRuns = {pre: {}, post: {}};
  events.forEach(function (record) {
    if (record.event_type === 'started' && startedRuns[record.phase]) {
      startedRuns[record.phase][String(record.run_id)] = true;
    }
  });

  const scored = {pre: [], post: []};
  let unsupported = 0;
  events.forEach(function (record) {
    if (record.event_type !== 'completed' || !scored[record.phase]) return;
    const version = String(record.assessment_version);
    if (!DASHBOARD_SCORING_RULES[version]) {
      unsupported += 1;
      return;
    }
    const result = scoreDashboardRecord_(record);
    scored[record.phase].push(result);
  });

  const latest = {};
  ['pre', 'post'].forEach(function (phase) {
    scored[phase].forEach(function (entry) {
      const key = dashboardPairingKey_(entry.record);
      if (!latest[key]) latest[key] = {};
      const prior = latest[key][phase];
      if (!prior || dashboardTime_(entry.record.received_at) > dashboardTime_(prior.record.received_at)) {
        latest[key][phase] = entry;
      }
    });
  });
  const matched = Object.keys(latest).map(function (key) { return latest[key]; })
    .filter(function (pair) { return pair.pre && pair.post; });

  const itemIds = Object.keys(DASHBOARD_ITEM_LABELS);
  const itemRows = itemIds.map(function (id) {
    const pre = dashboardCorrectPercent_(scored.pre, id);
    const post = dashboardCorrectPercent_(scored.post, id);
    return {
      id: id,
      label: DASHBOARD_ITEM_LABELS[id],
      pre: pre,
      post: post,
      change: pre === null || post === null ? null : post - pre
    };
  });

  const confidenceRows = Object.keys(DASHBOARD_CONFIDENCE_LABELS).map(function (id) {
    return {
      label: DASHBOARD_CONFIDENCE_LABELS[id],
      pre: dashboardAnswerMean_(scored.pre, 'confidence', id),
      post: dashboardAnswerMean_(scored.post, 'confidence', id)
    };
  });

  const venues = ['NYU', 'UMN', 'TAU', 'ONLINE'].map(function (venue) {
    const pre = scored.pre.filter(function (entry) { return entry.record.venue === venue; });
    const post = scored.post.filter(function (entry) { return entry.record.venue === venue; });
    return {venue: venue, preN: pre.length, postN: post.length, postMean: dashboardMean_(post.map(function (entry) { return entry.score; }))};
  });

  const changes = matched.map(function (pair) { return pair.post.score - pair.pre.score; });
  const durations = {
    pre: dashboardMedian_(scored.pre.map(dashboardDurationMinutes_).filter(dashboardIsNumber_)),
    post: dashboardMedian_(scored.post.map(dashboardDurationMinutes_).filter(dashboardIsNumber_))
  };
  const scoreDistribution = [];
  for (let score = 0; score <= 6; score += 1) {
    scoreDistribution.push([
      score,
      scored.pre.filter(function (entry) { return entry.score === score; }).length,
      scored.post.filter(function (entry) { return entry.score === score; }).length
    ]);
  }

  return {
    preN: scored.pre.length,
    postN: scored.post.length,
    matchedN: matched.length,
    startedN: {pre: Object.keys(startedRuns.pre).length, post: Object.keys(startedRuns.post).length},
    preMean: dashboardMean_(scored.pre.map(function (entry) { return entry.score; })),
    postMean: dashboardMean_(scored.post.map(function (entry) { return entry.score; })),
    pairedChange: dashboardMean_(changes),
    durationMinutes: durations,
    itemRows: itemRows,
    confidenceRows: confidenceRows,
    venues: venues,
    impactMean: dashboardAnswerMean_(scored.post, 'post_evaluation', 'impact_models_data'),
    relevanceMean: dashboardAnswerMean_(scored.post, 'post_evaluation', 'research_relevance'),
    scoreDistribution: scoreDistribution,
    unsupported: unsupported,
    insights: dashboardInsights_(itemRows, matched.length, dashboardMean_(changes), scored.post.length)
  };
}

function scoreDashboardRecord_(record) {
  const rules = DASHBOARD_SCORING_RULES[String(record.assessment_version)];
  const answers = dashboardAnswerMap_(record.payload, 'knowledge');
  const itemScores = {};
  Object.keys(DASHBOARD_ITEM_LABELS).forEach(function (id) {
    let expected = rules[id];
    if (id === 'observation_design') expected = expected[record.phase];
    itemScores[id] = dashboardCanonical_(answers[id]) === dashboardCanonical_(expected);
  });
  return {
    record: record,
    score: Object.keys(itemScores).filter(function (id) { return itemScores[id]; }).length,
    itemScores: itemScores
  };
}

function dashboardAnswerMap_(payload, family) {
  const output = {};
  const answers = payload && payload.answers && payload.answers[family];
  if (!Array.isArray(answers)) return output;
  answers.forEach(function (answer) { output[answer.question_id] = answer.response; });
  return output;
}

function dashboardCanonical_(value) {
  if (Array.isArray(value)) return '[' + value.map(dashboardCanonical_).join(',') + ']';
  if (value && typeof value === 'object') {
    return '{' + Object.keys(value).sort().map(function (key) {
      return JSON.stringify(key) + ':' + dashboardCanonical_(value[key]);
    }).join(',') + '}';
  }
  return JSON.stringify(value);
}

function dashboardPairingKey_(record) {
  const code = record.payload && record.payload.pairing_code;
  return code ? 'code:' + String(code).toUpperCase() : 'participant:' + String(record.participant_id);
}

function dashboardTime_(value) {
  const time = value instanceof Date ? value.getTime() : new Date(value).getTime();
  return isNaN(time) ? 0 : time;
}

function dashboardCorrectPercent_(entries, id) {
  if (!entries.length) return null;
  const correct = entries.filter(function (entry) { return entry.itemScores[id]; }).length;
  return 100 * correct / entries.length;
}

function dashboardAnswerMean_(entries, family, id) {
  const values = entries.map(function (entry) {
    return dashboardAnswerMap_(entry.record.payload, family)[id];
  }).map(Number).filter(dashboardIsNumber_);
  return dashboardMean_(values);
}

function dashboardDurationMinutes_(entry) {
  const milliseconds = Number(entry.record.payload && entry.record.payload.client_duration_ms);
  return milliseconds > 0 ? milliseconds / 60000 : null;
}

function dashboardIsNumber_(value) {
  return typeof value === 'number' && isFinite(value);
}

function dashboardMean_(values) {
  if (!values.length) return null;
  return values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
}

function dashboardMedian_(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort(function (a, b) { return a - b; });
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function dashboardInsights_(items, matchedN, pairedChange, postN) {
  const lines = [];
  if (matchedN) lines.push('Matched learning change: ' + dashboardSigned_(pairedChange, 2) + ' concepts out of 6 across ' + matchedN + ' paired participant' + (matchedN === 1 ? '' : 's') + '.');
  else lines.push('No paired pre/post responses yet; aggregate pre and post summaries are still shown.');
  const comparable = items.filter(function (item) { return item.change !== null; });
  if (comparable.length) {
    const largestGain = comparable.slice().sort(function (a, b) { return b.change - a.change; })[0];
    lines.push('Largest item-level change: ' + largestGain.label + ' (' + dashboardSigned_(largestGain.change, 1) + ' percentage points).');
  }
  const withPost = items.filter(function (item) { return item.post !== null; });
  if (withPost.length) {
    const lowest = withPost.slice().sort(function (a, b) { return a.post - b.post; })[0];
    lines.push('Lowest post-workshop correct percentage: ' + lowest.label + ' (' + lowest.post.toFixed(1) + '%).');
  }
  if (postN < 10) lines.push('Early signal only: with fewer than 10 post responses, one answer can move percentages substantially.');
  return lines;
}

function dashboardSigned_(value, digits) {
  if (value === null || !dashboardIsNumber_(value)) return '—';
  return (value > 0 ? '+' : '') + value.toFixed(digits);
}

function emptyDashboardSummary_() {
  return {
    preN: 0, postN: 0, matchedN: 0, startedN: {pre: 0, post: 0},
    preMean: null, postMean: null, pairedChange: null,
    durationMinutes: {pre: null, post: null},
    itemRows: Object.keys(DASHBOARD_ITEM_LABELS).map(function (id) {
      return {id: id, label: DASHBOARD_ITEM_LABELS[id], pre: null, post: null, change: null};
    }),
    confidenceRows: Object.keys(DASHBOARD_CONFIDENCE_LABELS).map(function (id) {
      return {label: DASHBOARD_CONFIDENCE_LABELS[id], pre: null, post: null};
    }),
    venues: ['NYU', 'UMN', 'TAU', 'ONLINE'].map(function (venue) { return {venue: venue, preN: 0, postN: 0, postMean: null}; }),
    impactMean: null, relevanceMean: null,
    scoreDistribution: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]],
    unsupported: 0,
    insights: ['No completed assessments yet. The dashboard will populate after the first completed response.']
  };
}

function renderAssessmentDashboard_(sheet, summary) {
  const green = '#2f8f70';
  const orange = '#ef9b55';
  const ink = '#17352f';
  const pale = '#eef7f3';
  const warm = '#fff5ea';
  sheet.setColumnWidth(1, 240);
  [2, 3, 4].forEach(function (column) { sheet.setColumnWidth(column, 105); });
  sheet.setColumnWidth(5, 28);
  sheet.setColumnWidth(6, 210);
  [7, 8].forEach(function (column) { sheet.setColumnWidth(column, 115); });

  sheet.getRange('A1:H1').merge().setValue('Workshop learning dashboard')
    .setFontSize(22).setFontWeight('bold').setFontColor('#ffffff').setBackground(ink)
    .setHorizontalAlignment('left').setVerticalAlignment('middle');
  sheet.setRowHeight(1, 48);
  const versionNote = summary.unsupported ? ' · ' + summary.unsupported + ' unsupported-version response(s) excluded' : '';
  sheet.getRange('A2:H2').merge().setValue('Aggregate view · completed assessments only · refreshed ' + new Date().toISOString() + versionNote)
    .setFontColor('#5f716d').setFontSize(10).setBackground('#f7faf9');

  dashboardMetric_(sheet, 'A4:B4', 'A5:B6', 'PRE COMPLETED', summary.preN, pale, ink);
  dashboardMetric_(sheet, 'C4:D4', 'C5:D6', 'POST COMPLETED', summary.postN, warm, ink);
  dashboardMetric_(sheet, 'E4:F4', 'E5:F6', 'PAIRED', summary.matchedN, '#edf3fb', ink);
  dashboardMetric_(sheet, 'G4:H4', 'G5:H6', 'PAIRED CHANGE', dashboardSigned_(summary.pairedChange, 2), '#f1edfb', ink);

  sheet.getRange('A8:H8').merge().setValue('DESCRIPTIVE SIGNALS').setFontWeight('bold').setFontColor(green);
  sheet.getRange('A9:H12').merge().setValue(summary.insights.join('\n'))
    .setWrap(true).setVerticalAlignment('top').setFontColor(ink).setBackground('#f7faf9');
  sheet.setRowHeights(9, 4, 23);

  dashboardSectionTitle_(sheet, 'A14:D14', 'Knowledge by concept', green);
  sheet.getRange('A15:D15').setValues([['Construct', 'Pre correct', 'Post correct', 'Change']]);
  const itemValues = summary.itemRows.map(function (item) { return [item.label, item.pre, item.post, item.change]; });
  sheet.getRange(16, 1, itemValues.length, 4).setValues(itemValues);
  sheet.getRange(16, 2, itemValues.length, 3).setNumberFormat('0.0"%"');
  dashboardTable_(sheet, 15, 1, itemValues.length + 1, 4, ink, pale);

  dashboardSectionTitle_(sheet, 'F14:H14', 'Response overview', orange);
  sheet.getRange('F15:H15').setValues([['Measure', 'Pre', 'Post']]);
  sheet.getRange('F16:H19').setValues([
    ['Mean knowledge / 6', summary.preMean, summary.postMean],
    ['Median duration (min)', summary.durationMinutes.pre, summary.durationMinutes.post],
    ['Started', summary.startedN.pre, summary.startedN.post],
    ['Completion rate', dashboardRate_(summary.preN, summary.startedN.pre), dashboardRate_(summary.postN, summary.startedN.post)]
  ]);
  sheet.getRange('G16:H17').setNumberFormat('0.00');
  sheet.getRange('G19:H19').setNumberFormat('0.0"%"');
  dashboardTable_(sheet, 15, 6, 5, 3, ink, warm);

  dashboardSectionTitle_(sheet, 'A24:D24', 'Confidence', green);
  sheet.getRange('A25:C25').setValues([['Statement', 'Pre mean', 'Post mean']]);
  const confidenceValues = summary.confidenceRows.map(function (item) { return [item.label, item.pre, item.post]; });
  sheet.getRange(26, 1, confidenceValues.length, 3).setValues(confidenceValues);
  sheet.getRange(26, 2, confidenceValues.length, 2).setNumberFormat('0.00');
  dashboardTable_(sheet, 25, 1, confidenceValues.length + 1, 3, ink, pale);

  dashboardSectionTitle_(sheet, 'F24:H24', 'Post-workshop reflections', orange);
  sheet.getRange('F25:G25').setValues([['Statement', 'Mean / 5']]);
  sheet.getRange('F26:G27').setValues([
    ['Changed how I connect models and data', summary.impactMean],
    ['Can imagine using an idea in my research', summary.relevanceMean]
  ]);
  sheet.getRange('G26:G27').setNumberFormat('0.00');
  dashboardTable_(sheet, 25, 6, 3, 2, ink, warm);

  dashboardSectionTitle_(sheet, 'A30:D30', 'Completed responses by venue', green);
  sheet.getRange('A31:D31').setValues([['Venue', 'Pre n', 'Post n', 'Post mean / 6']]);
  const venueValues = summary.venues.map(function (row) { return [row.venue, row.preN, row.postN, row.postMean]; });
  sheet.getRange(32, 1, venueValues.length, 4).setValues(venueValues);
  sheet.getRange(32, 4, venueValues.length, 1).setNumberFormat('0.00');
  dashboardTable_(sheet, 31, 1, venueValues.length + 1, 4, ink, pale);

  sheet.getRange('J1:L1').setValues([['Score', 'Pre', 'Post']]);
  sheet.getRange(2, 10, summary.scoreDistribution.length, 3).setValues(summary.scoreDistribution);

  const itemChart = sheet.newChart().asBarChart()
    .addRange(sheet.getRange(15, 1, summary.itemRows.length + 1, 3))
    .setPosition(38, 1, 0, 0)
    .setOption('title', 'Correct responses by concept')
    .setOption('legend', {position: 'top'})
    .setOption('colors', [green, orange])
    .setOption('hAxis', {title: 'Correct (%)', viewWindow: {min: 0, max: 100}})
    .setOption('width', 720).setOption('height', 350).build();
  sheet.insertChart(itemChart);

  const confidenceChart = sheet.newChart().asColumnChart()
    .addRange(sheet.getRange(25, 1, summary.confidenceRows.length + 1, 3))
    .setPosition(38, 7, 0, 0)
    .setOption('title', 'Confidence before and after')
    .setOption('legend', {position: 'top'})
    .setOption('colors', [green, orange])
    .setOption('vAxis', {title: 'Mean response', viewWindow: {min: 1, max: 5}})
    .setOption('width', 540).setOption('height', 350).build();
  sheet.insertChart(confidenceChart);

  const distributionChart = sheet.newChart().asColumnChart()
    .addRange(sheet.getRange(1, 10, summary.scoreDistribution.length + 1, 3))
    .setPosition(56, 1, 0, 0)
    .setOption('title', 'Knowledge-score distribution')
    .setOption('legend', {position: 'top'})
    .setOption('colors', [green, orange])
    .setOption('hAxis', {title: 'Concepts correct (0–6)'})
    .setOption('vAxis', {title: 'Participants', minValue: 0})
    .setOption('width', 720).setOption('height', 330).build();
  sheet.insertChart(distributionChart);
  sheet.hideColumns(10, 3);

  sheet.getRange('A73:H74').merge().setValue('Descriptive workshop-evaluation summaries only. Interpret small samples cautiously. The private raw event tab remains the source of truth; this dashboard contains no participant-level rows.')
    .setWrap(true).setFontSize(9).setFontColor('#6b7b77').setBackground('#f7faf9');
}

function dashboardMetric_(sheet, labelRange, valueRange, label, value, background, ink) {
  sheet.getRange(labelRange).merge().setValue(label).setFontSize(9).setFontWeight('bold')
    .setFontColor('#60716d').setBackground(background).setHorizontalAlignment('center');
  sheet.getRange(valueRange).merge().setValue(value === null ? '—' : value).setFontSize(22)
    .setFontWeight('bold').setFontColor(ink).setBackground(background).setHorizontalAlignment('center').setVerticalAlignment('middle');
}

function dashboardSectionTitle_(sheet, range, title, color) {
  sheet.getRange(range).merge().setValue(title).setFontSize(13).setFontWeight('bold').setFontColor(color);
}

function dashboardTable_(sheet, row, column, rowCount, columnCount, ink, headerBackground) {
  const range = sheet.getRange(row, column, rowCount, columnCount);
  range.setFontColor(ink).setVerticalAlignment('middle').setWrap(true)
    .setBorder(true, true, true, true, true, true, '#dce6e2', SpreadsheetApp.BorderStyle.SOLID);
  sheet.getRange(row, column, 1, columnCount).setFontWeight('bold').setBackground(headerBackground);
}

function dashboardRate_(completed, started) {
  return started ? 100 * completed / started : null;
}
