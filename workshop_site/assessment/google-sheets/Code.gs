/*
 * Private Google Sheet receiver for the anonymous workshop assessment.
 * Bind this script to a private Sheet and deploy it as a Web app that executes
 * as you. Anonymous visitors can append validated events; doGet never exposes data.
 */

const ASSESSMENT_SHEET_NAME = 'assessment_events';
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
