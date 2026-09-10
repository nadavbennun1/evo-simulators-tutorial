(function () {
  "use strict";
  const CONSENT_VERSION = "1.0";

  window.ASSESSMENT_CONFIG = Object.freeze({
    assessmentSchemaVersion: "1.0",
    assessmentVersion: "1.0.0",
    consentVersion: CONSENT_VERSION,
    questionBankPath: "questions/v1.0.0.json",
    buildMetaPath: "build-meta.json",
    supabaseUrl: "",
    supabaseAnonKey: "",
    participantIdKey: "evoSbiWorkshopParticipantId",
    pairingCodeKey: "evoSbiWorkshopPairingCode",
    queueKey: "evoSbiWorkshopAssessmentQueueV1",
    draftKeyPrefix: "evoSbiWorkshopAssessmentDraftV1",
    chapterOneUrl: "../evolution.html",
    workshopHomeUrl: "../index.html",
    consent: Object.freeze({
      heading: "Anonymous workshop evaluation",
      body: "This optional check helps us evaluate and improve the workshop. The study dataset contains your answers, workshop location, timestamps, questionnaire/workshop version, and a randomly generated anonymous code used to pair the beginning and end of the workshop.",
      exclusions: "We do not ask for names, email addresses, demographic information, precise location, or other identifying information."
    })
  });
})();
