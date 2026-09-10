(function () {
  "use strict";

  const config = window.ASSESSMENT_CONFIG;

  function readQueue() {
    try {
      const value = JSON.parse(localStorage.getItem(config.queueKey) || "[]");
      return Array.isArray(value) ? value : [];
    } catch (_) {
      return [];
    }
  }

  function writeQueue(queue) {
    localStorage.setItem(config.queueKey, JSON.stringify(queue));
  }

  function configured() {
    return Boolean(config.supabaseUrl && config.supabaseAnonKey);
  }

  async function transmit(event) {
    if (!configured()) throw new Error("Assessment storage is not configured");
    const endpoint = config.supabaseUrl.replace(/\/$/, "") + "/rest/v1/assessment_events";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        apikey: config.supabaseAnonKey,
        Authorization: "Bearer " + config.supabaseAnonKey,
        "Content-Type": "application/json",
        Prefer: "return=minimal"
      },
      body: JSON.stringify(event),
      keepalive: true
    });
    if (!response.ok && response.status !== 409) {
      throw new Error("Assessment storage returned " + response.status);
    }
  }

  async function flush() {
    const queue = readQueue();
    if (!queue.length) return {sent: 0, pending: 0, configured: configured()};
    if (!configured() || !navigator.onLine) return {sent: 0, pending: queue.length, configured: configured()};
    const remaining = [];
    let sent = 0;
    for (const event of queue) {
      try {
        await transmit(event);
        sent += 1;
      } catch (_) {
        remaining.push(event);
      }
    }
    writeQueue(remaining);
    return {sent: sent, pending: remaining.length, configured: true};
  }

  async function enqueue(event) {
    const queue = readQueue();
    if (!queue.some((item) => item.id === event.id)) queue.push(event);
    writeQueue(queue);
    return flush();
  }

  window.AssessmentBackend = Object.freeze({enqueue: enqueue, flush: flush, configured: configured});
  window.addEventListener("online", () => flush());
})();
