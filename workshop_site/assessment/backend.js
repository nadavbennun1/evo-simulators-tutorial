(function () {
  "use strict";

  const config = window.ASSESSMENT_CONFIG;
  let activeFlush = null;

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
    const localPreview = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
    return !localPreview && /^https:\/\/script\.google\.com\/macros\/s\/.+\/exec$/.test(config.googleSheetsEndpoint || "");
  }

  async function transmit(event) {
    if (!configured()) throw new Error("Assessment storage is not configured");
    await fetch(config.googleSheetsEndpoint, {
      method: "POST",
      mode: "no-cors",
      headers: {"Content-Type": "text/plain;charset=utf-8"},
      body: JSON.stringify(event),
      keepalive: true
    });
    if (!await acknowledged(event.id)) throw new Error("Assessment storage did not acknowledge the event");
  }

  function acknowledged(eventId) {
    return new Promise((resolve) => {
      const callback = "AssessmentAck_" + eventId.replace(/-/g, "");
      const script = document.createElement("script");
      let settled = false;
      const timeout = setTimeout(() => finish(false), 4000);
      function finish(ok) {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        delete window[callback];
        script.remove();
        resolve(ok);
      }
      window[callback] = (result) => finish(Boolean(result && result.ok));
      script.onerror = () => finish(false);
      script.src = config.googleSheetsEndpoint + "?event_id=" + encodeURIComponent(eventId) + "&callback=" + callback;
      document.head.appendChild(script);
    });
  }

  async function performFlush() {
    if (!configured() || !navigator.onLine) return {sent: 0, pending: readQueue().length, configured: configured()};
    let sent = 0;
    while (readQueue().length) {
      const event = readQueue()[0];
      try {
        await transmit(event);
        sent += 1;
        writeQueue(readQueue().filter((item) => item.id !== event.id));
      } catch (_) {
        break;
      }
    }
    return {sent: sent, pending: readQueue().length, configured: true};
  }

  function flush() {
    if (activeFlush) return activeFlush;
    activeFlush = performFlush().finally(() => {activeFlush = null;});
    return activeFlush;
  }

  function queue(event) {
    const queue = readQueue();
    if (!queue.some((item) => item.id === event.id)) queue.push(event);
    writeQueue(queue);
  }

  async function enqueue(event) {
    queue(event);
    return flush();
  }

  window.AssessmentBackend = Object.freeze({queue: queue, enqueue: enqueue, flush: flush, configured: configured});
  window.addEventListener("online", () => flush());
})();
