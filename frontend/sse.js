/* Incremental parser for a server-sent-events stream read with fetch().
 *
 *   const { events, rest } = parseSSE(buffer + decodedChunk);  buffer = rest;
 *
 * Returns only *complete* events (those followed by a blank line); the unfinished tail is returned
 * as ``rest`` to be prepended to the next chunk.  Comment lines (": keep-alive") are ignored, and
 * data lines of one event are joined with newlines, as the SSE spec says.
 */
(function (root) {
  "use strict";

  function parseSSE(buffer) {
    const text = String(buffer).replace(/\r\n?/g, "\n");
    const blocks = text.split("\n\n");
    const rest = blocks.pop(); // the last piece has no terminating blank line yet
    const events = [];
    for (const block of blocks) {
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line === "" || line.startsWith(":")) continue;
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (data.length) events.push({ event: event, data: data.join("\n") });
    }
    return { events: events, rest: rest };
  }

  const api = { parseSSE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CricketSSE = api;
})(typeof window !== "undefined" ? window : this);
