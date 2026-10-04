/* CricketMind chat page: DOM glue around /api/chat/stream.
 * Pure helpers live in render.js (safe markdown) and sse.js (stream parsing). */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  const memory = {
    get(k) { try { return sessionStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { if (v == null) sessionStorage.removeItem(k); else sessionStorage.setItem(k, v); } catch (_) { /* private mode */ } },
  };

  const LABELS = {
    find_player: "Looking up a player", find_team: "Looking up a team",
    player_profile: "Reading a player profile", rank_batters: "Ranking batters",
    rank_bowlers: "Ranking bowlers", matchup: "Working out a matchup",
    recommend_bowlers: "Comparing bowlers", best_pairs: "Comparing batting pairs",
    head_to_head: "Checking head-to-head history", fielding_profile: "Reading fielding records",
    predict_match: "Running the match model", batting_stats: "Fetching batting stats",
    bowling_stats: "Fetching bowling stats", leaderboard: "Building a leaderboard",
    team_record: "Reading a team record", standings: "Reading the standings",
    venue_stats: "Checking the ground", search_matches: "Searching matches",
    search_facts: "Searching stored records",
  };
  const label = (name) => LABELS[name] || "Using " + name;

  const state = {
    sessionId: memory.get("cm.session"), token: memory.get("cm.token"),
    busy: false, info: { max_message_chars: 1000, examples: [], auth_required: false }, last: "",
  };
  const log = $("#log"), input = $("#input"), sendBtn = $("#send"), count = $("#count");
  const status = $("#status"), welcome = $("#welcome"), gate = $("#token-gate");

  // ── status pill ───────────────────────────────────────────────────
  let healthTimer = null;
  async function refreshHealth() {
    try {
      const h = await (await fetch("/api/health")).json();
      if (h.warmup === "running") {
        status.className = "pill wait"; status.textContent = "Loading data…";
        healthTimer = setTimeout(refreshHealth, 3000);
        return;
      }
      let missing = [];
      if (h.warmup && typeof h.warmup === "object") {
        missing = Object.entries(h.warmup).filter(([, v]) => v !== true).map(([k]) => k);
      }
      if (missing.length) {
        status.className = "pill wait"; status.textContent = "Ready (limited)";
        status.title = "Not available: " + missing.join(", ");
      } else {
        status.className = "pill"; status.textContent = "Ready"; status.title = "";
      }
    } catch (_) {
      status.className = "pill bad"; status.textContent = "Offline";
    }
  }

  // ── helpers for the transcript ────────────────────────────────────
  function scrollDown() {
    const last = log.lastElementChild;
    if (last) last.scrollIntoView({ block: "end" });
  }
  function addUser(text) {
    welcome.hidden = true;
    const li = el("li", "msg user");
    li.appendChild(el("div", "bubble", text));
    log.appendChild(li);
    scrollDown();
  }
  function addAssistant() {
    const li = el("li", "msg assistant");
    const bubble = el("div", "bubble");
    const progress = el("div", "progress");
    const now = el("div", "now");
    now.appendChild(el("span", "spinner"));
    const nowText = el("span", null, "Thinking…");
    now.appendChild(nowText);
    const done = el("ul");
    progress.appendChild(done);
    progress.appendChild(now);
    bubble.appendChild(progress);
    li.appendChild(bubble);
    log.appendChild(li);
    scrollDown();
    return { li, bubble, progress, now, nowText, done };
  }
  function notice(ui, kind, text, action) {
    ui.bubble.replaceChildren();
    const n = el("div", "notice " + kind, text);
    if (action) {
      const b = el("button", null, action.label);
      b.type = "button";
      b.addEventListener("click", action.run);
      n.appendChild(b);
    }
    ui.bubble.appendChild(n);
    scrollDown();
  }
  function showAnswer(ui, a) {
    ui.bubble.replaceChildren();
    const body = el("div", "answer");
    body.innerHTML = window.CricketRender.renderMarkdown(a.answer); // safe: escape-first renderer
    ui.bubble.appendChild(body);
    const calls = a.tool_calls || [];
    if (calls.length) {
      const d = el("details", "trace");
      d.appendChild(el("summary", null, "How I got this (" + calls.length + " tool call" + (calls.length === 1 ? "" : "s") + ")"));
      const ul = el("ul");
      for (const c of calls) {
        const li = el("li");
        li.appendChild(el("code", null, c.name + "(" + JSON.stringify(c.arguments) + ")"));
        if (!c.ok) li.appendChild(el("span", "bad", " — returned an error"));
        ul.appendChild(li);
      }
      d.appendChild(ul);
      ui.bubble.appendChild(d);
    }
    const u = a.usage || {};
    ui.bubble.appendChild(el("div", "meta",
      calls.length + " tool call" + (calls.length === 1 ? "" : "s") + " · " +
      (u.output_tokens || 0).toLocaleString() + " tokens written"));
    scrollDown();
  }

  // ── sending ───────────────────────────────────────────────────────
  function setBusy(b) {
    state.busy = b;
    sendBtn.disabled = b; input.disabled = b;
    sendBtn.textContent = b ? "Working…" : "Ask";
    if (!b) input.focus();
  }
  function headers() {
    const h = { "Content-Type": "application/json" };
    if (state.token) h["Authorization"] = "Bearer " + state.token;
    return h;
  }
  async function errorMessage(res) {
    let detail = null;
    try { detail = (await res.json()).detail; } catch (_) { /* not JSON */ }
    const text = detail && typeof detail === "object" ? detail.message : detail;
    return { text: text || "The server returned an error (" + res.status + ").", code: detail && detail.code };
  }

  async function send(text) {
    text = text.trim();
    if (!text || state.busy) return;
    if (text.length > state.info.max_message_chars) {
      count.textContent = "Too long: " + text.length + " / " + state.info.max_message_chars;
      count.className = "count near";
      return;
    }
    state.last = text;
    addUser(text);
    input.value = ""; resize(); updateCount();
    const ui = addAssistant();
    setBusy(true);
    const retry = { label: "Try again", run: () => { ui.li.remove(); log.lastElementChild && log.lastElementChild.remove(); send(text); } };
    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ message: text, session_id: state.sessionId }),
      });
      if (!res.ok) {
        const err = await errorMessage(res);
        if (res.status === 401) {
          gate.hidden = false; $("#token").focus();
          notice(ui, "warn", "This server needs an access token. Enter it below, then ask again.");
        } else if (res.status === 429) {
          const wait = res.headers.get("Retry-After");
          notice(ui, "warn", "Too many requests." + (wait ? " Please wait " + wait + " seconds." : ""), retry);
        } else if (err.code === "session_full") {
          notice(ui, "warn", err.text, { label: "Start a new chat", run: newChat });
        } else {
          notice(ui, "err", err.text, res.status >= 500 ? retry : null);
        }
        return;
      }
      await readStream(res, ui, retry);
    } catch (e) {
      notice(ui, "err", "Could not reach the server. Check your connection.", retry);
    } finally {
      setBusy(false);
    }
  }

  async function readStream(res, ui, retry) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", finished = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = window.CricketSSE.parseSSE(buffer);
      buffer = parsed.rest;
      for (const ev of parsed.events) {
        let data = {};
        try { data = JSON.parse(ev.data); } catch (_) { continue; }
        if (ev.event === "session") {
          state.sessionId = data.session_id; memory.set("cm.session", data.session_id);
        } else if (ev.event === "step") {
          if (data.n > 1) ui.nowText.textContent = "Thinking…";
        } else if (ev.event === "tool_call") {
          ui.nowText.textContent = label(data.name) + "…";
        } else if (ev.event === "tool_result") {
          const li = el("li", data.ok ? "" : "failed", label(data.name));
          ui.done.appendChild(li);
          ui.nowText.textContent = "Thinking…";
          scrollDown();
        } else if (ev.event === "answer") {
          finished = true; showAnswer(ui, data);
        } else if (ev.event === "error") {
          finished = true;
          notice(ui, "err", data.message || "Something went wrong.", data.status >= 500 ? retry : null);
        }
      }
    }
    if (!finished) notice(ui, "err", "The connection ended before an answer arrived.", retry);
  }

  async function newChat() {
    const id = state.sessionId;
    state.sessionId = null; memory.set("cm.session", null);
    log.replaceChildren(); welcome.hidden = false;
    if (id) {
      try { await fetch("/api/chat/" + encodeURIComponent(id), { method: "DELETE", headers: headers() }); } catch (_) { /* best effort */ }
    }
    input.focus();
  }

  // ── composer behaviour ────────────────────────────────────────────
  function resize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }
  function updateCount() {
    const max = state.info.max_message_chars, n = input.value.length;
    if (n >= max * 0.8) {
      count.textContent = n + " / " + max;
      count.className = "count" + (n > max ? " near" : "");
    } else count.textContent = "";
  }
  input.addEventListener("input", () => { resize(); updateCount(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); }
  });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  $("#new-chat").addEventListener("click", newChat);
  gate.addEventListener("submit", (e) => {
    e.preventDefault();
    state.token = $("#token").value.trim() || null;
    memory.set("cm.token", state.token);
    gate.hidden = true; input.focus();
  });

  // ── start-up ──────────────────────────────────────────────────────
  async function init() {
    refreshHealth();
    try {
      state.info = await (await fetch("/api/info")).json();
    } catch (_) { /* keep defaults; the status pill already shows Offline */ }
    input.maxLength = state.info.max_message_chars * 2; // allow typing past it to see the counter
    const chips = $("#examples");
    for (const q of state.info.examples || []) {
      const b = el("button", "chip", q);
      b.type = "button";
      b.addEventListener("click", () => send(q));
      chips.appendChild(b);
    }
    if (state.info.auth_required && !state.token) gate.hidden = false;
    input.focus();
  }
  init();
})();
