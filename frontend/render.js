/* A small, safe markdown renderer for assistant answers.
 *
 * Safety by construction: every character of the input is HTML-escaped first, and the renderer only
 * ever *adds* a fixed set of tags (p, br, strong, em, code, pre, h3-h6, ul, ol, li, table parts, hr).
 * It never creates links, images or attributes from the text, so there is nothing to inject into.
 * Supported: paragraphs, **bold**, *italic*, `code`, fenced code, # headings, - / * / 1. lists,
 * pipe tables, --- rules.  Anything else is shown as plain (escaped) text.
 */
(function (root) {
  "use strict";

  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ESC[c]);
  }

  // inline: operates on already-escaped text
  function inline(escaped) {
    const codes = [];
    let s = escaped.replace(/`([^`\n]+)`/g, (_, c) => {
      codes.push("<code>" + c + "</code>");
      return "\u0000" + (codes.length - 1) + "\u0000";
    });
    s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[\s(])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => codes[+i]);
  }

  const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l) && l.includes("-");
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  function renderMarkdown(text) {
    const lines = String(text == null ? "" : text).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (/^\s*$/.test(line)) { i++; continue; }

      // fenced code block
      if (/^\s*```/.test(line)) {
        const code = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        i++; // closing fence (or end of text)
        out.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>");
        continue;
      }

      // horizontal rule
      if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) { out.push("<hr>"); i++; continue; }

      // heading
      const h = /^\s{0,3}(#{1,4})\s+(.*?)\s*#*\s*$/.exec(line);
      if (h) {
        const level = h[1].length + 2; // # -> h3 ... #### -> h6 (the page title is the only h1)
        out.push("<h" + level + ">" + inline(escapeHtml(h[2])) + "</h" + level + ">");
        i++;
        continue;
      }

      // table
      if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const body = [];
        while (i < lines.length && isTableRow(lines[i])) body.push(cells(lines[i++]));
        let t = "<table><thead><tr>" + head.map((c) => "<th>" + inline(escapeHtml(c)) + "</th>").join("") + "</tr></thead><tbody>";
        for (const r of body) t += "<tr>" + r.map((c) => "<td>" + inline(escapeHtml(c)) + "</td>").join("") + "</tr>";
        out.push(t + "</tbody></table>");
        continue;
      }

      // lists (flat; indentation is kept as a class so nesting still reads sensibly)
      const li = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/.exec(line);
      if (li) {
        const ordered = /\d/.test(li[2]);
        const items = [];
        while (i < lines.length) {
          const m = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/.exec(lines[i]);
          if (!m || /\d/.test(m[2]) !== ordered) break;
          const nested = m[1].length >= 2 ? ' class="nested"' : "";
          items.push("<li" + nested + ">" + inline(escapeHtml(m[3])) + "</li>");
          i++;
        }
        const tag = ordered ? "ol" : "ul";
        out.push("<" + tag + ">" + items.join("") + "</" + tag + ">");
        continue;
      }

      // paragraph: consecutive plain lines, joined with line breaks
      const para = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^\s*```/.test(lines[i]) &&
             !/^\s{0,3}#{1,4}\s/.test(lines[i]) && !/^(\s*)([-*•]|\d+[.)])\s+/.test(lines[i]) &&
             !(isTableRow(lines[i]) && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
        para.push(inline(escapeHtml(lines[i].trim())));
        i++;
      }
      if (para.length) out.push("<p>" + para.join("<br>") + "</p>");
      else i++; // safety: never loop forever
    }
    return out.join("");
  }

  const api = { renderMarkdown, escapeHtml };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CricketRender = api;
})(typeof window !== "undefined" ? window : this);
