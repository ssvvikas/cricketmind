"""The page's JavaScript (safe markdown renderer, SSE parser), run under Node."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


def node(script: str, payload):
    """Run `script` under Node with `payload` (JSON) available as `input`; return its JSON output."""
    prelude = (f"const R = require({json.dumps(str(FRONTEND / 'render.js'))});"
               f"const S = require({json.dumps(str(FRONTEND / 'sse.js'))});"
               "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));")
    out = subprocess.run(["node", "-e", prelude + script], input=json.dumps(payload), text=True,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr[-500:]
    return json.loads(out.stdout)


def render(*texts):
    return node("console.log(JSON.stringify(input.map(R.renderMarkdown)))", list(texts))


ALLOWED_TAG = re.compile(r"</?(p|br|strong|em|code|pre|h[3-6]|ul|ol|li|table|thead|tbody|tr|th|td|hr)"
                         r'(\sclass="nested")?/?>')


class TestMarkdown:
    def test_inline_formatting(self):
        assert render("**a** *b* `c`") == ["<p><strong>a</strong> <em>b</em> <code>c</code></p>"]

    def test_headings_start_at_h3(self):
        assert render("# T", "## U", "#### V") == ["<h3>T</h3>", "<h4>U</h4>", "<h6>V</h6>"]

    def test_lists(self):
        assert render("- a\n- b") == ["<ul><li>a</li><li>b</li></ul>"]
        assert render("1. a\n2. b") == ["<ol><li>a</li><li>b</li></ol>"]
        assert render("- a\n  - b") == ['<ul><li>a</li><li class="nested">b</li></ul>']
        assert render("- a\n1. b") == ["<ul><li>a</li></ul><ol><li>b</li></ol>"]

    def test_tables(self):
        assert render("| x | y |\n|---|---|\n| 1 | **2** |") == [
            "<table><thead><tr><th>x</th><th>y</th></tr></thead>"
            "<tbody><tr><td>1</td><td><strong>2</strong></td></tr></tbody></table>"]

    def test_paragraphs_and_line_breaks(self):
        assert render("a\nb\n\nc") == ["<p>a<br>b</p><p>c</p>"]

    def test_code_fences_keep_markup_as_text(self):
        out = render("```\n<b>x</b> **y**\n```")[0]
        assert out == "<pre><code>&lt;b&gt;x&lt;/b&gt; **y**</code></pre>"

    def test_numbers_snake_case_and_maths_are_not_italicised(self):
        for text in ("2*3*4", "snake_case_name", "5 * 3 = 15", "a*b*c", "56.1% win rate, 134.8 SR"):
            assert "<em>" not in render(text)[0]
        assert render("*this* works")[0] == "<p><em>this</em> works</p>"
        assert render("56.1% win rate")[0] == "<p>56.1% win rate</p>"

    def test_empty_and_odd_input(self):
        assert render("", "   ", "\n\n") == ["", "", ""]
        assert node("console.log(JSON.stringify([R.renderMarkdown(null), R.renderMarkdown(undefined)]))",
                    None) == ["", ""]
        assert "<pre>" in render("```\nunterminated")[0]  # an open fence must not hang or throw

    def test_horizontal_rule(self):
        assert render("a\n\n---\n\nb") == ["<p>a</p><hr><p>b</p>"]


PAYLOADS = [
    "<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "**<b onmouseover=x>hi</b>**",
    "[x](javascript:alert(1))", '<a href="javascript:alert(1)">x</a>', '"><svg onload=alert(1)>',
    "'';!--\"<XSS>=&{()}", "`<script>alert(1)</script>`", "| <img src=x onerror=1> | y |\n|---|---|\n| <b> | c |",
    "# <script>alert(1)</script>", "- <iframe src=//evil.example>", "1. <style>*{x:y}</style>",
    "```\n</code></pre><script>alert(1)</script>\n```", "&lt;script&gt;alert(1)&lt;/script&gt;",
    "<scr<script>ipt>alert(1)</scr</script>ipt>", "<img src=x onerror=alert(1)>",
    "**bold<**<script>", "*em<*<img src=x onerror=alert(1)>", "` `<img onerror=x>` `",
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==", "<math><mi//xlink:href=\"data:x,<script>\">",
]


class TestMarkdownCannotInjectMarkup:
    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_after_removing_the_fixed_tags_no_angle_bracket_is_left(self, payload):
        """Every `<` / `>` that came from the text must have been escaped: only the renderer's
        own tags (no attributes except class="nested") may contain raw angle brackets."""
        out = render(payload)[0]
        stripped = ALLOWED_TAG.sub("", out)
        assert "<" not in stripped and ">" not in stripped, out

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_the_only_attribute_ever_produced_is_class_nested(self, payload):
        out = render(payload)[0]
        attrs = re.findall(r"<[a-z0-9]+\s+([^>]*)>", out)
        assert all(a == 'class="nested"' for a in attrs), out

    def test_links_and_images_are_inert_text(self):
        out = render("[click](javascript:alert(1)) ![img](http://evil/x.png)")[0]
        assert "<a" not in out and "<img" not in out and "href" not in out.replace("&", "")
        assert "[click](javascript:alert(1))" in out

    def test_existing_entities_are_escaped_again_not_decoded(self):
        assert render("&lt;b&gt;")[0] == "<p>&amp;lt;b&amp;gt;</p>"

    def test_quotes_are_escaped(self):
        assert render("a \"b\" 'c'")[0] == "<p>a &quot;b&quot; &#39;c&#39;</p>"


class TestMarkdownIsFast:
    """Pathological inputs must not cause catastrophic regex backtracking (ReDoS)."""

    @pytest.mark.parametrize("name,expr", [
        ("stars", "'*'.repeat(20000)"), ("bold openers", "'**a'.repeat(5000)"),
        ("pipes", "'|'.repeat(20000)"), ("bullets", "'- '.repeat(10000)"),
        ("backticks", "'`a'.repeat(10000)"), ("underscores", "'a_'.repeat(10000)"),
        ("table", "'| a '.repeat(5000) + '|\\n' + '|---'.repeat(5000) + '|'"),
        ("long line", "'word '.repeat(40000)"), ("many lines", "'line\\n'.repeat(20000)"),
        ("em candidates", "'* a '.repeat(8000)"),
    ])
    def test_hostile_input_renders_quickly(self, name, expr):
        seconds = node(f"const t = {expr}; const s = Date.now(); R.renderMarkdown(t);"
                       "console.log(JSON.stringify((Date.now() - s) / 1000))", None)
        assert seconds < 2.0, f"{name} took {seconds}s"


class TestSSEParser:
    def parse(self, *chunks):
        """Feed chunks one by one the way app.js does; return all events."""
        return node("""
            let buffer = '', all = [];
            for (const c of input) { const p = S.parseSSE(buffer + c); buffer = p.rest; all.push(...p.events); }
            console.log(JSON.stringify({events: all, rest: buffer}));""", list(chunks))

    def test_a_complete_event(self):
        r = self.parse('event: answer\ndata: {"a": 1}\n\n')
        assert r == {"events": [{"event": "answer", "data": '{"a": 1}'}], "rest": ""}

    def test_an_event_split_across_chunks_waits_for_its_end(self):
        r = self.parse("event: tool_ca", 'll\ndata: {"n"', ': 1}\n', "\nevent: done\ndata: {}\n\n")
        assert [e["event"] for e in r["events"]] == ["tool_call", "done"]
        assert r["events"][0]["data"] == '{"n": 1}'

    def test_an_unfinished_tail_is_kept_not_emitted(self):
        r = self.parse('event: a\ndata: {}\n\nevent: b\ndata: {"x"')
        assert [e["event"] for e in r["events"]] == ["a"] and r["rest"].startswith("event: b")

    def test_crlf_comments_and_multi_line_data(self):
        r = self.parse(": keep-alive\r\n\r\nevent: x\r\ndata: one\r\ndata: two\r\n\r\n")
        assert r["events"] == [{"event": "x", "data": "one\ntwo"}]

    def test_default_event_name_and_events_without_data_are_dropped(self):
        r = self.parse("data: hello\n\nevent: empty\n\n")
        assert r["events"] == [{"event": "message", "data": "hello"}]

    def test_the_real_server_stream_parses_identically_at_every_split_point(self):
        """Parse the actual bytes the API produces, cut at every possible position."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.llm.agent import CricketAgent
        from tests.test_api import settings
        from tests.test_llm_agent import FakeClient, StubToolbox, reply, text, tool_use

        app = create_app(settings(), toolbox=StubToolbox(), agent_factory=lambda: CricketAgent(
            toolbox=StubToolbox(), client=FakeClient(
                reply(tool_use("a", "find_player", query="Kohli"), stop="tool_use"),
                reply(text("It has a – dash and an emoji \U0001F3CF, and \"quotes\"")))))
        with TestClient(app).stream("POST", "/api/chat/stream", json={"message": "hi"}) as r:
            raw = "".join(r.iter_text())
        whole = self.parse(raw)["events"]
        assert [e["event"] for e in whole] == ["session", "step", "tool_call", "tool_result",
                                              "step", "answer", "done"]
        assert json.loads(whole[5]["data"])["answer"].startswith("It has a – dash")
        result = node("""
            const raw = input; let bad = 0;
            const whole = S.parseSSE(raw).events;
            for (let cut = 0; cut <= raw.length; cut++) {
              let buffer = '', all = [];
              for (const c of [raw.slice(0, cut), raw.slice(cut)]) {
                const p = S.parseSSE(buffer + c); buffer = p.rest; all.push(...p.events);
              }
              if (JSON.stringify(all) !== JSON.stringify(whole)) bad++;
            }
            console.log(JSON.stringify({bad, cuts: raw.length + 1}));""", raw)
        assert result["bad"] == 0 and result["cuts"] > 200


class TestPageFiles:
    @pytest.mark.parametrize("name", ["render.js", "sse.js", "app.js"])
    def test_scripts_are_valid_javascript(self, name):
        out = subprocess.run(["node", "--check", str(FRONTEND / name)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_the_page_loads_every_script_it_needs_in_order(self):
        html = (FRONTEND / "index.html").read_text()
        srcs = re.findall(r'<script src="(/static/[^"]+)"', html)
        assert srcs == ["/static/render.js", "/static/sse.js", "/static/app.js"]
        assert all((FRONTEND / s.split("/")[-1]).exists() for s in srcs)
        assert 'rel="stylesheet" href="/static/style.css"' in html

    def test_app_js_only_uses_the_safe_renderer_for_html(self):
        js = (FRONTEND / "app.js").read_text()
        assert js.count("innerHTML") == 1 and "CricketRender.renderMarkdown(a.answer)" in js
        assert "eval(" not in js and "document.write" not in js and "insertAdjacentHTML" not in js

    def test_every_element_app_js_looks_up_exists_in_the_page(self):
        js = (FRONTEND / "app.js").read_text()
        html = (FRONTEND / "index.html").read_text()
        for ident in set(re.findall(r'\$\("#([\w-]+)"\)', js)):
            assert f'id="{ident}"' in html, ident
