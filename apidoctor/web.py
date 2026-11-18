"""Local web page. A thin layer: HTTP safety checks live here, all logic lives in service.py."""
import threading

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response

from apidoctor.config import PROJECT_ROOT, load_config
from apidoctor.llm.base import get_provider
from apidoctor.service import (
    MAX_BODY_BYTES, MAX_LOG_LINES, RequestError, diagnose_request, parse_payload,
)

HOST, PORT = "127.0.0.1", 8000   # the server only ever binds to this address
CACHE_PATH = PROJECT_ROOT / ".cache" / "llm_cache.json"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SAMPLE_LOGS = PROJECT_ROOT / "sample_data" / "logs.jsonl"
SAMPLE_SPEC = PROJECT_ROOT / "sample_data" / "spec.yaml"
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "connect-src 'self'; base-uri 'none'; form-action 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}
_LOCK = threading.Lock()   # one diagnosis at a time: protects the cache file

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>APIDoctor</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f7f9; --fg: #1b1f24; --muted: #4b5563; --card: #ffffff; --border: #cfd4dc;
    --accent: #1d4ed8; --accent-fg: #ffffff; --focus: #1d4ed8;
    --warn-bg: #fff4e5; --warn-bd: #b45309; --warn-fg: #5c2e00;
    --err-bg: #fdecec; --err-bd: #b42318; --err-fg: #7a1212;
    --auth-bg: #fee2e2; --auth-fg: #7f1d1d;
    --validation-bg: #dbeafe; --validation-fg: #1e3a8a;
    --notfound-bg: #e5e7eb; --notfound-fg: #1f2937;
    --conflict-bg: #fef3c7; --conflict-fg: #78350f;
    --ratelimit-bg: #ede9fe; --ratelimit-fg: #4c1d95;
    --servererror-bg: #fce7f3; --servererror-fg: #831843;
    --timeout-bg: #ccfbf1; --timeout-fg: #134e4a;
    --unknown-bg: #e5e7eb; --unknown-fg: #374151;
    --retryyes-bg: #dcfce7; --retryyes-fg: #14532d;
    --retryno-bg: #ffedd5; --retryno-fg: #7c2d12;
    --retrymixed-bg: #e5e7eb; --retrymixed-fg: #374151;
    --srcrule-bg: #e5e7eb; --srcrule-fg: #1f2937;
    --srcllm-bg: #e0e7ff; --srcllm-fg: #312e81;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1419; --fg: #e6e8eb; --muted: #a8afb8; --card: #171d24; --border: #2f3a46;
      --accent: #6ea8fe; --accent-fg: #0b1220; --focus: #9ec5fe;
      --warn-bg: #3a2a0d; --warn-bd: #d9a441; --warn-fg: #fde9bd;
      --err-bg: #3d1717; --err-bd: #f08a8a; --err-fg: #fcd0d0;
      --auth-bg: #4a1d1d; --auth-fg: #fecaca;
      --validation-bg: #1e3a5f; --validation-fg: #bfdbfe;
      --notfound-bg: #2d333b; --notfound-fg: #d1d5db;
      --conflict-bg: #4a3510; --conflict-fg: #fde68a;
      --ratelimit-bg: #35275a; --ratelimit-fg: #ddd6fe;
      --servererror-bg: #531a3a; --servererror-fg: #fbcfe8;
      --timeout-bg: #134e4a; --timeout-fg: #99f6e4;
      --unknown-bg: #2d333b; --unknown-fg: #d1d5db;
      --retryyes-bg: #14392a; --retryyes-fg: #bbf7d0;
      --retryno-bg: #4a2a12; --retryno-fg: #fed7aa;
      --retrymixed-bg: #2d333b; --retrymixed-fg: #d1d5db;
      --srcrule-bg: #2d333b; --srcrule-fg: #d1d5db;
      --srcllm-bg: #2a2f5a; --srcllm-fg: #c7d2fe;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  main { max-width: 1080px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
  h1 { margin: 0 0 .25rem; font-size: 1.9rem; }
  h2 { margin: 2rem 0 .75rem; font-size: 1.25rem; }
  .lead { margin: 0 0 1rem; color: var(--muted); font-size: 1.05rem; }
  .steps { display: flex; flex-wrap: wrap; gap: .5rem 1.25rem; list-style: none; margin: 0 0 1.5rem; padding: 0; counter-reset: step; }
  .steps li { counter-increment: step; display: flex; align-items: center; gap: .5rem; font-weight: 600; }
  .steps li::before { content: counter(step); display: inline-flex; align-items: center; justify-content: center;
                      width: 1.6rem; height: 1.6rem; border-radius: 50%; background: var(--accent); color: var(--accent-fg); font-size: .9rem; }
  .toolbar { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin-bottom: 1rem; }
  .inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
  .panel h3 { margin: 0 0 .5rem; font-size: 1.05rem; }
  .panel .filerow { margin-bottom: .6rem; font-size: .92rem; color: var(--muted); }
  textarea { width: 100%; min-height: 11rem; resize: vertical; padding: .6rem; border-radius: 8px;
             border: 1px solid var(--border); background: var(--bg); color: var(--fg);
             font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .controls { display: flex; flex-wrap: wrap; gap: 1rem 2rem; align-items: flex-start; justify-content: space-between; margin: 1.25rem 0; }
  .check { display: flex; gap: .5rem; align-items: center; font-weight: 600; }
  .check input { width: 1.15rem; height: 1.15rem; }
  .hint { margin: .25rem 0 0; color: var(--muted); font-size: .92rem; max-width: 46rem; }
  .btn { font: inherit; font-weight: 600; padding: .55rem 1.2rem; border-radius: 8px; cursor: pointer;
         border: 1px solid var(--border); background: var(--card); color: var(--fg); }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
  .btn:disabled { opacity: .55; cursor: not-allowed; }
  :focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
  .msg { margin: 1rem 0; padding: .75rem 1rem; border-radius: 8px; border: 1px solid var(--err-bd); background: var(--err-bg); color: var(--err-fg); font-weight: 600; }
  .msg[hidden] { display: none; }
  .warn { margin: 1rem 0; padding: .75rem 1rem; border-radius: 8px; border: 1px solid var(--warn-bd); background: var(--warn-bg); color: var(--warn-fg); }
  .warn ul { margin: .4rem 0 0; padding-left: 1.2rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: .8rem 1rem; }
  .stat .label { color: var(--muted); font-size: .9rem; }
  .stat .value { font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
  .stat .sub { color: var(--muted); font-size: .85rem; }
  .groups { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: .75rem; }
  .group { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: .9rem 1rem; }
  .group .op { font-weight: 700; font-size: 1.05rem; overflow-wrap: anywhere; }
  .group .count { color: var(--muted); margin: .15rem 0 .5rem; }
  .badges { display: flex; flex-wrap: wrap; gap: .4rem; }
  .badge { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .85rem; font-weight: 700; white-space: nowrap; border: 1px solid transparent; }
  .cat-auth { background: var(--auth-bg); color: var(--auth-fg); }
  .cat-validation { background: var(--validation-bg); color: var(--validation-fg); }
  .cat-not_found { background: var(--notfound-bg); color: var(--notfound-fg); }
  .cat-conflict { background: var(--conflict-bg); color: var(--conflict-fg); }
  .cat-rate_limit { background: var(--ratelimit-bg); color: var(--ratelimit-fg); }
  .cat-server_error { background: var(--servererror-bg); color: var(--servererror-fg); }
  .cat-network_timeout { background: var(--timeout-bg); color: var(--timeout-fg); }
  .cat-unknown { background: var(--unknown-bg); color: var(--unknown-fg); border-style: dashed; border-color: var(--muted); }
  .retry-yes { background: var(--retryyes-bg); color: var(--retryyes-fg); }
  .retry-no { background: var(--retryno-bg); color: var(--retryno-fg); }
  .retry-mixed { background: var(--retrymixed-bg); color: var(--retrymixed-fg); }
  .src-rule { background: var(--srcrule-bg); color: var(--srcrule-fg); }
  .src-llm { background: var(--srcllm-bg); color: var(--srcllm-fg); }
  .tablewrap { overflow-x: auto; background: var(--card); border: 1px solid var(--border); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; min-width: 820px; font-size: .92rem; }
  th, td { padding: .55rem .7rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .03em; }
  tr:last-child td { border-bottom: 0; }
  .req { display: flex; gap: .5rem; align-items: baseline; }
  .method { font-weight: 700; }
  .urlbtn { font: inherit; background: none; border: 0; padding: 0; color: inherit; text-align: left; cursor: pointer;
            max-width: 24ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block; }
  .urlbtn[aria-expanded="true"] { max-width: none; white-space: normal; overflow-wrap: anywhere; }
  details summary { cursor: pointer; }
  details .full { margin: .4rem 0 0; overflow-wrap: anywhere; }
  .empty { background: var(--card); border: 1px dashed var(--border); border-radius: 10px; padding: 1.5rem; text-align: center; color: var(--muted); }
  .empty strong { display: block; color: var(--fg); font-size: 1.1rem; margin-bottom: .25rem; }
</style>
</head>
<body>
<main>
<header>
  <h1>APIDoctor</h1>
  <p class="lead">Paste API call logs and an OpenAPI spec. APIDoctor explains why calls failed and whether a retry can help.</p>
  <ol class="steps"><li>Add logs</li><li>Add spec</li><li>Diagnose</li></ol>
</header>

<div class="toolbar">
  <button id="sample" type="button" class="btn primary">Try with sample data</button>
  <button id="clear" type="button" class="btn">Clear</button>
  <span class="hint">The sample is a small made-up example. It fills both boxes and runs the diagnosis.</span>
</div>

<div class="inputs">
  <section class="panel">
    <h3><label for="logs">1. Logs</label></h3>
    <div class="filerow"><label for="logsFile">Choose a .jsonl file, or paste below: </label><input type="file" id="logsFile"></div>
    <textarea id="logs" spellcheck="false" placeholder="One JSON log record per line. Each needs timestamp, method, url, status_code and duration_ms."></textarea>
  </section>
  <section class="panel">
    <h3><label for="spec">2. OpenAPI spec</label></h3>
    <div class="filerow"><label for="specFile">Choose a .yaml or .json file, or paste below: </label><input type="file" id="specFile"></div>
    <textarea id="spec" spellcheck="false" placeholder="Paste the OpenAPI spec here (YAML or JSON)."></textarea>
  </section>
</div>

<div class="controls">
  <div>
    <label class="check"><input type="checkbox" id="useLlm" disabled> Use LLM for cases the rules cannot decide</label>
    <p id="llmNote" class="hint">Checking server status...</p>
  </div>
  <button id="run" type="button" class="btn primary">3. Diagnose</button>
</div>

<div id="message" class="msg" role="alert" hidden></div>
<section id="results" aria-live="polite"></section>
</main>

<script>
(function () {
  "use strict";
  // Class names come only from this fixed table, never from log data.
  var CAT_CLASS = {
    auth: "cat-auth", validation: "cat-validation", not_found: "cat-not_found", conflict: "cat-conflict",
    rate_limit: "cat-rate_limit", server_error: "cat-server_error", network_timeout: "cat-network_timeout",
    unknown: "cat-unknown"
  };
  var busy = false;

  function $(id) { return document.getElementById(id); }
  // All data goes into the page as plain text (textContent), never as markup: logs are untrusted.
  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    if (cls) { e.className = cls; }
    return e;
  }
  function badge(text, cls) { return el("span", text, "badge " + cls); }
  function categoryBadge(c) { return badge(String(c).replace(/_/g, " "), CAT_CLASS[c] || "cat-unknown"); }
  function retryBadge(v) {
    if (v === true) { return badge("Retry may help", "retry-yes"); }
    if (v === false) { return badge("Do not retry as-is", "retry-no"); }
    return badge("Mixed: check each failure", "retry-mixed");
  }
  function sourceBadge(s) { return s === "llm" ? badge("LLM", "src-llm") : badge("Rule", "src-rule"); }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function shortUrl(u) {
    try { var x = new URL(u); return x.pathname + x.search; } catch (e) { return String(u); }
  }
  // Long text collapses into a native <details> element; short text stays plain.
  function expandable(text, limit) {
    var s = String(text);
    if (s.length <= limit) { return el("span", s); }
    var d = el("details");
    d.appendChild(el("summary", s.slice(0, limit) + "..."));
    d.appendChild(el("p", s, "full"));
    return d;
  }

  function showMessage(text) { var m = $("message"); m.textContent = text; m.hidden = false; }
  function clearMessage() { var m = $("message"); m.textContent = ""; m.hidden = true; }
  function setBusy(b) {
    busy = b;
    $("run").disabled = b; $("sample").disabled = b; $("clear").disabled = b;
    $("run").textContent = b ? "Diagnosing..." : "3. Diagnose";
    $("run").setAttribute("aria-busy", b ? "true" : "false");
  }
  function getJson(response) {
    return response.json().then(function (j) { return { ok: response.ok, body: j }; },
                                function () { return { ok: false, body: {} }; });
  }

  function loadFile(input, target) {
    input.addEventListener("change", function () {
      var file = input.files[0];
      if (!file) { return; }
      var reader = new FileReader();
      reader.onload = function () { target.value = reader.result; };
      reader.readAsText(file);
    });
  }
  loadFile($("logsFile"), $("logs"));
  loadFile($("specFile"), $("spec"));

  fetch("/api/status").then(function (r) { return r.json(); }).then(function (s) {
    if (s.llm_available) {
      $("useLlm").disabled = false;
      $("llmNote").textContent = "Only redacted data is sent to the LLM, and it is limited to " +
        s.max_llm_calls_per_request + " calls per run.";
    } else {
      $("llmNote").textContent = "Unavailable: no LLM key is configured on this machine, so only rules are used. " +
        "Only redacted data is ever sent to an LLM.";
    }
  }).catch(function () { $("llmNote").textContent = "Could not read the server status."; });

  function card(label, value, sub) {
    var c = el("div", null, "stat");
    c.appendChild(el("div", label, "label"));
    c.appendChild(el("div", value, "value"));
    if (sub) { c.appendChild(el("div", sub, "sub")); }
    return c;
  }

  function renderGroups(d, out) {
    out.appendChild(el("h2", "Root causes"));
    var wrap = el("div", null, "groups");
    d.groups.forEach(function (g) {
      var c = el("div", null, "group");
      c.appendChild(el("div", g.operation, "op"));
      c.appendChild(el("div", plural(g.count, "failure", "failures"), "count"));
      var b = el("div", null, "badges");
      b.appendChild(categoryBadge(g.category));
      b.appendChild(retryBadge(g.retryable));
      c.appendChild(b);
      wrap.appendChild(c);
    });
    out.appendChild(wrap);
  }

  function renderFailures(d, out) {
    out.appendChild(el("h2", "Failures"));
    var wrap = el("div", null, "tablewrap"), t = el("table"), head = el("tr"), body = el("tbody");
    ["Request", "Status", "Category", "Source", "Explanation", "Suggested fix"].forEach(function (h) {
      head.appendChild(el("th", h));
    });
    t.appendChild(el("thead")).appendChild(head);
    d.results.forEach(function (r) {
      var tr = el("tr");
      var req = el("td"), line = el("div", null, "req");
      line.appendChild(el("span", r.method, "method"));
      var btn = el("button", shortUrl(r.url), "urlbtn");
      btn.type = "button"; btn.title = r.url; btn.setAttribute("aria-expanded", "false");
      btn.addEventListener("click", function () {
        var open = btn.getAttribute("aria-expanded") === "true";
        btn.setAttribute("aria-expanded", open ? "false" : "true");
        btn.textContent = open ? shortUrl(r.url) : r.url;
      });
      line.appendChild(btn); req.appendChild(line); tr.appendChild(req);
      tr.appendChild(el("td", r.status_code === null ? "no response" : r.status_code));
      var cat = el("td"); cat.appendChild(categoryBadge(r.category));
      cat.appendChild(document.createElement("br")); cat.appendChild(retryBadge(r.retryable)); tr.appendChild(cat);
      var src = el("td"); src.appendChild(sourceBadge(r.source)); tr.appendChild(src);
      var why = el("td"); why.appendChild(expandable(r.explanation, 140)); tr.appendChild(why);
      var fix = el("td"); fix.appendChild(expandable(r.suggested_fix, 90)); tr.appendChild(fix);
      body.appendChild(tr);
    });
    t.appendChild(body); wrap.appendChild(t); out.appendChild(wrap);
  }

  function render(d) {
    var out = $("results");
    out.textContent = "";
    var cards = el("div", null, "cards");
    cards.appendChild(card("Records read", d.total_records));
    cards.appendChild(card("Failures found", d.failures));
    cards.appendChild(card("Root-cause groups", d.groups.length));
    if (d.llm) {
      var sub = d.llm.cache_hits ? plural(d.llm.cache_hits, "answer", "answers") + " from cache" : "";
      if (d.llm_skipped) { sub += (sub ? ", " : "") + d.llm_skipped + " skipped (call limit)"; }
      cards.appendChild(card("LLM calls used", d.llm.real_calls, sub));
    } else {
      cards.appendChild(card("Mode", "Rules only"));
    }
    out.appendChild(cards);

    if (d.load_errors.length) {
      var warn = el("div", null, "warn");
      warn.setAttribute("role", "alert");
      warn.appendChild(el("strong", plural(d.load_errors.length, "log line", "log lines") + " could not be read"));
      var ul = el("ul");
      d.load_errors.slice(0, 50).forEach(function (e) { ul.appendChild(el("li", "Line " + e.line + ": " + e.message)); });
      if (d.load_errors.length > 50) { ul.appendChild(el("li", "...and " + (d.load_errors.length - 50) + " more")); }
      warn.appendChild(ul);
      out.appendChild(warn);
    }

    if (d.results.length) {
      renderGroups(d, out);
      renderFailures(d, out);
    } else {
      var empty = el("div", null, "empty");
      empty.appendChild(el("strong", "No failures found"));
      empty.appendChild(el("span", d.total_records
        ? "None of the " + plural(d.total_records, "record", "records") + " read looks like a failed call."
        : "No readable log records were found. Check the log format."));
      out.appendChild(empty);
    }
    out.scrollIntoView();
  }

  function diagnose() {
    clearMessage();
    var logs = $("logs").value, spec = $("spec").value;
    if (!logs.trim() || !spec.trim()) {
      showMessage("Add both logs and an OpenAPI spec first, or try the sample data.");
      return Promise.resolve();
    }
    setBusy(true);
    return fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ logs_text: logs, spec_text: spec, use_llm: $("useLlm").checked && !$("useLlm").disabled })
    }).then(getJson).then(function (res) {
      if (!res.ok) { throw new Error(res.body.error || "The server could not process the request."); }
      render(res.body);
    }).catch(function (e) {
      $("results").textContent = "";
      showMessage(e instanceof TypeError ? "Could not reach the server. Is it still running?" : e.message);
    }).then(function () { setBusy(false); });
  }

  $("run").addEventListener("click", function () { if (!busy) { diagnose(); } });

  $("sample").addEventListener("click", function () {
    if (busy) { return; }
    clearMessage();
    setBusy(true);
    fetch("/api/sample").then(getJson).then(function (res) {
      if (!res.ok) { throw new Error(res.body.error || "Could not load the sample data."); }
      $("logs").value = res.body.logs_text;
      $("spec").value = res.body.spec_text;
      setBusy(false);
      return diagnose();
    }).catch(function (e) {
      setBusy(false);
      showMessage(e instanceof TypeError ? "Could not reach the server. Is it still running?" : e.message);
    });
  });

  $("clear").addEventListener("click", function () {
    if (busy) { return; }
    $("logs").value = ""; $("spec").value = "";
    $("logsFile").value = ""; $("specFile").value = "";
    $("results").textContent = "";
    clearMessage();
  });
})();
</script>
</body>
</html>
"""

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_only_and_headers(request: Request, call_next):
    # Rejects requests whose Host header is not localhost (blocks DNS-rebinding tricks).
    host = request.headers.get("host", "").split(":")[0].lower()
    if host not in LOCAL_HOSTS:
        response: Response = JSONResponse({"error": "bad host"}, status_code=400)
    else:
        response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers[name] = value
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_HTML


@app.get("/api/status")
def status() -> dict:
    config = load_config()
    return {   # never includes the key itself
        "llm_available": config.llm_enabled,
        "max_llm_calls_per_request": config.max_llm_calls_per_run,
        "max_log_lines": MAX_LOG_LINES,
        "max_body_mb": MAX_BODY_BYTES // (1024 * 1024),
    }


@app.get("/api/sample")
def sample():
    """The repo's synthetic example, so the page can offer a one-click demo."""
    try:
        return {"logs_text": SAMPLE_LOGS.read_text(encoding="utf-8"),
                "spec_text": SAMPLE_SPEC.read_text(encoding="utf-8")}
    except OSError:
        return JSONResponse({"error": "the sample data files are missing"}, status_code=404)


async def _read_limited(request: Request) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise RequestError(413, "request body too large (limit 2 MB)")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise RequestError(413, "request body too large (limit 2 MB)")
        chunks.append(chunk)
    return b"".join(chunks)


def _diagnose(payload: dict) -> dict:
    config = load_config()
    with _LOCK:
        return diagnose_request(payload, config, get_provider(config), CACHE_PATH)


@app.post("/api/diagnose")
async def api_diagnose(request: Request):
    try:
        # Requiring application/json stops other websites from sending "simple" cross-site POSTs.
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise RequestError(415, "Content-Type must be application/json")
        payload = parse_payload(await _read_limited(request))
        return JSONResponse(await run_in_threadpool(_diagnose, payload))
    except RequestError as problem:
        return JSONResponse({"error": problem.message}, status_code=problem.status)
    except Exception:   # never return raw error text
        return JSONResponse({"error": "internal error"}, status_code=500)


def serve() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
