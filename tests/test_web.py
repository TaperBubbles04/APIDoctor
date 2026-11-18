"""HTTP-layer tests. Calls the ASGI app directly (no httpx, no sockets); skipped if FastAPI is missing."""
import asyncio
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from apidoctor.config import load_config
from apidoctor.models import Diagnosis
from apidoctor.service import MAX_BODY_BYTES, MAX_LOG_LINES

try:
    from apidoctor import web
except ImportError:   # FastAPI not installed
    web = None

ROOT = Path(__file__).resolve().parent.parent
SPEC_TEXT = (ROOT / "sample_data" / "spec.yaml").read_text(encoding="utf-8")
LOGS_TEXT = (ROOT / "sample_data" / "logs.jsonl").read_text(encoding="utf-8")
BASE = load_config(env_path=Path("/nonexistent/.env"))
NO_KEY = replace(BASE, api_key=None, delay_between_llm_calls_s=0, max_llm_calls_per_run=1)
WITH_KEY = replace(NO_KEY, api_key="TESTKEY-0123456789abcdef")
JSON_HEADERS = {"content-type": "application/json"}


class Scripted:
    def __init__(self):
        self.calls = 0

    def diagnose(self, record, spec_excerpt, rule_hint):
        self.calls += 1
        return Diagnosis("conflict", False, 0.8, "scripted", "scripted fix", "llm")


def call(method, path, body=b"", headers=None, host="127.0.0.1:8000"):
    """Run one request through the app. Returns (status, lower-cased headers dict, body bytes)."""
    header_list = [(b"host", host.encode())] + [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    if body and "content-length" not in (headers or {}):
        header_list.append((b"content-length", str(len(body)).encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": header_list, "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000)}
    sent = []

    async def run():
        delivered = asyncio.Event()

        async def receive():
            if not delivered.is_set():
                delivered.set()
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)   # a real client stays connected until the response is sent

        async def send(message):
            sent.append(message)

        await web.app(scope, receive, send)

    asyncio.run(asyncio.wait_for(run(), timeout=10))
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers_out = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return start["status"], headers_out, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def post(payload=None, raw_body=None, headers=JSON_HEADERS):
    body = raw_body if raw_body is not None else json.dumps(payload).encode()
    return call("POST", "/api/diagnose", body, headers)


def good(**over):
    return {"logs_text": LOGS_TEXT, "spec_text": SPEC_TEXT, "use_llm": False, **over}


@unittest.skipIf(web is None, "FastAPI is not installed")
class WebTests(unittest.TestCase):
    def setUp(self):
        for patcher in (mock.patch("apidoctor.web.load_config", return_value=NO_KEY),
                        mock.patch("apidoctor.web.get_provider", return_value=None),
                        mock.patch("apidoctor.web.CACHE_PATH", Path("/nonexistent/cache.json"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_index_page_is_self_contained_and_safe(self):
        status, headers, body = call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        html = body.decode()
        self.assertIn("APIDoctor", html)
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
                          "<script src", "http://", "https://", " style=", "@import", "<link"):
            self.assertNotIn(forbidden, html, forbidden)
        self.assertIn("textContent", html)
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")

    def test_csp_is_the_original_strict_policy(self):
        _, headers, _ = call("GET", "/")
        self.assertEqual(headers["content-security-policy"],
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; base-uri 'none'; form-action 'none'")

    def test_page_has_the_designed_content(self):
        html = call("GET", "/")[2].decode()
        for needle in ("Paste API call logs and an OpenAPI spec. APIDoctor explains why calls failed and whether a retry can help.",
                       "<li>Add logs</li>", "<li>Add spec</li>", "<li>Diagnose</li>",
                       "Try with sample data", "/api/sample", ">Clear<", "prefers-color-scheme: dark", ":focus-visible",
                       "Root causes", "Failures", "Rules only", "Retry may help", "Do not retry as-is",
                       "Records read", "Failures found", "Root-cause groups", "LLM calls used",
                       "No failures found", "could not be read", "role=\"alert\"",
                       "Only redacted data is sent to the LLM", "textContent"):
            self.assertIn(needle, html, needle)

    def test_page_never_builds_class_names_from_data(self):
        html = call("GET", "/")[2].decode()
        self.assertIn("CAT_CLASS[c]", html)              # fixed lookup table
        self.assertNotIn('"cat-" +', html)               # no concatenation with data
        self.assertNotIn("className = r.", html)

    def test_sample_route_returns_the_repo_files(self):
        status, _, body = call("GET", "/api/sample")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(data["logs_text"], LOGS_TEXT)
        self.assertEqual(data["spec_text"], SPEC_TEXT)
        self.assertEqual(set(data), {"logs_text", "spec_text"})

    def test_sample_route_missing_files_is_a_clean_404(self):
        with mock.patch("apidoctor.web.SAMPLE_LOGS", Path("/nonexistent/logs.jsonl")):
            status, _, body = call("GET", "/api/sample")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body), {"error": "the sample data files are missing"})
        with mock.patch("apidoctor.web.SAMPLE_SPEC", Path("/nonexistent/spec.yaml")):
            self.assertEqual(call("GET", "/api/sample")[0], 404)

    def test_sample_route_obeys_host_check_and_headers(self):
        self.assertEqual(call("GET", "/api/sample", host="evil.example.com")[0], 400)
        _, headers, _ = call("GET", "/api/sample")
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertFalse([h for h in headers if h.startswith("access-control")])

    def test_sample_data_round_trips_through_diagnose(self):
        _, _, body = call("GET", "/api/sample")
        sample = json.loads(body)
        status, _, out = post({**sample, "use_llm": False})
        self.assertEqual((status, json.loads(out)["failures"]), (200, 5))

    def test_no_cors_headers(self):
        _, headers, _ = call("GET", "/api/status")
        self.assertFalse([h for h in headers if h.startswith("access-control")])

    def test_status_without_key(self):
        status, _, body = call("GET", "/api/status")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIs(data["llm_available"], False)
        self.assertEqual(data["max_log_lines"], MAX_LOG_LINES)

    def test_status_with_key_never_returns_it(self):
        with mock.patch("apidoctor.web.load_config", return_value=WITH_KEY):
            _, _, body = call("GET", "/api/status")
        self.assertIs(json.loads(body)["llm_available"], True)
        self.assertNotIn(b"TESTKEY", body)

    def test_non_local_host_header_is_rejected(self):
        for host in ("evil.example.com", "evil.example.com:8000", ""):
            self.assertEqual(call("GET", "/api/status", host=host)[0], 400, host)
        self.assertEqual(call("GET", "/api/status", host="localhost:8000")[0], 200)

    def test_post_requires_json_content_type(self):
        status, _, body = post(raw_body=json.dumps(good()).encode(), headers={"content-type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertEqual(post(raw_body=b"{}", headers={})[0], 415)

    def test_oversized_body_rejected(self):
        status, _, _ = call("POST", "/api/diagnose", b"{}", JSON_HEADERS | {"content-length": str(MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)
        status, _, body = post(raw_body=b" " * (MAX_BODY_BYTES + 1), headers=JSON_HEADERS | {"content-length": "2"})
        self.assertEqual(status, 413)   # a lying Content-Length does not help

    def test_too_many_lines_rejected(self):
        many = "\n".join(["{}"] * (MAX_LOG_LINES + 1))
        self.assertEqual(post(good(logs_text=many))[0], 413)

    def test_diagnose_sample_data(self):
        status, _, body = post(good())
        data = json.loads(body)
        self.assertEqual((status, data["failures"], data["non_failures"]), (200, 5, 1))
        self.assertNotIn(b"SYNTHETIC", body)

    def test_bad_requests_get_fixed_error_messages(self):
        for kwargs, expected in ((dict(raw_body=b"{nope"), "valid JSON"),
                                 (dict(payload=good(spec_text="a: [unclosed SYNTHETIC-SECRET")), "spec is not valid"),
                                 (dict(payload=good(use_llm=True)), "no LLM key")):
            status, _, body = post(**kwargs)
            self.assertEqual(status, 400)
            message = json.loads(body)["error"]
            self.assertIn(expected, message)
            self.assertNotIn("SYNTHETIC", message)

    def test_llm_cap_per_request(self):
        fake = Scripted()
        logs = "\n".join(json.dumps({"timestamp": "t", "method": "PUT", "url": f"https://api.example.test/v1/orders/o{i}",
                                     "status_code": 409, "request_headers": {}, "duration_ms": 5}) for i in (1, 2, 3))
        with mock.patch("apidoctor.web.load_config", return_value=WITH_KEY), \
                mock.patch("apidoctor.web.get_provider", return_value=fake), \
                mock.patch("apidoctor.web.CACHE_PATH", Path(self.enterContext_tmp()) / "c.json"):
            status, _, body = post(good(logs_text=logs, use_llm=True))
        data = json.loads(body)
        self.assertEqual((status, fake.calls, data["llm_skipped"]), (200, 1, 2))

    def enterContext_tmp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_unexpected_errors_never_leak_text(self):
        with mock.patch("apidoctor.web.diagnose_request", side_effect=RuntimeError("SYNTHETIC-SECRET-leak")):
            status, _, body = post(good())
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body), {"error": "internal error"})


if __name__ == "__main__":
    unittest.main()
