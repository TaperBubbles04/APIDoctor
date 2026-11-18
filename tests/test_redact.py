import json
import unittest

from apidoctor.redact import REDACTED, redact_body, redact_headers, redact_string, redact_url

N = 20  # token-length threshold used in these tests
TOKEN = "SYNTHETIC-TOKEN-0123456789abcdef"


class RedactTests(unittest.TestCase):
    def test_authorization_header(self):
        out = redact_headers({"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}, N)
        self.assertEqual(out["Authorization"], REDACTED)
        self.assertEqual(out["Accept"], "application/json")

    def test_secret_header_name_case_insensitive(self):
        self.assertEqual(redact_headers({"x-API-key": "abc"}, N)["x-API-key"], REDACTED)

    def test_token_in_query_string(self):
        url = redact_url(f"https://h.test/v1/x?api_key={TOKEN}&limit=5", N)
        self.assertNotIn("SYNTHETIC", url)
        self.assertIn("limit=5", url)

    def test_long_token_in_innocent_query_param(self):
        url = redact_url(f"https://h.test/v1/x?ref={TOKEN}", N)
        self.assertNotIn("SYNTHETIC", url)

    def test_userinfo_in_url(self):
        url = redact_url("https://bob:hunter2@h.test/v1/x", N)
        self.assertNotIn("hunter2", url)
        self.assertIn("h.test", url)

    def test_token_in_nested_json_body(self):
        body = {"a": [{"b": {"note": f"x {TOKEN} y", "password": "pw"}}]}
        out = redact_body(body, N)
        self.assertNotIn("SYNTHETIC", json.dumps(out))
        self.assertEqual(out["a"][0]["b"]["password"], REDACTED)

    def test_json_string_body(self):
        out = redact_body(json.dumps({"k": TOKEN, "keep": "ok"}), N)
        self.assertNotIn("SYNTHETIC", out)
        self.assertIn("ok", out)

    def test_short_id_not_redacted_but_long_token_is(self):
        self.assertEqual(redact_string("ord_8f3a12", N), "ord_8f3a12")
        self.assertEqual(redact_body({"id": "ord_8f3a12"}, N), {"id": "ord_8f3a12"})
        self.assertEqual(redact_string(TOKEN, N), REDACTED)

    def test_uuid_not_redacted(self):
        uid = "123e4567-e89b-12d3-a456-426614174000"
        self.assertEqual(redact_string(uid, N), uid)

    def test_threshold_is_configurable(self):
        self.assertEqual(redact_string("ord_8f3a12", 8), REDACTED)

    def test_plain_words_and_numbers_untouched(self):
        body = {"error": "must be an integer", "count": 3, "ok": True, "none": None}
        self.assertEqual(redact_body(body, N), body)

    def test_auth_matches_whole_name_only(self):
        body = {"auth": "x", "Authorization": "y", "author": "Jane", "authenticated": True}
        out = redact_body(body, N)
        self.assertEqual(out["auth"], REDACTED)
        self.assertEqual(out["Authorization"], REDACTED)
        self.assertEqual(out["author"], "Jane")
        self.assertIs(out["authenticated"], True)

    def test_auth_whole_name_in_query(self):
        url = redact_url("https://h.test/x?auth=abc&author=jane", N)
        self.assertIn("auth=%5BREDACTED%5D", url.replace("[", "%5B").replace("]", "%5D"))
        self.assertIn("author=jane", url)

    def test_response_headers_redacted(self):
        out = redact_headers({"Set-Cookie": "sid=abc", "Retry-After": "30"}, N)
        self.assertEqual(out["Set-Cookie"], REDACTED)
        self.assertEqual(out["Retry-After"], "30")

    def test_idempotent(self):
        once = redact_body({"t": TOKEN}, N)
        self.assertEqual(redact_body(once, N), once)


if __name__ == "__main__":
    unittest.main()
