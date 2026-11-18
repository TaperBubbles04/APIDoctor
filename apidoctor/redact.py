"""Replace secrets with [REDACTED] in headers, URLs and JSON bodies.

Two ways something gets redacted:
  1. its NAME looks secret (header Authorization, query param api_key, JSON key password)
  2. its VALUE looks like a token: a run of token characters at least `min_len`
     long that contains both a letter and a digit (UUIDs are kept, they are IDs).
Known gap: a long token made only of letters is not caught by rule 2.
"""
import json
import re
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

SECRET_HEADERS = {
    "authorization", "proxy-authorization", "cookie", "set-cookie",
    "x-api-key", "api-key", "x-auth-token", "x-access-token",
}
# A key/param name is secret if, ignoring case and - _ characters, it contains one of these...
SECRET_NAME_PARTS = ("token", "secret", "password", "passwd", "apikey", "credential", "privatekey")
# ...or equals one of these exactly (so "author" and "authenticated" are NOT redacted).
SECRET_NAME_EXACT = ("auth", "authorization", "proxyauthorization")

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SCHEME_TOKEN = re.compile(r"\b(Bearer|Basic)\s+[^\s\"',;]+", re.IGNORECASE)


def is_secret_name(name: str) -> bool:
    normalized = name.lower().replace("-", "").replace("_", "")
    return normalized in SECRET_NAME_EXACT or any(part in normalized for part in SECRET_NAME_PARTS)


def redact_string(text: str, min_len: int) -> str:
    text = _SCHEME_TOKEN.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    run = re.compile(rf"[A-Za-z0-9_\-+=.]{{{min_len},}}")

    def replace(match: re.Match) -> str:
        s = match.group(0)
        has_letter = any(c.isalpha() for c in s)
        has_digit = any(c.isdigit() for c in s)
        if has_letter and has_digit and not _UUID.match(s):
            return REDACTED
        return s

    return run.sub(replace, text)


def redact_headers(headers: dict[str, str], min_len: int) -> dict[str, str]:
    out = {}
    for name, value in headers.items():
        if name.lower() in SECRET_HEADERS or is_secret_name(name):
            out[name] = REDACTED
        else:
            out[name] = redact_string(value, min_len)
    return out


def redact_url(url: str, min_len: int) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:  # user:password@host
        netloc = REDACTED + "@" + netloc.rsplit("@", 1)[1]
    pairs = []
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        value = REDACTED if is_secret_name(name) else redact_string(value, min_len)
        pairs.append(f"{quote(name, safe='')}={quote(value, safe='[]')}")
    path = redact_string(parts.path, min_len)
    fragment = redact_string(parts.fragment, min_len)
    return urlunsplit((parts.scheme, netloc, path, "&".join(pairs), fragment))


def redact_body(body: Any, min_len: int) -> Any:
    if isinstance(body, dict):
        return {
            key: REDACTED if is_secret_name(str(key)) else redact_body(value, min_len)
            for key, value in body.items()
        }
    if isinstance(body, list):
        return [redact_body(item, min_len) for item in body]
    if isinstance(body, str):
        stripped = body.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                return json.dumps(redact_body(json.loads(body), min_len))
            except ValueError:
                pass  # not valid JSON: treat as plain text
        return redact_string(body, min_len)
    return body  # numbers, bool, None
