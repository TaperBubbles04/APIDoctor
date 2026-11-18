"""Read an OpenAPI spec and answer questions about ONE operation.

Deliberately small. Known limits:
  - Only LOCAL refs of the form '#/components/...' are resolved, to a depth of 5,
    with cycle protection. Any other $ref (external file, URL, '#/paths/...',
    or too deep, or a cycle) is left in place as {"$ref": "..."}.
  - URL paths are matched as-is AND with each servers[].url base path stripped.
    Server urls containing {variables} are skipped (not substituted).
  - allOf / oneOf / anyOf are not followed. Only top-level body fields are checked,
    and only the types string, integer, number, boolean.
"""
import json
import re
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from apidoctor.models import LogRecord
from apidoctor.redact import REDACTED

MAX_REF_DEPTH = 5
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


class TypeMismatch(NamedTuple):
    field: str
    expected: str
    actual: str


def load_spec(path: Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ValueError(f"unsupported spec file type: {suffix!r} (use .json, .yaml or .yml)")
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    import yaml  # imported here so JSON specs work without PyYAML
    return yaml.safe_load(text)


def load_spec_text(text: str) -> dict:
    """Parse spec text that is JSON or YAML. Raises ValueError if it is neither, or not an object."""
    try:
        spec = json.loads(text)
    except ValueError:
        import yaml
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError:
            raise ValueError("spec is not valid JSON or YAML") from None
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON or YAML object")
    return spec


def resolve_refs(node: Any, spec: dict, depth: int = 0, active: tuple = ()) -> Any:
    """Return a copy of node with local #/components/ refs replaced by their targets."""
    if isinstance(node, list):
        return [resolve_refs(item, spec, depth, active) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        if not ref.startswith("#/components/") or ref in active or depth >= MAX_REF_DEPTH:
            return dict(node)  # left untouched
        target: Any = spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return dict(node)  # dangling ref: leave it
            target = target[part]
        return resolve_refs(target, spec, depth + 1, active + (ref,))
    return {key: resolve_refs(value, spec, depth, active) for key, value in node.items()}


def _server_base_paths(spec: dict) -> list[str]:
    """Base paths from servers[].url, e.g. 'https://h/v1' -> '/v1'. Templated ({x}) urls are skipped."""
    bases = []
    for server in spec.get("servers") or []:
        url = server.get("url") if isinstance(server, dict) else None
        if not isinstance(url, str) or "{" in url:
            continue
        base = urlsplit(url).path.rstrip("/")
        if base and base not in bases:
            bases.append(base)
    return bases


def _match(spec: dict, method: str, url: str) -> tuple[str, dict] | None:
    """Find (path template, raw path item) for the URL. The path is tried as-is and
    with each server base path stripped. Fewest {params} wins; on a tie, as-is wins."""
    path = urlsplit(url).path.rstrip("/") or "/"
    candidates = [path]
    for base in _server_base_paths(spec):
        if path == base:
            candidates.append("/")
        elif path.startswith(base + "/"):
            candidates.append(path[len(base):])
    best = None
    for candidate in candidates:
        found = _match_path(spec, method, candidate)
        if found and (best is None or found[0] < best[0]):
            best = found
    return (best[1], best[2]) if best else None


def _match_path(spec: dict, method: str, path: str) -> tuple[int, str, dict] | None:
    segments = path.split("/")
    best = None
    for template, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict) or method.lower() not in item:
            continue
        t_segments = (template.rstrip("/") or "/").split("/")
        if len(t_segments) != len(segments):
            continue
        params = 0
        for t, s in zip(t_segments, segments):
            if re.fullmatch(r"\{[^{}]+\}", t):
                if not s:
                    break
                params += 1
            elif t != s:
                break
        else:
            if best is None or params < best[0]:
                best = (params, template, item)
    return best


def find_operation(spec: dict, method: str, url: str) -> dict | None:
    found = _match(spec, method, url)
    if not found:
        return None
    _, item = found
    operation = resolve_refs(item[method.lower()], spec)
    # Path-level parameters apply to every operation; operation ones come first.
    shared = resolve_refs(item.get("parameters", []), spec)
    operation["parameters"] = operation.get("parameters", []) + shared
    return operation


def operation_name(spec: dict, method: str, url: str) -> str | None:
    found = _match(spec, method, url)
    if not found:
        return None
    template, item = found
    return item[method.lower()].get("operationId") or f"{method.upper()} {template}"


def _body_schema(operation: dict) -> dict | None:
    content = (operation.get("requestBody") or {}).get("content") or {}
    media = content.get("application/json") or next(iter(content.values()), None)
    schema = (media or {}).get("schema")
    return schema if isinstance(schema, dict) else None


def _body_dict(body: Any) -> dict | None:
    """The request body as a dict, or None. Bodies may be JSON text after redaction."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    return body if isinstance(body, dict) else None


def required_fields_missing(record: LogRecord, operation: dict) -> list[str]:
    schema = _body_schema(operation)
    required = (schema or {}).get("required") or []
    if not required:
        return []
    body = _body_dict(record.request_body)
    if body is None:
        return list(required) if record.request_body is None else []
    return [name for name in required if name not in body]


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object" if isinstance(value, dict) else "null"


def type_mismatches(record: LogRecord, operation: dict) -> list[TypeMismatch]:
    schema = _body_schema(operation)
    properties = (schema or {}).get("properties") or {}
    body = _body_dict(record.request_body)
    if body is None:
        return []
    found = []
    for name, prop in properties.items():
        expected = prop.get("type") if isinstance(prop, dict) else None
        check = _TYPE_CHECKS.get(expected)
        if name not in body or check is None:
            continue
        value = body[name]
        if value is None or value == REDACTED:  # redaction replaced the real value
            continue
        if not check(value):
            found.append(TypeMismatch(name, expected, _json_type(value)))
    return found


def excerpt_for(record: LogRecord, spec: dict) -> dict | None:
    """A small dict about the matching operation only, for the LLM later."""
    operation = find_operation(spec, record.method, record.url)
    if operation is None:
        return None
    schema = _body_schema(operation) or {}
    required = schema.get("required") or []
    fields = {
        name: (prop.get("type") if isinstance(prop, dict) else None)
        for name, prop in (schema.get("properties") or {}).items()
    }
    return {
        "operation": operation_name(spec, record.method, record.url),
        "summary": operation.get("summary"),
        "parameters": [
            {
                "name": p.get("name"),
                "in": p.get("in"),
                "required": bool(p.get("required")),
                "type": (p.get("schema") or {}).get("type"),
            }
            for p in operation.get("parameters", [])
            if isinstance(p, dict)
        ],
        "request_body_required_fields": {n: fields.get(n) for n in required},
        "request_body_optional_fields": {n: t for n, t in fields.items() if n not in required},
        "response_codes": {
            str(code): (resp or {}).get("description")
            for code, resp in (operation.get("responses") or {}).items()
        },
    }
