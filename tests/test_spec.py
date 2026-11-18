import json
import tempfile
import unittest
from pathlib import Path

from apidoctor.models import LogRecord
from apidoctor.spec import (
    excerpt_for, find_operation, load_spec, operation_name, required_fields_missing,
    resolve_refs, type_mismatches,
)

ROOT = Path(__file__).resolve().parent.parent
SPEC = load_spec(ROOT / "sample_data" / "spec.yaml")


def rec(method="POST", url="https://api.example.test/v1/orders", body=None, status=422):
    return LogRecord("t", method, url, status, {}, body, None, 10)


class SpecTests(unittest.TestCase):
    def test_load_yaml_and_json(self):
        self.assertIn("/v1/orders", SPEC["paths"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"paths": {}}))
            self.assertEqual(load_spec(p), {"paths": {}})
            with self.assertRaises(ValueError):
                load_spec(Path(d) / "s.txt")

    def test_path_template_match(self):
        op = find_operation(SPEC, "GET", "https://api.example.test/v1/orders/ord_8f3a12?x=1")
        self.assertEqual(op["summary"], "Get one order")
        self.assertEqual(operation_name(SPEC, "GET", "https://api.example.test/v1/orders/ord_8f3a12"), "getOrder")
        # path-level parameter {id} was merged in
        self.assertEqual(op["parameters"][0]["name"], "id")

    def test_no_match(self):
        self.assertIsNone(find_operation(SPEC, "GET", "https://api.example.test/v9/nothing"))
        self.assertIsNone(find_operation(SPEC, "DELETE", "https://api.example.test/v1/orders"))

    def test_server_base_path_stripped(self):
        spec = {"servers": [{"url": "https://api.example.com/v1"}],
                "paths": {"/orders": {"get": {"summary": "list"}}, "/orders/{id}": {"get": {"summary": "one"}}}}
        self.assertEqual(find_operation(spec, "GET", "https://api.example.com/v1/orders")["summary"], "list")
        self.assertEqual(find_operation(spec, "GET", "https://api.example.com/v1/orders/o_1")["summary"], "one")
        self.assertEqual(operation_name(spec, "GET", "https://api.example.com/v1/orders"), "GET /orders")
        self.assertIsNone(find_operation(spec, "GET", "https://api.example.com/v2/orders"))

    def test_paths_written_with_base_path_still_match(self):
        spec = {"servers": [{"url": "https://api.example.com/v1"}],
                "paths": {"/v1/orders": {"get": {"summary": "full"}}}}
        self.assertEqual(find_operation(spec, "GET", "https://api.example.com/v1/orders")["summary"], "full")

    def test_as_is_match_wins_ties_and_templated_servers_skipped(self):
        spec = {"servers": [{"url": "https://h/v1"}, {"url": "https://h/{version}"}],
                "paths": {"/v1/orders": {"get": {"summary": "as-is"}}, "/orders": {"get": {"summary": "stripped"}}}}
        self.assertEqual(find_operation(spec, "GET", "https://h/v1/orders")["summary"], "as-is")

    def test_literal_path_beats_template(self):
        spec = {"paths": {"/o/{id}": {"get": {"summary": "by id"}}, "/o/search": {"get": {"summary": "search"}}}}
        self.assertEqual(find_operation(spec, "GET", "http://h/o/search")["summary"], "search")

    def test_refs_resolved(self):
        op = find_operation(SPEC, "POST", "https://api.example.test/v1/orders")
        schema = op["requestBody"]["content"]["application/json"]["schema"]
        self.assertEqual(schema["required"], ["customer", "quantity"])
        self.assertEqual(schema["properties"]["customer"]["required"], ["name"])  # nested ref too

    def test_ref_cycle_does_not_loop(self):
        spec = {"components": {"schemas": {
            "A": {"properties": {"b": {"$ref": "#/components/schemas/B"}}},
            "B": {"properties": {"a": {"$ref": "#/components/schemas/A"}}},
        }}}
        out = resolve_refs({"$ref": "#/components/schemas/A"}, spec)
        self.assertEqual(out["properties"]["b"]["properties"]["a"], {"$ref": "#/components/schemas/A"})

    def test_ref_depth_limit(self):
        schemas = {f"S{i}": {"next": {"$ref": f"#/components/schemas/S{i + 1}"}} for i in range(10)}
        out = resolve_refs({"$ref": "#/components/schemas/S0"}, {"components": {"schemas": schemas}})
        depth = 0
        while "next" in out:
            out, depth = out["next"], depth + 1
        self.assertEqual(depth, 5)
        self.assertIn("$ref", out)

    def test_non_local_refs_untouched(self):
        for ref in ("other.yaml#/components/schemas/X", "#/paths/~1x", "http://h/x.json"):
            self.assertEqual(resolve_refs({"$ref": ref}, SPEC), {"$ref": ref})

    def test_required_field_missing(self):
        op = find_operation(SPEC, "POST", "https://api.example.test/v1/orders")
        self.assertEqual(required_fields_missing(rec(body={"customer": {"name": "x"}}), op), ["quantity"])
        self.assertEqual(required_fields_missing(rec(body={"customer": {}, "quantity": 1}), op), [])
        self.assertEqual(required_fields_missing(rec(body=None), op), ["customer", "quantity"])
        # body may be JSON text (redaction re-serialises string bodies)
        self.assertEqual(required_fields_missing(rec(body='{"quantity": 1}'), op), ["customer"])

    def test_type_mismatch(self):
        op = find_operation(SPEC, "POST", "https://api.example.test/v1/orders")
        found = type_mismatches(rec(body={"customer": {}, "quantity": "three", "express": "yes"}), op)
        self.assertEqual([(m.field, m.expected, m.actual) for m in found],
                         [("quantity", "integer", "string"), ("express", "boolean", "string")])
        self.assertEqual(type_mismatches(rec(body={"quantity": True}), op)[0].actual, "boolean")
        self.assertEqual(type_mismatches(rec(body={"quantity": 3, "note": "hi"}), op), [])

    def test_redacted_value_is_not_a_type_mismatch(self):
        op = find_operation(SPEC, "POST", "https://api.example.test/v1/orders")
        self.assertEqual(type_mismatches(rec(body={"quantity": "[REDACTED]"}), op), [])

    def test_excerpt_is_small_and_about_one_operation(self):
        ex = excerpt_for(rec(), SPEC)
        self.assertEqual(ex["operation"], "createOrder")
        self.assertEqual(ex["request_body_required_fields"], {"customer": "object", "quantity": "integer"})
        self.assertEqual(ex["request_body_optional_fields"], {"note": "string", "express": "boolean"})
        self.assertEqual(set(ex["response_codes"]), {"201", "422"})
        self.assertNotIn("listItems", json.dumps(ex))
        self.assertIsNone(excerpt_for(rec(url="https://api.example.test/nope"), SPEC))


if __name__ == "__main__":
    unittest.main()
