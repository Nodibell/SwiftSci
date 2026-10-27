import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "verify_doc_coverage", Path(__file__).resolve().parents[1] / "verify_doc_coverage.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DocumentationReportingTests(unittest.TestCase):
    def run_check(self, source):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "Fixture.swift").write_text(source, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as exit_result:
                    checker.check_doc_coverage(directory)
            return exit_result.exception.code, output.getvalue()

    def test_documented_declaration(self):
        status, output = self.run_check("/// A value.\npublic struct Value {}\n")
        self.assertEqual(status, 0)
        self.assertIn("100.00% (1/1", output)
        self.assertIn("declaration scan", output)
        self.assertIn("No unresolved documentation placeholders", output)

    def test_undocumented_declaration(self):
        status, output = self.run_check("public struct Value {}\n")
        self.assertEqual(status, 0)
        self.assertIn("0.00% (0/1", output)
        self.assertNotIn("100.00%", output)

    def test_mixed_documentation(self):
        status, output = self.run_check(
            "/// A value.\npublic struct Documented {}\npublic struct Undocumented {}\n"
        )
        self.assertEqual(status, 0)
        self.assertIn("50.00% (1/2", output)
        self.assertNotIn("100.00%", output)

    def test_no_matching_declarations(self):
        status, output = self.run_check("struct Internal {}\n")
        self.assertEqual(status, 0)
        self.assertIn("N/A", output)
        self.assertIn("No public/open declarations matched", output)
        self.assertNotIn("100.00%", output)

    def test_placeholder_still_fails(self):
        status, output = self.run_check("/// <#description#>\npublic struct Value {}\n")
        self.assertEqual(status, 1)
        self.assertIn("unresolved Xcode placeholders", output)
        self.assertNotIn("No unresolved documentation placeholders", output)


if __name__ == "__main__":
    unittest.main()
