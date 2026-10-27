import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
MODULES = ("SwiftAgent", "SwiftDataFrame", "SwiftNLP")


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "requires Apple LLVM")
class CoverageExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="coverage fixtures ")
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.root = Path(cls.fixture.name)
        cls.bundles = cls.root / "bundles"
        profiles = []
        for index, module in enumerate(MODULES):
            source = cls.root / "Sources" / module / "Fixture.c"
            source.parent.mkdir(parents=True)
            source.write_text(f"int fixture_{index}(void) {{ return 0; }}\n", encoding="utf-8")
            runner = cls.root / f"Runner{index}.c"
            runner.write_text(
                f"int fixture_{index}(void);\nint main(void) {{ return fixture_{index}(); }}\n",
                encoding="utf-8",
            )
            binary = cls.bundles / f"{module}.xctest" / "Contents" / "MacOS" / module
            binary.parent.mkdir(parents=True)
            subprocess.run(
                ["xcrun", "clang", "-fprofile-instr-generate", "-fcoverage-mapping",
                 str(source), str(runner), "-o", str(binary)], check=True, capture_output=True,
            )
            profile = cls.root / f"{module}.profraw"
            subprocess.run([str(binary)], check=True, env=dict(os.environ, LLVM_PROFILE_FILE=str(profile)))
            profiles.append(str(profile))
        cls.profile = cls.root / "default.profdata"
        subprocess.run(
            ["xcrun", "llvm-profdata", "merge", "-sparse", *profiles, "-o", str(cls.profile)],
            check=True, capture_output=True,
        )
        workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        step = workflow.split("      - name: Export coverage\n", 1)[1].split("      - name:", 1)[0]
        cls.export_command = textwrap.dedent(step.split("run: |\n", 1)[1])

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="swiftsci-coverage-test-")
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        self.build = self.work / "debug"
        shutil.copytree(self.bundles, self.build)
        shutil.copytree(REPOSITORY / "scripts", self.work / "scripts")
        self.profile_path = self.work / ".build/debug/codecov/default.profdata"
        self.profile_path.parent.mkdir(parents=True)
        shutil.copyfile(self.profile, self.profile_path)
        commands = self.work / "commands"
        commands.mkdir()
        swift = commands / "swift"
        swift.write_text('#!/bin/sh\nprintf "%s\\n" "$COVERAGE_BUILD_DIR"\n', encoding="utf-8")
        swift.chmod(0o755)
        self.env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])

    def export(self):
        return subprocess.run(
            ["bash", "-euo", "pipefail", "-c", self.export_command],
            cwd=self.work, env=dict(self.env, COVERAGE_BUILD_DIR=str(self.build)),
            text=True, capture_output=True,
        )

    def coverage(self):
        return (self.work / "coverage.lcov").read_text(encoding="utf-8")

    def test_exports_every_bundle(self):
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        for module in MODULES:
            self.assertIn(f"/Sources/{module}/Fixture.c\n", self.coverage())
        self.assertEqual(self.coverage().count("DA:1,1\n"), len(MODULES))

    def test_preserves_paths_with_spaces(self):
        destination = self.work / "build output with spaces"
        self.build.rename(destination)
        self.build = destination
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        for module in MODULES:
            self.assertIn(f"/Sources/{module}/Fixture.c\n", self.coverage())

    def test_missing_profile_fails(self):
        self.profile_path.unlink()
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("profile", result.stderr.lower())
        self.assertEqual(self.coverage(), "")

    def test_no_bundles_fails(self):
        shutil.rmtree(self.build)
        self.build.mkdir()
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".xctest", result.stderr)
        self.assertEqual(self.coverage(), "")

    def test_missing_bundle_executable_fails(self):
        (self.build / "SwiftNLP.xctest/Contents/MacOS/SwiftNLP").unlink()
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SwiftNLP", result.stderr)
        self.assertEqual(self.coverage(), "")

    def test_llvm_failure_preserves_status_and_diagnostic(self):
        self.profile_path.write_text("invalid LLVM profile\n", encoding="utf-8")
        expected = subprocess.run(
            ["xcrun", "llvm-cov", "export", "-format=lcov",
             "-instr-profile=.build/debug/codecov/default.profdata",
             str(self.build / "SwiftAgent.xctest/Contents/MacOS/SwiftAgent")],
            cwd=self.work, text=True, capture_output=True,
        )
        result = self.export()
        self.assertNotEqual(expected.returncode, 0)
        self.assertEqual(result.returncode, expected.returncode)
        self.assertIn(expected.stderr.strip(), result.stderr)
        self.assertEqual(self.coverage(), "")

    def test_missing_required_module_fails(self):
        shutil.rmtree(self.build / "SwiftDataFrame.xctest")
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SwiftDataFrame", result.stderr)
        self.assertEqual(self.coverage(), "")

    def test_unexecuted_required_module_fails(self):
        subprocess.run(
            ["xcrun", "llvm-profdata", "merge", "-sparse",
             str(self.root / "SwiftAgent.profraw"), "-o", str(self.profile_path)],
            check=True, capture_output=True,
        )
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SwiftDataFrame", result.stderr)
        self.assertIn("SwiftNLP", result.stderr)
        self.assertEqual(self.coverage(), "")


if __name__ == "__main__":
    unittest.main()
