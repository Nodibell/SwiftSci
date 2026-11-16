import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from test_contracts import ROOT
from contracts import *
from datasets import *
from runner import build, metal_build_record, plan, run, source_identity
from reporting import summarize, comparison


class RunnerTests(unittest.TestCase):
    def make_plan(self):
        profile = dict(
            schema_version=1,
            id="smoke",
            warmups=0,
            samples=1,
            batches=1,
            timeout_seconds=1,
            cases=[dict(id="tiny-target", dataset="tiny", workload="target-v1")],
        )
        dataset = load_manifest(ROOT, "tiny")
        workload = load_workload(ROOT, "target-v1")
        return dict(
            schema_version=1,
            profile=profile,
            cases=[
                dict(
                    case=profile["cases"][0],
                    dataset=dataset,
                    workload=workload,
                    case_key="a" * 64,
                )
            ],
            engines={},
            contract_hash="c" * 64,
            environment={},
        )

    def test_crash_timeout_missing_and_wrong_response_cannot_certify(self):
        scripts = [
            "raise SystemExit(7)",
            "import time;time.sleep(5)",
            "pass",
            'import json,sys;json.dump({},open(sys.argv[2],"w"))',
        ]
        for script in scripts:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                p = self.make_plan()
                prepare(root, p["cases"][0]["dataset"])
                worker = root / "fake.py"
                worker.write_text(script)
                p["engines"] = {"fake": dict(command=[sys.executable, str(worker)])}
                p["source"] = {}
                with patch("builtins.print"):
                    result = run(root, p, root / "run")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(
                    read_json(root / "run/certificate.json")["status"], "failed"
                )
                self.assertIsNone(summarize(result)[0]["median_ns"])

    def test_incomplete_batches_are_not_passed(self):
        p = self.make_plan()
        p["profile"]["batches"] = 2
        result = dict(
            schema_version=1,
            status="passed",
            plan=p,
            events=[
                dict(
                    case_id="tiny-target",
                    engine="x",
                    status="passed",
                    result=dict(
                        samples=[dict(elapsed_ns=10, maximum_absolute_error=0)],
                        peak_rss_bytes=1,
                    ),
                )
            ],
        )
        self.assertEqual(summarize(result)[0]["status"], "failed")


class AuditTests(unittest.TestCase):
    def passing_run(self):
        p = RunnerTests().make_plan()
        p["engines"] = {"swiftsci": {}}
        result = dict(
            schema_version=1,
            case_key="a" * 64,
            status="passed",
            peak_rss_bytes=100,
            engine_version="test",
            samples=[
                dict(
                    elapsed_ns=2000,
                    output_sha256="b" * 64,
                    maximum_absolute_error=0,
                    validated=True,
                )
            ],
        )
        event = dict(
            case_id="tiny-target",
            case_key="a" * 64,
            engine="swiftsci",
            batch=0,
            status="passed",
            result=result,
        )
        return dict(schema_version=1, status="passed", plan=p, events=[event])

    def test_complete_run_compares_but_missing_duplicate_or_mismatched_cases_fail(self):
        from reporting import audit

        good = self.passing_run()
        self.assertTrue(audit(good))
        self.assertEqual(comparison(good, good)[0]["speedup"], 1)
        for events in [[], good["events"] * 2]:
            bad = copy.deepcopy(good)
            bad["events"] = events
            with self.assertRaises(ContractError):
                audit(bad)
        for field, value in [
            ("engine", "unknown"),
            ("batch", 1),
            ("case_key", "z" * 64),
            ("status", "failed"),
        ]:
            bad = copy.deepcopy(good)
            bad["events"][0][field] = value
            with self.assertRaises(ContractError):
                audit(bad)

    def test_summary_retains_worst_measured_absolute_error(self):
        good = self.passing_run()
        good["plan"]["profile"]["batches"] = 2
        second = copy.deepcopy(good["events"][0])
        second["batch"] = 1
        good["events"][0]["result"]["samples"][0]["maximum_absolute_error"] = 1e-9
        second["result"]["samples"][0]["maximum_absolute_error"] = 3e-9
        good["events"].append(second)
        self.assertEqual(summarize(good)[0]["maximum_absolute_error"], 3e-9)

    def test_unresolved_timing_has_no_speedup(self):
        good = self.passing_run()
        good["events"][0]["result"]["samples"][0]["elapsed_ns"] = 10
        self.assertIsNone(comparison(good, good)[0]["speedup"])

    def test_explicit_unresolved_zero_is_accurate_but_never_a_speedup(self):
        from reporting import audit

        good = self.passing_run()
        sample = good["events"][0]["result"]["samples"][0]
        for elapsed in [0, 999, 1000]:
            sample.update(elapsed_ns=elapsed, timing_resolved=elapsed >= 1000)
            self.assertTrue(audit(good))
            summary = summarize(good)[0]
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["median_ns"], elapsed)
            self.assertEqual(summary["timing_resolved"], elapsed >= 1000)
            self.assertEqual(
                comparison(good, good)[0]["speedup"], 1 if elapsed >= 1000 else None
            )
        for elapsed, resolved in [
            (0, True),
            (1000, False),
            (-1, False),
            (True, False),
            (0.0, False),
        ]:
            sample.update(elapsed_ns=elapsed, timing_resolved=resolved)
            with self.assertRaises(ContractError):
                audit(good)

    def test_unresolved_sample_cannot_hide_behind_resolved_median(self):
        good = self.passing_run()
        good["plan"]["profile"]["samples"] = 3
        samples = good["events"][0]["result"]["samples"]
        samples.extend([copy.deepcopy(samples[0]), copy.deepcopy(samples[0])])
        samples[0].update(elapsed_ns=0, timing_resolved=False)
        self.assertEqual(summarize(good)[0]["median_ns"], 2000)
        self.assertIsNone(comparison(good, good)[0]["speedup"])

    def test_environment_changes_refuse_comparison(self):
        good = self.passing_run()
        other = copy.deepcopy(good)
        other["plan"]["environment"] = {"machine": "different"}
        with self.assertRaises(ContractError):
            comparison(good, other)


class InstrumentationTests(unittest.TestCase):
    def test_coverage_sections_reject_worker_even_when_build_settings_claim_no_coverage(
        self,
    ):
        from runner import verify_uninstrumented

        for name in [
            "__llvm_prf_cnts",
            "__llvm_prf_data",
            "__llvm_covmap",
            "__llvm_covfun",
        ]:
            with (
                self.subTest(section=name),
                patch(
                    "runner.command",
                    return_value=f"  sectname __text\n  sectname {name}\n",
                ),
            ):
                with self.assertRaises(ContractError):
                    verify_uninstrumented(Path("worker"))
        with patch(
            "runner.command", return_value="  sectname __text\n  sectname __cstring\n"
        ):
            verify_uninstrumented(Path("worker"))
        with patch("runner.command", return_value=""):
            with self.assertRaises(ContractError):
                verify_uninstrumented(Path("worker"))


class MetalProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.products = Path(self.temp.name)
        self.worker = self.products / "SwiftSciBenchmarkWorker"
        self.worker.write_bytes(b"worker")
        self.library = self.products / "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"compiled safe math")
        self.record = dict(
            binary_sha256=digest(self.worker.read_bytes()),
            coverage_instrumentation="absent",
            environment={},
            source={"tree_sha256": "source"},
            metal=dict(
                build_settings={"MTL_FAST_MATH": "NO", "MTL_MATH_MODE": "SAFE",
                                "MTL_MATH_FP32_FUNCTIONS": "PRECISE"},
                libraries={str(self.library.relative_to(self.products)):
                           digest(self.library.read_bytes())},
            ),
        )

    def resolve(self):
        from runner import plan

        write_json(Path(str(self.worker) + ".build.json"), self.record)
        with (
            patch("runner.verify_uninstrumented"),
            patch("runner.environment", return_value={}),
            patch("runner.source_identity", return_value={"tree_sha256": "source"}),
        ):
            return plan(ROOT, {"cases": []}, ["swiftsci"], self.worker, sys.executable)

    def test_missing_or_fast_math_provenance_requires_rebuild(self):
        original = copy.deepcopy(self.record)
        for mutation in ["missing", "fast", "missing_libraries"]:
            with self.subTest(mutation=mutation):
                self.record = copy.deepcopy(original)
                if mutation == "missing":
                    self.record.pop("metal")
                elif mutation == "fast":
                    self.record["metal"]["build_settings"]["MTL_FAST_MATH"] = "YES"
                else:
                    self.record["metal"].pop("libraries")
                with self.assertRaisesRegex(ContractError, "Metal.*rebuild"):
                    self.resolve()

    def test_replaced_removed_or_added_metal_library_requires_rebuild(self):
        extra = self.products / "other.bundle/Contents/Resources/default.metallib"
        extra.parent.mkdir(parents=True)
        for mutation in ["replaced", "removed", "added"]:
            with self.subTest(mutation=mutation):
                self.library.write_bytes(b"compiled safe math")
                if mutation == "replaced":
                    self.library.write_bytes(b"compiled fast math")
                elif mutation == "removed":
                    self.library.unlink()
                else:
                    extra.write_bytes(b"new library")
                with self.assertRaisesRegex(ContractError, "Metal.*rebuild"):
                    self.resolve()
                if extra.exists():
                    extra.unlink()

    def test_matching_metal_artifacts_are_included_in_run_fingerprint(self):
        result = self.resolve()
        self.assertEqual(result["engines"]["swiftsci"]["build"]["metal"], self.record["metal"])

class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=self.root, check=True)
        self.source = self.root / "Sources/Example.swift"
        self.source.parent.mkdir()
        self.original = b"let value = 1\n"
        self.source.write_bytes(self.original)
        self.git("add", "--", "Sources/Example.swift")
        self.git("commit", "--quiet", "--no-gpg-sign", "-m", "Fixture source")

        self.products = Path(temp.name) / "build/derived"
        self.worker = self.products / "Build/Products/Release/SwiftSciBenchmarkWorker"
        self.record_path = self.worker.with_suffix(".build.json")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.root,
            text=True,
        ).strip()

    def test_clean_worktree_records_commit_and_source_hashes(self):
        ident = source_identity(self.root)
        expected_files = {"Sources/Example.swift": digest(self.original)}
        self.assertEqual(ident["commit"], self.git("rev-parse", "HEAD"))
        self.assertFalse(ident["dirty"])
        self.assertEqual(ident["dirty_entries"], [])
        self.assertEqual(ident["files"], expected_files)
        self.assertEqual(ident["tree_sha256"], identity(expected_files))

    def test_modified_tracked_source_changes_fingerprint(self):
        before = source_identity(self.root)
        changed = b"let value = 2\n"
        self.source.write_bytes(changed)
        after = source_identity(self.root)
        self.assertEqual(after["commit"], before["commit"])
        self.assertTrue(after["dirty"])
        self.assertEqual(after["dirty_entries"], ["M Sources/Example.swift"])
        self.assertEqual(after["files"], {"Sources/Example.swift": digest(changed)})
        self.assertNotEqual(after["tree_sha256"], before["tree_sha256"])

    def test_untracked_source_is_included_in_fingerprint(self):
        before = source_identity(self.root)
        added = b"let added = 3\n"
        (self.root / "Sources/Added.swift").write_bytes(added)
        after = source_identity(self.root)
        self.assertEqual(after["commit"], before["commit"])
        self.assertTrue(after["dirty"])
        self.assertEqual(after["dirty_entries"], ["?? Sources/Added.swift"])
        self.assertEqual(after["files"], {**before["files"], "Sources/Added.swift": digest(added)})
        self.assertNotEqual(after["tree_sha256"], before["tree_sha256"])

    def test_source_changed_during_build_cannot_be_recorded(self):
        real_run = subprocess.run

        def build_worker(args, **kwargs):
            if args[0] != "xcodebuild":
                return real_run(args, **kwargs)
            self.assertEqual((kwargs["cwd"] / "Sources/Example.swift").read_bytes(), self.original)
            self.source.write_bytes(b"let value = 2\n")
            self.worker.parent.mkdir(parents=True)
            self.worker.write_bytes(b"worker")

        with (
            patch("runner.subprocess.run", side_effect=build_worker),
            patch("runner.verify_uninstrumented"),
            self.assertRaisesRegex(ContractError, "Source changed during build; rebuild"),
        ):
            build(self.root, self.products, self.products.parent / "packages")
        self.assertFalse(self.record_path.exists())

    def test_plan_rejects_worker_built_from_different_source_fingerprint(self):
        shutil.copytree(ROOT / "Benchmarks/Tools", self.root / "Benchmarks/Tools",
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.git("add", "--", "Benchmarks/Tools")
        self.git("commit", "--quiet", "--no-gpg-sign", "-m", "Fixture benchmark inputs")
        self.worker.parent.mkdir(parents=True)
        self.worker.write_bytes(b"worker")
        library = self.worker.parent / "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"compiled safe math")
        original_source = source_identity(self.root)
        write_json(self.record_path, dict(
            binary_sha256=digest(self.worker.read_bytes()),
            coverage_instrumentation="absent",
            environment={},
            source=original_source,
            metal=metal_build_record(self.worker),
        ))

        with patch("runner.verify_uninstrumented"), patch("runner.environment", return_value={}):
            resolved = plan(self.root, {"cases": []}, ["swiftsci"], self.worker, sys.executable)
            self.assertEqual(resolved["source"], original_source)
            self.source.write_bytes(b"let value = 2\n")
            with self.assertRaisesRegex(ContractError, "Swift worker source is stale; rebuild"):
                plan(self.root, {"cases": []}, ["swiftsci"], self.worker, sys.executable)


if __name__ == "__main__":
    unittest.main()
