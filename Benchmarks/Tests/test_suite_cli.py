"""Unit tests for the unified `bench.py` CLI facade and `suite_cli` orchestration helpers."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
import suite_cli  # noqa: E402


class UnifiedBenchCLITests(unittest.TestCase):
    def test_catalog_covers_all_28_profiles_and_trials(self):
        catalog = suite_cli.build_catalog(ROOT)
        std = catalog["standardized"]
        self.assertEqual(std["profiles_count"], 28)
        self.assertEqual(len(std["tiers"]["cpu"]), 16)
        self.assertEqual(len(std["tiers"]["apple"]), 4)
        self.assertEqual(len(std["tiers"]["sweep"]), 8)
        self.assertEqual(catalog["scientific_validation"]["workloads_count"], 39)
        self.assertEqual(
            set(catalog["trials"].keys()),
            {"coreml", "bounded", "fused", "fused-coreml", "mixed", "persistent"},
        )

    def test_cli_list_and_doctor_json(self):
        bench = ROOT / "Benchmarks/Tools/bench.py"
        list_proc = subprocess.run(
            [sys.executable, str(bench), "list", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(list_proc.returncode, 0, msg=list_proc.stderr)
        listed = json.loads(list_proc.stdout)
        self.assertEqual(listed["standardized"]["profiles_count"], 28)

        doc_proc = subprocess.run(
            [sys.executable, str(bench), "doctor", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(doc_proc.returncode, 0, msg=doc_proc.stderr)
        doc = json.loads(doc_proc.stdout)
        for key in ("source", "worker", "python", "environments", "power", "ready"):
            self.assertIn(key, doc)

    def test_build_trial_command_maps_worker_and_forwards_flags(self):
        for name, rel in suite_cli.TRIAL_SCRIPTS.items():
            cmd = suite_cli.build_trial_command(
                ROOT,
                name,
                Path("/tmp/worker"),
                Path("/tmp/out"),
                extra_args=["--smoke", "--models", "/tmp/models"],
                python_exe="/tmp/python",
            )
            self.assertEqual(cmd[0], "/tmp/python")
            self.assertEqual(cmd[1], str(ROOT / rel))
            self.assertIn("--worker", cmd)
            self.assertNotIn("--swift-worker", cmd)
            self.assertEqual(cmd[cmd.index("--worker") + 1], "/tmp/worker")
            self.assertEqual(cmd[cmd.index("--output") + 1], "/tmp/out")
            self.assertEqual(cmd[-3:], ["--smoke", "--models", "/tmp/models"])

    def test_release_gate_dry_run_via_cli(self):
        bench = ROOT / "Benchmarks/Tools/bench.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(bench),
                "release-gate",
                "--dry-run",
                "--skip-scientific",
                "--output",
                "/tmp/swiftsci-nonexistent-dry-run",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["dry_run"])
        step_names = [s["name"] for s in payload["steps"]]
        self.assertEqual(
            step_names,
            ["build", "inventory", "unittest", "acceptance-cpu", "pretrained-controls", "workflow-controls"],
        )

    def test_run_release_gate_envelope_and_fail_closed_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_ok = Path(tmp) / "gate-ok"
            summary_ok, rc_ok = suite_cli.run_release_gate(
                ROOT,
                tier="cpu",
                swift_worker=Path("/tmp/fake-worker"),
                output_dir=out_ok,
                include_scientific=False,
                step_executor=lambda cmd, cwd, env: (0, 0.25),
            )
            self.assertEqual(rc_ok, 0)
            self.assertTrue(summary_ok["all_passed"])
            self.assertEqual(summary_ok["status"], "passed")
            written = json.loads((out_ok / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["command"], "release-gate")
            self.assertEqual(len(written["steps"]), 6)

            out_fail = Path(tmp) / "gate-fail"

            def fail_acceptance(cmd, cwd, env):
                if "acceptance.py" in cmd[1]:
                    return 2, 0.1
                return 0, 0.05

            summary_fail, rc_fail = suite_cli.run_release_gate(
                ROOT,
                tier="cpu",
                swift_worker=Path("/tmp/fake-worker"),
                output_dir=out_fail,
                include_scientific=False,
                step_executor=fail_acceptance,
            )
            self.assertEqual(rc_fail, 1)
            self.assertFalse(summary_fail["all_passed"])
            self.assertEqual(summary_fail["status"], "failed")
            failed_steps = [s for s in summary_fail["steps"] if s["status"] == "failed"]
            self.assertEqual([s["name"] for s in failed_steps], ["acceptance-cpu"])


if __name__ == "__main__":
    unittest.main()
