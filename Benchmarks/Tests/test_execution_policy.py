import copy
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from test_contracts import ROOT
import test_runner
from contracts import ContractError, identity
from execution_policy import (LEGACY_MODE, PRODUCTION_MODE, THREAD_VARIABLES,
                              policy, worker_environment, engine_order, runtime_record)
from reporting import audit, comparison
from runner import plan


class ExecutionPolicyTests(unittest.TestCase):
    def test_defaults_remove_inherited_caps_before_child_imports(self):
        inherited = dict(os.environ, **{name: "1" for name in THREAD_VARIABLES})
        inherited["OMP_NUM_THREADS_EXTRA"] = "2"
        before = inherited.copy()
        environment = worker_environment(PRODUCTION_MODE, inherited)
        self.assertEqual(inherited, before)
        self.assertFalse(set(THREAD_VARIABLES) & environment.keys())
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);from execution_policy import runtime_record;print(json.dumps(runtime_record()))"
        import json
        record = json.loads(subprocess.check_output([sys.executable, "-c", code,
                            str(ROOT / "Benchmarks/Tools")], env=environment, text=True))
        self.assertEqual(record["mode"], PRODUCTION_MODE)
        self.assertEqual(record["thread_environment"], {})
        self.assertFalse(record["active_threads_measured"])

    def test_legacy_policy_preserves_requested_caps_without_claiming_one_core(self):
        environment = worker_environment(LEGACY_MODE, {})
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(runtime_record()["thread_environment"], policy(LEGACY_MODE)["thread_environment"])
        with self.assertRaises(ContractError):
            worker_environment("single-core")

    def test_native_plan_raises_rounds_without_mutating_profile(self):
        profile = dict(cases=[], batches=1)
        with (patch("runner.command", return_value="versions"),
              patch("runner.environment", return_value={}),
              patch("runner.source_identity", return_value={}),
              patch("runner.subprocess.check_output", return_value="{}")):
            native = plan(ROOT, profile, ["pandas"], None, sys.executable, PRODUCTION_MODE)
            legacy = plan(ROOT, profile, ["pandas"], None, sys.executable, LEGACY_MODE)
        self.assertEqual(profile["batches"], 1)
        self.assertEqual(native["profile"]["batches"], 5)
        self.assertEqual(legacy["profile"]["batches"], 1)
        self.assertNotEqual(native["contract_hash"], legacy["contract_hash"])
        self.assertEqual(identity(native["contract"]), native["contract_hash"])

    def test_rotation_visits_each_starting_position(self):
        engines = ["swiftsci", "pandas", "polars", "duckdb"]
        self.assertEqual([engine_order(engines, i)[0] for i in range(4)], engines)
        self.assertEqual(engine_order(engines, 4), engines)

    def modern_run(self):
        run = test_runner.AuditTests().passing_run()
        p = run["plan"]
        p["profile"]["batches"] = 5
        p["execution"] = policy(PRODUCTION_MODE)
        p["engine_order"] = ["swiftsci"]
        p["contract"] = {k: p[k] for k in ("execution", "profile", "cases", "engine_order")}
        p["contract_hash"] = identity(p["contract"])
        event = run["events"][0]
        event["result"]["execution"] = dict(mode=PRODUCTION_MODE, thread_environment={},
                      configured_pool_sizes={}, active_threads_measured=False)
        run["events"] = [dict(copy.deepcopy(event), batch=b) for b in range(5)]
        run.update(power_start={"source": "AC"}, power_end={"source": "AC"}, power_unchanged=True)
        return run

    def test_missing_mismatched_and_tampered_policy_cannot_pass_audit(self):
        good = self.modern_run()
        self.assertTrue(audit(good))
        self.assertEqual(comparison(good, good)[0]["speedup"], 1)
        mutations = [
            lambda r: r["events"][0]["result"].pop("execution"),
            lambda r: r["events"][0]["result"]["execution"].update(mode=LEGACY_MODE),
            lambda r: r["events"][0]["result"]["execution"]["thread_environment"].update(OMP_NUM_THREADS="1"),
            lambda r: r["plan"]["execution"].update(minimum_process_rounds=1),
            lambda r: r["events"].reverse(),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(good); mutate(bad)
            with self.assertRaises(ContractError):
                audit(bad)

    def test_comparison_rejects_power_or_pool_changes_and_legacy_evidence(self):
        good = self.modern_run()
        for change in ("power", "pool", "legacy"):
            bad = copy.deepcopy(good)
            if change == "power":
                bad["power_end"] = {"source": "battery"}
            elif change == "pool":
                bad["events"][0]["result"]["execution"]["configured_pool_sizes"] = {"example": 2}
            else:
                bad["plan"].pop("execution")
            with self.assertRaises(ContractError):
                comparison(good, bad)
