#!/usr/bin/env python3
"""Verify this sanitized evidence export without rerunning its benchmarks."""

import hashlib
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1] / "Tools"))
from contracts import require
from reporting import comparison, summarize


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    manifest = read(directory / "manifest.json")
    inventory = {str(p.relative_to(directory)): sha256(p)
                 for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    require(inventory == manifest["export_files"], "Export inventory or checksum mismatch")
    rows = read(directory / "measurements.json")
    profile_names = [p["profile"] for p in manifest["profiles"]]
    require(len(profile_names) == len(set(profile_names)) == 6, "Invalid profile inventory")
    require({r["profile"] for r in rows} == set(profile_names), "Summary profile mismatch")
    executions = samples = 0
    for entry in manifest["profiles"]:
        name = entry["profile"]
        folder = directory / name
        run = read(folder / "metadata.json")
        run["events"] = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        run["summary"] = [{k: v for k, v in row.items() if k != "profile"}
                          for row in rows if row["profile"] == name]
        comparison(run, run)
        require(summarize(run) == run["summary"], f"Summary mismatch: {name}")
        plan = run["plan"]
        require(plan["source"] == manifest["measured_source"], f"Source mismatch: {name}")
        build_source = plan["engines"]["swiftsci"]["build"]["source"]
        for key in ("files", "tree_sha256"):
            require(build_source[key] == plan["source"][key], f"Build source mismatch: {name}")
        require(plan["execution"]["mode"] == "production-default", f"Wrong mode: {name}")
        require([plan["profile"][key] for key in ("batches", "warmups", "samples")] == [5, 2, 5],
                f"Sampling policy mismatch: {name}")
        certificate = read(folder / "certificate.original.json")
        original = manifest["original_artifacts"][name]
        require(sha256(folder / "certificate.original.json") == original["certificate_sha256"],
                f"Original certificate checksum mismatch: {name}")
        require(certificate["run_sha256"] == original["run_sha256"], f"Original run identity mismatch: {name}")
        require(certificate["source"] == plan["source"] and certificate["status"] == "passed",
                f"Original certificate source or status mismatch: {name}")
        require(certificate["contract_hash"] == entry["contract_hash"] == plan["contract_hash"],
                f"Contract mismatch: {name}")
        count = sum(len(e["result"]["samples"]) for e in run["events"])
        require(count == entry["samples"] == certificate["validated_samples"], f"Sample count mismatch: {name}")
        require(len(run["events"]) == entry["executions"] and entry["engines"] == plan["engine_order"],
                f"Execution inventory mismatch: {name}")
        executions += len(run["events"])
        samples += count
    require(manifest["totals"] == dict(executions=executions, samples=samples, summaries=len(rows),
                                      unresolved_summaries=sum(not r["timing_resolved"] for r in rows)),
            "Suite totals mismatch")
    acceptance = read(directory / "cpu-acceptance.json")
    policy = read(directory / "cpu-regression-policy.json")
    for filename in ("cpu-acceptance.json", "cpu-regression-policy.json"):
        require(sha256(directory / filename) == manifest["original_artifacts"][filename],
                f"Original CPU record checksum mismatch: {filename}")
    require(policy["acceptance_sha256"] == sha256(directory / "cpu-acceptance.json"), "CPU policy input mismatch")
    require(policy["source"] == acceptance["source"] == manifest["measured_source"], "CPU source mismatch")
    require(policy["status"] == "passed" and policy["blocking"] == [], "CPU policy failed")
    require(policy["conformance_status"] == acceptance["status"] == "failed", "Raw conformance status changed")
    require(acceptance["coverage_complete"] is True, "Incomplete CPU coverage")
    failures = []
    for profile in acceptance["profiles"]:
        require(profile["coverage_complete"] and not profile["infrastructure_errors"], "Incomplete CPU profile")
        require(profile["passed"] + len(profile["failures"]) == profile["engine_cases"], "CPU counts mismatch")
        failures.extend(dict(profile=profile["profile"], **f) for f in profile["failures"])
    require(len(failures) == len(policy["known_failures"]) == 4, "CPU failure inventory changed")
    for failure, known in zip(failures, policy["known_failures"]):
        require(all(known[k] == v for k, v in failure.items()), "Retained CPU failure mismatch")
    print(f"PASS: {executions:,} executions, {samples:,} samples, {len(rows)} recomputed summaries.")
    print("CPU regression policy passed; four recorded conformance failures remain failed.")
    print("Original certificates bind unredacted runs. This export does not repeat numerical or Parquet file validation.")


if __name__ == "__main__":
    verify(DIRECTORY)
