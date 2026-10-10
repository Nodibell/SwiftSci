#!/usr/bin/env python3
"""Run controlled Core ML inference comparisons in bounded fresh processes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys

POLICIES = ("coreml-cpu", "coreml-gpu", "coreml-neural", "coreml-all")
MODES = ("native-cpu", "native-gpu", *POLICIES)
WORKLOADS = ("linear128", "mlp128", "mlp512")


def plan_cases(sweep, row_counts=(1, 32, 256)):
    if not sweep:
        return [("linear128", 256, 1, mode) for mode in MODES]
    cases = []
    for workload in WORKLOADS:
        for rows in row_counts:
            if workload == "linear128":
                cases.extend((workload, rows, 1, mode) for mode in ("native-cpu", "native-gpu"))
            cases.extend((workload, rows, 1, mode) for mode in ("mlx-cpu", "mlx-gpu"))
            for batch in sorted({1, min(32, rows)}):
                cases.extend((workload, rows, batch, mode) for mode in POLICIES)
            for suffix in ("-matrix", "-matrix-adapter"):
                cases.extend((workload, rows, rows, mode + suffix) for mode in POLICIES)
    return cases


def validate_record(record, case, samples):
    workload, rows, batch, mode = case
    values = [record["firstPrediction"], *record["warmPredictions"]]
    identity = (record["workload"], record["rows"], record["maximumBatchSize"], record["mode"])
    if identity != case or len(values) != samples + 1:
        raise RuntimeError(f"Incomplete validation for {case}")
    warmup_exceedances = record.get("warmupFailures", 0)
    if type(warmup_exceedances) is not int or not 0 <= warmup_exceedances <= 4:
        raise RuntimeError("Invalid warmup accuracy evidence")
    outputs = record.get("outputColumns", 1)
    if type(outputs) is not int or not 1 <= outputs <= 1024:
        raise RuntimeError("Invalid output column count")
    for sample in values:
        matched = sample.get("validated")
        mismatches = sample.get("mismatchedValues")
        if (type(matched) is not bool or type(mismatches) is not int
                or not 0 <= mismatches <= rows * outputs or matched != (mismatches == 0)):
            raise RuntimeError("Inconsistent reference-accuracy evidence")
        for field in ("maximum_absolute_error", "elapsed_ns"):
            value = sample.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid measurement: {field}")
    matrix_input = mode.endswith(("-matrix", "-matrix-adapter"))
    if record["matrixInput"] != matrix_input or record["executionTraceVerified"]:
        raise RuntimeError("Unexpected matrix contract or unsubstantiated execution trace")
    if mode.endswith("-matrix-adapter") and record.get("publicAdapter") is not True:
        raise RuntimeError("Matrix adapter result did not use the public adapter")
    if record["reservedBytesAfterPrediction"] != 0:
        raise RuntimeError(f"Admission reservation retained for {case}")
    if len(record["residentBytesPerSample"]) != samples:
        raise RuntimeError("Missing per-sample resident measurements")


def validate_comparability(records):
    groups = {}
    for record in records:
        group = (record["workload"], record["rows"])
        identity = (record["parameterSHA256"], record["inputSHA256"])
        if groups.setdefault(group, identity) != identity:
            raise RuntimeError("Compared paths used different parameters or inputs")
    # Rank-one and rank-two artifacts intentionally differ. Within each schema they must match.
    models = {}
    for record in records:
        group = (record["workload"], record["rows"], record["matrixInput"])
        if models.setdefault(group, record["modelSHA256"]) != record["modelSHA256"]:
            raise RuntimeError("Model artifacts differ within the same shape contract")


def run_case(worker, output, case, samples, env):
    workload, rows, batch, mode = case
    stem = f"{workload}-r{rows}-b{batch}-{mode}"
    result_path = output / f"{stem}.json"
    env = dict(env, SWIFTSCI_COREML_SAMPLES=str(samples), SWIFTSCI_COREML_WORKLOAD=workload,
               SWIFTSCI_COREML_ROWS=str(rows), SWIFTSCI_COREML_BATCH=str(batch))
    with (output / f"{stem}.log").open("w") as log:
        process = subprocess.Popen([str(worker), "--coreml-qualification", mode, str(result_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=120)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    if code not in (0, 2):
        raise RuntimeError(f"{stem} failed with exit {code}; see {stem}.log")
    record = json.loads(result_path.read_text())
    validate_record(record, case, samples)
    passed = reference_accuracy_met(record)
    if (code == 0) != passed:
        raise RuntimeError("Worker exit code disagrees with its reference-accuracy evidence")
    return record


def reference_accuracy_met(record):
    return record.get("warmupFailures", 0) == 0 and all(
        s["validated"] for s in [record["firstPrediction"], *record["warmPredictions"]])


def completion_exit_code(records, require_reference_accuracy):
    return 2 if require_reference_accuracy and not all(reference_accuracy_met(r) for r in records) else 0


def assessment(records):
    return {
        "reference_contract": {
            "id": "double-reference-1e-5",
            "reference": "independent scalar Double",
            "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-5,
        },
        "cases": [{
            "workload": r["workload"], "rows": r["rows"],
            "maximumBatchSize": r["maximumBatchSize"], "mode": r["mode"],
            "execution": "completed",
            "reference_accuracy": "within_tolerance" if reference_accuracy_met(r) else "outside_tolerance",
            "application_suitability": "not_assessed",
        } for r in records],
    }


def report(records):
    lines = ["# Core ML inference qualification", "",
             "Controlled fixed-weight inference. Each table entry runs in a fresh process.", "",
             "| Workload | Rows | Maximum batch | Path | Double-reference accuracy | First (ms) | Warm median (ms) | Rows/s | Warm RSS (MiB) | Final RSS (MiB) | Max error |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        warm = record["warmPredictions"]
        median = statistics.median(s["elapsed_ns"] for s in warm) / 1e6
        error = max(s["maximum_absolute_error"] for s in [record["firstPrediction"], *warm])
        throughput = f'{record["rows"] * 1000 / median:.0f}' if median >= 0.001 else 'unresolved'
        warm_text = f'{median:.3f}' if median >= 0.001 else '<0.001'
        first_ms = record["firstPrediction"]["elapsed_ns"] / 1e6
        first_text = f'{first_ms:.3f}' if first_ms >= 0.001 else '<0.001'
        passed = reference_accuracy_met(record)
        timing = f'{first_text} | {warm_text} | {throughput}'
        status = "Within tolerance" if passed else "Outside tolerance"
        lines.append(f'| {record["workload"]} | {record["rows"]} | {record["maximumBatchSize"]} | {record["mode"]} | '
                     f'{status} | {timing} | '
                     f'{record["residentBytesAfterWarmup"] / 2**20:.2f} | {record["residentBytesAfterSamples"] / 2**20:.2f} | {error:.3g} |')
    lines += ["", "Every listed case completed execution and structural checks. Accuracy describes agreement with the Double reference, including warmups. Exceeding this requirement alone does not establish an integration or hardware defect. Timings are measurements at the observed accuracy, not claims of equivalent numerical quality. Application suitability is not assessed.", "", "## Interpretation", "",
              "Native paths use public LinearRegression for the linear workload. MLX paths are benchmark-only fixed-weight matrix references, not SwiftSci's public MLP estimators.", "",
              "Core ML example paths submit rows or bounded example batches through CoreMLPredictor. Paths ending in `-matrix-adapter` submit one fixed matrix through the public adapter; `-matrix` paths call Core ML directly as references. Both matrix paths use the same rank-two artifact. Their Maximum batch column reports matrix rows; the public matrix API leaves maximumBatchSize at 1.", "",
              "Packing, execution, synchronization needed for output, and complete result extraction are timed. Model loading and the independent scalar oracle are outside timing. The double-reference-1e-5 contract uses abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected) for every path. No reduced-precision application contract is inferred from these fixtures.", "",
              "Model parameters and inputs are identical within each workload and row count. Float32 MLX arithmetic and Core ML's selected internal precision need not match native Double arithmetic bit for bit.", "",
              "RSS includes caches and framework allocations. Stable samples do not prove absence of leaks. Process peak RSS includes compilation. Public example and matrix adapter paths use memory admission; zero reserved bytes on a reference path is not an admission test.", "",
              "Compute plans describe anticipated placement. Actual Neural Engine execution remains unverified. These results are qualification evidence, not a production performance baseline.", "",
              "## Planned placement", ""]
    for record in records:
        lines.append(f'- {record["workload"]}, {record["rows"]} rows, batch {record["maximumBatchSize"]}, {record["mode"]}: {"; ".join(record["plannedDevices"])}')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32, choices=range(1, 513), metavar="1..512")
    parser.add_argument("--sweep", action="store_true", help="Include bounded batches, matrix models, and two multilayer networks")
    parser.add_argument("--rows", nargs="+", type=int, choices=range(1, 1025), metavar="1..1024",
                        help="Override row counts for --sweep")
    parser.add_argument("--require-reference-accuracy", action="store_true",
                        help="Exit 2 if any result exceeds the unchanged Double-reference tolerance")
    args = parser.parse_args()
    if args.rows and not args.sweep:
        parser.error("--rows requires --sweep")
    worker = args.worker.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "Benchmarks/Tools"))
    from runner import source_identity, verify_uninstrumented, metal_build_record
    source = source_identity(root)
    build = json.loads(Path(str(worker) + ".build.json").read_text())
    if build["binary_sha256"] != hashlib.sha256(worker.read_bytes()).hexdigest():
        raise RuntimeError("Worker differs from its recorded build")
    if build["source"]["tree_sha256"] != source["tree_sha256"]:
        raise RuntimeError("Worker source differs from this checkout; rebuild")
    verify_uninstrumented(worker)
    if build["metal"] != metal_build_record(worker):
        raise RuntimeError("Metal build settings differ from the recorded build")
    env = os.environ.copy()
    env.update(VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cases = plan_cases(args.sweep, tuple(dict.fromkeys(args.rows)) if args.rows else (1, 32, 256))
    records = []
    for index, case in enumerate(cases, 1):
        print(f"{index}/{len(cases)}: {case}", flush=True)
        records.append(run_case(worker, output, case, args.samples, env))
    validate_comparability(records)
    if source_identity(root) != source:
        raise RuntimeError("Sources changed during qualification; discard and rerun")
    source_files = subprocess.check_output(["git", "ls-files", "Sources", "Benchmarks/Worker", "Package.swift", "Package.resolved"], cwd=root, text=True).splitlines()
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}
    metadata = {
        "worker": str(worker), "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "source_files": source_hashes, "platform": platform.platform(), "cases": cases,
        "xcode": subprocess.check_output(["xcodebuild", "-version"], text=True).strip(),
        "swift": subprocess.check_output(["swift", "--version"], text=True).strip(),
        "threads": {k: env[k] for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "build": build, "require_reference_accuracy": args.require_reference_accuracy, "qualification_only": True, "formal_performance_baseline": False,
    }
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "assessment.json").write_text(json.dumps(assessment(records), indent=2) + "\n")
    (output / "REPORT.md").write_text(report(records))
    print(output / "REPORT.md")
    raise SystemExit(completion_exit_code(records, args.require_reference_accuracy))


if __name__ == "__main__":
    main()
