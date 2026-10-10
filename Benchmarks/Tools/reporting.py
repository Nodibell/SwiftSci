"""One summary implementation for every worker language."""

import math
import statistics
from contracts import fields, positive, require, repository_file, verified_file, identity
from execution_policy import policy, validate_runtime, engine_order, PRODUCTION_MODE


def validate_worker(result, request):
    fields(
        result,
        [
            "schema_version",
            "case_key",
            "status",
            "samples",
            "peak_rss_bytes",
            "engine_version",
        ],
        ["error", "execution"],
    )
    require(
        result["schema_version"] == 1 and result["case_key"] == request["case_key"],
        "Worker identity mismatch",
    )
    require(result["status"] == "passed", result.get("error", "Worker did not pass"))
    if "execution_mode" in request:
        validate_runtime(result.get("execution"), request["execution_mode"], request.get("engine"))
    positive(result["peak_rss_bytes"], "peak_rss_bytes")
    require(
        isinstance(result["samples"], list)
        and len(result["samples"]) == request["samples"],
        "Worker sample count mismatch",
    )
    for sample in result["samples"]:
        fields(
            sample,
            ["elapsed_ns", "output_sha256", "maximum_absolute_error", "validated"],
            ["timing_resolved"],
        )
        elapsed = sample["elapsed_ns"]
        require(type(elapsed) is int and elapsed >= 0, "Invalid elapsed_ns")
        if "timing_resolved" in sample:
            require(
                sample["timing_resolved"] is (elapsed >= 1000),
                "Timing resolution flag disagrees with duration",
            )
        else:
            positive(elapsed, "elapsed_ns")
        require(sample["validated"] is True, "Unvalidated sample")
        require(
            isinstance(sample["output_sha256"], str)
            and len(sample["output_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in sample["output_sha256"]),
            "Missing output identity",
        )
        error = sample["maximum_absolute_error"]
        require(
            type(error) in (int, float) and math.isfinite(error) and error >= 0,
            "Invalid validation error",
        )


def summarize(run):
    groups = {}
    for event in run["events"]:
        key = (event["case_id"], event["engine"])
        groups.setdefault(key, []).append(event)
    output = []
    for (case, engine), events in groups.items():
        complete = (
            all(e["status"] == "passed" for e in events)
            and len(events) == run["plan"]["profile"]["batches"]
        )
        times = [
            s["elapsed_ns"]
            for e in events
            if e["status"] == "passed"
            for s in e["result"]["samples"]
        ]
        process_medians = [
            statistics.median(s["elapsed_ns"] for s in e["result"]["samples"])
            for e in events
            if e["status"] == "passed"
        ]
        median = statistics.median(process_medians) if complete else None
        output.append(
            dict(
                case_id=case,
                engine=engine,
                status="passed" if complete else "failed",
                median_ns=median,
                process_medians_ns=process_medians,
                process_min_ns=min(process_medians) if complete else None,
                process_max_ns=max(process_medians) if complete else None,
                process_stdev_ns=statistics.stdev(process_medians) if complete and len(process_medians) > 1 else None,
                raw_sample_count=len(times),
                maximum_absolute_error=max(
                    (
                        sample["maximum_absolute_error"]
                        for event in events
                        if event["status"] == "passed"
                        for sample in event["result"]["samples"]
                    ),
                    default=None,
                ),
                timing_resolved=complete and all(t >= 1000 for t in times),
                peak_rss_bytes=max(
                    (
                        e["result"]["peak_rss_bytes"]
                        for e in events
                        if e["status"] == "passed"
                    ),
                    default=None,
                ),
            )
        )
    return output


def audit_artifacts(event, profile, directory=None):
    records = event.get("artifacts", [])
    require(len(records) == profile["samples"], "Missing Parquet artifact evidence")
    token = f"{event['case_id']}-{event['engine']}-{event['batch']}"
    for index, record in enumerate(records, start=profile["warmups"]):
        expected_name = f"{token}.response.json.sample{index}.parquet"
        fields(record, ["file", "sha256", "bytes", "independently_validated"])
        require(record["file"] == expected_name, "Parquet artifact name mismatch")
        require(
            record["independently_validated"] is True,
            "Parquet artifact was not validated",
        )
        require(
            isinstance(record["sha256"], str)
            and len(record["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in record["sha256"]),
            "Invalid artifact digest",
        )
        positive(record["bytes"], "artifact bytes")
        if directory is not None:
            verified_file(
                repository_file(directory, record["file"]),
                record["sha256"],
                record["bytes"],
            )


def audit(run, directory=None):
    require(
        run.get("schema_version") == 1 and run.get("status") == "passed",
        "Run is incomplete or failed",
    )
    if "execution" in run["plan"]:
        execution = run["plan"]["execution"]
        require(execution == policy(execution.get("mode")), "Invalid execution policy")
        contract = run["plan"].get("contract", {})
        require(identity(contract) == run["plan"]["contract_hash"], "Contract checksum mismatch")
        require(contract.get("execution") == execution and
                contract.get("profile") == run["plan"]["profile"] and
                contract.get("cases") == run["plan"]["cases"] and
                contract.get("engine_order") == run["plan"].get("engine_order"), "Resolved contract mismatch")
        require(run["plan"]["profile"]["batches"] >= execution["minimum_process_rounds"],
                "Insufficient process rounds for execution mode")
    expected = {
        (c["case"]["id"], e, b): c["case_key"]
        for c in run["plan"]["cases"]
        for e in run["plan"]["engines"]
        for b in range(run["plan"]["profile"]["batches"])
    }
    cases = {c["case"]["id"]: c for c in run["plan"]["cases"]}
    observed = set()
    for event in run["events"]:
        key = (event["case_id"], event["engine"], event["batch"])
        require(
            key in expected and key not in observed,
            "Unexpected or duplicate case result",
        )
        require(
            event["status"] == "passed" and event["case_key"] == expected[key],
            "Failed or mismatched case",
        )
        validate_worker(
            event["result"],
            dict(case_key=expected[key], samples=run["plan"]["profile"]["samples"],
                 **(dict(execution_mode=run["plan"]["execution"]["mode"], engine=event["engine"])
                    if "execution" in run["plan"] else {})),
        )
        if (
            cases[event["case_id"]].get("workload", {}).get("operation")
            == "parquet-write"
        ):
            audit_artifacts(event, run["plan"]["profile"], directory)
        observed.add(key)
    require(observed == expected.keys(), "Incomplete run coverage")
    if "execution" in run["plan"]:
        # JSON objects are serialized with sorted keys; preserve the scheduled order separately.
        engines = run["plan"]["engine_order"]
        require(set(engines) == set(run["plan"]["engines"]) and len(engines) == len(set(engines)),
                "Invalid engine order")
        schedule = [(c["case"]["id"], e, b) for c in run["plan"]["cases"]
                    for b in range(run["plan"]["profile"]["batches"])
                    for e in engine_order(engines, b)]
        require([(e["case_id"], e["engine"], e["batch"]) for e in run["events"]] == schedule,
                "Engine execution order differs from policy")
    return True


def comparison(left, right):
    require(
        left["schema_version"] == right["schema_version"] == 1, "Unsupported run schema"
    )
    require(left["status"] == right["status"] == "passed", "Cannot compare failed runs")
    require(
        left["plan"]["contract_hash"] == right["plan"]["contract_hash"],
        "Experiment contracts differ",
    )
    require(
        left["plan"]["environment"] == right["plan"]["environment"],
        "Machine/toolchain settings differ",
    )
    require(left["plan"].get("execution") == right["plan"].get("execution"),
            "Execution modes differ")
    audit(left)
    audit(right)
    if left["plan"].get("execution", {}).get("mode") == PRODUCTION_MODE:
        for run in (left, right):
            require(run.get("power_start") == run.get("power_end") and run.get("power_unchanged") is True,
                    "Power settings changed during benchmark")
        require(left.get("power_start") == right.get("power_start"), "Power settings differ")
        def pools(run):
            result = {}
            for event in run["events"]:
                value = event["result"]["execution"]["configured_pool_sizes"]
                previous = result.setdefault(event["engine"], value)
                require(previous == value, "Engine pool settings changed during run")
            return result
        require(pools(left) == pools(right), "Configured engine pool sizes differ")
    for engine in left["plan"]["engines"]:
        require(engine in right["plan"]["engines"], "Engine coverage differs")
        if engine != "swiftsci":
            require(left["plan"]["engines"][engine].get("numerical_backend") == right["plan"]["engines"][engine].get("numerical_backend"), "Numerical backends differ")
            require(
                left["plan"]["engines"][engine]["version"]
                == right["plan"]["engines"][engine]["version"],
                "Reference engine versions differ",
            )
    a = {(v["case_id"], v["engine"]): v for v in summarize(left)}
    b = {(v["case_id"], v["engine"]): v for v in summarize(right)}
    require(a.keys() == b.keys(), "Run coverage differs")
    results = []
    for key in sorted(a):
        x, y = a[key], b[key]
        require(x["status"] == y["status"] == "passed", "Case not complete")
        results.append(
            dict(
                case_id=key[0],
                engine=key[1],
                baseline_ns=x["median_ns"],
                candidate_ns=y["median_ns"],
                speedup=x["median_ns"] / y["median_ns"]
                if x["timing_resolved"] and y["timing_resolved"]
                else None,
            )
        )
    return results
