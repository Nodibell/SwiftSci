"""One summary implementation for every worker language."""

import math
import statistics
from contracts import fields, positive, require


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
        ["error"],
    )
    require(
        result["schema_version"] == 1 and result["case_key"] == request["case_key"],
        "Worker identity mismatch",
    )
    require(result["status"] == "passed", result.get("error", "Worker did not pass"))
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
        )
        positive(sample["elapsed_ns"], "elapsed_ns")
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
                raw_sample_count=len(times),
                timing_resolved=complete and median >= 1000,
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


def audit(run):
    require(
        run.get("schema_version") == 1 and run.get("status") == "passed",
        "Run is incomplete or failed",
    )
    expected = {
        (c["case"]["id"], e, b): c["case_key"]
        for c in run["plan"]["cases"]
        for e in run["plan"]["engines"]
        for b in range(run["plan"]["profile"]["batches"])
    }
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
            dict(case_key=expected[key], samples=run["plan"]["profile"]["samples"]),
        )
        observed.add(key)
    require(observed == expected.keys(), "Incomplete run coverage")
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
    audit(left)
    audit(right)
    for engine in left["plan"]["engines"]:
        require(engine in right["plan"]["engines"], "Engine coverage differs")
        if engine != "swiftsci":
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
