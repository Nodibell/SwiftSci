"""Strict versioned benchmark records; no fallback data or fuzzy case matching."""

import hashlib
import json
import math
import os
from pathlib import Path


class ContractError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(record):
    return digest(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def read_json(path):
    def reject(value):
        raise ContractError(f"Nonfinite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"Duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected an object")
    require(
        set(required) <= value.keys(), f"Missing fields: {set(required) - value.keys()}"
    )
    require(
        value.keys() <= set(required) | set(optional),
        f"Unknown fields: {value.keys() - set(required) - set(optional)}",
    )


def positive(value, name, minimum=1):
    require(
        type(value) is int and value >= minimum,
        f"{name} must be an integer >= {minimum}",
    )


def verified_file(path, sha256, size=None):
    data = Path(path).read_bytes()
    require(digest(data) == sha256, f"Checksum mismatch: {path}")
    if size is not None:
        require(len(data) == size, f"Size mismatch: {path}")
    return data


def repository_file(root, relative):
    require(isinstance(relative, str), "Path must be a string")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes repository")
    return path


def validate_values(actual, expected, atol, rtol):
    require(len(actual) == len(expected), "Output length mismatch")
    maximum = 0.0
    for index, (a, e) in enumerate(zip(actual, expected)):
        if math.isnan(e):
            require(math.isnan(a), f"Expected NaN at {index}")
        elif math.isinf(e):
            require(a == e, f"Infinity mismatch at {index}")
        else:
            require(math.isfinite(a), f"Nonfinite output at {index}")
            error = abs(a - e)
            require(
                error <= atol + rtol * abs(e), f"Output mismatch at {index}: {a} != {e}"
            )
            maximum = max(maximum, error)
    return maximum


WORKLOADS = {
    "ols-cpu", "nist-anova", "pca-cpu", "multinomial-nb-cpu",
    "h2o-q1", "h2o-q2", "h2o-q3", "h2o-q4", "h2o-q5", "wine-pipeline",
    "welch",
    "student",
    "paired",
    "anova",
    "regression-metrics",
    "roc-auc",
    "onehot",
    "tfidf",
    "sqlite-ingest",
    "rag-summary",
    "pool-dice",
    "parquet-read",
    "parquet-write",
    "csv-stream-read",
    "csv-stream-filter",
    "csv-stream-group",
    "group-sum-mean",
    "row-sum",
    "inner-join",
    "pearson",
    "spearman",
    "csv-read",
    "filter",
    "sort",
    "group-sum",
    "flat-matrix",
    "target",
    "standard-scale",
    "minmax-scale",
    "mean",
    "variance",
    "stddev",
}


def load_workload(root, name):
    value = read_json(root / "Benchmarks/Specs/workloads" / (name + ".json"))
    fields(
        value,
        [
            "schema_version",
            "id",
            "operation",
            "scope",
            "output",
            "atol",
            "rtol",
            "semantics",
        ],
    )
    require(
        value["schema_version"] == 1 and value["id"] == name,
        "Workload identity/version mismatch",
    )
    require(value["operation"] in WORKLOADS, "Unsupported operation")
    require(
        value["scope"] in ["ingestion", "operation", "fit-transform-export", "full-pipeline"],
        "Invalid timing scope",
    )
    for k in ["atol", "rtol"]:
        require(
            type(value[k]) in (int, float)
            and math.isfinite(value[k])
            and value[k] >= 0,
            "Invalid tolerance",
        )
    return value


def load_profile(root, name):
    require(
        name
        in [
            "smoke",
            "standard",
            "extended",
            "certification",
            "nist",
            "migration-smoke",
            "migration",
            "public-smoke",
            "public-data",
            "numerical-conformance",
            "numerical-binary64",
            "model-conformance",
        ],
        "Unknown profile",
    )
    value = read_json(root / "Benchmarks/Specs/profiles" / (name + ".json"))
    fields(
        value,
        [
            "schema_version",
            "id",
            "warmups",
            "samples",
            "batches",
            "timeout_seconds",
            "cases",
        ],
    )
    require(
        value["schema_version"] == 1 and value["id"] == name,
        "Profile identity/version mismatch",
    )
    for key in ["samples", "batches", "timeout_seconds"]:
        positive(value[key], key)
    positive(value["warmups"], "warmups", 0)
    require(isinstance(value["cases"], list) and value["cases"], "Empty profile")
    seen = set()
    for case in value["cases"]:
        fields(case, ["id", "dataset", "workload"])
        require(case["id"] not in seen, "Duplicate case ID")
        seen.add(case["id"])
        for key in ["id", "dataset", "workload"]:
            require(
                isinstance(case[key], str)
                and case[key]
                and all(c.isalnum() or c in "-_" for c in case[key]),
                "Invalid case identifier",
            )
    return value
