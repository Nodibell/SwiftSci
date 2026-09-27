"""Strict numerical fixture inputs and pinned independent expected values."""

import math
from contracts import fields, positive, require, read_json, repository_file, verified_file

OPERATIONS = {"ols-cpu", "nist-anova"}


def finite_vector(values, name):
    require(isinstance(values, list) and values, f"Empty or invalid {name}")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), f"Nonfinite {name}")


def validate_manifest(manifest):
    require(manifest.get("operation") in OPERATIONS, "Unknown numerical fixture operation")
    for key in ["fixture", "source_fixture", "reference_fixture"]:
        require(isinstance(manifest.get(key), str) and manifest[key], f"Missing {key}")
    require(manifest.get("reference_basis") in ("nist-decimal-v1", "binary64-v1"), "Unknown reference basis")
    sha = manifest.get("source_sha256", "")
    require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Invalid numerical source checksum")
    positive(manifest.get("source_size_bytes"), "source_size_bytes")
    sha = manifest.get("reference_sha256", "")
    require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Invalid reference checksum")
    positive(manifest.get("reference_size_bytes"), "reference_size_bytes")
    fields(manifest.get("tolerances"), ["atol", "rtol"])
    for v in manifest["tolerances"].values():
        require(type(v) in (int, float) and math.isfinite(v) and v >= 0, "Invalid numerical tolerance")


def validate_input(payload, operation, rows):
    if operation == "ols-cpu":
        fields(payload, ["operation", "features", "targets"])
        finite_vector(payload["targets"], "targets")
        matrix = payload["features"]
        require(isinstance(matrix, list) and len(matrix) == rows == len(payload["targets"]), "OLS row count mismatch")
        for row in matrix:
            finite_vector(row, "features")
        require(all(len(row) == len(matrix[0]) for row in matrix), "Ragged OLS features")
        require(rows >= len(matrix[0]) + 1, "Underdetermined OLS fixture")
    elif operation == "nist-anova":
        fields(payload, ["operation", "groups"])
        groups = payload["groups"]
        require(isinstance(groups, list) and len(groups) >= 2, "ANOVA needs groups")
        for group in groups:
            finite_vector(group, "ANOVA group")
        require(sum(map(len, groups)) == rows and rows > len(groups), "ANOVA row count mismatch")
    else:
        require(False, "Unsupported numerical fixture")
    require(payload["operation"] == operation, "Numerical input operation mismatch")
    return payload


def source_and_input(root, manifest):
    verified_file(repository_file(root, manifest["source_fixture"]), manifest["source_sha256"], manifest["source_size_bytes"])
    path = repository_file(root, manifest["fixture"])
    content = verified_file(path, manifest["sha256"], manifest["size_bytes"])
    validate_input(read_json(path), manifest["operation"], manifest["rows"])
    return content


def expected_values(root, manifest, payload):
    path = repository_file(root, manifest["reference_fixture"])
    verified_file(path, manifest["reference_sha256"], manifest["reference_size_bytes"])
    reference = read_json(path)
    require(reference["inputIdentity"] == {"sha256": manifest["sha256"], "bytes": manifest["size_bytes"]}, "Reference input identity mismatch")
    require(reference["source"]["sha256"] == manifest["source_sha256"], "Reference source mismatch")
    require(reference["model"]["observations"] == manifest["rows"], "Reference row count mismatch")
    if manifest["reference_basis"] == "nist-decimal-v1":
        values = list(reference["nistCertifiedValues"])
        derived = reference["independentDecimalReference"]
    else:
        derived = reference["binary64InputDiagnostic"]
        values = list(derived["values"])
    if manifest["operation"] == "ols-cpu":
        values += derived["predictions"]
        count = len(payload["features"][0]) + 2 + manifest["rows"]
    else:
        count = 3
    require(len(values) == count, "Numerical reference shape mismatch")
    values = [float(v) for v in values]
    finite_vector(values, "numerical answers")
    return values
