"""Strict numerical fixture inputs and pinned independent expected values."""

from workflow_fixtures import OPERATIONS as WORKFLOW_OPERATIONS, validate_input as validate_workflow
from workflow_reference import reference as workflow_reference
import math
import re
from contracts import fields, positive, require, read_json, repository_file, verified_file

from controlled_fixtures import OPERATIONS as CONTROLLED_OPERATIONS, validate_input as validate_controlled, output_count

from supervised_fixtures import OPERATIONS as SUPERVISED_OPERATIONS, validate_input as validate_supervised, output_count as supervised_output_count

from dataframe_fixtures import validate_input as validate_dataframe, expected_values as dataframe_expected

from neural_shapes import validate_input as validate_shaped, output_count as shaped_output_count

from neural_fixtures import validate_input as validate_neural, output_count as neural_output_count

from vision_fixtures import validate_input as validate_vision, output_count as vision_output_count

from boundary_fixtures import validate_input as validate_boundary, output_count as boundary_output_count
from boundary_sweep import validate_descriptor, reference as sweep_reference, output_count as sweep_output_count
from contracts import digest
import struct

OPERATIONS = {"dataframe-model", "dataframe-model-sweep", "vision-letterbox-cpu", "decoder-fixed-f32", "decoder-shaped-f32", "dataframe-semantics", "ols-cpu", "nist-anova", "nist-anova-decimal", "pca-cpu", "multinomial-nb-cpu"} | CONTROLLED_OPERATIONS | SUPERVISED_OPERATIONS | WORKFLOW_OPERATIONS


def finite_vector(values, name):
    require(isinstance(values, list) and values, f"Empty or invalid {name}")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), f"Nonfinite {name}")


def validate_manifest(manifest):
    require(manifest.get("operation") in OPERATIONS, "Unknown numerical fixture operation")
    for operation, basis in [("dataframe-model", "boundary-reference-v1"),("dataframe-model-sweep", "sweep-reference-v1")]:
        require((manifest.get("operation") == operation) == (manifest.get("reference_basis") == basis), "Boundary reference basis mismatch")
        if manifest.get("operation") == operation:
            require(manifest.get("tolerances") in ({"atol":2e-5,"rtol":0},{"atol":1e-12,"rtol":0}), "Boundary tolerance contract mismatch")
    workflow = manifest.get("operation") in WORKFLOW_OPERATIONS
    require(workflow == (manifest.get("reference_basis") == "workflow-reference-v1"), "Workflow reference basis mismatch")
    if workflow:
        require(manifest.get("tolerances") == {"atol":1e-8,"rtol":1e-10}, "Workflow tolerance contract mismatch")
    vision = manifest.get("operation") == "vision-letterbox-cpu"
    require(vision == (manifest.get("reference_basis") == "vision-reference-v1"), "Vision reference basis mismatch")
    if vision:
        require(manifest.get("tolerances") == {"atol": 2e-6, "rtol": 2e-6}, "Vision tolerance contract mismatch")
    shaped = manifest.get("operation") == "decoder-shaped-f32"
    require(shaped == (manifest.get("reference_basis") == "neural-shaped-reference-v1"), "Shaped reference basis mismatch")
    if shaped:
        require(manifest.get("tolerances") == {"atol": 2e-5, "rtol": 2e-5}, "Shaped tolerance contract mismatch")
    neural = manifest.get("operation") == "decoder-fixed-f32"
    require(neural == (manifest.get("reference_basis") == "neural-reference-v1"), "Neural reference basis mismatch")
    if neural:
        require(manifest.get("tolerances") == {"atol": 2e-5, "rtol": 2e-5}, "Neural tolerance contract mismatch")
    dataframe = manifest.get("operation") == "dataframe-semantics"
    require(dataframe == (manifest.get("reference_basis") == "dataframe-reference-v1"), "Dataframe reference basis mismatch")
    if dataframe:
        require(manifest.get("tolerances") == {"atol": 0, "rtol": 0}, "Dataframe encoding requires exact comparison")
    supervised = manifest.get("operation") in SUPERVISED_OPERATIONS
    require(supervised == (manifest.get("reference_basis") == "supervised-reference-v1"), "Supervised reference basis mismatch")
    controlled = manifest.get("operation") in CONTROLLED_OPERATIONS
    require(controlled == (manifest.get("reference_basis") == "controlled-reference-v1"), "Controlled reference basis mismatch")
    exact = manifest.get("operation") in ("pca-cpu", "multinomial-nb-cpu")
    require(exact == (manifest.get("reference_basis") == "exact-rational-v1"), "Reference basis/operation mismatch")
    if manifest.get("operation") == "nist-anova-decimal":
        require(manifest.get("reference_basis") == "nist-decimal-v1", "Decimal ANOVA requires original-decimal references")
    if manifest.get("operation") == "nist-anova":
        require(manifest.get("reference_basis") == "binary64-v1", "Binary64 ANOVA requires binary64 references")
    for key in ["fixture", "source_fixture", "reference_fixture"]:
        require(isinstance(manifest.get(key), str) and manifest[key], f"Missing {key}")
    require(manifest.get("reference_basis") in ("workflow-reference-v1", "nist-decimal-v1", "binary64-v1", "exact-rational-v1", "controlled-reference-v1", "supervised-reference-v1", "dataframe-reference-v1", "neural-reference-v1", "neural-shaped-reference-v1", "vision-reference-v1", "boundary-reference-v1", "sweep-reference-v1"), "Unknown reference basis")
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
    if operation in WORKFLOW_OPERATIONS:
        return validate_workflow(payload, operation, rows)
    if operation == "dataframe-model":
        return validate_boundary(payload, operation, rows)
    if operation == "dataframe-model-sweep":
        return validate_descriptor(payload, operation, rows)
    if operation == "vision-letterbox-cpu":
        return validate_vision(payload, operation, rows)
    if operation == "decoder-shaped-f32":
        return validate_shaped(payload, operation, rows)
    if operation == "decoder-fixed-f32":
        return validate_neural(payload, operation, rows)
    if operation == "dataframe-semantics":
        return validate_dataframe(payload, operation, rows)
    if operation in SUPERVISED_OPERATIONS:
        return validate_supervised(payload, operation, rows)
    if operation in CONTROLLED_OPERATIONS:
        return validate_controlled(payload, operation, rows)
    if operation == "ols-cpu":
        fields(payload, ["operation", "features", "targets"])
        finite_vector(payload["targets"], "targets")
        matrix = payload["features"]
        require(isinstance(matrix, list) and len(matrix) == rows == len(payload["targets"]), "OLS row count mismatch")
        for row in matrix:
            finite_vector(row, "features")
        require(all(len(row) == len(matrix[0]) for row in matrix), "Ragged OLS features")
        require(rows >= len(matrix[0]) + 1, "Underdetermined OLS fixture")
    elif operation in ("nist-anova", "nist-anova-decimal"):
        fields(payload, ["operation", "groups"])
        groups = payload["groups"]
        require(isinstance(groups, list) and len(groups) >= 2, "ANOVA needs groups")
        for group in groups:
            if operation == "nist-anova-decimal":
                require(isinstance(group, list) and group, "Empty decimal ANOVA group")
                require(all(isinstance(v, str) and re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", v) for v in group), "Invalid decimal ANOVA token")
            else:
                finite_vector(group, "ANOVA group")
        require(sum(map(len, groups)) == rows and rows > len(groups), "ANOVA row count mismatch")
    elif operation in ("pca-cpu", "multinomial-nb-cpu"):
        required = ["operation", "features", "query", "n_components"] if operation == "pca-cpu" else ["operation", "features", "targets", "query", "alpha"]
        fields(payload, required)
        matrix = payload["features"]
        require(isinstance(matrix, list) and len(matrix) == rows and rows > 1, "Invalid model training rows")
        for row in matrix:
            finite_vector(row, "training features")
        width = len(matrix[0])
        query = payload["query"]
        require(isinstance(query, list) and query, "Missing model query rows")
        for row in query:
            finite_vector(row, "query features")
        require(all(len(row) == width for row in matrix + query), "Ragged model matrix")
        if operation == "pca-cpu":
            k = payload["n_components"]
            require(type(k) is int and 1 <= k <= min(rows, width), "Invalid PCA component count")
        else:
            finite_vector(payload["targets"], "class labels")
            require(len(payload["targets"]) == rows and len(set(payload["targets"])) >= 2, "Invalid class labels")
            require(all(v >= 0 for row in matrix + query for v in row), "Negative multinomial feature")
            alpha = payload["alpha"]
            require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0, "Invalid smoothing")
    else:
        require(False, "Unsupported numerical fixture")
    require(payload["operation"] == operation, "Numerical input operation mismatch")
    return payload


def source_and_input(root, manifest):
    verified_file(repository_file(root, manifest["source_fixture"]), manifest["source_sha256"], manifest["source_size_bytes"])
    if manifest["operation"] in WORKFLOW_OPERATIONS or manifest["operation"] in SUPERVISED_OPERATIONS or manifest["operation"] in ("decoder-fixed-f32", "decoder-shaped-f32", "vision-letterbox-cpu", "dataframe-model", "dataframe-model-sweep"):
        lock_path = repository_file(root, manifest["source_fixture"])
        for source in read_json(lock_path)["sources"]:
            relative = str((lock_path.parent / source["path"]).relative_to(root))
            verified_file(repository_file(root, relative), source["sha256"], source["bytes"])
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
    if manifest["reference_basis"] == "workflow-reference-v1":
        require(reference["operation"] == manifest["operation"], "Workflow reference operation mismatch")
        values = workflow_reference(payload)
        require(reference["independentReference"]["values"] == values, "Independent workflow reference differs")
        finite_vector(values, "workflow answers")
        return values
    if manifest["reference_basis"] in ("boundary-reference-v1", "sweep-reference-v1"):
        require(manifest["tolerances"] == {"atol": 2e-5 if payload["dtype"] == "float32" else 1e-12, "rtol": 0}, "Boundary dtype tolerance mismatch")
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        if manifest["operation"] == "dataframe-model-sweep":
            values = sweep_reference(payload)
            raw = struct.pack('<' + 'd'*len(values), *values)
            require(len(values) == reference["independentReference"]["count"] == sweep_output_count(payload), "Sweep reference count mismatch")
            require(digest(raw) == reference["independentReference"]["binary64_sha256"], "Sweep reference digest mismatch")
        else:
            values = reference["independentReference"]["values"]
            require(len(values) == boundary_output_count(payload), "Boundary reference count mismatch")
        finite_vector(values, "boundary answers")
        return values
    if manifest["reference_basis"] == "vision-reference-v1":
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        values = reference["independentReference"]["values"]
        require(len(values) == vision_output_count(payload), "Vision reference shape mismatch")
        finite_vector(values, "vision answers")
        return values
    if manifest["reference_basis"] in ("neural-reference-v1", "neural-shaped-reference-v1"):
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        values = reference["independentReference"]["values"]
        require(len(values) == (shaped_output_count(payload) if manifest["operation"] == "decoder-shaped-f32" else neural_output_count(payload)), "Neural reference shape mismatch")
        finite_vector(values, "neural answers")
        return values
    if manifest["reference_basis"] == "dataframe-reference-v1":
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        values = reference["independentReference"]["values"]
        require(values == dataframe_expected(payload), "Dataframe pinned reference disagrees with scalar contract")
        finite_vector(values, "dataframe encoding")
        return values
    if manifest["reference_basis"] in ("controlled-reference-v1", "supervised-reference-v1"):
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        values = reference["independentReference"]["values"]
        require(len(values) == (supervised_output_count(payload) if manifest["operation"] in SUPERVISED_OPERATIONS else output_count(payload)), "Controlled output shape mismatch")
        finite_vector(values, "controlled answers")
        return values
    if manifest["reference_basis"] == "exact-rational-v1":
        require(reference["operation"] == manifest["operation"], "Reference operation mismatch")
        values = list(reference["exactRationalReference"]["values"])
        n, p, q = manifest["rows"], len(payload["features"][0]), len(payload["query"])
        if manifest["operation"] == "pca-cpu":
            k = payload["n_components"]
            count = p + 2*k + k*p*p + n*n + q*q + n*q
        else:
            c = len(set(payload["targets"]))
            count = c + q*c + q
        require(len(values) == count, "Exact model reference shape mismatch")
        finite_vector(values, "exact model answers")
        return values
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
