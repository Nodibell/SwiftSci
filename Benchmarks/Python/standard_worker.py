#!/usr/bin/env python3
"""Pandas/NumPy worker for the versioned benchmark protocol."""

import hashlib
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Tools"))
from contracts import read_json, write_json, require, verified_file, validate_values
import numpy as np
import pandas as pd

request = read_json(sys.argv[1])
destination = Path(sys.argv[2])
try:
    verified_file(
        request["input_path"], request["input_sha256"], request["input_bytes"]
    )
    expected = np.frombuffer(
        verified_file(request["expected_path"], request["expected_sha256"]), dtype="<f8"
    )
    op = request["operation"]
    if request["dataset_kind"] == "nist-univariate-v1":
        x = np.loadtxt(
            request["input_path"], skiprows=request["input_skip_rows"], dtype=np.float64
        )
        require(
            len(x) == request["rows"] and np.isfinite(x).all(),
            "Invalid NIST values or row count",
        )
        frame = None
    else:
        frame = (
            pd.read_csv(
                request["input_path"],
                dtype={"id": "int64", "group": "int64", "x": "float64", "y": "float64"},
            )
            if op != "csv-read" and not op.startswith("csv-stream-")
            else None
        )
        x = frame["x"].to_numpy() if frame is not None else None
    if frame is not None:
        require(len(frame) == request["rows"], "Row count mismatch")

    y = frame["y"].to_numpy() if frame is not None else None
    right = (
        pd.DataFrame(
            {
                "id": np.arange(request["rows"], dtype=np.int64),
                "weight": np.arange(request["rows"], dtype=np.float64) / 4,
            }
        )
        if op == "inner-join"
        else None
    )

    def execute():
        if op.startswith("csv-stream-"):
            chunks = []
            for chunk in pd.read_csv(
                request["input_path"],
                chunksize=10000,
                dtype={"id": "int64", "group": "int64", "x": "float64", "y": "float64"},
            ):
                if op == "csv-stream-read":
                    chunks.append(chunk)
                elif op == "csv-stream-filter":
                    chunks.append(chunk.loc[chunk.x > 0])
                elif op == "csv-stream-group":
                    chunks.append(
                        chunk.groupby("group", sort=False, as_index=False).agg(
                            x_sum=("x", "sum"), y_mean=("y", "mean")
                        )
                    )
                else:
                    raise ValueError("Unsupported streaming operation")
            return chunks
        if op == "row-sum":
            total = 0.0
            for row in frame.itertuples(index=False):
                total += row.x
            return total
        if op == "inner-join":
            return frame.merge(right, on="id", how="inner", sort=False)
        if op == "group-sum-mean":
            return frame.groupby("group", sort=False, as_index=False).agg(
                x_sum=("x", "sum"), y_mean=("y", "mean")
            )
        if op == "pearson":
            return np.corrcoef(x, y)[0, 1]
        if op == "spearman":
            return np.corrcoef(
                pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average")
            )[0, 1]
        if op == "csv-read":
            return pd.read_csv(
                request["input_path"],
                dtype={"id": "int64", "group": "int64", "x": "float64", "y": "float64"},
            )
        if op == "filter":
            return frame.loc[frame.x > 0]
        if op == "sort":
            return frame.sort_values("x", kind="stable")
        if op == "group-sum":
            return frame.groupby("group", sort=False, as_index=False)["x"].sum()
        if op == "flat-matrix":
            return (
                frame[["x", "y"]].to_numpy(dtype=np.float64, copy=True).ravel(order="C")
            )
        if op == "target":
            return frame["x"].to_numpy(dtype=np.float64, copy=True)
        if op == "mean":
            return np.mean(x)
        if op == "variance":
            return np.var(x, ddof=1)
        if op == "stddev":
            return np.std(x, ddof=1)
        values = frame[["x", "y"]].to_numpy(dtype=np.float64, copy=True)
        if op == "standard-scale":
            mean = values.mean(axis=0)
            scale = values.std(axis=0, ddof=0)
            scale[scale < 1e-12] = 1
            return ((values - mean) / scale).ravel(order="C")
        if op == "minmax-scale":
            low = values.min(axis=0)
            span = values.max(axis=0) - low
            return (
                (values - low)
                * np.divide(1.0, span, out=np.zeros_like(span), where=span >= 1e-12)
            ).ravel(order="C")
        raise ValueError("Unsupported operation: " + op)

    def canonical(output):
        if op.startswith("csv-stream-"):
            arrays = []
            for index, chunk in enumerate(output):
                if op == "csv-stream-group":
                    chunk = chunk.sort_values("group")[["group", "x_sum", "y_mean"]]
                    arrays.append(
                        np.column_stack((np.full(len(chunk), index), chunk)).ravel()
                    )
                else:
                    arrays.append(chunk.to_numpy(dtype=np.float64).ravel())
            return np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
        if op == "inner-join":
            output = output.sort_values("id")[["id", "group", "x", "y", "weight"]]
        if op == "group-sum-mean":
            output = output.sort_values("group")[["group", "x_sum", "y_mean"]]
        if op == "group-sum":
            output = output.sort_values("group")[["group", "x"]]
        return np.asarray(output, dtype="<f8").ravel(order="C")

    samples = []
    for index in range(request["warmups"] + request["samples"]):
        start = time.perf_counter_ns()
        output = execute()
        duration = time.perf_counter_ns() - start
        actual = canonical(output)
        error = validate_values(actual, expected, request["atol"], request["rtol"])
        if index >= request["warmups"]:
            samples.append(
                dict(
                    elapsed_ns=duration,
                    timing_resolved=duration >= 1000,
                    output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    maximum_absolute_error=error,
                    validated=True,
                )
            )
        del output, actual
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(
        destination,
        dict(
            schema_version=1,
            case_key=request["case_key"],
            status="passed",
            samples=samples,
            peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
            engine_version=f"pandas {pd.__version__}; numpy {np.__version__}; python {sys.version}",
        ),
    )
except Exception as error:
    write_json(
        destination,
        dict(
            schema_version=1,
            case_key=request["case_key"],
            status="failed",
            samples=[],
            peak_rss_bytes=0,
            engine_version=f"pandas {pd.__version__}; numpy {np.__version__}",
            error=str(error),
        ),
    )
    raise SystemExit(1)
