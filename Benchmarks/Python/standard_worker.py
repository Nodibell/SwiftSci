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
from scipy import stats
import sqlite3
from collections import Counter

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
    mixed = request["dataset_kind"] in ["mixed-table-v1", "parquet-table-v1"]
    dtypes = {
        "id": "int64",
        "group": "str" if mixed else "int64",
        "x": "float64",
        "y": "float64",
    }
    if mixed:
        dtypes["flag"] = "bool"
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
                dtype=dtypes,
            )
            if op not in ["csv-read", "parquet-read"]
            and not op.startswith("csv-stream-")
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

    labels = np.arange(request["rows"], dtype=np.int64) % 2 if op == "roc-auc" else None
    categories = (
        pd.DataFrame(
            {
                "dept": [f"dept_{i % 8}" for i in range(request["rows"])],
                "region": [f"region_{i % 4}" for i in range(request["rows"])],
            }
        )
        if op == "onehot"
        else None
    )
    documents = (
        [
            "alpha beta beta" if i % 2 == 0 else "beta gamma"
            for i in range(request["rows"])
        ]
        if op == "tfidf"
        else None
    )

    def execute():
        if op in ["welch", "student", "paired"]:
            result = (
                stats.ttest_rel(y, x)
                if op == "paired"
                else stats.ttest_ind(x, y, equal_var=op == "student")
            )
            interval = result.confidence_interval(confidence_level=0.95)
            pooled = np.sqrt(
                ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
                / (len(x) + len(y) - 2)
            )
            return [
                result.statistic,
                result.pvalue,
                result.df,
                interval.low,
                interval.high,
                np.mean(y - x) / np.std(y - x, ddof=1)
                if op == "paired"
                else (np.mean(x) - np.mean(y)) / pooled,
            ]
        if op == "anova":
            result = stats.f_oneway(x, y)
            mean = (np.sum(x) + np.sum(y)) / (len(x) + len(y))
            between = (
                len(x) * (np.mean(x) - mean) ** 2 + len(y) * (np.mean(y) - mean) ** 2
            )
            within = np.sum((x - np.mean(x)) ** 2) + np.sum((y - np.mean(y)) ** 2)
            return [
                result.statistic,
                result.pvalue,
                1,
                len(x) + len(y) - 2,
                between / (between + within),
            ]
        if op == "regression-metrics":
            err = x - y
            valid = np.abs(x) > 1e-12
            return [
                np.sqrt(np.mean(err * err)),
                np.mean(np.abs(err)),
                100 * np.mean(np.abs(err[valid] / x[valid])) if valid.any() else 0,
                1 - np.sum(err * err) / np.sum((x - np.mean(x)) ** 2),
            ]
        if op == "roc-auc":
            order = np.argsort(-x, kind="stable")
            scores, truth = x[order], labels[order]
            ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
            tp, fp = np.cumsum(truth)[ends], np.cumsum(1 - truth)[ends]
            return np.trapezoid(np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]])
        if op == "onehot":
            return (
                pd.get_dummies(categories, dtype=np.float64).to_numpy().ravel(order="C")
            )
        if op == "tfidf":
            counts = [Counter(doc.lower().split()) for doc in documents]
            vocab = sorted(set().union(*(set(row) for row in counts)))
            matrix = np.array(
                [[row.get(term, 0) for term in vocab] for row in counts],
                dtype=np.float64,
            )
            df = np.count_nonzero(matrix, axis=0)
            return (
                matrix
                / matrix.sum(axis=1, keepdims=True)
                * (np.log((1 + len(counts)) / (1 + df)) + 1)
            ).ravel(order="C")
        if op == "sqlite-ingest":
            connection = sqlite3.connect(":memory:")
            try:
                connection.execute("CREATE TABLE sample (id INT, val REAL)")
                connection.execute("INSERT INTO sample VALUES (1,10.5),(2,20.0)")
                return (
                    pd.read_sql_query(
                        "SELECT id, val FROM sample ORDER BY id", connection
                    )
                    .to_numpy(dtype=np.float64)
                    .ravel(order="C")
                )
            finally:
                connection.close()
        if op == "rag-summary":
            empty = pd.DataFrame()
            return f"## BenchDF Profile\n- Rows: {len(empty)}, Columns: {len(empty.columns)}\n- Columns: {', '.join(empty.columns)}\n"
        if op == "pool-dice":
            image = np.full((3, 32, 32), 0.8)
            pooled = image.mean(axis=(1, 2))
            pred = pooled > 0.5
            dice = 2 * np.count_nonzero(pred & pred) / (np.count_nonzero(pred) * 2)
            return np.r_[pooled, dice]
        if op.startswith("csv-stream-"):
            chunks = []
            input_rows = []
            for chunk in pd.read_csv(
                request["input_path"],
                chunksize=10000,
                dtype=dtypes,
            ):
                input_rows.append(len(chunk))
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
            return chunks, input_rows
        if op == "parquet-read":
            return pd.read_parquet(request["input_path"], engine="pyarrow")
        if op == "parquet-write":
            path = Path(str(destination) + f".sample{index}.parquet")
            frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
            return path
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
                dtype=dtypes,
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

    def numeric_frame(frame):
        if not mixed:
            return frame
        frame = frame.copy()
        require(
            frame["group"].str.fullmatch(r"category[0-9]+").all(), "Malformed category"
        )
        require(pd.api.types.is_bool_dtype(frame["flag"]), "Lost Boolean type")
        frame["group"] = frame["group"].str[8:].astype("int64")
        return frame

    def numeric_groups(frame):
        frame = frame.copy()
        if mixed:
            require(
                frame["group"].str.fullmatch(r"category[0-9]+").all(),
                "Malformed group key",
            )
            frame["group"] = frame["group"].str[8:].astype("int64")
        return frame.sort_values("group")

    def canonical(output):
        if op == "rag-summary":
            return np.frombuffer(output.encode("utf-8"), dtype=np.uint8).astype(
                np.float64
            )
        if op == "parquet-write":
            output = pd.read_parquet(output, engine="pyarrow")
        if op.startswith("csv-stream-"):
            chunks, sizes = output
            require(
                sizes
                == [
                    min(10000, request["rows"] - start)
                    for start in range(0, request["rows"], 10000)
                ],
                "Streaming chunk boundaries differ",
            )
            arrays = []
            for index, chunk in enumerate(chunks):
                if op == "csv-stream-group":
                    chunk = numeric_groups(chunk)[["group", "x_sum", "y_mean"]]
                    arrays.append(
                        np.column_stack((np.full(len(chunk), index), chunk)).ravel()
                    )
                else:
                    arrays.append(
                        numeric_frame(chunk).to_numpy(dtype=np.float64).ravel()
                    )
            return np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
        if op == "inner-join":
            output = output.sort_values("id")[
                ["id", "group", "x", "y"] + (["flag"] if mixed else []) + ["weight"]
            ]
        if op == "group-sum-mean":
            output = numeric_groups(output)[["group", "x_sum", "y_mean"]]
        if op == "group-sum":
            output = numeric_groups(output)[["group", "x"]]
        if isinstance(output, pd.DataFrame) and op not in [
            "group-sum",
            "group-sum-mean",
        ]:
            output = numeric_frame(output)
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
