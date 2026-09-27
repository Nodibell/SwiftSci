"""Canonical fixture preparation and independent reference evaluation."""

import csv
import io
import math
import struct
from contracts import (
    digest,
    fields,
    positive,
    read_json,
    repository_file,
    require,
    verified_file,
)


def load_manifest(root, name):
    value = read_json(root / "Benchmarks/Specs/datasets" / (name + ".json"))
    fields(
        value,
        [
            "schema_version",
            "id",
            "kind",
            "rows",
            "sha256",
            "size_bytes",
            "source",
            "license",
            "generator_version",
        ],
        ["groups", "fixture", "certified", "data_start_line", "tolerances"],
    )
    require(
        value["schema_version"] == 1 and value["id"] == name,
        "Dataset identity/version mismatch",
    )
    require(
        value["kind"]
        in ["table-v1", "mixed-table-v1", "parquet-table-v1", "nist-univariate-v1"],
        "Unknown dataset generator",
    )
    positive(value["rows"], "rows")
    positive(value["size_bytes"], "size_bytes")
    require(
        isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"]),
        "Invalid SHA-256",
    )
    require(value["generator_version"] == 1, "Unknown generator version")
    if value["kind"] != "nist-univariate-v1":
        positive(value.get("groups"), "groups")
    else:
        require(
            "fixture" in value and "certified" in value,
            "Missing NIST source or answers",
        )
        positive(value["rows"], "rows", 2)
        positive(value.get("data_start_line"), "data_start_line")
        fields(value["certified"], ["mean", "variance", "stddev"])
        fields(value.get("tolerances"), ["mean", "variance", "stddev"])
        for operation, answer in value["certified"].items():
            require(
                type(answer) in (int, float) and math.isfinite(answer),
                "Invalid reference answer",
            )
            if operation != "mean":
                require(answer >= 0, "Negative dispersion reference")
            tolerance = value["tolerances"][operation]
            fields(tolerance, ["atol", "rtol"])
            for number in tolerance.values():
                require(
                    type(number) in (int, float)
                    and math.isfinite(number)
                    and number >= 0,
                    "Invalid reference tolerance",
                )
    return value


def parse_univariate(content, data_start_line, rows):
    positive(data_start_line, "data_start_line")
    positive(rows, "rows", 2)
    lines = content.decode("ascii").splitlines()
    require(data_start_line <= len(lines), "Missing NIST data section")
    data = [
        float(line.strip()) for line in lines[data_start_line - 1 :] if line.strip()
    ]
    require(len(data) == rows, "NIST row count mismatch")
    require(all(math.isfinite(x) for x in data), "Nonfinite NIST input")
    return data


def resolve_workload(dataset, workload):
    if dataset["kind"] != "nist-univariate-v1":
        return workload
    operation = workload["operation"]
    require(
        operation in dataset["certified"],
        "NIST dataset only supports declared statistics",
    )
    return dict(workload, **dataset["tolerances"][operation])


def generate_table(rows, groups):
    # Exact quarter fractions and integers avoid platform-specific decimal formatting.
    out = io.StringIO(newline="")
    out.write("id,group,x,y\n")
    for i in range(rows):
        out.write(
            f"{i},{(i * 17) % groups},{((i * 37) % 1009 - 504) / 4:.2f},{((i * 13) % 701 - 350) / 4:.2f}\n"
        )
    return out.getvalue().encode("ascii")


def generate_mixed_table(rows, groups):
    numeric = generate_table(rows, groups).decode("ascii").splitlines()
    result = ["id,group,x,y,flag"]
    for line in numeric[1:]:
        i, group, x, y = line.split(",")
        result.append(
            f"{i},category{group},{x},{y},{'true' if int(i) % 2 == 0 else 'false'}"
        )
    return ("\n".join(result) + "\n").encode("ascii")


def generate_parquet(rows, groups):
    import pyarrow as pa
    import pyarrow.parquet as pq

    require(
        pa.__version__ == "25.0.1",
        "Parquet fixture generation requires pinned pyarrow 25.0.1",
    )
    records = list(
        csv.DictReader(io.StringIO(generate_mixed_table(rows, groups).decode()))
    )
    table = pa.table(
        {
            "id": pa.array([int(r["id"]) for r in records], type=pa.int64()),
            "group": [r["group"] for r in records],
            "x": [float(r["x"]) for r in records],
            "y": [float(r["y"]) for r in records],
            "flag": [r["flag"] == "true" for r in records],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        version="1.0",
        data_page_version="1.0",
        compression="snappy",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=rows,
    )
    return sink.getvalue().to_pybytes()


def cache_path(root, manifest):
    return root / "Benchmarks/Data/standardized" / (manifest["sha256"] + ".data")


def prepare(root, manifest):
    path = cache_path(root, manifest)
    if path.exists():
        verified_file(path, manifest["sha256"], manifest["size_bytes"])
        return path
    if manifest["kind"] == "table-v1":
        content = generate_table(manifest["rows"], manifest["groups"])
    elif manifest["kind"] == "mixed-table-v1":
        content = generate_mixed_table(manifest["rows"], manifest["groups"])
    elif manifest["kind"] == "parquet-table-v1":
        content = generate_parquet(manifest["rows"], manifest["groups"])
    else:
        content = repository_file(root, manifest["fixture"]).read_bytes()
    require(
        digest(content) == manifest["sha256"]
        and len(content) == manifest["size_bytes"],
        "Generated fixture disagrees with manifest",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return path


def values(root, manifest):
    content = verified_file(
        cache_path(root, manifest), manifest["sha256"], manifest["size_bytes"]
    )
    if manifest["kind"] == "nist-univariate-v1":
        return parse_univariate(content, manifest["data_start_line"], manifest["rows"])
    if manifest["kind"] == "parquet-table-v1":
        content = generate_mixed_table(manifest["rows"], manifest["groups"])
    reader = csv.DictReader(io.StringIO(content.decode("ascii")))
    if manifest["kind"] in ["mixed-table-v1", "parquet-table-v1"]:
        require(
            reader.fieldnames == ["id", "group", "x", "y", "flag"],
            "Unexpected mixed CSV schema",
        )
        result = []
        for r in reader:
            require(
                r["group"].startswith("category") and r["flag"] in ["true", "false"],
                "Malformed mixed table",
            )
            result.append(
                (
                    int(r["id"]),
                    int(r["group"][8:]),
                    float(r["x"]),
                    float(r["y"]),
                    float(r["flag"] == "true"),
                )
            )
        require(len(result) == manifest["rows"], "CSV row count mismatch")
        return result
    require(reader.fieldnames == ["id", "group", "x", "y"], "Unexpected CSV schema")
    data = [
        (int(r["id"]), int(r["group"]), float(r["x"]), float(r["y"])) for r in reader
    ]
    require(len(data) == manifest["rows"], "CSV row count mismatch")
    return data


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        for index in order[begin:end]:
            ranks[index] = (begin + 1 + end) / 2
        begin = end
    return ranks


def correlation(x, y):
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    cx, cy = [v - mx for v in x], [v - my for v in y]
    return math.fsum(a * b for a, b in zip(cx, cy)) / math.sqrt(
        math.fsum(a * a for a in cx) * math.fsum(b * b for b in cy)
    )


def grouped_reference(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    return [
        [
            float(k),
            math.fsum(r[2] for r in groups[k]),
            math.fsum(r[3] for r in groups[k]) / len(groups[k]),
        ]
        for k in sorted(groups)
    ]


def reference(root, manifest, workload):
    rows = values(root, manifest)
    operation = workload["operation"]
    if manifest["kind"] == "nist-univariate-v1":
        require(
            operation in ["mean", "stddev", "variance"],
            "NIST dataset only supports statistics",
        )
        return [manifest["certified"][operation]]
    x = [r[2] for r in rows]
    if operation in [
        "welch",
        "student",
        "paired",
        "anova",
        "regression-metrics",
        "roc-auc",
        "tfidf",
    ]:
        import numerical_reference as nr

        y = [r[3] for r in rows]
        if operation in ["welch", "student", "paired"]:
            return nr.welch(x, y, method=operation)
        if operation == "anova":
            return nr.anova([x, y, [(a + b) / 2 for a, b in zip(x, y)]])
        if operation == "regression-metrics":
            return nr.regression_metrics(x, y)
        if operation == "roc-auc":
            return [nr.auc([int(r[0]) % 2 for r in rows], x)]
        return nr.tfidf(len(rows))
    if operation == "onehot":
        return [
            float(level == r[0] % modulus)
            for r in rows
            for modulus in [8, 4]
            for level in range(modulus)
        ]
    if operation == "sqlite-ingest":
        return [1, 10.5, 2, 20]
    if operation == "rag-summary":
        return list(b"## BenchDF Profile\n- Rows: 0, Columns: 0\n- Columns: \n")
    if operation == "pool-dice":
        return [0.8, 0.8, 0.8, 1.0]
    if operation in ["mean", "stddev", "variance"]:
        mean = math.fsum(x) / len(x)
        variance = math.fsum((v - mean) ** 2 for v in x) / (len(x) - 1)
        return [
            {"mean": mean, "variance": variance, "stddev": math.sqrt(variance)}[
                operation
            ]
        ]
    if operation in ["pearson", "spearman"]:
        y = [r[3] for r in rows]
        return [
            correlation(average_ranks(x), average_ranks(y))
            if operation == "spearman"
            else correlation(x, y)
        ]
    if operation == "row-sum":
        return [math.fsum(x)]
    if operation == "group-sum-mean":
        return [v for r in grouped_reference(rows) for v in r]
    if operation == "csv-stream-group":
        return [
            v
            for start in range(0, len(rows), 10000)
            for r in grouped_reference(rows[start : start + 10000])
            for v in [start // 10000, *r]
        ]
    if operation == "inner-join":
        return [v for r in rows for v in [*r, r[0] / 4]]
    if operation == "target":
        return x
    if operation == "flat-matrix":
        return [v for r in rows for v in r[2:4]]
    if operation in ["csv-read", "csv-stream-read", "parquet-read", "parquet-write"]:
        return [v for r in rows for v in r]
    if operation in ["filter", "csv-stream-filter"]:
        return [v for r in rows if r[2] > 0 for v in r]
    if operation == "sort":
        return [v for r in sorted(rows, key=lambda r: (r[2], r[0])) for v in r]
    if operation == "group-sum":
        groups = {}
        for row in rows:
            groups.setdefault(row[1], []).append(row[2])
        return [v for k in sorted(groups) for v in [float(k), math.fsum(groups[k])]]
    require(
        operation in ["standard-scale", "minmax-scale"],
        "Unsupported reference operation",
    )
    columns = [[r[c] for r in rows] for c in [2, 3]]
    scaled = []
    for col in columns:
        if operation == "standard-scale":
            offset = math.fsum(col) / len(col)
            span = math.sqrt(math.fsum((v - offset) ** 2 for v in col) / len(col))
            scaled.append([(v - offset) / (span if span >= 1e-12 else 1) for v in col])
        else:
            offset = min(col)
            span = max(col) - offset
            scaled.append(
                [(v - offset) * (1 / span if span >= 1e-12 else 0) for v in col]
            )
    return [v for pair in zip(*scaled) for v in pair]


def binary(values):
    return b"".join(struct.pack("<d", v) for v in values)
