"""Public-data contracts and independent references, without dataframe dependencies."""

import csv
import io
from decimal import Decimal, localcontext
from contracts import require, verified_file, repository_file

H2O_COLUMNS = ["id1", "id2", "id3", "id4", "id5", "id6", "v1", "v2", "v3"]
WINE_FEATURES = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]
WINE_COLUMNS = ["id", *WINE_FEATURES, "quality"]
H2O_QUERIES = {
    "h2o-q1": (["id1"], [("v1", "sum")]),
    "h2o-q2": (["id1", "id2"], [("v1", "sum")]),
    "h2o-q3": (["id3"], [("v1", "sum"), ("v3", "mean")]),
    "h2o-q4": (["id4"], [("v1", "mean"), ("v2", "mean"), ("v3", "mean")]),
    "h2o-q5": (["id6"], [("v1", "sum"), ("v2", "sum"), ("v3", "sum")]),
}


def generate_h2o(rows, groups, distribution):
    require(distribution in ["uniform", "hot-key"], "Unknown key distribution")
    # Independent fixed-width streams avoid tying composite keys to one row modulo.
    state = 0x243F6A88
    def next_int(limit):
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return (state >> 8) % limit
    out = io.StringIO(newline="")
    out.write(",".join(H2O_COLUMNS) + "\n")
    high = max(1, rows // groups)
    for i in range(rows):
        keys = [next_int(n) for n in [groups, groups, high, groups, groups, high]]
        if distribution == "hot-key" and i % 5 != 0:
            keys = [0] * 6
        v1, v2, v3 = next_int(5) + 1, next_int(15) + 1, next_int(400) / 4
        out.write(",".join([*(f"key{k}" for k in keys[:3]), *(str(k) for k in keys[3:]), str(v1), str(v2), f"{v3:.2f}"]) + "\n")
    return out.getvalue().encode("ascii")


def derive_wine(root, manifest):
    source = verified_file(repository_file(root, manifest["fixture"]), manifest["source_sha256"], manifest["source_size_bytes"])
    reader = csv.reader(io.StringIO(source.decode("ascii")), delimiter=";")
    require(next(reader) == [x.replace("_", " ") for x in WINE_FEATURES] + ["quality"], "Unexpected UCI header")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(WINE_COLUMNS)
    count = 0
    for index, row in enumerate(reader):
        require(len(row) == 12 and all(Decimal(v).is_finite() for v in row), "Invalid UCI row")
        writer.writerow([index, *row])
        count += 1
    require(count == manifest["rows"], "UCI row count mismatch")
    return out.getvalue().encode("ascii")


def parse_rows(content, manifest):
    reader = csv.DictReader(io.StringIO(content.decode("ascii")))
    columns = H2O_COLUMNS if manifest["kind"] == "h2o-group-v1" else WINE_COLUMNS
    require(reader.fieldnames == columns, "Public dataset schema mismatch")
    rows = list(reader)
    require(len(rows) == manifest["rows"], "Public dataset row count mismatch")
    for row in rows:
        require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
        for column in columns:
            if column in ["id1", "id2", "id3"]:
                require(row[column].startswith("key") and row[column][3:].isdigit(), "Invalid group key")
            else:
                require(Decimal(row[column]).is_finite(), "Nonfinite public input")
    return rows


def group_reference(rows, operation):
    keys, aggregates = H2O_QUERIES[operation]
    sums, counts = {}, {}
    for row in rows:
        key = tuple(int(row[k][3:]) if k in ["id1", "id2", "id3"] else int(row[k]) for k in keys)
        if key not in sums:
            sums[key] = [Decimal(0)] * len(aggregates)
            counts[key] = 0
        for i, (column, _) in enumerate(aggregates):
            sums[key][i] += Decimal(row[column])
        counts[key] += 1
    return [value for key in sorted(sums) for value in [*key, *(float(total / counts[key] if aggregate == "mean" else total) for total, (_, aggregate) in zip(sums[key], aggregates))]]


def wine_reference(rows):
    selected = sorted((r for r in rows if Decimal(r["quality"]) >= 6), key=lambda r: Decimal(r["alcohol"]))
    require(len(selected) > 1, "Wine pipeline needs multiple selected rows")
    with localcontext() as ctx:
        ctx.prec = 50
        columns = [[Decimal(r[k]) for r in selected] for k in WINE_FEATURES]
        means = [sum(c) / len(c) for c in columns]
        scales = [(sum((v - mean) ** 2 for v in c) / len(c)).sqrt() for c, mean in zip(columns, means)]
        matrix = [float((Decimal(r[k]) - mean) / scale) if scale >= Decimal("1e-12") else 0.0 for r in selected for k, mean, scale in zip(WINE_FEATURES, means, scales)]
    return [int(r["id"]) for r in selected] + [int(r["quality"]) for r in selected] + matrix
