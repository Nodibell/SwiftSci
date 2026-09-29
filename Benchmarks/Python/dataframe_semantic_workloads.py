"""Pandas semantic comparisons with explicit null and scalar-type handling.

Object columns preserve Python None separately from present NaN and keep Int64
values exact. These small diagnostic workloads include construction and encoding;
they do not measure pandas' optimized numeric-column throughput. Empty gather
and filter results deliberately discard columns to match Swift API compatibility.
"""

import operator

import numpy as np
import pandas as pd

from dataframe_fixtures import encode_frames, float_words


def _decode(value, kind):
    if value is None:
        return None
    if kind == "int64":
        return int(value)
    if kind == "float64":
        if isinstance(value, str):
            return {"NaN": float("nan"), "+Inf": float("inf"), "-Inf": -float("inf")}[value]
        return float(value)
    return value


def prepare(payload):
    """Decode typed cells before timing; frame construction remains in execute."""
    columns = [
        {"name": column["name"], "type": column["type"],
         "values": [_decode(value, column["type"]) for value in column["values"]]}
        for column in payload["columns"]
    ]
    parameters = dict(payload["parameters"])
    types = {column["name"]: column["type"] for column in columns}
    if payload["action"] == "filter":
        parameters["value"] = _decode(parameters["value"], types[parameters["column"]])
    elif payload["action"] == "replace":
        parameters["values"] = [
            _decode(value, types[parameters["column"]]) for value in parameters["values"]
        ]
    return {"action": payload["action"], "columns": columns, "parameters": parameters}


def _records(frame, types):
    return [
        {"name": name, "type": types[name], "values": frame[name].tolist()}
        for name in frame.columns
    ]


def _group_counts(frame, types, parameters):
    keys = parameters["columns"]
    # Extension arrays preserve full-width integers and distinguish null strings
    # from literal sentinel text when pandas constructs its grouping index.
    grouping = pd.DataFrame({
        name: pd.array(frame[name].tolist(), dtype="Int64" if types[name] == "int64" else "string")
        for name in keys
    })
    groups = grouping.groupby(keys, dropna=False, sort=False, observed=True)
    positions = sorted(groups.indices.values(), key=lambda rows: int(rows[0]))
    key_columns = [
        {"name": name, "type": "utf8", "values": [
            None if frame[name].iloc[int(rows[0])] is None else str(frame[name].iloc[int(rows[0])])
            for rows in positions
        ]}
        for name in keys
    ]
    numeric_names = [
        name for name in frame.columns if name not in keys and types[name] in ("int64", "float64")
    ]
    row_counts = key_columns + [
        {"name": name, "type": "int64", "values": [len(rows) for rows in positions]}
        for name in numeric_names
    ]
    if not numeric_names:
        row_counts.append({"name": "count", "type": "int64", "values": [len(rows) for rows in positions]})
    value = parameters["value"]
    present_counts = key_columns + [{
        "name": value + "_count", "type": "float64", "values": [
            float(frame[value].iloc[rows].map(lambda cell: cell is not None).sum())
            for rows in positions
        ]
    }]
    return row_counts, present_counts


def execute(prepared):
    """Construct, operate on, and serialize a pandas frame inside timing."""
    frame = pd.DataFrame({
        column["name"]: pd.Series(column["values"], dtype=object)
        for column in prepared["columns"]
    })
    types = {column["name"]: column["type"] for column in prepared["columns"]}
    action = prepared["action"]
    parameters = prepared["parameters"]
    source = _records(frame, types)

    if action == "gather":
        result = frame.iloc[parameters["indices"]]
    elif action == "select":
        result = frame.loc[:, parameters["columns"]]
    elif action == "sort":
        result = frame.sort_values(
            parameters["column"], ascending=parameters["ascending"],
            kind="stable", na_position="last",
        )
    elif action == "filter":
        values = frame[parameters["column"]]
        predicate = parameters["predicate"]
        if predicate == "isNull":
            mask = values.map(lambda cell: cell is None)
        elif predicate == "isNotNull":
            mask = values.map(lambda cell: cell is not None)
        else:
            compare = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
                       "le": operator.le, "gt": operator.gt, "ge": operator.ge}[predicate]
            threshold = parameters["value"]
            mask = values.map(lambda cell: cell is not None and compare(cell, threshold))
        result = frame.loc[mask]
    elif action == "replace":
        name = parameters["column"]
        retained = frame[name].copy(deep=True)
        selected = frame.loc[:, [name]]
        result = frame.copy(deep=True)
        replacement = pd.Series(parameters["values"], dtype=object, name="temporary")
        result[name] = replacement
        return encode_frames([
            source, _records(result, types), _records(selected, types),
            _records(retained.to_frame(), types),
        ])
    elif action == "group-count":
        row_counts, present_counts = _group_counts(frame, types, parameters)
        return encode_frames([source, row_counts, present_counts])
    elif action == "matrix":
        selected = frame.loc[:, parameters["columns"]]
        matrix = selected.to_numpy(dtype=np.float64, copy=True)
        flat_matrix = np.ascontiguousarray(selected.to_numpy(dtype=np.float64, copy=True))
        if flat_matrix.shape != matrix.shape:
            raise ValueError("Nested and flat matrix dimensions differ")
        target = frame[parameters["target"]].to_numpy(dtype=np.float64, copy=True)
        words = encode_frames([source]) + [float(matrix.shape[0]), float(matrix.shape[1])]
        for value in matrix.ravel(order="C"):
            words.extend(float_words(float(value)))
        for value in flat_matrix.ravel(order="C"):
            words.extend(float_words(float(value)))
        words.append(float(target.shape[0]))
        for value in target:
            words.extend(float_words(float(value)))
        return words
    else:
        raise ValueError(f"Unsupported dataframe semantic action: {action}")
    if action in ("gather", "filter") and result.empty:
        return encode_frames([source, []])
    return encode_frames([source, _records(result, types)])
