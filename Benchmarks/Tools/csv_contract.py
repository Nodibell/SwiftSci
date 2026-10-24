"""Matched known-schema CSV ingestion; output types are checked outside timing."""
from contracts import require


MATCHED_OPERATIONS = frozenset(("csv-read", "csv-stream-read", "csv-stream-filter",
                                "csv-stream-group", "wine-pipeline"))
WINE_FEATURES = ("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                 "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                 "pH", "sulphates", "alcohol")


def schema_for(kind):
    if kind == "wine-quality-v1":
        return {"id": "Int64", **dict.fromkeys(WINE_FEATURES, "Double"), "quality": "Int64"}
    require(kind in ('table-v1', 'mixed-table-v1'), 'Unsupported matched CSV dataset')
    schema = dict(id='Int64', group='Int64', x='Double', y='Double')
    if kind == 'mixed-table-v1':
        schema.update(group='String', flag='Bool')
    return schema


def requested_schema(request):
    expected = schema_for(request['dataset_kind'])
    require(request.get('csv_schema') == expected,
            'CSV comparison requires the complete provided schema; inference is a separate workload')
    return expected


def validate_schema(actual, request):
    require(actual == requested_schema(request), 'CSV output schema differs from the comparison contract')
    return actual


def pandas_schema(frame):
    types = {"int64": "Int64", "float64": "Double", "str": "String", "bool": "Bool"}
    return {name: types.get(str(dtype), str(dtype)) for name, dtype in frame.dtypes.items()}
