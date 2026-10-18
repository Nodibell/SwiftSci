"""Matched known-schema CSV ingestion; output types are checked outside timing."""
from contracts import require


def schema_for(kind):
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
