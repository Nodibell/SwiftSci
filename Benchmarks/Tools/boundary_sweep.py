"""Small deterministic sweep descriptors, expanded before the sample timer starts."""
from contracts import fields, require
from boundary_fixtures import validate_input

OPERATION = 'dataframe-model-sweep'


def validate_descriptor(p, operation, rows):
    fields(p, ['operation', 'recipe', 'device', 'dtype', 'stage', 'rows', 'columns'])
    require(p['operation'] == operation == OPERATION and p['recipe']=='dyadic-v1', 'Unknown boundary sweep recipe')
    require(type(p['rows']) is int and p['rows'] in (128,1024,8192) and rows == p['rows'], 'Invalid sweep rows')
    require(type(p['columns']) is int and p['columns'] in (8,64), 'Invalid sweep width')
    require(p['device'] in ('cpu','gpu') and p['dtype'] in ('float32','float64') and
            (p['device'],p['dtype']) != ('gpu','float64'), 'Unsupported sweep device/dtype')
    require(p['stage'] in ('conversion','prepared','pipeline'), 'Invalid sweep stage')
    return p


def expand(p):
    validate_descriptor(p,p['operation'],p['rows'])
    rows, width = p['rows'],p['columns']
    names=[f'x{i}' for i in range(width)]
    result={'operation':'dataframe-model','device':p['device'],'dtype':p['dtype'],
        'row_ids':[(i*37)%rows for i in range(rows)], 'feature_names':names,'feature_order':list(reversed(names)),
        'features':[[((i*7+j*3)%19-9)/8 for j in range(width)] for i in range(rows)],
        'targets':[(i%17)/4-1 for i in range(rows)],'filter_values':[i%4 for i in range(rows)],
        'filter_threshold':1,'sort_keys':[i%11 for i in range(rows)],'ascending':True,
        'weights':[(j+1)/8 for j in range(width)],'bias':-.375}
    return validate_input(result,result['operation'],rows)


def reference(p):
    from boundary_reference import reference as full_reference
    full=full_reference(expand(p))
    rows,width=int(full[0]),int(full[1])
    stop=3+rows*(2*width+2)
    if p['stage']=='conversion': return full[:stop]
    if p['stage']=='prepared': return full[:3]+full[stop:stop+2*rows]
    return full


def output_count(p):
    rows=p['rows']*3//4
    width=p['columns']
    if p['stage']=='conversion':return 3+rows*(2*width+2)
    if p['stage']=='prepared':return 3+2*rows
    return 4+rows*(2*width+4)


def prepare(p):
    from boundary_workloads import source_frame, convert
    concrete=expand(p)
    source=source_frame(concrete) if p['stage'] in ('conversion','prepared') else None
    prepared=convert(concrete,source) if p['stage']=='prepared' else None
    return p,concrete,source,prepared


def execute(state):
    from boundary_workloads import convert, compute, execute as pipeline
    descriptor,p,source,prepared=state
    stage=descriptor['stage']
    if stage=='pipeline':return pipeline(p)
    if stage=='conversion':
        selected,matrix,target,tensor,_,_=convert(p,source)
        return [len(selected),matrix.shape[1],tensor.dtype.itemsize*8]+selected['row_id'].tolist()+matrix.ravel().tolist()+target.tolist()+tensor.ravel().tolist()
    prediction,residual=compute(prepared)
    tensor=prepared[3]
    return [tensor.shape[0],tensor.shape[1],tensor.dtype.itemsize*8]+prediction.tolist()+residual.tolist()
