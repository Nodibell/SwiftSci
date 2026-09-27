"""Pure Python benchmark comparators. No golden-reference or generator imports."""
import math


def vector_cosine(payload):
    query=payload['query']
    norm=math.sqrt(math.fsum(x*x for x in query))
    if norm==0:
        raise ValueError('Zero-query cosine has no approved fixture contract')
    matches=[]
    for index,row in enumerate(payload['vectors']):
        if len(row)!=len(query):
            raise ValueError('Vector dimensions differ')
        denominator=norm*math.sqrt(math.fsum(x*x for x in row))
        if denominator==0:
            raise ValueError('Zero-entry cosine has no approved fixture contract')
        score=math.fsum(x*y for x,y in zip(query,row))/denominator
        matches.append((index,score))
    matches.sort(key=lambda match:-match[1])
    return matches[:payload['top_k']]


def canonical_cosine(matches,entry_count,expected_count):
    if len(matches)!=expected_count or len({row for row,_ in matches})!=len(matches):
        raise ValueError('Wrong count or duplicate vector result')
    if any(type(row) is not int or not 0<=row<entry_count or not math.isfinite(score) for row,score in matches):
        raise ValueError('Invalid vector result identity or score')
    if any(a[1] < b[1] for a,b in zip(matches,matches[1:])):
        raise ValueError('Vector results are not closest first')
    matches=sorted(matches,key=lambda match:(-match[1],match[0]))
    return [value for row,score in matches for value in (float(row),score)]


def kernel_shap(payload):
    model=payload['model']
    def predict(row):
        return (model['bias']+model['weights'][0]*row[0]+model['weights'][1]*row[1]
                +model['interaction']*row[0]*row[1])
    background=payload['background']
    base=[math.fsum(row[col] for row in background)/len(background) for col in range(2)]
    instance=payload['instance']
    empty=predict(base)
    full=predict(instance)
    first=predict([instance[0],base[1]])
    second=predict([base[0],instance[1]])
    return [empty,((first-empty)+(full-second))/2,((second-empty)+(full-first))/2,full]
