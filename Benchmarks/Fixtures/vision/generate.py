#!/usr/bin/env python3
"""Generate original small images and analytically determined complete outputs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'Benchmarks/Tools'))
from contracts import digest, write_json
from vision_fixtures import OPERATION, TOLERANCES, validate_input
from vision_reference import reference


def cases():
    specifications = [
        ('rgb-odd-identity', 5, 3, 3, 5, 3, None, 114/255),
        ('gray-odd-identity', 3, 5, 1, 3, 5, None, 114/255),
        ('rgb-wide-odd-padding', 5, 2, 3, 5, 5, None, 114/255),
        ('gray-tall-odd-padding', 2, 5, 1, 5, 5, None, .125),
        ('rgb-up-round-half', 3, 2, 3, 8, 3, [.125, .5, .875], 114/255),
        ('gray-down-round-half', 8, 5, 1, 4, 4, [.75], .125),
        ('rgb-up-asymmetric', 2, 3, 3, 9, 8, [.25, .625, 1], .0625),
        ('rgb-down-asymmetric', 7, 3, 3, 4, 5, [0, .375, .875], 114/255),
    ]
    for name, width, height, channels, tw, th, constants, padding in specifications:
        count = width*height
        pixels = [constants[channel] if constants is not None else
                  ((channel*13 + y*7 + x*3) % 33)/32
                  for channel in range(channels) for y in range(height) for x in range(width)]
        yield 'vision-cpu-'+name, {'operation': OPERATION, 'device': 'cpu',
            'layout': 'CHW', 'encoding': 'normalized-f32', 'width': width, 'height': height,
            'channels': channels, 'target_width': tw, 'target_height': th,
            'padding_color': padding, 'pixels': pixels}


def generate():
    lock_path = ROOT/'Benchmarks/Fixtures/vision/sources.lock.json'
    sources = []
    for relative in ['generate.py', '../../Tools/vision_reference.py', '../../Tools/vision_fixtures.py',
                     '../../Python/vision_workloads.py']:
        data = (lock_path.parent/relative).read_bytes()
        sources.append({'path': relative, 'sha256': digest(data), 'bytes': len(data)})
    write_json(lock_path, {'schema_version': 1, 'sources': sources})
    source = lock_path.read_bytes()
    profile = []
    for name, payload in cases():
        rows = payload['width']*payload['height']
        validate_input(payload, OPERATION, rows)
        path = ROOT/f'Benchmarks/Fixtures/vision/inputs/{name}.json'
        write_json(path, payload)
        raw = path.read_bytes()
        refpath = ROOT/f'Benchmarks/Fixtures/vision/references/{name}.json'
        write_json(refpath, {'operation': OPERATION,
            'inputIdentity': {'sha256': digest(raw), 'bytes': len(raw)},
            'source': {'sha256': digest(source)}, 'model': {'observations': rows},
            'independentReference': {'method': 'rational geometry; identity or constant plane; Float32 output',
                                     'values': reference(payload)}})
        ref = refpath.read_bytes()
        write_json(ROOT/f'Benchmarks/Specs/datasets/{name}.json', {
            'schema_version': 1, 'id': name, 'kind': 'numerical-fixture-v1', 'rows': rows,
            'sha256': digest(raw), 'size_bytes': len(raw),
            'source': 'Original images; see Benchmarks/Fixtures/vision/README.md',
            'license': 'MIT, repository LICENSE', 'generator_version': 1,
            'fixture': str(path.relative_to(ROOT)), 'source_fixture': str(lock_path.relative_to(ROOT)),
            'source_sha256': digest(source), 'source_size_bytes': len(source),
            'reference_fixture': str(refpath.relative_to(ROOT)), 'reference_sha256': digest(ref),
            'reference_size_bytes': len(ref), 'reference_basis': 'vision-reference-v1',
            'operation': OPERATION, 'tolerances': TOLERANCES})
        profile.append({'id': name, 'dataset': name, 'workload': 'vision-letterbox-cpu-v1'})
    write_json(ROOT/'Benchmarks/Specs/profiles/vision-conformance.json', {
        'schema_version': 1, 'id': 'vision-conformance', 'warmups': 1,
        'samples': 2, 'batches': 1, 'timeout_seconds': 60, 'cases': profile})


if __name__ == '__main__':
    generate()
