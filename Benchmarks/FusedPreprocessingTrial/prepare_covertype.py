#!/usr/bin/env python3
"""Extract the recorded training and held-out rows without retraining a model."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepared = args.existing / 'prepared'
    provenance = json.loads((prepared / 'provenance.json').read_text())
    source = args.existing / 'data/covtype.data.gz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != provenance['source_sha256']:
        raise RuntimeError('Original Covertype source checksum differs')
    selection = prepared / 'selected-row-indices.json'
    if hashlib.sha256(selection.read_bytes()).hexdigest() != provenance['files'][selection.name]:
        raise RuntimeError('Recorded held-out selection checksum differs')
    selected = json.loads(selection.read_text())
    if len(selected) != 8192 or len(set(selected)) != 8192 or min(selected) < 15120:
        raise RuntimeError('Invalid held-out partition')
    positions = {row: index for index, row in enumerate(selected)}
    training = []
    held_out = [None] * len(selected)
    with gzip.open(source, 'rt') as stream:
        for index, line in enumerate(stream):
            if index < 11340 or index in positions:
                row = [float(x) for x in line.strip().split(',')[:54]]
                if len(row) != 54:
                    raise RuntimeError('Wrong source width')
                if index < 11340:
                    training.append(row)
                else:
                    held_out[positions[index]] = row
    if len(training) != 11340 or any(row is None for row in held_out):
        raise RuntimeError('Missing recorded rows')
    document = {'sourceSHA256': digest, 'selectedRows': selected,
                'trainingColumns': list(map(list, zip(*training))),
                'heldOutColumns': list(map(list, zip(*held_out)))}
    with args.output.open('x') as out:
        json.dump(document, out, allow_nan=False)
    print(f'Extracted {len(training)} training and {len(held_out)} held-out rows; source SHA256 {digest}')


if __name__ == '__main__':
    main()
