"""Bounded CHW normalized inputs for public YOLO preprocessing conformance."""
import math
import struct
from contracts import fields, require

OPERATION = 'vision-letterbox-cpu'
TOLERANCES = {'atol': 2e-6, 'rtol': 2e-6}


def geometry(payload):
    width, height = payload['width'], payload['height']
    target_width, target_height = payload['target_width'], payload['target_height']
    scale = min(target_width / width, target_height / height)
    # Positive half ties round away from zero, matching the public implementation.
    resized_width = min(target_width, max(1, math.floor(width * scale + .5)))
    resized_height = min(target_height, max(1, math.floor(height * scale + .5)))
    return resized_width, resized_height, (target_width-resized_width)//2, (target_height-resized_height)//2


def validate_input(payload, operation, rows):
    fields(payload, ['operation', 'device', 'layout', 'encoding', 'width', 'height',
                     'channels', 'target_width', 'target_height', 'padding_color', 'pixels'])
    require(operation == OPERATION == payload['operation'], 'Vision operation mismatch')
    require(payload['device'] == 'cpu', 'Vision requires explicit CPU')
    require(payload['layout'] == 'CHW' and payload['encoding'] == 'normalized-f32', 'Invalid vision layout or encoding')
    for key in ('width', 'height', 'target_width', 'target_height'):
        require(type(payload[key]) is int and 1 <= payload[key] <= 32, 'Invalid bounded image dimension')
    require(type(payload['channels']) is int and payload['channels'] in (1, 3), 'Vision requires gray or RGB')
    count = payload['width'] * payload['height']
    require(type(rows) is int and rows == count, 'Vision pixel count mismatch')
    pixels = payload['pixels']
    require(isinstance(pixels, list) and len(pixels) == count * payload['channels'], 'Invalid image buffer length')
    for value in pixels:
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, 'Invalid normalized pixel')
        require(struct.unpack('<f', struct.pack('<f', value))[0] == value, 'Pixel must be exactly Float32')
    padding = payload['padding_color']
    require(type(padding) in (int, float) and math.isfinite(padding) and 0 <= padding <= 1, 'Invalid padding color')
    resized_width, resized_height, _, _ = geometry(payload)
    if (resized_width, resized_height) != (payload['width'], payload['height']):
        require(all(len(set(pixels[channel*count:(channel+1)*count])) == 1
                    for channel in range(payload['channels'])),
                'This contract certifies resize only for constant planes')
    return payload


def output_count(payload):
    return 4 + payload['target_width'] * payload['target_height'] * 3
