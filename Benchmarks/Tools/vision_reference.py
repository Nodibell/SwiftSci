"""Independent rational geometry and scalar constant/identity image oracle."""
from fractions import Fraction
import struct


def reference(payload):
    width, height = payload['width'], payload['height']
    tw, th = payload['target_width'], payload['target_height']
    scale = min(Fraction(tw, width), Fraction(th, height))
    def rounded(value):
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)
    rw, rh = min(tw, max(1, rounded(width*scale))), min(th, max(1, rounded(height*scale)))
    left, top = (tw-rw)//2, (th-rh)//2
    f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    padding = f32(payload['padding_color'])
    pixels, count = payload['pixels'], width*height
    resized = (rw, rh) != (width, height)
    if resized and any(len(set(pixels[c*count:(c+1)*count])) != 1 for c in range(payload['channels'])):
        raise ValueError('Nonconstant interpolation has no oracle in this bounded contract')
    values = [1, th, tw, 3]
    for y in range(th):
        for x in range(tw):
            inside = left <= x < left+rw and top <= y < top+rh
            for channel in range(3):
                source_channel = 0 if payload['channels'] == 1 else channel
                offset = 0 if resized else (y-top)*width + x-left
                values.append(f32(pixels[source_channel*count+offset]) if inside else padding)
    return values
