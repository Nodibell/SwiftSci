"""NumPy CPU comparator for the bounded identity/constant letterbox contract."""
import numpy as np


def prepare(payload):
    return {'image': np.asarray(payload['pixels'], dtype=np.float32).reshape(
                payload['channels'], payload['height'], payload['width']),
            'target': (payload['target_height'], payload['target_width']),
            'padding': np.float32(payload['padding_color'])}


def execute(prepared):
    image = prepared['image']
    channels, height, width = image.shape
    target_height, target_width = prepared['target']
    scaled = np.floor(np.array([height, width]) * min(target_height/height, target_width/width) + .5).astype(int)
    resized_height, resized_width = np.maximum(1, np.minimum(scaled, prepared['target']))
    top, left = (target_height-resized_height)//2, (target_width-resized_width)//2
    if (resized_height, resized_width) == (height, width):
        content = image.transpose(1, 2, 0)
    else:
        if not np.all(image == image[:, :1, :1]):
            raise ValueError('Nonconstant resize is outside this comparator')
        content = np.broadcast_to(image[:, 0, 0], (resized_height, resized_width, channels))
    result = np.full((1, target_height, target_width, 3), prepared['padding'], dtype=np.float32)
    result[0, top:top+resized_height, left:left+resized_width, :] = content
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite letterbox result')
    return list(result.shape) + result.astype(np.float64).ravel().tolist()
