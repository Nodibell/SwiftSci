# Object Detection Optimizations & SIMD Bounding Boxes

Hardware-accelerated zero-heap bounding box processing and Non-Maximum Suppression (NMS) for real-time vision pipelines.

## Overview

In real-time computer vision models such as YOLO, post-processing can generate tens of thousands of candidate detections per frame. Standard bounding box objects with heap-allocated `String` class names induce significant ARC and memory allocation overhead.

`SwiftVision` introduces ``BoundingBoxSIMD``:
1. **Zero-Heap Value Representation**: Encapsulates bounding coordinates in hardware `SIMD4<Float>` registers and replaces strings with integer `classId: Int32`.
2. **Accelerated Vector IoU**: Computes intersection, union, and clamping entirely within SIMD registers without memory allocations.
3. **High-Throughput NMS**: Provides ``NonMaximumSuppression/filter(boxes:iouThreshold:scoreThreshold:)->[BoundingBoxSIMD]`` operating directly on SIMD bounding boxes.

## Usage Example

```swift
import SwiftVision

// Create candidates using SIMD coordinates:
let candidate = BoundingBoxSIMD(
    xMin: 12.5, yMin: 34.0,
    xMax: 150.0, yMax: 220.0,
    confidence: 0.92,
    classId: 0 // e.g. "person"
)

// Filter detections with zero-heap NMS:
let detections = NonMaximumSuppression.filter(
    boxes: candidates,
    iouThreshold: 0.45,
    scoreThreshold: 0.25
)
```

## Topics

### Structures
- ``BoundingBoxSIMD``
- ``BoundingBox``
- ``NonMaximumSuppression``
