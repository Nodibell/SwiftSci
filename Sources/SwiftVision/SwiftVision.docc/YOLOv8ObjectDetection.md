# YOLOv8 Object Detection & ONNX Weight Parsing

GPU-accelerated real-time object detection inference using YOLOv8 architectures and native binary ONNX weight parsing in SwiftVision.

## Overview

`SwiftVision` provides a complete GPU-accelerated forward-pass inference engine for YOLOv8 object detection models (`YOLOv8n`, `YOLOv8s`, `YOLOv8m`, `YOLOv8l`, `YOLOv8x`) running natively on Apple Silicon UMA via MLX / MLXNN.

### 1. Preprocessing Images with Letterbox

`YOLOPreprocessor` resizes input images to the model's expected `640 × 640` resolution while strictly preserving original aspect ratios using symmetric `(114, 114, 114)` gray padding:

```swift
import SwiftVision
import MLX

let (inputTensor, letterboxMeta) = try YOLOPreprocessor.preprocess(
    imagePath: "sample.jpg",
    targetSize: (640, 640)
)
```

### 2. Loading ONNX Weights

Parse binary `.onnx` model files directly into MLX weight dictionaries:

```swift
let weightData = try Data(contentsOf: URL(fileURLWithPath: "yolov8n.onnx"))
let weights = ONNXWeightReader.parse(data: weightData)

let detector = YOLOv8Detector(version: .nano)
detector.loadWeights(weights)
```

### 3. Running Object Detection

Execute GPU-accelerated forward pass inference, Distribution Focal Loss (DFL) decoding, and Non-Maximum Suppression (NMS):

```swift
let detections = try await detector.detect(
    imagePath: "sample.jpg",
    confidenceThreshold: 0.25,
    iouThreshold: 0.45
)

for detection in detections {
    print("Class \(detection.classIndex) (\(detection.confidence)): \(detection.boundingBox)")
}
```

### 4. Neural Network Architecture Components

- ``YOLOBackbone``: CSPDarknet with `ConvBlock`, `BottleneckBlock`, `C2fBlock`, and `SPPFBlock` producing multi-scale feature pyramids (`P3`, `P4`, `P5`).
- ``YOLONeck``: PANet feature pyramid network with top-down upsampling and bottom-up strided convolutions.
- ``YOLOHead``: Decoupled anchor-free detection head with DFL regression and classification branches evaluating 8,400 anchor cells.
