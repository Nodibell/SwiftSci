# ``SwiftVision``

Computer Vision Neural Inference, Object Detection & Image Datasets.

## Overview

`SwiftVision` provides GPU-accelerated computer vision neural architectures, real-time YOLOv8 object detection inference, native ONNX weight parsing, U-Net semantic segmentation, and dataset preprocessing pipelines.

### Key Capabilities

- **Real YOLOv8 Object Detection**: Full GPU-accelerated forward pass (`YOLOv8Detector`) featuring CSPDarknet backbone, PANet feature pyramid, decoupled DFL detection head, and non-maximum suppression (NMS).
- **ONNX Weight Parsing**: `ONNXWeightReader` native binary Protobuf wire-format parser extracting model weights directly into `[String: MLXArray]` instances.
- **Letterbox Preprocessor**: `YOLOPreprocessor` aspect-ratio preserving resizer with `(114, 114, 114)` gray padding and normalization.
- **Segmentation Models**: `UNetSegmentationModel` neural architecture for semantic image segmentation.
- **Image Dataset Loader**: `ImageDataset` directory scanner with configurable resizing and batching.
- **Evaluation Metrics**: `UNetMetrics` with `diceScore` and `iouScore` (Intersection over Union).

### Example Usage

```swift
import SwiftVision

// 1. Initialize YOLOv8 object detector
let detector = YOLOv8Detector(version: .nano)

// 2. Run real GPU-accelerated object detection
let detections = try await detector.detect(
    imagePath: "image.jpg",
    confidenceThreshold: 0.25,
    iouThreshold: 0.45
)
```

## Topics

### Guides & Tutorials
- <doc:YOLOv8ObjectDetection>
- <doc:ImageDatasetsAndSegmentation>
