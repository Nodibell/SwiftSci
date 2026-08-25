# ``SwiftVision``

Computer Vision Neural Inference, Object Detection & Image Datasets.

## Overview

`SwiftVision` provides GPU-accelerated computer vision neural architectures, real-time YOLOv8 object detection inference, native ONNX weight parsing, U-Net semantic segmentation, and dataset preprocessing pipelines.

### Key Capabilities

- **Real YOLOv8 Object Detection & Instance Segmentation**: Full GPU-accelerated forward pass featuring CSPDarknet backbone, PANet feature pyramid, decoupled DFL detection head, `YOLOSegHead` prototype mask generator (`160x160` proto masks), and `decodeMask` pixel-level segmentation.
- **CLIP Vision-Language Multimodal Projector**: Dual visual and text linear projection layers (`CLIPProjector`) mapping multi-modal embeddings into a shared latent metric space with temperature-scaled cosine similarity logits and zero-shot softmax classification.
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
