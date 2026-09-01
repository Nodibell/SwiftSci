# Real-Time YOLOv8 Object Detection & Hardware Acceleration

Execute real-time anchor-free object detection on Apple Silicon GPU and Neural Engine using `YOLOv8Detector`, `YOLOPreprocessor`, and `ImageDataset`.

## Overview

`SwiftVision` implements the full **YOLOv8** deep learning architecture in native Swift and MLX. The pipeline combines Apple Accelerate `vImageScale_PlanarF` for bilinear letterbox scaling, deep CSPDarknet backbone evaluation, decoupled PANet feature pyramid routing, and anchorless box decoding with Non-Maximum Suppression (NMS).

```
Input Image (RGB / File / CVPixelBuffer)
               │
               ▼
┌────────────────────────────────────────┐
│ Apple Accelerate vImage Letterbox      │ ───► Resizes to 640x640 with (114, 114, 114) padding
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ YOLOBackbone (CSPDarknet + SPPF)       │ ───► P3, P4, P5 multi-scale feature maps
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ YOLONeck (PANet Feature Pyramid)       │ ───► Top-down & bottom-up cross-scale fusion
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ YOLOHead (Decoupled Class & Box DFL)   │ ───► 8,400 raw candidate anchor predictions
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ Vectorized Non-Maximum Suppression     │ ───► Final BoundingBoxes with class IDs & confidences
└────────────────────────────────────────┘
```

---

## 1. Loading Weights and Executing Detection

Load pre-trained `.onnx` or `.safetensors` model weights and detect objects:

```swift
import Foundation
import SwiftVision

// 1. Initialize YOLOv8 detector from model weights file
let modelURL = URL(fileURLWithPath: "yolov8n.onnx")
let detector = try YOLOv8Detector(weightsURL: modelURL)

// 2. Perform object detection on an image file
let imageURL = URL(fileURLWithPath: "street_scene.jpg")
let detections = try await detector.detect(
    imageURL: imageURL,
    confidenceThreshold: 0.4,
    iouThreshold: 0.45
)

print("=== YOLOv8 Detection Results ===")
for box in detections {
    let rect = box.rect
    print("Class: '\(box.className)' (ID: \(box.classId)) | Confidence: \(String(format: "%.2f%%", box.confidence * 100)) | Box: [x=\(rect.minX), y=\(rect.minY), w=\(rect.width), h=\(rect.height)]")
}
```

---

## 2. Hardware-Accelerated Letterbox Preprocessing (`vImage`)

`YOLOPreprocessor` uses Apple Accelerate's `vImageScale_PlanarF` with `kvImageHighQualityResampling`, performing aspect-ratio preserving downsampling directly in hardware:

```swift
import SwiftVision

// Preprocess raw image buffer into normalized Float32 planar tensor
let inputImage = URL(fileURLWithPath: "input_1920x1080.jpg")
let (tensorBuffer, scaleFactor, padX, padY) = try YOLOPreprocessor.preprocess(
    imageURL: inputImage,
    targetSize: (width: 640, height: 640)
)

print("Preprocessed tensor count: \(tensorBuffer.count) Float elements.")
print("Scale Factor: \(scaleFactor), Padding: (\(padX), \(padY))")
```

---

## 3. Training & Batching Image Datasets

Load and normalize computer vision datasets in compact `Float32` tensors:

```swift
let dataset = try ImageDataset(
    directoryURL: URL(fileURLWithPath: "coco_val2017/"),
    targetSize: (width: 256, height: 256),
    normalize: true
)

print("Loaded \(dataset.count) images. Memory per image: \(dataset.data.count * 4 / 1024) KB.")
```
