# Real-Time YOLOv8 Object Detection on Apple GPU

Deploy real-time anchorless object detection models with ONNX weight parsing and Metal acceleration.

## Overview

`SwiftVision` features real-time object detection via **YOLOv8**, including letterbox preprocessing, convolutional backbone inference on Apple Silicon MLX GPU, and accelerated Non-Maximum Suppression (NMS).

## 1. Running Object Detection

```swift
import SwiftVision

let detector = try YOLOv8Detector(weightsURL: URL(fileURLWithPath: "yolov8n.onnx"))

let imageURL = URL(fileURLWithPath: "street_scene.jpg")
let detections = try await detector.detect(imageURL: imageURL, confidenceThreshold: 0.5, iouThreshold: 0.45)

for box in detections {
    print("Detected \(box.className) [Conf: \(box.confidence)] at \(box.rect)")
}
```

## Topics

### YOLO Types
- ``YOLOv8Detector``
- ``BoundingBox``
