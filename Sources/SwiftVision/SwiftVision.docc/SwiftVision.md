# ``SwiftVision``

Computer Vision Pipelines & Image Datasets.

## Overview

`SwiftVision` provides image dataset loading, augmentation pipelines, and U-Net segmentation neural architectures.

### Key Capabilities

- **Image Dataset Loader**: `ImageDataset` directory scanner with target resizing and batching.
- **Segmentation Models**: `UNet` neural network architecture for semantic segmentation.
- **Evaluation Metrics**: `DiceScore` and `IoU` (Intersection over Union) segmentation metrics.

### Example Usage

```swift
import SwiftVision

let dataset = try ImageDataset(directoryURL: imageDir, targetSize: (224, 224))
```

## Topics

### Guides & Tutorials
- <doc:ImageDatasetsAndSegmentation>
