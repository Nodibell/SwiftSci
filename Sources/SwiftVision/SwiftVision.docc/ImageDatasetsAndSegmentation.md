# Computer Vision & Image Datasets

Load image datasets, perform image augmentation, and train U-Net neural network architectures for image segmentation.

## Overview

Native computer vision tools with batch loading and U-Net metrics.

### 1. Loading Image Datasets

```swift
import SwiftVision

let dataset = try ImageDataset(directoryURL: imageDir, targetSize: (224, 224))
print("Loaded \(dataset.count) images across \(dataset.classNames) classes.")
```

### 2. U-Net Segmentation Metrics

```swift
let dice = UNetMetrics.diceScore(predMask: pred, trueMask: target)
print("Dice Coefficient: \(dice)")
```
