# Deep Convolutional U-Net Semantic Segmentation

Train and deploy deep convolutional U-Net neural networks for image segmentation directly on Apple Silicon MLX GPU.

## Overview

The U-Net architecture is the gold standard for semantic image segmentation, consisting of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) for precise spatial localization, linked via skip connections.

```
Input Image ──► [ DoubleConv ] ────────────────────── Skip ─────────────────────► [ Up + Conv ] ──► Mask
                     │                                                                ▲
                  [ Down ] ──► [ DoubleConv ] ──────── Skip ───────► [ Up + Conv ] ───┘
                                   │                                     ▲
                                [ Down ] ──► [ Bottleneck Conv ] ────────┘
```

## 1. Initializing and Running U-Net

```swift
import SwiftVision

// Initialize U-Net on Apple Silicon Metal GPU
let unet = UNetArchitecture(inChannels: 3, outChannels: 1, baseFeatures: 32)

// Forward pass on image tensor (Batch x Height x Width x Channels)
let imageTensor = try ImageTensorLoader.load(url: URL(fileURLWithPath: "sample.jpg"))
let maskOutput = try unet.forward(image: imageTensor)

print("Output mask dimensions: \(maskOutput.shape)")
```

## Topics

### Vision Network Types
- ``UNetArchitecture``
- ``UNetDoubleConv``
- ``UNetDown``
- ``UNetUp``
- ``UNetOutConv``
