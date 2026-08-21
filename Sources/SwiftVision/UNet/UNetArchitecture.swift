import Foundation
import MLX
import MLXNN

// MARK: - Deep Convolutional U-Net Architecture (Apple Silicon MLX)

/// Double 3x3 Convolution Block with Batch Normalization and ReLU/SiLU activation.
public class UNetDoubleConv: Module, UnaryLayer {
    @ModuleInfo public var conv1: Conv2d
    @ModuleInfo public var bn1: BatchNorm
    @ModuleInfo public var conv2: Conv2d
    @ModuleInfo public var bn2: BatchNorm

    /// Creates a Double Convolution block.
    /// - Parameters:
    ///   - inChannels: Number of input channels.
    ///   - outChannels: Number of output channels.
    ///   - midChannels: Number of intermediate channels (defaults to `outChannels`).
    public init(inChannels: Int, outChannels: Int, midChannels: Int? = nil) {
        let mid = midChannels ?? outChannels
        self.conv1 = Conv2d(inputChannels: inChannels, outputChannels: mid, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1), bias: false)
        self.bn1 = BatchNorm(featureCount: mid)
        self.conv2 = Conv2d(inputChannels: mid, outputChannels: outChannels, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1), bias: false)
        self.bn2 = BatchNorm(featureCount: outChannels)
        super.init()
    }

    /// Evaluates forward pass: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h1 = relu(bn1(conv1(x)))
        return relu(bn2(conv2(h1)))
    }
}

/// Downsampling block: 2x2 Max Pooling followed by DoubleConv.
public class UNetDown: Module, UnaryLayer {
    @ModuleInfo public var pool: MaxPool2d
    @ModuleInfo public var doubleConv: UNetDoubleConv

    /// Creates a Downsampling block.
    /// - Parameters:
    ///   - inChannels: Number of input channels.
    ///   - outChannels: Number of output channels.
    public init(inChannels: Int, outChannels: Int) {
        self.pool = MaxPool2d(kernelSize: IntOrPair(2), stride: IntOrPair(2))
        self.doubleConv = UNetDoubleConv(inChannels: inChannels, outChannels: outChannels)
        super.init()
    }

    /// Evaluates downsampling: MaxPool2d -> DoubleConv.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = pool(x)
        return doubleConv(pooled)
    }
}

/// Upsampling block: Bilinear/Stride Upsample + Skip Connection Concatenation + DoubleConv.
public class UNetUp: Module {
    @ModuleInfo public var upConv: Conv2d
    @ModuleInfo public var doubleConv: UNetDoubleConv

    /// Creates an Upsampling block.
    /// - Parameters:
    ///   - inChannels: Number of input channels from previous decoder layer.
    ///   - outChannels: Number of output channels.
    public init(inChannels: Int, outChannels: Int) {
        self.upConv = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0), bias: false)
        self.doubleConv = UNetDoubleConv(inChannels: inChannels, outChannels: outChannels)
        super.init()
    }

    /// Evaluates upsampling with skip-connection feature fusion:
    /// - Parameters:
    ///   - x1: Decoder feature map to be upsampled.
    ///   - x2: Encoder skip-connection feature map.
    /// - Returns: Fused high-resolution feature map.
    public func callAsFunction(x1: MLXArray, x2: MLXArray) -> MLXArray {
        // Upsample x1 spatially by a factor of 2 via nearest/bilinear interpolation
        let shape = x1.shape // [B, H, W, C] in NHWC format
        let b = shape[0]
        let h = shape[1]
        let w = shape[2]
        let c = shape[3]

        // Spatial 2x repeat interpolation
        let x1Up = upConv(x1)
        let x1Reshaped = x1Up.reshaped([b, h, 1, w, 1, x1Up.shape[3]])
        let x1Repeated = broadcast(x1Reshaped, to: [b, h, 2, w, 2, x1Up.shape[3]])
        let x1Expanded = x1Repeated.reshaped([b, h * 2, w * 2, x1Up.shape[3]])

        // Slice / crop x1Expanded to match x2 spatial dimensions if needed
        let targetH = x2.shape[1]
        let targetW = x2.shape[2]
        let x1Cropped = x1Expanded[0..<b, 0..<min(targetH, x1Expanded.shape[1]), 0..<min(targetW, x1Expanded.shape[2]), 0..<x1Expanded.shape[3]]
        let x2Cropped = x2[0..<b, 0..<x1Cropped.shape[1], 0..<x1Cropped.shape[2], 0..<x2.shape[3]]

        let concatenated = concatenated([x2Cropped, x1Cropped], axis: -1)
        return doubleConv(concatenated)
    }
}

/// Final 1x1 Convolution Head mapping feature maps to segmentation class probability logits.
public class UNetOutConv: Module, UnaryLayer {
    @ModuleInfo public var conv: Conv2d

    /// Creates an Output Convolution head.
    /// - Parameters:
    ///   - inChannels: Number of incoming decoder feature channels.
    ///   - outChannels: Number of target segmentation classes.
    public init(inChannels: Int, outChannels: Int) {
        self.conv = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0), bias: true)
        super.init()
    }

    /// Evaluates 1x1 Conv output layer.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

/// Deep Convolutional U-Net Neural Network for semantic image segmentation.
public class UNetArchitecture: Module, UnaryLayer {
    @ModuleInfo public var inc: UNetDoubleConv
    @ModuleInfo public var down1: UNetDown
    @ModuleInfo public var down2: UNetDown
    @ModuleInfo public var up1: UNetUp
    @ModuleInfo public var up2: UNetUp
    @ModuleInfo public var outc: UNetOutConv

    /// Number of input channels.
    public let inChannels: Int
    /// Number of output segmentation classes.
    public let numClasses: Int

    /// Creates a complete U-Net architecture.
    /// - Parameters:
    ///   - inChannels: Input image channels (e.g. 1 for grayscale, 3 for RGB).
    ///   - numClasses: Number of output segmentation classes.
    ///   - baseChannels: Base feature channel multiplier (default 16 for efficient mobile/UMA inference).
    public init(inChannels: Int = 3, numClasses: Int = 2, baseChannels: Int = 16) {
        self.inChannels = inChannels
        self.numClasses = numClasses

        let c1 = baseChannels
        let c2 = baseChannels * 2
        let c3 = baseChannels * 4

        self.inc = UNetDoubleConv(inChannels: inChannels, outChannels: c1)
        self.down1 = UNetDown(inChannels: c1, outChannels: c2)
        self.down2 = UNetDown(inChannels: c2, outChannels: c3)
        self.up1 = UNetUp(inChannels: c3, outChannels: c2)
        self.up2 = UNetUp(inChannels: c2, outChannels: c1)
        self.outc = UNetOutConv(inChannels: c1, outChannels: numClasses)
        super.init()
    }

    /// Evaluates full U-Net forward pass over input image tensor.
    /// - Parameter x: Input image tensor of shape `[Batch, Height, Width, Channels]` (NHWC).
    /// - Returns: Output class probability logits of shape `[Batch, Height, Width, NumClasses]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = inc(x)
        let x2 = down1(x1)
        let x3 = down2(x2)
        let xUp1 = up1(x1: x3, x2: x2)
        let xUp2 = up2(x1: xUp1, x2: x1)
        return outc(xUp2)
    }
}
