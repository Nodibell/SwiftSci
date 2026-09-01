import Foundation
import MLX
import MLXNN

/// Conv2d + BatchNorm + SiLU block for YOLOv8.
public class ConvBlock: Module, UnaryLayer {
    @ModuleInfo public var conv: Conv2d
    @ModuleInfo public var bn: BatchNorm

    /// Creates a Convolutional block.
    /// - Parameters:
    ///   - cIn: Input channel dimension count.
    ///   - cOut: Output channel dimension count.
    ///   - k: Kernel spatial dimension size (default `1`).
    ///   - s: Convolution stride rate (default `1`).
    ///   - p: Explicit spatial padding width.
    ///   - g: Convolution group count (default `1`).
    public init(cIn: Int, cOut: Int, k: Int = 1, s: Int = 1, p: Int? = nil, g: Int = 1) {
        let pad = p ?? (k / 2)
        self.conv = Conv2d(inputChannels: cIn, outputChannels: cOut, kernelSize: IntOrPair(k), stride: IntOrPair(s), padding: IntOrPair(pad), groups: g, bias: false)
        self.bn = BatchNorm(featureCount: cOut, eps: 1e-3, momentum: 0.03)
        super.init()
    }

    /// Evaluates forward pass of Conv2d -> BatchNorm -> SiLU activation.
    /// - Parameter x: Input feature tensor.
    /// - Returns: Processed output feature map array.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return silu(bn(conv(x)))
    }
}

/// Bottleneck block with optional residual connection.
public class BottleneckBlock: Module, UnaryLayer {
    @ModuleInfo public var cv1: ConvBlock
    @ModuleInfo public var cv2: ConvBlock
    /// Flag indicating whether residual identity shortcut is enabled.
    public let shortcut: Bool

    /// Creates a Bottleneck block.
    /// - Parameters:
    ///   - cIn: Input channel dimension count.
    ///   - cOut: Output channel dimension count.
    ///   - shortcut: Enable identity addition shortcut (default `true`).
    ///   - g: Group count for convolution.
    ///   - e: Expansion factor for inner hidden channel depth.
    public init(cIn: Int, cOut: Int, shortcut: Bool = true, g: Int = 1, e: Float = 0.5) {
        let cHidden = Int(Float(cOut) * e)
        self.cv1 = ConvBlock(cIn: cIn, cOut: cHidden, k: 3, s: 1)
        self.cv2 = ConvBlock(cIn: cHidden, cOut: cOut, k: 3, s: 1, g: g)
        self.shortcut = shortcut && (cIn == cOut)
        super.init()
    }

    /// Evaluates forward pass through dual 3x3 convolutions with optional identity shortcut addition.
    /// - Parameter x: Input feature tensor.
    /// - Returns: Processed bottleneck output tensor.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = cv2(cv1(x))
        return shortcut ? (x + y) : y
    }
}

/// C2f block: split channels + sequential bottlenecks + multi-stage channel concatenation.
public class C2fBlock: Module, UnaryLayer {
    @ModuleInfo public var cv1: ConvBlock
    @ModuleInfo public var cv2: ConvBlock
    @ModuleInfo var bottleneckList: [BottleneckBlock]
    /// Output channel count for C2f block.
    public let cOut: Int
    /// Hidden inner channel dimension count.
    public let cHidden: Int

    /// Creates a C2f block.
    /// - Parameters:
    ///   - cIn: Input channel count.
    ///   - cOut: Output channel count.
    ///   - n: Number of sequential bottleneck blocks.
    ///   - shortcut: Enable identity shortcut in sub-bottlenecks.
    ///   - g: Group count.
    ///   - e: Expansion ratio.
    public init(cIn: Int, cOut: Int, n: Int = 1, shortcut: Bool = true, g: Int = 1, e: Float = 0.5) {
        self.cOut = cOut
        self.cHidden = Int(Float(cOut) * e)
        self.cv1 = ConvBlock(cIn: cIn, cOut: 2 * self.cHidden, k: 1, s: 1)

        var blocks = [BottleneckBlock]()
        for _ in 0..<n {
            blocks.append(BottleneckBlock(cIn: self.cHidden, cOut: self.cHidden, shortcut: shortcut, g: g, e: 1.0))
        }
        self.bottleneckList = blocks
        self.cv2 = ConvBlock(cIn: (2 + n) * self.cHidden, cOut: cOut, k: 1, s: 1)
        super.init()
    }

    /// Evaluates forward pass over split channels, sequential bottlenecks, and final channel fusion.
    /// - Parameter x: Input feature map tensor.
    /// - Returns: Processed C2f output tensor.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x12 = cv1(x)
        let parts = split(x12, parts: 2, axis: 3)
        var y = [parts[0], parts[1]]
        var current = parts[1]
        for b in bottleneckList {
            current = b(current)
            y.append(current)
        }
        return cv2(concatenated(y, axis: 3))
    }
}

/// Spatial Pyramid Pooling - Fast (SPPF) block.
public class SPPFBlock: Module, UnaryLayer {
    @ModuleInfo public var cv1: ConvBlock
    @ModuleInfo public var cv2: ConvBlock
    @ModuleInfo public var m: MaxPool2d

    /// Kernel size for SPPF max pooling.
    public let k: Int

    /// Creates an SPPF block.
    /// - Parameters:
    ///   - cIn: Input channel count.
    ///   - cOut: Output channel count.
    ///   - k: Max pooling kernel size.
    public init(cIn: Int, cOut: Int, k: Int = 5) {
        self.k = k
        let cHidden = cIn / 2
        self.cv1 = ConvBlock(cIn: cIn, cOut: cHidden, k: 1, s: 1)
        self.cv2 = ConvBlock(cIn: cHidden * 4, cOut: cOut, k: 1, s: 1)
        self.m = MaxPool2d(kernelSize: IntOrPair(k), stride: IntOrPair(1), padding: IntOrPair(0))
        super.init()
    }

    private func pool(_ x: MLXArray) -> MLXArray {
        let padVal = (-Float.infinity).asMLXArray(dtype: x.dtype)
        let padSize = k / 2
        let widths: [IntOrPair] = [IntOrPair(0), IntOrPair(padSize), IntOrPair(padSize), IntOrPair(0)]
        let paddedX = padded(x, widths: widths, mode: .constant, value: padVal)
        return m(paddedX)
    }

    /// Evaluates forward pass through sequential 5x5 max pooling pyramids and channel fusion.
    /// - Parameter x: Input feature map.
    /// - Returns: SPPF output tensor.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = cv1(x)
        let y1 = pool(x1)
        let y2 = pool(y1)
        let y3 = pool(y2)
        return cv2(concatenated([x1, y1, y2, y3], axis: 3))
    }
}

/// Backbone output containing multi-scale feature maps P3, P4, P5.
public struct YOLOBackboneOutput {
    /// P3 feature map (stride 8, 80x80x64 for 640x640 input).
    public let p3: MLXArray
    /// P4 feature map (stride 16, 40x40x128 for 640x640 input).
    public let p4: MLXArray
    /// P5 feature map (stride 32, 20x20x256 for 640x640 input).
    public let p5: MLXArray

    /// Creates a YOLOBackboneOutput structure.
    /// - Parameters:
    ///   - p3: P3 feature map tensor.
    ///   - p4: P4 feature map tensor.
    ///   - p5: P5 feature map tensor.
    public init(p3: MLXArray, p4: MLXArray, p5: MLXArray) {
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
    }
}

/// CSPDarknet Backbone for YOLOv8n (3.2M model).
public class YOLOBackbone: Module {
    @ModuleInfo public var b0: ConvBlock  // 3 -> 16, s=2
    @ModuleInfo public var b1: ConvBlock  // 16 -> 32, s=2
    @ModuleInfo public var b2: C2fBlock   // 32 -> 32, n=1
    @ModuleInfo public var b3: ConvBlock  // 32 -> 64, s=2 (P3)
    @ModuleInfo public var b4: C2fBlock   // 64 -> 64, n=2
    @ModuleInfo public var b5: ConvBlock  // 64 -> 128, s=2 (P4)
    @ModuleInfo public var b6: C2fBlock   // 128 -> 128, n=2
    @ModuleInfo public var b7: ConvBlock  // 128 -> 256, s=2 (P5)
    @ModuleInfo public var b8: C2fBlock   // 256 -> 256, n=1
    @ModuleInfo public var b9: SPPFBlock  // 256 -> 256, k=5

    /// Initializes a new CSPDarknet YOLOBackbone instance.
    public override init() {
        self.b0 = ConvBlock(cIn: 3, cOut: 16, k: 3, s: 2)
        self.b1 = ConvBlock(cIn: 16, cOut: 32, k: 3, s: 2)
        self.b2 = C2fBlock(cIn: 32, cOut: 32, n: 1)
        self.b3 = ConvBlock(cIn: 32, cOut: 64, k: 3, s: 2)
        self.b4 = C2fBlock(cIn: 64, cOut: 64, n: 2)
        self.b5 = ConvBlock(cIn: 64, cOut: 128, k: 3, s: 2)
        self.b6 = C2fBlock(cIn: 128, cOut: 128, n: 2)
        self.b7 = ConvBlock(cIn: 128, cOut: 256, k: 3, s: 2)
        self.b8 = C2fBlock(cIn: 256, cOut: 256, n: 1)
        self.b9 = SPPFBlock(cIn: 256, cOut: 256, k: 5)
        super.init()
    }

    /// Evaluates forward pass over CSPDarknet backbone generating multi-scale feature maps P3, P4, P5.
    /// - Parameter x: Preprocessed image tensor [Batch, 640, 640, 3].
    /// - Returns: `YOLOBackboneOutput` containing multi-scale P3, P4, P5 arrays.
    public func callAsFunction(_ x: MLXArray) -> YOLOBackboneOutput {
        let x0 = b0(x)
        let x1 = b1(x0)
        let x2 = b2(x1)
        let x3 = b3(x2)
        let p3 = b4(x3)  // P3 feature map (64 ch)
        let x5 = b5(p3)
        let p4 = b6(x5)  // P4 feature map (128 ch)
        let x7 = b7(p4)
        let x8 = b8(x7)
        let p5 = b9(x8)  // P5 feature map (256 ch)
        return YOLOBackboneOutput(p3: p3, p4: p4, p5: p5)
    }

    /// Loads weights for all CSPDarknet backbone stages from the given loader.
    public func loadWeights(from loader: YOLOWeightLoader, prefix: String = "model") {
        var params: [String: MLXArray] = [:]
        for i in 0...9 {
            let pfx = "\(prefix).\(i)"
            let bName = "b\(i)"
            if let w = loader.get("\(pfx).conv.weight") { params["\(bName).conv.weight"] = w }
            if let b = loader.get("\(pfx).conv.bias") { params["\(bName).conv.bias"] = b }
            if let w = loader.get("\(pfx).bn.weight") { params["\(bName).bn.weight"] = w }
            if let b = loader.get("\(pfx).bn.bias") { params["\(bName).bn.bias"] = b }
            if let rm = loader.get("\(pfx).bn.running_mean") { params["\(bName).bn.runningMean"] = rm }
            if let rv = loader.get("\(pfx).bn.running_var") { params["\(bName).bn.runningVar"] = rv }

            if let w = loader.get("\(pfx).cv1.conv.weight") { params["\(bName).cv1.conv.weight"] = w }
            if let w = loader.get("\(pfx).cv1.bn.weight") { params["\(bName).cv1.bn.weight"] = w }
            if let b = loader.get("\(pfx).cv1.bn.bias") { params["\(bName).cv1.bn.bias"] = b }
            if let w = loader.get("\(pfx).cv2.conv.weight") { params["\(bName).cv2.conv.weight"] = w }
            if let w = loader.get("\(pfx).cv2.bn.weight") { params["\(bName).cv2.bn.weight"] = w }
            if let b = loader.get("\(pfx).cv2.bn.bias") { params["\(bName).cv2.bn.bias"] = b }

            for mIdx in 0..<10 {
                if let w = loader.get("\(pfx).m.\(mIdx).cv1.conv.weight") { params["\(bName).bottleneckList.\(mIdx).cv1.conv.weight"] = w }
                if let w = loader.get("\(pfx).m.\(mIdx).cv1.bn.weight") { params["\(bName).bottleneckList.\(mIdx).cv1.bn.weight"] = w }
                if let b = loader.get("\(pfx).m.\(mIdx).cv1.bn.bias") { params["\(bName).bottleneckList.\(mIdx).cv1.bn.bias"] = b }
                if let w = loader.get("\(pfx).m.\(mIdx).cv2.conv.weight") { params["\(bName).bottleneckList.\(mIdx).cv2.conv.weight"] = w }
                if let w = loader.get("\(pfx).m.\(mIdx).cv2.bn.weight") { params["\(bName).bottleneckList.\(mIdx).cv2.bn.weight"] = w }
                if let b = loader.get("\(pfx).m.\(mIdx).cv2.bn.bias") { params["\(bName).bottleneckList.\(mIdx).cv2.bn.bias"] = b }
            }
        }
        if !params.isEmpty {
            self.update(parameters: NestedDictionary.unflattened(params))
        }
    }
}
