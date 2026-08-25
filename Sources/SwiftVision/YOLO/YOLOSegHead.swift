#if os(macOS)
import Foundation
import MLX
import MLXNN

/// Raw predictions output by YOLOSegHead containing bounding boxes, class scores, mask coefficients, and prototype masks.
public struct YOLOSegOutput {
    /// Decoded bounding boxes in xyxy format `[Batch, numAnchors, 4]`.
    public let boxes: MLXArray
    /// Class probabilities `[Batch, numAnchors, numClasses]`.
    public let scores: MLXArray
    /// Instance mask coefficients `[Batch, numAnchors, numMasks]`.
    public let maskCoefficients: MLXArray
    /// Prototype mask feature maps `[Batch, numMasks, protoH, protoW]`.
    public let protos: MLXArray

    /// Creates a new YOLOSegOutput container.
    ///
    /// - Parameters:
    ///   - boxes: Decoded bounding boxes `[Batch, numAnchors, 4]`.
    ///   - scores: Classification probabilities `[Batch, numAnchors, numClasses]`.
    ///   - maskCoefficients: Mask coefficient tensor `[Batch, numAnchors, numMasks]`.
    ///   - protos: Prototype masks tensor `[Batch, numMasks, protoH, protoW]`.
    public init(boxes: MLXArray, scores: MLXArray, maskCoefficients: MLXArray, protos: MLXArray) {
        self.boxes = boxes
        self.scores = scores
        self.maskCoefficients = maskCoefficients
        self.protos = protos
    }
}

/// YOLOv8-Seg Instance Segmentation Head with prototype mask generator and mask coefficient branches.
///
/// `YOLOSegHead` extends YOLO detection with dual prototype mask rendering and coefficient regression
/// to generate pixel-level object segmentation masks on Apple Silicon GPU.
public class YOLOSegHead: Module {
    /// Number of object classification categories (default: 80).
    public let numClasses: Int
    /// Number of prototype masks (default: 32).
    public let numMasks: Int
    /// Distribution Focal Loss maximum regression bin count (default: 16).
    public let regMax: Int

    // Standard detection head
    @ModuleInfo public var head: YOLOHead

    // Mask Coefficient regression branches for P3, P4, P5
    @ModuleInfo public var cv4_0_0: ConvBlock
    @ModuleInfo public var cv4_0_1: ConvBlock
    @ModuleInfo public var cv4_0_2: Conv2d

    @ModuleInfo public var cv4_1_0: ConvBlock
    @ModuleInfo public var cv4_1_1: ConvBlock
    @ModuleInfo public var cv4_1_2: Conv2d

    @ModuleInfo public var cv4_2_0: ConvBlock
    @ModuleInfo public var cv4_2_1: ConvBlock
    @ModuleInfo public var cv4_2_2: Conv2d

    // Prototype mask generator (Proto head on P3)
    @ModuleInfo public var proto_cv1: ConvBlock
    @ModuleInfo public var proto_cv2: ConvBlock
    @ModuleInfo public var proto_cv3: Conv2d

    /// Initializes a YOLOv8 instance segmentation head.
    ///
    /// - Parameters:
    ///   - numClasses: Target category count (default: 80).
    ///   - numMasks: Prototype mask count (default: 32).
    ///   - regMax: Regression bin count (default: 16).
    public init(numClasses: Int = 80, numMasks: Int = 32, regMax: Int = 16) {
        self.numClasses = numClasses
        self.numMasks = numMasks
        self.regMax = regMax

        self.head = YOLOHead(numClasses: numClasses, regMax: regMax)

        let ch = [64, 128, 256]
        let cMask = 32

        // P3 Mask Branch
        self.cv4_0_0 = ConvBlock(cIn: ch[0], cOut: cMask, k: 3, s: 1)
        self.cv4_0_1 = ConvBlock(cIn: cMask, cOut: cMask, k: 3, s: 1)
        self.cv4_0_2 = Conv2d(inputChannels: cMask, outputChannels: numMasks, kernelSize: IntOrPair(1))

        // P4 Mask Branch
        self.cv4_1_0 = ConvBlock(cIn: ch[1], cOut: cMask, k: 3, s: 1)
        self.cv4_1_1 = ConvBlock(cIn: cMask, cOut: cMask, k: 3, s: 1)
        self.cv4_1_2 = Conv2d(inputChannels: cMask, outputChannels: numMasks, kernelSize: IntOrPair(1))

        // P5 Mask Branch
        self.cv4_2_0 = ConvBlock(cIn: ch[2], cOut: cMask, k: 3, s: 1)
        self.cv4_2_1 = ConvBlock(cIn: cMask, cOut: cMask, k: 3, s: 1)
        self.cv4_2_2 = Conv2d(inputChannels: cMask, outputChannels: numMasks, kernelSize: IntOrPair(1))

        // Proto Head (expands P3 feature map to 160x160 resolution with numMasks channels)
        self.proto_cv1 = ConvBlock(cIn: ch[0], cOut: 64, k: 3, s: 1)
        self.proto_cv2 = ConvBlock(cIn: 64, cOut: 64, k: 3, s: 1)
        self.proto_cv3 = Conv2d(inputChannels: 64, outputChannels: numMasks, kernelSize: IntOrPair(1))

        super.init()
    }

    /// Evaluates instance segmentation forward pass over feature pyramid maps.
    ///
    /// - Parameter neckOut: Feature pyramid output maps from YOLONeck.
    /// - Returns: `YOLOSegOutput` containing boxes, scores, mask coefficients, and prototype feature maps.
    public func callAsFunction(_ neckOut: YOLONeckOutput) -> YOLOSegOutput {
        // 1. Detection outputs (boxes and scores)
        let detOut = head(neckOut)

        // 2. Mask coefficients from P3, P4, P5
        let mc0 = cv4_0_2(cv4_0_1(cv4_0_0(neckOut.headP3)))
        let mc1 = cv4_1_2(cv4_1_1(cv4_1_0(neckOut.headP4)))
        let mc2 = cv4_2_2(cv4_2_1(cv4_2_0(neckOut.headP5)))

        // Flatten spatial dims: [B, H, W, numMasks] -> [B, H*W, numMasks]
        let b0 = mc0.reshaped([mc0.shape[0], mc0.shape[1] * mc0.shape[2], numMasks])
        let b1 = mc1.reshaped([mc1.shape[0], mc1.shape[1] * mc1.shape[2], numMasks])
        let b2 = mc2.reshaped([mc2.shape[0], mc2.shape[1] * mc2.shape[2], numMasks])

        let allCoeffs = MLX.concatenated([b0, b1, b2], axis: 1)

        // 3. Proto mask generation on P3
        let p = proto_cv3(proto_cv2(proto_cv1(neckOut.headP3)))

        return YOLOSegOutput(
            boxes: detOut.boxes,
            scores: detOut.scores,
            maskCoefficients: allCoeffs,
            protos: p
        )
    }

    /// Computes instance segmentation binary mask for a detected object by combining mask coefficients and prototype maps.
    ///
    /// - Parameters:
    ///   - coeff: Detection mask coefficients tensor `[numMasks]`.
    ///   - protos: Prototype mask feature maps `[H, W, numMasks]` or `[numMasks, H, W]`.
    ///   - threshold: Binary activation cutoff threshold (default: 0.5).
    /// - Returns: Binary 2D mask array `[H, W]`.
    public static func decodeMask(coeff: MLXArray, protos: MLXArray, threshold: Float = 0.5) -> MLXArray {
        var p = protos
        if p.ndim == 3 && p.shape[2] == coeff.shape[0] {
            // [H, W, numMasks] @ [numMasks, 1] -> [H, W]
            let c = coeff.reshaped([coeff.shape[0], 1])
            let rawMask = matmul(p, c).squeezed(axis: -1)
            let sigMask = sigmoid(rawMask)
            return sigMask .> Float(threshold)
        } else if p.ndim == 3 && p.shape[0] == coeff.shape[0] {
            // [numMasks, H, W] -> transpose to [H, W, numMasks]
            p = p.transposed(1, 2, 0)
            let c = coeff.reshaped([coeff.shape[0], 1])
            let rawMask = matmul(p, c).squeezed(axis: -1)
            let sigMask = sigmoid(rawMask)
            return sigMask .> Float(threshold)
        }
        let sigMask = sigmoid(protos)
        return sigMask .> Float(threshold)
    }
}
#endif
