import Foundation
import MLX
import MLXNN

/// Raw predictions output by YOLOHead across all 8,400 anchor cells.
public struct YOLOHeadOutput {
    /// Bounding boxes in xyxy format [Batch, 8400, 4].
    public let boxes: MLXArray
    /// Class probabilities [Batch, 8400, numClasses].
    public let scores: MLXArray

    /// Creates a YOLOHeadOutput container.
    /// - Parameters:
    ///   - boxes: Decoded bounding box tensor [Batch, 8400, 4].
    ///   - scores: Classification probabilities tensor [Batch, 8400, numClasses].
    public init(boxes: MLXArray, scores: MLXArray) {
        self.boxes = boxes
        self.scores = scores
    }
}

/// Decoupled Anchor-free Detection Head with Distribution Focal Loss (DFL) decoding for YOLOv8n.
public class YOLOHead: Module {
    /// Number of object classification categories (defaults to 80 for COCO).
    public let numClasses: Int
    /// Distribution Focal Loss maximum regression bin count (default 16).
    public let regMax: Int
    
    // Scale 0 (P3, stride 8, cIn=64)
    @ModuleInfo public var cv2_0_0: ConvBlock
    @ModuleInfo public var cv2_0_1: ConvBlock
    @ModuleInfo public var cv2_0_2: Conv2d
    @ModuleInfo public var cv3_0_0: ConvBlock
    @ModuleInfo public var cv3_0_1: ConvBlock
    @ModuleInfo public var cv3_0_2: Conv2d

    // Scale 1 (P4, stride 16, cIn=128)
    @ModuleInfo public var cv2_1_0: ConvBlock
    @ModuleInfo public var cv2_1_1: ConvBlock
    @ModuleInfo public var cv2_1_2: Conv2d
    @ModuleInfo public var cv3_1_0: ConvBlock
    @ModuleInfo public var cv3_1_1: ConvBlock
    @ModuleInfo public var cv3_1_2: Conv2d

    // Scale 2 (P5, stride 32, cIn=256)
    @ModuleInfo public var cv2_2_0: ConvBlock
    @ModuleInfo public var cv2_2_1: ConvBlock
    @ModuleInfo public var cv2_2_2: Conv2d
    @ModuleInfo public var cv3_2_0: ConvBlock
    @ModuleInfo public var cv3_2_1: ConvBlock
    @ModuleInfo public var cv3_2_2: Conv2d

    /// Pre-computed DFL projection array [16] = [0, 1, 2, ..., 15]
    public let dflProj: MLXArray

    /// Initializes a decoupled detection head.
    /// - Parameters:
    ///   - numClasses: Target category count (default 80).
    ///   - regMax: Regression bin count (default 16).
    public init(numClasses: Int = 80, regMax: Int = 16) {
        self.numClasses = numClasses
        self.regMax = regMax
        self.dflProj = MLXArray(Array(0..<regMax).map { Float($0) })

        let ch = [64, 128, 256]
        let cReg = 64
        let cCls = max(numClasses, ch[0])

        // P3 Branch
        self.cv2_0_0 = ConvBlock(cIn: ch[0], cOut: cReg, k: 3, s: 1)
        self.cv2_0_1 = ConvBlock(cIn: cReg, cOut: cReg, k: 3, s: 1)
        self.cv2_0_2 = Conv2d(inputChannels: cReg, outputChannels: 4 * regMax, kernelSize: IntOrPair(1))
        self.cv3_0_0 = ConvBlock(cIn: ch[0], cOut: cCls, k: 3, s: 1)
        self.cv3_0_1 = ConvBlock(cIn: cCls, cOut: cCls, k: 3, s: 1)
        self.cv3_0_2 = Conv2d(inputChannels: cCls, outputChannels: numClasses, kernelSize: IntOrPair(1))

        // P4 Branch
        self.cv2_1_0 = ConvBlock(cIn: ch[1], cOut: cReg, k: 3, s: 1)
        self.cv2_1_1 = ConvBlock(cIn: cReg, cOut: cReg, k: 3, s: 1)
        self.cv2_1_2 = Conv2d(inputChannels: cReg, outputChannels: 4 * regMax, kernelSize: IntOrPair(1))
        self.cv3_1_0 = ConvBlock(cIn: ch[1], cOut: cCls, k: 3, s: 1)
        self.cv3_1_1 = ConvBlock(cIn: cCls, cOut: cCls, k: 3, s: 1)
        self.cv3_1_2 = Conv2d(inputChannels: cCls, outputChannels: numClasses, kernelSize: IntOrPair(1))

        // P5 Branch
        self.cv2_2_0 = ConvBlock(cIn: ch[2], cOut: cReg, k: 3, s: 1)
        self.cv2_2_1 = ConvBlock(cIn: cReg, cOut: cReg, k: 3, s: 1)
        self.cv2_2_2 = Conv2d(inputChannels: cReg, outputChannels: 4 * regMax, kernelSize: IntOrPair(1))
        self.cv3_2_0 = ConvBlock(cIn: ch[2], cOut: cCls, k: 3, s: 1)
        self.cv3_2_1 = ConvBlock(cIn: cCls, cOut: cCls, k: 3, s: 1)
        self.cv3_2_2 = Conv2d(inputChannels: cCls, outputChannels: numClasses, kernelSize: IntOrPair(1))

        super.init()
    }

    /// Evaluates decoupled classification and DFL bounding box regression over all 8,400 anchor cells.
    /// - Parameter neckOut: Feature pyramid output maps from YOLONeck.
    /// - Returns: `YOLOHeadOutput` containing decoded bounding boxes and class scores.
    public func callAsFunction(_ neckOut: YOLONeckOutput) -> YOLOHeadOutput {
        let inputs = [neckOut.headP3, neckOut.headP4, neckOut.headP5]
        let strides: [Float] = [8.0, 16.0, 32.0]
        let batchSize = inputs[0].dim(0)

        var allDecodedBoxes = [MLXArray]()
        var allScores = [MLXArray]()

        for i in 0..<3 {
            let feat = inputs[i]
            let stride = strides[i]
            let h = feat.dim(1)
            let w = feat.dim(2)

            // Box regression branch
            let regFeat: MLXArray
            if i == 0 {
                regFeat = cv2_0_2(cv2_0_1(cv2_0_0(feat)))
            } else if i == 1 {
                regFeat = cv2_1_2(cv2_1_1(cv2_1_0(feat)))
            } else {
                regFeat = cv2_2_2(cv2_2_1(cv2_2_0(feat)))
            }

            // Classification branch
            let clsFeat: MLXArray
            if i == 0 {
                clsFeat = cv3_0_2(cv3_0_1(cv3_0_0(feat)))
            } else if i == 1 {
                clsFeat = cv3_1_2(cv3_1_1(cv3_1_0(feat)))
            } else {
                clsFeat = cv3_2_2(cv3_2_1(cv3_2_0(feat)))
            }

            // Reshape regFeat [B, H, W, 4*16] -> [B, H*W, 4, 16]
            let regReshaped = regFeat.reshaped(batchSize, h * w, 4, regMax)
            let regSoftmax = softmax(regReshaped, axis: -1)
            // Dot product with dflProj [16] -> [B, H*W, 4]
            let distances = matmul(regSoftmax, dflProj.reshaped(regMax, 1)).reshaped(batchSize, h * w, 4)

            // Sigmoid class scores [B, H, W, numClasses] -> [B, H*W, numClasses]
            let scores = sigmoid(clsFeat.reshaped(batchSize, h * w, numClasses))

            // Build anchor center grid coordinates for (H, W) at this stride
            var gridX = [Float]()
            var gridY = [Float]()
            gridX.reserveCapacity(h * w)
            gridY.reserveCapacity(h * w)

            for gy in 0..<h {
                for gx in 0..<w {
                    gridX.append((Float(gx) + 0.5) * stride)
                    gridY.append((Float(gy) + 0.5) * stride)
                }
            }

            let cx = MLXArray(gridX).reshaped(1, h * w, 1)
            let cy = MLXArray(gridY).reshaped(1, h * w, 1)

            let distL = distances[.ellipsis, 0].reshaped(batchSize, h * w, 1) * stride
            let distT = distances[.ellipsis, 1].reshaped(batchSize, h * w, 1) * stride
            let distR = distances[.ellipsis, 2].reshaped(batchSize, h * w, 1) * stride
            let distB = distances[.ellipsis, 3].reshaped(batchSize, h * w, 1) * stride


            let xMin = cx - distL
            let yMin = cy - distT
            let xMax = cx + distR
            let yMax = cy + distB

            let decodedBoxes = concatenated([xMin, yMin, xMax, yMax], axis: 2)

            allDecodedBoxes.append(decodedBoxes)
            allScores.append(scores)
        }

        let totalBoxes = concatenated(allDecodedBoxes, axis: 1)
        let totalScores = concatenated(allScores, axis: 1)

        return YOLOHeadOutput(boxes: totalBoxes, scores: totalScores)
    }
}
