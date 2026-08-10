import Foundation
import MLX
import MLXNN

/// Neck feature maps for detection head input (head_p3, head_p4, head_p5).
public struct YOLONeckOutput {
    /// P3 neck output (stride 8, 80x80x64).
    public let headP3: MLXArray
    /// P4 neck output (stride 16, 40x40x128).
    public let headP4: MLXArray
    /// P5 neck output (stride 32, 20x20x256).
    public let headP5: MLXArray

    public init(headP3: MLXArray, headP4: MLXArray, headP5: MLXArray) {
        self.headP3 = headP3
        self.headP4 = headP4
        self.headP5 = headP5
    }
}

/// PANet Feature Pyramid Network (Neck) for YOLOv8n.
public class YOLONeck: Module {
    @ModuleInfo public var upsample: Upsample
    @ModuleInfo public var c2f_p4: C2fBlock   // 384 -> 128, n=1
    @ModuleInfo public var c2f_p3: C2fBlock   // 192 -> 64, n=1
    @ModuleInfo public var conv_p4: ConvBlock  // 64 -> 64, s=2
    @ModuleInfo public var c2f_n4: C2fBlock   // 192 -> 128, n=1
    @ModuleInfo public var conv_p5: ConvBlock  // 128 -> 128, s=2
    @ModuleInfo public var c2f_n5: C2fBlock   // 384 -> 256, n=1

    public override init() {
        self.upsample = Upsample(scaleFactor: 2.0, mode: .nearest)
        self.c2f_p4 = C2fBlock(cIn: 384, cOut: 128, n: 1, shortcut: false)
        self.c2f_p3 = C2fBlock(cIn: 192, cOut: 64, n: 1, shortcut: false)
        self.conv_p4 = ConvBlock(cIn: 64, cOut: 64, k: 3, s: 2)
        self.c2f_n4 = C2fBlock(cIn: 192, cOut: 128, n: 1, shortcut: false)
        self.conv_p5 = ConvBlock(cIn: 128, cOut: 128, k: 3, s: 2)
        self.c2f_n5 = C2fBlock(cIn: 384, cOut: 256, n: 1, shortcut: false)
        super.init()
    }

    public func callAsFunction(_ backboneOut: YOLOBackboneOutput) -> YOLONeckOutput {
        let p3 = backboneOut.p3  // [B, 80, 80, 64]
        let p4 = backboneOut.p4  // [B, 40, 40, 128]
        let p5 = backboneOut.p5  // [B, 20, 20, 256]

        // Top-down path
        let p5_up = upsample(p5)                      // [B, 40, 40, 256]
        let cat_p4 = concatenated([p5_up, p4], axis: 3) // [B, 40, 40, 384]
        let n_p4 = c2f_p4(cat_p4)                       // [B, 40, 40, 128]

        let p4_up = upsample(n_p4)                     // [B, 80, 80, 128]
        let cat_p3 = concatenated([p4_up, p3], axis: 3) // [B, 80, 80, 192]
        let headP3 = c2f_p3(cat_p3)                    // [B, 80, 80, 64]  (stride 8)

        // Bottom-up path
        let p3_down = conv_p4(headP3)                  // [B, 40, 40, 64]
        let cat_n4 = concatenated([p3_down, n_p4], axis: 3) // [B, 40, 40, 192]
        let headP4 = c2f_n4(cat_n4)                    // [B, 40, 40, 128] (stride 16)

        let p4_down = conv_p5(headP4)                  // [B, 20, 20, 128]
        let cat_n5 = concatenated([p4_down, p5], axis: 3)   // [B, 20, 20, 384]
        let headP5 = c2f_n5(cat_n5)                    // [B, 20, 20, 256] (stride 32)

        return YOLONeckOutput(headP3: headP3, headP4: headP4, headP5: headP5)
    }
}
