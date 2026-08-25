#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftVision

@Suite("CLIP Multimodal Feature Projector Tests")
struct CLIPProjectorTests {

    @Test("CLIPProjector normalizes embeddings and computes similarities")
    func testCLIPProjectorOperations() throws {
        let visualDim = 64
        let textDim = 32
        let projDim = 16

        let projector = CLIPProjector(
            visualDimensions: visualDim,
            textDimensions: textDim,
            projectionDimensions: projDim
        )

        let numImages = 2
        let numTexts = 3

        let imgFeatures = MLXArray([Float](repeating: 1.0, count: numImages * visualDim), [numImages, visualDim])
        let txtFeatures = MLXArray([Float](repeating: 0.5, count: numTexts * textDim), [numTexts, textDim])

        let imgEmb = projector.projectVision(imgFeatures)
        let txtEmb = projector.projectText(txtFeatures)

        #expect(imgEmb.shape == [numImages, projDim])
        #expect(txtEmb.shape == [numTexts, projDim])

        // Verify L2 normalization (sum of squares == 1.0)
        let imgNormSq = sum(imgEmb * imgEmb, axis: -1).asArray(Float.self)
        for n in imgNormSq {
            #expect(abs(n - 1.0) < 1e-4)
        }

        let similarity = projector.similarity(visualFeatures: imgFeatures, textFeatures: txtFeatures)
        #expect(similarity.shape == [numImages, numTexts])

        let probs = projector.zeroShotClassify(visualFeatures: imgFeatures, candidateTextFeatures: txtFeatures)
        #expect(probs.shape == [numImages, numTexts])

        // Softmax sums to 1.0
        let probSum = sum(probs, axis: -1).asArray(Float.self)
        for s in probSum {
            #expect(abs(s - 1.0) < 1e-4)
        }
    }
}
#endif
