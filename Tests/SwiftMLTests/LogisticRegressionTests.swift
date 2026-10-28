import Testing
import Foundation
import SwiftPreprocessing
@testable import SwiftML

@Suite("LogisticRegression Tests")
struct LogisticRegressionTests {
    
    @Test("Supplied logistic parameters predict without fitting", arguments: [ExecutionDevice.cpu, .auto, .gpu, .ane])
    func suppliedParameters(device: ExecutionDevice) async throws {
        let model = LogisticRegression(weights: [2, -3], bias: 1, device: device)
        let probabilities = try await model.predictProbability(features: [[1, 1], [0, 0]])
        #expect(probabilities[0] == [0.5, 0.5])
        #expect(abs(probabilities[1][1] - 1 / (1 + exp(-1.0))) < 1e-7)
        #expect(try await model.predict(features: [[1, 1], [-1, 1]]) == [0, 0])
        #expect(await model.resolvedDevice == ((device == .cpu || device == .auto) ? .cpu : .gpu))
    }

    @Test("LogisticRegression classification of linearly separable data")
    func testLogisticClassification() async throws {
        // Linearly separable dataset: feature > 0 -> class 1, else class 0
        let features: [[Double]] = [[-2.0], [-1.0], [1.0], [2.0]]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        
        let model = LogisticRegression()
        try await model.fit(features: features, targets: targets, learningRate: 0.5, epochs: 600)
        
        let w = await model.getWeights()
        let b = await model.getBias()
        
        #expect(w != nil)
        #expect(b != nil)
        
        let testFeatures: [[Double]] = [[-3.0], [3.0]]
        let probs = try await model.predictProbability(features: testFeatures)
        #expect(probs[0][1] < 0.2) // Prob of class 1 for -3.0 should be very small
        #expect(probs[1][1] > 0.8) // Prob of class 1 for 3.0 should be very high
        
        let classes = try await model.predict(features: testFeatures)
        #expect(classes[0] == 0)
        #expect(classes[1] == 1)
    }
    
    @Test("LogisticRegression errors")
    func testLogisticRegressionErrors() async throws {
        let model = LogisticRegression()
        
        // Empty input throws
        await #expect(throws: MLError.self) {
            try await model.fit(features: [], targets: [])
        }
        
        // Transform before fit throws
        await #expect(throws: MLError.self) {
            _ = try await model.predict(features: [[1.0]])
        }
    }

    @Test("ProbabilityMatrix honors RandomAccessCollection contract across empty, single, and multi-row shapes")
    func testProbabilityMatrixCollectionContract() async throws {
        // 1. Empty matrix
        let empty = ProbabilityMatrix([])
        #expect(empty.isEmpty)
        #expect(empty.count == 0)
        #expect(empty.startIndex == 0)
        #expect(empty.endIndex == 0)
        #expect(Array(empty).isEmpty)
        #expect(empty.asArray.isEmpty)

        // 2. Single-row matrix (previously padded to 6 rows)
        let single: ProbabilityMatrix = [[0.25, 0.75]]
        #expect(!single.isEmpty)
        #expect(single.count == 1)
        #expect(single.startIndex == 0)
        #expect(single.endIndex == 1)
        #expect(single[single.startIndex] == [0.25, 0.75])
        #expect(single[single.index(before: single.endIndex)] == [0.25, 0.75])
        #expect(single.map { $0[1] } == [0.75])
        #expect(single == ProbabilityMatrix([[0.25, 0.75]]))

        // 3. Multi-row matrices on both sides of the old 6-row threshold
        for rowCount in [2, 5, 6, 8] {
            let rawRows = (0..<rowCount).map { i -> [Double] in
                let p = Double(i) / Double(max(1, rowCount))
                return [1.0 - p, p]
            }
            let matrix = ProbabilityMatrix(rawRows)
            #expect(matrix.count == rowCount)
            #expect(matrix.startIndex == 0)
            #expect(matrix.endIndex == rowCount)
            #expect(matrix[matrix.startIndex] == rawRows.first)
            #expect(matrix[matrix.index(before: matrix.endIndex)] == rawRows.last)
            #expect(Array(matrix) == rawRows)
            #expect(matrix.asArray == rawRows)
            #expect(matrix.description == rawRows.description)
        }

        // 4. End-to-end predictProba parity with predictProbability
        let model = LogisticRegression(weights: [2.0, -3.0], bias: 1.0, device: .cpu)
        let singleBatch = [[1.0, 1.0]]
        let probaSingle = try await model.predictProba(features: singleBatch)
        let rawSingle = try await model.predictProbability(features: singleBatch)
        #expect(probaSingle.count == 1)
        #expect(probaSingle.asArray == rawSingle)

        let twoRowBatch = [[1.0, 1.0], [0.0, 0.0]]
        let probaTwo = try await model.predictProba(features: twoRowBatch)
        let rawTwo = try await model.predictProbability(features: twoRowBatch)
        #expect(probaTwo.count == 2)
        #expect(probaTwo.asArray == rawTwo)
    }
}
