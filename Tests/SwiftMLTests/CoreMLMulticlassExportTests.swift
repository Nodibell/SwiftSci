import CoreML
import Foundation
import Testing
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing

@Suite("Core ML multiclass export", .serialized)
struct CoreMLMulticlassExportTests {
    @Test("Exported multiclass probabilities match independent stable softmax")
    func suppliedMulticlassProbabilities() async throws {
        let layer = LayerWeights(W: [-1, 0, 1], b: [0, 0, 0], inDim: 1, outDim: 3)
        let data = CoreMLExporter.exportBinaryMLPClassifier(
            inputNames: ["x"], outputName: "probabilities", layers: [layer])
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[-1_000, -1, 0, 1, 1_000]])
        let input = try source.selectingRows([4, 0, 2, 3, 3, 1, 2])
        let budget = try MemoryBudget(limit: 1_048_576)
        try await withCompiledCoreML(data) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled,
                inputColumns: ["x"], outputName: "probabilities", computeUnits: .cpuOnly)
            for batchSize in [1, 4] {
                let prediction = try await predictor.predict(input, budget: budget,
                    workspaceBytes: 65_536, maximumBatchSize: batchSize)
                #expect(prediction.values.columnNames == ["probabilities[0]", "probabilities[1]", "probabilities[2]"])
                #expect(prediction.values.originalRowIndices == input.originalRowIndices)
                #expect(prediction.inputPackingBytes == input.rowCount * MemoryLayout<Double>.stride)
                #expect(prediction.outputCopyBytes == input.rowCount * 3 * MemoryLayout<Double>.stride)
                for row in 0..<input.rowCount {
                    let x = try #require(input[row, 0])
                    let logits = [-x, 0, x]
                    let largest = try #require(logits.max())
                    let exponentials = logits.map { exp($0 - largest) }
                    let denominator = exponentials.reduce(0, +)
                    var sum = 0.0
                    for column in 0..<3 {
                        let actual = try #require(prediction.values[row, column])
                        #expect(actual.isFinite && actual >= 0 && actual <= 1)
                        #expect(abs(actual - exponentials[column] / denominator) < 1e-6)
                        sum += actual
                    }
                    #expect(abs(sum - 1) < 1e-6)
                }
                #expect(await budget.reservedBytes == 0)
            }
        }
    }

    @Test("Fitted public multiclass classifier survives export and prepared inference")
    func trainedClassifierExport() async throws {
        let features: [[Double]] = [[-1], [0], [1]]
        let targets: [Double] = [10, 20, 30]
        let model = MLPClassifier(hiddenLayerSizes: [], maxIter: 1,
                                  batchSize: 3, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: features, targets: targets)
        #expect(await model.trainedClasses == targets)
        let expected = try await model.predictProbability(features: features)
        let data = try await model.exportCoreML(featureNames: ["x"], outputName: "probabilities")
        let input = try PreparedNumericBatch(columnNames: ["x"], columns: [[-1, 0, 1]])
        let budget = try MemoryBudget(limit: 1_048_576)
        try await withCompiledCoreML(data) { compiled in
            let predictor = try CoreMLPredictor(compiledModelURL: compiled,
                inputColumns: ["x"], outputName: "probabilities", computeUnits: .cpuOnly)
            let prediction = try await predictor.predict(input, budget: budget, workspaceBytes: 65_536)
            for row in features.indices {
                for column in 0..<3 {
                    let actual = try #require(prediction.values[row, column])
                    #expect(abs(actual - expected[row][column]) < 1e-6)
                }
            }
            #expect(await budget.reservedBytes == 0)
        }
    }
}
