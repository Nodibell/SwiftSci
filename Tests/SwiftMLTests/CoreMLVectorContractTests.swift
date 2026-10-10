import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Core ML numeric contracts", .serialized)
struct CoreMLVectorContractTests {
    @Test func float32ConversionAndOverflow() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            let model = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"],
                                            outputName: "result", computeUnits: .cpuOnly)
            let budget = try MemoryBudget(limit: 4096)
            let columns: [[Double]] = [[1.00000001, -2, 4], [3, -1, 7]]
            let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: columns)
            let output = try await model.predict(input, budget: budget, workspaceBytes: 128)
            for row in 0..<3 {
                for column in 0..<2 {
                    let expected = Double(max(Float(columns[column][row]), 0))
                    #expect(output.values[row, column] == expected)
                }
            }
            #expect(output.inputPackingBytes == 24)
            #expect(output.outputCopyBytes == 48)
            var tooLarge = input
            try tooLarge.updateColumn(at: 0, rows: [0], values: [.greatestFiniteMagnitude])
            await #expect(throws: (any Error).self) {
                try await model.predict(tooLarge, budget: budget, workspaceBytes: 128)
            }
            #expect(await budget.reservedBytes == 0)
            #expect(output.values[0, 0] == 1)
        }
    }

    @Test func rejectsTensorRankWidthAndIntegerOutput() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [1, 2])) { url in
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"], outputName: "result")
            }
        }
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"], outputName: "result")
            }
        }
        let classifier = CoreMLExporter.exportBinaryLogisticModel(inputNames: ["x"],
            outputName: "label", weights: [1], bias: 0, classLabels: [0, 1])
        try await withCompiledCoreML(classifier) { url in
            #expect(throws: (any Error).self) {
                try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x"], outputName: "label")
            }
        }
    }

    @Test func predictorOwnerReleasesAfterCompletedRequest() async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [2])) { url in
            weak var owner: CoreMLPredictor?
            let budget = try MemoryBudget(limit: 4096)
            do {
                let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y"], outputName: "result")
                owner = predictor
                let input = try PreparedNumericBatch(columnNames: ["x", "y"], columns: [[1], [2]])
                let result = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
                #expect(result.values[0, 0] == 1)
            }
            #expect(owner == nil)
            #expect(await budget.reservedBytes == 0)
        }
    }
}
