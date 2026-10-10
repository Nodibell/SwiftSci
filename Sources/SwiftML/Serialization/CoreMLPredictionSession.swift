import CoreML
import Foundation
import SwiftDataFrame
import SwiftPreprocessing

// The actor owns model configuration. Async matrix predictions use distinct request buffers;
// only Core ML's documented thread-safe async call can overlap on the same model.
// Core ML documents native async prediction as thread-safe. This owner exposes only
// that operation; mutable providers are transferred per request and never shared.
final class CoreMLAsyncModel: @unchecked Sendable {
    private let model: MLModel
    init(_ model: MLModel) { self.model = model }

    func predict(_ provider: sending MLDictionaryFeatureProvider) async throws -> sending any MLFeatureProvider {
        try await model.prediction(from: provider)
    }

    func predict(_ buffers: CoreMLMatrixBuffers, inputName: String,
                 outputName: String) async throws -> sending any MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: buffers.input.array)])
        let options = MLPredictionOptions()
        options.outputBackings = [outputName: buffers.output.array]
        return try await model.prediction(from: provider, options: options)
    }

}

final class CoreMLPredictionSession {
    let model: MLModel
    let asyncModel: CoreMLAsyncModel
    let inputLayout: CoreMLInputLayout
    let inputColumns: [String]
    let inputArray: ArrayFeature?
    let outputArray: ArrayFeature?
    let outputName: String
    let outputColumns: [String]

    init(compiledModelURL: URL, inputColumns: [String], outputName: String,
                computeUnits: MLComputeUnits = .all, inputLayout: CoreMLInputLayout = .examples) throws {
        guard !inputColumns.isEmpty, Set(inputColumns).count == inputColumns.count else {
            throw SwiftMLError.invalidParameter("Core ML input column names must be nonempty and unique")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loaded = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        let inputs = loaded.modelDescription.inputDescriptionsByName
        if inputs.count == 1, let feature = inputs.values.first, feature.type == .multiArray {
            let vector = try ArrayFeature(feature, rank: inputLayout == .matrix ? 2 : 1)
            guard vector.width == inputColumns.count else {
                throw SwiftMLError.invalidParameter("Core ML vector width differs from input column count")
            }
            inputArray = vector
        } else {
            guard inputLayout == .examples, Set(inputs.keys) == Set(inputColumns), inputs.values.allSatisfy({ $0.type == .double }) else {
                throw SwiftMLError.invalidParameter("Core ML requires named Double scalars or one numeric vector input")
            }
            inputArray = nil
        }
        guard let output = loaded.modelDescription.outputDescriptionsByName[outputName] else {
            throw SwiftMLError.invalidParameter("Core ML output does not exist: \(outputName)")
        }
        if output.type == .multiArray {
            let vector = try ArrayFeature(output, rank: inputLayout == .matrix ? 2 : 1)
            outputArray = vector
            outputColumns = vector.width == 1 ? [outputName] : (0..<vector.width).map { "\(outputName)[\($0)]" }
        } else {
            guard inputLayout == .examples, output.type == .double else {
                throw SwiftMLError.invalidParameter("Core ML output must be a Double scalar or numeric vector")
            }
            outputArray = nil
            outputColumns = [outputName]
        }
        if inputLayout == .matrix, inputArray?.shape[0] != outputArray?.shape[0] {
            throw SwiftMLError.invalidParameter("Core ML matrix input and output row counts must match")
        }
        self.inputLayout = inputLayout
        self.model = loaded
        self.asyncModel = CoreMLAsyncModel(loaded)
        self.inputColumns = inputColumns
        self.outputName = outputName
    }

    func validateRequest(_ input: PreparedNumericBatch, maximumBatchSize: Int) throws {
        if inputLayout == .matrix {
            guard maximumBatchSize == 1, input.rowCount == inputArray?.shape[0] else {
                throw SwiftMLError.invalidParameter("Core ML matrix requests require the declared row count and maximumBatchSize 1")
            }
        }
    }

    func columnIndices(in input: PreparedNumericBatch) throws -> [Int] {
        let names = input.columnNames
        guard names.count == inputColumns.count else {
            throw SwiftMLError.invalidParameter("Prepared columns do not match the Core ML input contract")
        }
        if names == inputColumns { return Array(names.indices) }
        var positions = [String: Int](minimumCapacity: names.count)
        for (index, name) in names.enumerated() {
            guard positions.updateValue(index, forKey: name) == nil else {
                throw SwiftMLError.invalidParameter("Prepared columns do not match the Core ML input contract")
            }
        }
        return try inputColumns.map { name in
            guard let index = positions[name] else {
                throw SwiftMLError.invalidParameter("Prepared columns do not match the Core ML input contract")
            }
            return index
        }
    }

    func run(_ input: PreparedNumericBatch, indices: [Int], packingBytes: Int,
             outputBytes: Int, maximumBatchSize: Int) throws -> CoreMLPrediction {
        try autoreleasepool {
            try Task.checkCancellation()
            var columns = [[Double]](repeating: [Double](repeating: 0, count: input.rowCount), count: outputColumns.count)
            if inputLayout == .matrix {
                try runMatrix(input, indices: indices, into: &columns)
                try Task.checkCancellation()
                return CoreMLPrediction(values: input.replacingNumericColumns(columns, columnNames: outputColumns),
                                        inputPackingBytes: packingBytes, outputCopyBytes: outputBytes)
            }
            // Single-example calls reuse storage. Batch providers must own distinct vectors.
            let reusable = maximumBatchSize == 1 && input.rowCount > 0 ? try makeVector() : nil
            var first = 0
            while first < input.rowCount {
                try Task.checkCancellation()
                let end = first + min(maximumBatchSize, input.rowCount - first)
                try autoreleasepool {
                    if maximumBatchSize == 1 {
                        let provider = try features(input, row: first, indices: indices, vector: reusable)
                        try extract(model.prediction(from: provider), row: first, into: &columns)
                    } else {
                        var providers: [any MLFeatureProvider] = []
                        providers.reserveCapacity(end - first)
                        for row in first..<end {
                            try Task.checkCancellation()
                            let vector = try makeVector()
                            providers.append(try features(input, row: row, indices: indices, vector: vector))
                        }
                        let batch = MLArrayBatchProvider(array: providers)
                        let predictions = try model.predictions(from: batch, options: MLPredictionOptions())
                        guard predictions.count == end - first else {
                            throw SwiftMLError.invalidParameter("Core ML returned an unexpected batch size")
                        }
                        for offset in 0..<predictions.count {
                            try extract(predictions.features(at: offset), row: first + offset, into: &columns)
                        }
                    }
                }
                first = end
            }
            try Task.checkCancellation()
            let values = input.replacingNumericColumns(columns, columnNames: outputColumns)
            return CoreMLPrediction(values: values, inputPackingBytes: packingBytes, outputCopyBytes: outputBytes)
        }
    }

    private func matrixProvider(_ input: PreparedNumericBatch, indices: [Int]) throws -> sending MLDictionaryFeatureProvider {
        let schema = inputArray!
        let array = try MLMultiArray(shape: schema.shape.map(NSNumber.init(value:)), dataType: schema.dataType)
        try packMatrix(input, indices: indices, into: array)
        try Task.checkCancellation()
        return try MLDictionaryFeatureProvider(dictionary: [schema.name: MLFeatureValue(multiArray: array)])
    }

    private func runMatrix(_ input: PreparedNumericBatch, indices: [Int], into columns: inout [[Double]]) throws {
        let provider = try matrixProvider(input, indices: indices)
        let prediction = try model.prediction(from: provider)
        try extractMatrixPrediction(prediction, into: &columns)
    }

    func runMatrixAsynchronously(_ input: PreparedNumericBatch, indices: [Int],
                                packingBytes: Int, outputBytes: Int,
                                isolation: isolated any Actor) async throws -> CoreMLPrediction {
        try Task.checkCancellation()
        var columns = [[Double]](repeating: [Double](repeating: 0, count: input.rowCount), count: outputColumns.count)
        let provider = try matrixProvider(input, indices: indices)
        // Await completion before dropping request owners or releasing admission, including errors.
        let prediction = try await asyncModel.predict(provider)
        return try autoreleasepool {
            try extractMatrixPrediction(prediction, into: &columns)
            try Task.checkCancellation()
            return CoreMLPrediction(values: input.replacingNumericColumns(columns, columnNames: outputColumns),
                inputPackingBytes: packingBytes, outputCopyBytes: outputBytes)
        }
    }

    func extractMatrixPrediction(_ prediction: any MLFeatureProvider, into columns: inout [[Double]]) throws {
        try Task.checkCancellation()
        let schema = outputArray!
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.dataType == schema.dataType, output.shape == schema.shape.map(NSNumber.init(value:)) else {
            throw SwiftMLError.invalidParameter("Core ML output differs from its declared matrix contract")
        }
        try extractMatrix(output, into: &columns)
    }

    private func makeVector() throws -> MLMultiArray? {
        try inputArray.map { try MLMultiArray(shape: $0.shape.map(NSNumber.init(value:)), dataType: $0.dataType) }
    }

    private func features(_ input: PreparedNumericBatch, row: Int, indices: [Int],
                          vector: MLMultiArray?) throws -> MLDictionaryFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        if let vector, let schema = inputArray {
            if schema.dataType == .float16 {
                if #available(macOS 15, *) {
                    try packVector(input, row: row, indices: indices, into: vector, as: Float16.self)
                } else {
                    throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
                }
            } else if schema.dataType == .float32 {
                try packVector(input, row: row, indices: indices, into: vector, as: Float.self)
            } else {
                try packVector(input, row: row, indices: indices, into: vector, as: Double.self)
            }
            features[schema.name] = MLFeatureValue(multiArray: vector)
        } else {
            for (position, column) in indices.enumerated() {
                features[inputColumns[position]] = MLFeatureValue(double: try value(input, row: row, column: column))
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func value(_ input: PreparedNumericBatch, row: Int, column: Int) throws -> Double {
        guard let value = input[row, column], value.isFinite else {
            throw SwiftMLError.invalidParameter("Core ML input has a missing or nonfinite value at row \(row), column \(input.columnNames[column])")
        }
        return value
    }

    private func packVector<T: BinaryFloatingPoint & MLShapedArrayScalar>(
        _ input: PreparedNumericBatch, row: Int, indices: [Int], into vector: MLMultiArray, as: T.Type
    ) throws {
        try vector.withUnsafeMutableBufferPointer(ofType: T.self) { buffer, strides in
            for (position, column) in indices.enumerated() {
                let converted = T(try value(input, row: row, column: column))
                guard converted.isFinite else { throw SwiftMLError.invalidParameter("Core ML input overflows the model's element type") }
                buffer[position * strides[0]] = converted
            }
        }
    }

    private func extract(_ prediction: any MLFeatureProvider, row: Int, into columns: inout [[Double]]) throws {
        guard let output = prediction.featureValue(for: outputName) else {
            throw SwiftMLError.invalidParameter("Core ML omitted the requested output")
        }
        if let schema = outputArray {
            guard let array = output.multiArrayValue, array.count == schema.width,
                  array.dataType == schema.dataType, array.shape == [NSNumber(value: schema.width)] else {
                throw SwiftMLError.invalidParameter("Core ML output differs from its declared vector contract")
            }
            for column in columns.indices { columns[column][row] = array[column].doubleValue }
        } else {
            guard output.type == .double else {
                throw SwiftMLError.invalidParameter("Core ML output is not a Double scalar")
            }
            columns[0][row] = output.doubleValue
        }
    }

}


func coreMLByteCount(_ factors: Int...) throws -> Int {
    var result = 1
    for factor in factors {
        let (next, overflow) = result.multipliedReportingOverflow(by: factor)
        guard factor >= 0, !overflow else { throw MemoryAdmissionError.invalidCapacity }
        result = next
    }
    return result
}
