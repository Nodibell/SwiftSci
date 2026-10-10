import CoreML
import Darwin
import Foundation
import SwiftDataFrame
import SwiftPreprocessing

// Only an admitted request owns a pair. Async prediction completes before the pool reuses it.
final class CoreMLMatrixBuffers: @unchecked Sendable {
    private var inputOwner: CoreMLMatrixStorage?
    private var outputOwner: CoreMLMatrixStorage?
    var input: CoreMLMatrixStorage { inputOwner! }
    var output: CoreMLMatrixStorage { outputOwner! }
    func release() { inputOwner = nil; outputOwner = nil }

    init(input: ArrayFeature, output: ArrayFeature) throws {
        inputOwner = try CoreMLMatrixStorage(input)
        outputOwner = try CoreMLMatrixStorage(output)
    }
}

final class CoreMLMatrixStorage {
    let array: MLMultiArray

    static func capacity(_ schema: ArrayFeature) throws -> Int {
        let bytes = try coreMLByteCount(schema.shape[0], schema.width, schema.elementBytes)
        let page = Int(getpagesize())
        let rounded = try MemoryEstimate(capacities: [bytes, page - 1]).bytes / page
        return try coreMLByteCount(rounded, page)
    }

    init(_ schema: ArrayFeature) throws {
        let capacity = try Self.capacity(schema)
        var allocation: UnsafeMutableRawPointer?
        guard posix_memalign(&allocation, Int(getpagesize()), capacity) == 0, let allocation else {
            throw SwiftMLError.invalidParameter("Could not allocate Core ML matrix storage")
        }
        do {
            array = try MLMultiArray(dataPointer: allocation,
                shape: schema.shape.map(NSNumber.init(value:)), dataType: schema.dataType,
                strides: [NSNumber(value: schema.width), 1], deallocator: { free($0) })
        } catch { free(allocation); throw error }
    }
}

extension CoreMLPredictionSession {
    func matrixBufferCapacity(count: Int) throws -> Int {
        guard let inputArray, let outputArray else {
            throw SwiftMLError.invalidParameter("A matrix pool requires fixed matrix input and output")
        }
        let pair = try MemoryEstimate(capacities: [CoreMLMatrixStorage.capacity(inputArray),
                                                  CoreMLMatrixStorage.capacity(outputArray)])
        return try coreMLByteCount(count, pair.bytes)
    }

    func matrixRequestBytes(_ input: CoreMLMatrixRequest) throws -> Int {
        let ownedOutput = try coreMLByteCount(input.rowCount, outputColumns.count, MemoryLayout<Double>.stride)
        // outputBackings is advisory. Allow one complete fallback result even when buffers are supplied.
        let fallback = try CoreMLMatrixStorage.capacity(outputArray!)
        return try MemoryEstimate(capacities: [input.unreservedInputBytes, ownedOutput, fallback]).bytes
    }

    func runPooledMatrix(_ input: CoreMLMatrixRequest, buffers: CoreMLMatrixBuffers,
                         isolation: isolated any Actor) async throws -> (prediction: CoreMLPrediction, backingMatched: Bool) {
        try Task.checkCancellation()
        try input.pack(using: self, into: buffers.input)
        try Task.checkCancellation()
        let result = try await asyncModel.predict(buffers, inputName: inputArray!.name, outputName: outputName)
        return try autoreleasepool {
            var columns = [[Double]](repeating: [Double](repeating: 0, count: input.rowCount), count: outputColumns.count)
            try extractMatrixPrediction(result, into: &columns)
            try Task.checkCancellation()
            let packing = try coreMLByteCount(input.rowCount, inputArray!.width, inputArray!.elementBytes)
            let copied = try coreMLByteCount(input.rowCount, outputColumns.count, MemoryLayout<Double>.stride)
            let prediction = CoreMLPrediction(values: input.result(columns, names: outputColumns),
                                    inputPackingBytes: packing, outputCopyBytes: copied)
            let backingMatched = result.featureValue(for: outputName)?.multiArrayValue === buffers.output.array
            return (prediction, backingMatched)
        }
    }
}
