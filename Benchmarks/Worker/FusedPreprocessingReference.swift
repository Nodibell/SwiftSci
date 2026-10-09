import SwiftDataFrame
import SwiftPreprocessing

// Allocation and staged baselines are benchmark references, not library APIs.
extension StandardPreprocessingPlan {
    // Concrete entry points let this module specialize conversion and array access
    // before callers cross the module boundary. Generic methods retain dtype-wide validation coverage.
    func stagedFloat32(_ input: PreparedNumericBatch) throws -> [Float] {
        try staged(input, as: Float.self)
    }

    func fusedFloat32(_ input: PreparedNumericBatch, tiled: Bool) throws -> [Float] {
        if !tiled { return try fused(input, as: Float.self, tiled: false) }
        try validate(input)
        var output = [Float](repeating: 0, count: input.rowCount * input.columnCount)
        try output.withUnsafeMutableBufferPointer { try fillFloat32(input, into: $0) }
        return output
    }

    func stagedFloat16(_ input: PreparedNumericBatch) throws -> [Float16] {
        try staged(input, as: Float16.self)
    }

    func fusedFloat16(_ input: PreparedNumericBatch) throws -> [Float16] {
        try validate(input)
        var output = [Float16](repeating: 0, count: input.rowCount * input.columnCount)
        try output.withUnsafeMutableBufferPointer { try fillFloat16(input, into: $0) }
        return output
    }

    func staged<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch, as: T.Type) throws -> [T] {
        try validate(input)
        let imputed = try imputer.transform(input)
        let scaled = try scaler.transform(consuming: consume imputed)
        let width = input.columnCount
        var result = [T](repeating: 0, count: input.rowCount * width)
        // Direct packing avoids charging the reference for an extra Double matrix.
        // Use PR #63's 16x16 packing traversal for the staged reference too.
        result.withUnsafeMutableBufferPointer { output in
            for rowStart in stride(from: 0, to: input.rowCount, by: 16) {
                let rowEnd = min(rowStart + 16, input.rowCount)
                for columnStart in stride(from: 0, to: width, by: 16) {
                    let columnEnd = min(columnStart + 16, width)
                    for c in columnStart..<columnEnd {
                        scaled.columns[c].values.withUnsafeBufferPointer { column in
                            for r in rowStart..<rowEnd { output[r * width + c] = T(column[r]) }
                        }
                    }
                }
            }
        }
        return result
    }

    func fused<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch, as: T.Type,
                                      tiled: Bool) throws -> [T] {
        try validate(input)
        var result = [T](repeating: 0, count: input.rowCount * input.columnCount)
        try result.withUnsafeMutableBufferPointer { output in
            try fill(input, into: output, tiled: tiled)
        }
        return result
    }

}
