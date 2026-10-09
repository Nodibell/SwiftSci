import CoreML
import SwiftDataFrame

extension CoreMLPredictionSession {
    func packMatrix(_ input: PreparedNumericBatch, indices: [Int], into array: MLMultiArray) throws {
        if array.dataType == .float16 {
            if #available(macOS 15, *) {
                try packMatrix(input, indices: indices, into: array, as: Float16.self)
            } else {
                throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
            }
        } else if array.dataType == .float32 {
            try packMatrix(input, indices: indices, into: array, as: Float.self)
        } else {
            try packMatrix(input, indices: indices, into: array, as: Double.self)
        }
    }

    func extractMatrix(_ array: MLMultiArray, into columns: inout [[Double]]) throws {
        if array.dataType == .float16 {
            if #available(macOS 15, *) {
                try extractMatrix(array, into: &columns, as: Float16.self)
            } else {
                throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
            }
        } else if array.dataType == .float32 {
            try extractMatrix(array, into: &columns, as: Float.self)
        } else {
            try extractMatrix(array, into: &columns, as: Double.self)
        }
    }

    private func packMatrix<T: BinaryFloatingPoint & MLShapedArrayScalar>(
        _ input: PreparedNumericBatch, indices: [Int], into array: MLMultiArray, as: T.Type
    ) throws {
        for index in indices where input.columns[index].nullCount != 0 {
            throw SwiftMLError.invalidParameter("Core ML input has missing values in column \(input.columnNames[index])")
        }
        try array.withUnsafeMutableBufferPointer(ofType: T.self) { buffer, strides in
            // Bound the strided destination working set while reading contiguous column segments.
            for first in stride(from: 0, to: input.rowCount, by: 16) {
                try Task.checkCancellation()
                let end = first + min(16, input.rowCount - first)
                for (position, index) in indices.enumerated() {
                    try input.columns[index].values.withUnsafeBufferPointer { source in
                        for row in first..<end {
                            let value = source[row]
                            guard value.isFinite else {
                                throw SwiftMLError.invalidParameter("Core ML input has a nonfinite value at row \(row), column \(input.columnNames[index])")
                            }
                            let converted = T(value)
                            guard converted.isFinite else { throw SwiftMLError.invalidParameter("Core ML input overflows the model's element type") }
                            buffer[row * strides[0] + position * strides[1]] = converted
                        }
                    }
                }
            }
        }
    }

    private func extractMatrix<T: BinaryFloatingPoint & MLShapedArrayScalar>(
        _ array: MLMultiArray, into columns: inout [[Double]], as: T.Type
    ) throws {
        try array.withUnsafeBufferPointer(ofType: T.self) { buffer in
            let strides = array.strides.map(\.intValue)
            for first in stride(from: 0, to: columns[0].count, by: 16) {
                try Task.checkCancellation()
                let end = first + min(16, columns[0].count - first)
                for column in columns.indices {
                    try columns[column].withUnsafeMutableBufferPointer { destination in
                        for row in first..<end {
                            let value = Double(buffer[row * strides[0] + column * strides[1]])
                            guard value.isFinite else { throw SwiftMLError.invalidParameter("Core ML returned a nonfinite matrix value") }
                            destination[row] = value
                        }
                    }
                }
            }
        }
    }

}
