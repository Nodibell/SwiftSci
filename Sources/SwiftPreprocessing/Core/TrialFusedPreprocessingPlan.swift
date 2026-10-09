import Accelerate
import SwiftDataFrame

// Package-only experiment shared by the benchmark worker and owned Core ML input.
// The fitted plan is immutable; mutable output is borrowed only during synchronous filling.
package struct TrialFusedPreprocessingPlan: Sendable {
    package let imputer: Imputer
    package let scaler: StandardScaler
    package let names: [String]
    package let replacements: [Double]
    package let negativeMeans: [Double]
    package let deviations: [Double]

    package init(training: PreparedNumericBatch, strategy: Imputer.Strategy = .mean) throws {
        var imputer = Imputer(strategy: strategy)
        try imputer.fit(training)
        var scaler = StandardScaler()
        try scaler.fit(imputer.transform(training))
        self.imputer = imputer
        self.scaler = scaler
        names = training.columnNames
        replacements = imputer.statistics!
        negativeMeans = scaler.mean!.map { -$0 }
        deviations = scaler.std!
    }

    package func validate(_ input: PreparedNumericBatch) throws {
        guard input.columnNames == names else {
            throw SwiftMLError.invalidParameter("Fused trial requires the fitted column names and order")
        }
        let (_, overflow) = input.rowCount.multipliedReportingOverflow(by: input.columnCount)
        guard !overflow else { throw SwiftMLError.invalidParameter("Packed element count overflow") }
    }

    package func staged<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch, as: T.Type) throws -> [T] {
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

    package func fused<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch, as: T.Type,
                                      tiled: Bool) throws -> [T] {
        try validate(input)
        var result = [T](repeating: 0, count: input.rowCount * input.columnCount)
        try result.withUnsafeMutableBufferPointer { output in
            try fill(input, into: output, tiled: tiled)
        }
        return result
    }

    package func fill<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch,
        into output: UnsafeMutableBufferPointer<T>, tiled: Bool) throws {
        try validate(input)
        guard output.count == input.rowCount * input.columnCount else {
            throw SwiftMLError.invalidParameter("Fused output capacity differs")
        }
        let width = input.columnCount
        let capacity = tiled && input.rowCount >= 32 && width <= 512
            ? min(input.rowCount, 2048 / width) : 1
        var tile = [Double](repeating: 0, count: capacity * width)
        var shifted = [Double](repeating: 0, count: width)
        var normalized = [Double](repeating: 0, count: width)
        try tile.withUnsafeMutableBufferPointer { buffer in
            for start in stride(from: 0, to: input.rowCount, by: capacity) {
                try Task.checkCancellation()
                let rows = min(capacity, input.rowCount - start)
                for c in 0..<width {
                    input.columns[c].values.withUnsafeBufferPointer { column in
                        for r in 0..<rows {
                            let value = column[start + r]
                            buffer[r * width + c] = value.isNaN ? replacements[c] : value
                        }
                    }
                }
                for r in 0..<rows {
                    let row = buffer.baseAddress!.advanced(by: r * width)
                    vDSP_vaddD(row, 1, negativeMeans, 1, &shifted, 1, vDSP_Length(width))
                    // Match StandardScaler's array-backed, row-width division.
                    vDSP_vdivD(deviations, 1, shifted, 1, &normalized, 1, vDSP_Length(width))
                    for c in 0..<width { output[(start + r) * width + c] = T(normalized[c]) }
                }
            }
        }
    }
}

