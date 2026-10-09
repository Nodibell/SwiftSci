import Accelerate
import Foundation
import SwiftSciBenchmarkSupport
import SwiftDataFrame
import SwiftPreprocessing

// Experimental worker implementation. No public preprocessing API is changed.
struct FusedPreprocessingPlan: Sendable {
    let imputer: Imputer
    let scaler: StandardScaler
    let names: [String]
    let replacements: [Double]
    let negativeMeans: [Double]
    let deviations: [Double]

    init(training: PreparedNumericBatch, strategy: Imputer.Strategy = .mean) throws {
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

    func validate(_ input: PreparedNumericBatch) throws {
        guard input.columnNames == names else {
            throw BenchmarkFailure("Fused trial requires the fitted column names and order")
        }
        let (_, overflow) = input.rowCount.multipliedReportingOverflow(by: input.columnCount)
        guard !overflow else { throw BenchmarkFailure("Packed element count overflow") }
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
        let width = input.columnCount
        let capacity = tiled && input.rowCount >= 32 && width <= 512
            ? min(input.rowCount, 2048 / width) : 1
        var tile = [Double](repeating: 0, count: capacity * width)
        var shifted = [Double](repeating: 0, count: width)
        var normalized = [Double](repeating: 0, count: width)
        var result = [T](repeating: 0, count: input.rowCount * width)
        tile.withUnsafeMutableBufferPointer { buffer in
            for start in stride(from: 0, to: input.rowCount, by: capacity) {
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
                    for c in 0..<width { result[(start + r) * width + c] = T(normalized[c]) }
                }
            }
        }
        return result
    }
}

func fusedTrialFixture(rows: Int, width: Int, missing: Bool, offset: Int = 0) throws -> PreparedNumericBatch {
    var columns = [[Double]]()
    for c in 0..<width {
        var column = [Double]()
        column.reserveCapacity(rows)
        for r in 0..<rows {
            let integer: Int = ((r + offset) * 37 + c * 19) % 1009
            let value = Double(integer - 504) / 7.0
            let absent = missing && (r + c) % 17 == 0
            column.append(absent ? .nan : value)
        }
        columns.append(column)
    }
    let names: [String] = (0..<width).map { "x\($0)" }
    var batch = try PreparedNumericBatch(columnNames: names, columns: consume columns)
    if missing && rows > 2 {
        for c in 0..<width { try batch.updateColumn(at: c, rows: [1], values: [nil]) }
    }
    return batch
}

func fusedTrialEqual<T: BinaryFloatingPoint>(_ lhs: [T], _ rhs: [T]) -> Bool {
    guard lhs.count == rhs.count else { return false }
    return zip(lhs, rhs).allSatisfy { a, b in
        if a.isNaN && b.isNaN { return true }
        return a == b && a.sign == b.sign
    }
}
