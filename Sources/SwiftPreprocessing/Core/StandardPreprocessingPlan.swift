import Accelerate
import SwiftDataFrame

/// Immutable training-fitted imputation followed by standard scaling.
///
/// Fit once on training rows, then reuse for held-out or incoming batches. The operation
/// matches `Imputer` followed by `StandardScaler`, including NaN and constant-column rules.
/// Core ML matrix preparation can apply this plan directly to its owned input storage.
/// Column names and order must match training. Numeric conversion occurs after Double scaling.
public struct StandardPreprocessingPlan: Sendable {
    package let imputer: Imputer
    package let scaler: StandardScaler
    /// Feature names in the exact order used during fitting.
    public let columnNames: [String]
    package let replacements: [Double]
    package let negativeMeans: [Double]
    package let deviations: [Double]

    /// Fits imputation statistics, then scaling statistics on the imputed training batch.
    /// Empty training input throws. Fitting never reads later prediction batches.
    public init(training: PreparedNumericBatch, strategy: Imputer.Strategy = .mean) throws {
        var imputer = Imputer(strategy: strategy)
        try imputer.fit(training)
        var scaler = StandardScaler()
        try scaler.fit(imputer.transform(training))
        self.imputer = imputer
        self.scaler = scaler
        columnNames = training.columnNames
        replacements = imputer.statistics!
        negativeMeans = scaler.mean!.map { -$0 }
        deviations = scaler.std!
    }

    package func validate(_ input: PreparedNumericBatch) throws {
        guard input.columnNames == columnNames else {
            throw SwiftMLError.invalidParameter("Preprocessing requires the fitted column names and order")
        }
        let (_, overflow) = input.rowCount.multipliedReportingOverflow(by: input.columnCount)
        guard !overflow else { throw SwiftMLError.invalidParameter("Packed element count overflow") }
    }

    // Concrete module entry points preserve compiler specialization of the generic fill.
    package func fillFloat32(_ input: PreparedNumericBatch,
                            into output: UnsafeMutableBufferPointer<Float>) throws {
        try fill(input, into: output, tiled: true)
    }

    package func fillDouble(_ input: PreparedNumericBatch,
                           into output: UnsafeMutableBufferPointer<Double>) throws {
        try fill(input, into: output, tiled: true)
    }

    package func workspaceAllowance(for input: PreparedNumericBatch) throws -> MemoryEstimate {
        try workspaceAllowance(rowCount: input.rowCount)
    }

    package func workspaceAllowance(rowCount: Int) throws -> MemoryEstimate {
        let width = columnNames.count
        let rows = tileRows(rowCount: rowCount, width: width, tiled: true)
        let (tileElements, tileOverflow) = rows.multipliedReportingOverflow(by: width)
        guard !tileOverflow else { throw MemoryAdmissionError.invalidCapacity }
        let elements = try MemoryEstimate(capacities: [tileElements, width, width]).bytes
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: MemoryLayout<Double>.stride)
        guard !overflow else { throw MemoryAdmissionError.invalidCapacity }
        return try MemoryEstimate(capacities: [bytes, 4096])
    }

    private func tileRows(_ input: PreparedNumericBatch, tiled: Bool) -> Int {
        tileRows(rowCount: input.rowCount, width: input.columnCount, tiled: tiled)
    }

    private func tileRows(rowCount: Int, width: Int, tiled: Bool) -> Int {
        return tiled && rowCount >= 32 && width > 0 && width <= 512
            ? min(rowCount, 2048 / width) : 1
    }

    package func fillFloat16(_ input: PreparedNumericBatch,
                            into output: UnsafeMutableBufferPointer<Float16>) throws {
        try fill(input, into: output, tiled: true)
    }

    package func fill<T: BinaryFloatingPoint>(_ input: PreparedNumericBatch,
        into output: UnsafeMutableBufferPointer<T>, tiled: Bool) throws {
        try validate(input)
        guard output.count == input.rowCount * input.columnCount else {
            throw SwiftMLError.invalidParameter("Fused output capacity differs")
        }
        let width = input.columnCount
        let capacity = tileRows(input, tiled: tiled)
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
