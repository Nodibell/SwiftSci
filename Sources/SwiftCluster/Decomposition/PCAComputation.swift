#if os(macOS)
import Accelerate
import Foundation
import MLX
import SwiftDataFrame
import SwiftPreprocessing

protocol PCANumericInput: Sendable {
    var rowCount: Int { get }
    var columnCount: Int { get }
    subscript(row: Int, column: Int) -> Double { get }
}

struct PCARows: PCANumericInput {
    let values: [[Double]]
    var rowCount: Int { values.count }
    var columnCount: Int { values[0].count }
    subscript(row: Int, column: Int) -> Double { values[row][column] }
}

struct PCAColumns: PCANumericInput {
    let batch: PreparedNumericBatch
    var rowCount: Int { batch.rowCount }
    var columnCount: Int { batch.columnCount }
    subscript(row: Int, column: Int) -> Double { batch.columns[column].values[row] }
}

extension PreparedNumericMatrix: PCANumericInput {}

struct PCAFitResult: Sendable {
    let mean: [Double]
    let components: [[Double]]
    let variance: [Double]
    let totalVariance: Double
}

enum PCAComputation {
    static func fit<Input: PCANumericInput>(_ input: Input, components k: Int,
                                            solver: SVDSolver, device: ExecutionDevice) throws -> PCAFitResult {
        if device != .cpu {
            for row in 0..<input.rowCount {
                for column in 0..<input.columnCount where !Float(input[row, column]).isFinite {
                    throw ClusterError.invalidParameter("PCA GPU input overflows Float.")
                }
            }
            return try Device.withDefaultDevice(.gpu) {
                try ScopedGPURead.withMatrix(rows: input.rowCount, columns: input.columnCount,
                                             value: { input[$0, $1] }) { array in
                    fitGPU(array, rows: input.rowCount, components: k)
                }
            }
        }
        let randomized: Bool
        switch solver {
        case .full: randomized = false
        case .randomized: randomized = true
        case .auto: randomized = Double(k) < 0.8 * Double(min(input.rowCount, input.columnCount))
        }
        var means = [Double](repeating: 0, count: input.columnCount)
        for row in 0..<input.rowCount {
            for column in 0..<input.columnCount { means[column] += input[row, column] }
        }
        if randomized {
            var inverse = 1.0 / Double(input.rowCount)
            vDSP_vsmulD(means, 1, &inverse, &means, 1, vDSP_Length(means.count))
        } else {
            for column in means.indices { means[column] /= Double(input.rowCount) }
        }
        var centered = [Double](repeating: 0, count: input.rowCount * input.columnCount)
        var sumSquares = 0.0
        for row in 0..<input.rowCount {
            var rowSquares = 0.0
            for column in 0..<input.columnCount {
                let value = input[row, column] - means[column]
                guard value.isFinite else { throw ClusterError.invalidParameter("PCA centering overflowed.") }
                centered[column * input.rowCount + row] = value
                if randomized { rowSquares += value * value }
            }
            if randomized { sumSquares += rowSquares }
        }
        let singular: [Double]
        var vectors: [[Double]]
        let total: Double
        if randomized {
            let result = try RandomizedSVD.compute(columnMajor: centered, rows: input.rowCount,
                                                   columns: input.columnCount, nComponents: k,
                                                   computeLeftVectors: false)
            singular = result.S
            vectors = result.Vt
            total = sumSquares
        } else {
            (singular, vectors) = try fullSVD(&centered, rows: input.rowCount, columns: input.columnCount, components: k)
            total = singular.reduce(0) { $0 + $1 * $1 }
        }
        var unused: [[Double]]? = nil
        RandomizedSVD.svdFlip(u: &unused, vt: &vectors, uBasedDecision: false)
        let scale = 1.0 / Double(max(1, input.rowCount - 1))
        // Preserve division on full SVD and reciprocal multiplication on randomized SVD.
        let variance = singular.prefix(k).map { randomized ? $0 * $0 * scale : ($0 * $0) / Double(max(1, input.rowCount - 1)) }
        let totalVariance = randomized ? total * scale : total / Double(max(1, input.rowCount - 1))
        return PCAFitResult(mean: means, components: vectors, variance: variance, totalVariance: totalVariance)
    }

    static func score<Input: PCANumericInput>(_ input: Input, row: Int, component: Int,
                                              fitted: PCAFitResult) -> Double {
        var sum = 0.0
        for column in fitted.mean.indices {
            sum += (input[row, column] - fitted.mean[column]) * fitted.components[component][column]
        }
        return sum
    }

    private static func fullSVD(_ values: inout [Double], rows: Int, columns: Int,
                                components k: Int) throws -> ([Double], [[Double]]) {
        let layout = try PCAFullSVDLayout(rows: rows, columns: columns)
        var job: Int8 = 83
        var m = layout.rows, n = layout.columns, lda = m, ldu = m
        let rank = layout.rank
        var ldvt = min(m, n), info: Int32 = 0, lwork: Int32 = -1
        var singular = [Double](repeating: 0, count: rank)
        var u = [Double](repeating: 0, count: layout.leftCount)
        var vt = [Double](repeating: 0, count: layout.rightCount)
        var iwork = [Int32](repeating: 0, count: layout.integerWorkspaceCount)
        var query = [Double](repeating: 0, count: 1)
        dgesdd_(&job, &m, &n, &values, &lda, &singular, &u, &ldu, &vt, &ldvt, &query, &lwork, &iwork, &info)
        guard info == 0 else { throw ClusterError.svdFailed(info: info) }
        guard let workspace = Int32(exactly: query[0]), workspace > 0 else {
            throw ClusterError.invalidParameter("PCA SVD workspace exceeds backend limits.")
        }
        lwork = workspace
        var work = [Double](repeating: 0, count: Int(lwork))
        dgesdd_(&job, &m, &n, &values, &lda, &singular, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
        guard info == 0 else { throw ClusterError.svdFailed(info: info) }
        let vectors = (0..<k).map { component in
            (0..<columns).map { column in vt[column * rank + component] }
        }
        return (singular, vectors)
    }

    private static func fitGPU(_ input: MLXArray, rows: Int, components k: Int) -> PCAFitResult {
        let means = input.mean(axis: 0)
        let (_, singular, vectors) = svd(input - means, stream: .cpu)
        eval(means, singular, vectors)
        let mean = means.asArray(Float.self).map(Double.init)
        let s = singular.asArray(Float.self).map(Double.init)
        let flat = vectors.asArray(Float.self)
        let columns = vectors.shape[1]
        var components = (0..<k).map { component in
            (0..<columns).map { column in Double(flat[component * columns + column]) }
        }
        var unused: [[Double]]? = nil
        RandomizedSVD.svdFlip(u: &unused, vt: &components, uBasedDecision: false)
        let divisor = Double(max(1, rows - 1))
        return PCAFitResult(mean: mean, components: components,
                            variance: s.prefix(k).map { $0 * $0 / divisor },
                            totalVariance: s.reduce(0) { $0 + $1 * $1 } / divisor)
    }
}
#endif
