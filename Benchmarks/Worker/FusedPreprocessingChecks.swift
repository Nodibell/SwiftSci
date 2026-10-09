import Foundation
import SwiftSciBenchmarkSupport
import SwiftDataFrame
import SwiftPreprocessing

private func checkFused<T: BinaryFloatingPoint>(_ plan: FusedPreprocessingPlan,
    input: PreparedNumericBatch, as type: T.Type) throws {
    let before = input.rowValues()
    let expected = try plan.staged(input, as: type)
    for tiled in [false, true] {
        let output = try plan.fused(input, as: type, tiled: tiled)
        guard fusedTrialEqual(expected, output) else {
            throw BenchmarkFailure("Fusion differs: \(input.rowCount)x\(input.columnCount), \(type), tiled=\(tiled)")
        }
    }
    let after = input.rowValues()
    guard zip(before, after).allSatisfy({ fusedTrialEqual($0, $1) }) else {
        throw BenchmarkFailure("Fusion mutated its input snapshot")
    }
}

func verifyFusedPreprocessing() throws -> Int {
    var cases = 0
    for width in [1, 7, 64, 65, 512, 513] {
        for missing in [false, true] {
            let training = try fusedTrialFixture(rows: 67, width: width, missing: missing)
            let plan = try FusedPreprocessingPlan(training: training)
            for rows in [0, 1, 31, 32, 33, 65, 513] {
                let input = try fusedTrialFixture(rows: rows, width: width, missing: missing, offset: 119)
                try checkFused(plan, input: input, as: Double.self)
                try checkFused(plan, input: input, as: Float.self)
                try checkFused(plan, input: input, as: Float16.self)
                cases += 3
            }
        }
    }
    let trainingColumns: [[Double]] = [[1, 2, 3, .nan], [7, 7, 7, 7], [.nan, .nan, .nan, .nan]]
    let heldOutColumns: [[Double]] = [[-0.0, 1e100, .nan, .infinity], [8, -8, .nan, 7], [.nan, 3, -3, .nan]]
    let training = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns: trainingColumns)
    let heldOut = try PreparedNumericBatch(columnNames: ["a", "b", "c"], columns: heldOutColumns)
    let selected = try heldOut.selectingRows([3, 0, 3, 2])
    for strategy in [Imputer.Strategy.mean, .median, .mostFrequent, .constant(-0.0), .constant(.nan)] {
        let plan = try FusedPreprocessingPlan(training: training, strategy: strategy)
        let fitted = plan.replacements
        try checkFused(plan, input: selected, as: Double.self)
        try checkFused(plan, input: selected, as: Float.self)
        try checkFused(plan, input: selected, as: Float16.self)
        guard fusedTrialEqual(plan.replacements, fitted), selected.originalRowIndices == [3, 0, 3, 2] else {
            throw BenchmarkFailure("Fitted state or row provenance changed")
        }
        cases += 3
    }
    let plan = try FusedPreprocessingPlan(training: training)
    let mismatch = try PreparedNumericBatch(columnNames: ["b", "a", "c"], columns: heldOutColumns)
    do {
        _ = try plan.fused(mismatch, as: Float.self, tiled: true)
    } catch { return cases + 1 }
    throw BenchmarkFailure("Reordered fitted columns were accepted")
}
