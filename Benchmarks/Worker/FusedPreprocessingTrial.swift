import SwiftDataFrame
import SwiftPreprocessing

typealias FusedPreprocessingPlan = TrialFusedPreprocessingPlan

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
