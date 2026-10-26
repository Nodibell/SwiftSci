import SwiftDataFrame

struct PCAFullSVDLayout: Sendable {
    let rows: Int32
    let columns: Int32
    let rank: Int
    let leftCount: Int
    let rightCount: Int
    let integerWorkspaceCount: Int

    init(rows: Int, columns: Int) throws {
        guard rows > 0, columns > 0,
              let lapackRows = Int32(exactly: rows),
              let lapackColumns = Int32(exactly: columns) else {
            throw ClusterError.invalidParameter("PCA SVD dimensions exceed backend limits.")
        }
        self.rows = lapackRows
        self.columns = lapackColumns
        rank = min(rows, columns)

        // DGESDD JOBZ='S' documents LWORK >= 4*rank*rank + 7*rank.
        let square = try Self.product(rank, rank)
        let quadratic = try Self.product(4, square)
        let linear = try Self.product(7, rank)
        let (minimumWorkspace, overflow) = quadratic.addingReportingOverflow(linear)
        guard !overflow, Int32(exactly: minimumWorkspace) != nil else {
            throw ClusterError.invalidParameter("PCA SVD workspace exceeds backend limits.")
        }
        leftCount = try Self.bufferCount(rows, rank, stride: MemoryLayout<Double>.stride)
        rightCount = try Self.bufferCount(rank, columns, stride: MemoryLayout<Double>.stride)
        integerWorkspaceCount = try Self.bufferCount(8, rank, stride: MemoryLayout<Int32>.stride)
    }

    private static func product(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (count, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ClusterError.invalidParameter("PCA SVD workspace size overflowed Int.")
        }
        return count
    }

    private static func bufferCount(_ rows: Int, _ columns: Int, stride: Int) throws -> Int {
        let count = try product(rows, columns)
        _ = try product(count, stride)
        return count
    }
}
