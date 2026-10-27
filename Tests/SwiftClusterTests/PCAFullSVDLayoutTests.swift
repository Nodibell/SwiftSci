import Testing
@testable import SwiftCluster

@Suite("PCA full SVD workspace bounds")
struct PCAFullSVDLayoutTests {
    @Test(arguments: [[4, 3], [3, 4], [1, 1], [10, 1], [1, 10]])
    func economyShapes(_ shape: [Int]) throws {
        let rows = shape[0], columns = shape[1], rank = min(rows, columns)
        let layout = try PCAFullSVDLayout(rows: rows, columns: columns)
        #expect(layout.rows == Int32(rows) && layout.columns == Int32(columns))
        #expect(layout.rank == rank)
        #expect(layout.leftCount == rows * rank)
        #expect(layout.rightCount == rank * columns)
        #expect(layout.integerWorkspaceCount == 8 * rank)
    }

    @Test func rejectsUnsupportedDimensionsAndWorkspaceWithoutAllocation() throws {
        for invalid in [0, -1, Int(Int32.max) + 1, Int.max] {
            #expect(throws: ClusterError.self) { try PCAFullSVDLayout(rows: invalid, columns: 2) }
            #expect(throws: ClusterError.self) { try PCAFullSVDLayout(rows: 2, columns: invalid) }
        }
        #expect(throws: ClusterError.self) {
            try PCAFullSVDLayout(rows: Int(Int32.max), columns: Int(Int32.max))
        }
    }

    @Test func preservesBackendSupportedLargeLayoutWithoutAllocation() throws {
        let layout = try PCAFullSVDLayout(rows: 24_000, columns: 24_000)
        #expect(layout.leftCount == 576_000_000)
        #expect(layout.rightCount == 576_000_000)
        #expect(layout.integerWorkspaceCount == 192_000)
    }
}
