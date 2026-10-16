import Testing
@testable import SwiftDataFrame

@Suite("Concurrent compact row gathers")
struct CompactRowGatherConcurrencyTests {
    @Test func callersOwnIndependentGatherOutputs() async throws {
        let rows = 2048
        let width = 129
        let names = (0..<width).map { "x\($0)" }
        let columns = (0..<width).map { column in (0..<rows).map { Double(column * rows + $0) } }
        var prepared = try PreparedNumericBatch(columnNames: names, columns: columns)
        for column in 0..<width {
            try prepared.updateColumn(at: column, rows: [0, 63, 64, 2047], values: [nil, nil, nil, nil])
        }
        let source = prepared
        let passed = try await withThrowingTaskGroup(of: Bool.self) { group in
            for caller in 0..<8 {
                group.addTask {
                    let indices = (0..<rows).map { ($0 * 7919 + caller) % rows }
                    var selected = try source.selectingRows(indices)
                    let snapshot = selected
                    try selected.updateColumn(at: caller, rows: [0], values: [Double(-caller - 1)])
                    guard selected[0, caller] == Double(-caller - 1),
                          snapshot.originalRowIndices == indices else { return false }
                    for column in 0..<width {
                        for row in 0..<rows {
                            guard snapshot[row, column] == source[indices[row], column] else { return false }
                        }
                    }
                    return true
                }
            }
            var allPassed = true
            for try await result in group { allPassed = allPassed && result }
            return allPassed
        }
        #expect(passed)
        #expect(source.nullCount(inColumn: 0) == 4)
        #expect(source[1, 0] == 1)
    }
}
