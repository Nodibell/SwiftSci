import Testing
import SwiftDataFrame
@testable import SwiftCluster

enum PreparedPCAFixture {
    static let rows: [[Double]] = [[16, 6], [4, -10], [14, -5], [6, 1]]
    static let query: [[Double]] = [[13, 2], [14, -5], [10, -2]]
    static let scores: [[Double]] = [[10, 0], [-10, 0], [0, 5], [0, -5]]
    static let queryScores: [[Double]] = [[5, 0], [0, 5], [0, 0]]

    static func batch(_ rows: [[Double]] = PreparedPCAFixture.rows, names: [String] = ["a", "b"]) throws -> PreparedNumericBatch {
        let columns: [[Double]] = names.indices.map { column in rows.map { $0[column] } }
        return try PreparedNumericBatch(columnNames: names, columns: columns)
    }

    static func close(_ actual: [Double], _ expected: [Double], tolerance: Double = 1e-11) {
        #expect(actual.count == expected.count)
        for (a, b) in zip(actual, expected) { #expect(abs(a - b) <= tolerance * max(1, abs(b))) }
    }

    static func close(_ actual: PreparedNumericBatch, _ expected: [[Double]], tolerance: Double = 1e-11) {
        #expect(actual.rowCount == expected.count)
        for row in expected.indices {
            for column in expected[row].indices {
                let value = actual[row, column] ?? .nan
                #expect(abs(value - expected[row][column]) <= tolerance)
            }
        }
    }
}
