import XCTest
@testable import SwiftDataFrame

final class FastFilterTests: XCTestCase {

    // MARK: - Correctness: results must match the closure-path filter exactly

    func testFilterFastDoubleGreaterThan() throws {
        let scores = TypedColumn<Double>(name: "score", values: [88.5, 94.0, 72.0, 96.5, 81.0])
        let ids    = TypedColumn<Int64>(name: "id",    values: [1, 2, 3, 4, 5])
        let df     = try DataFrame(columns: [scores, ids])

        let fast = try df.filterFast(column: "score", where: .greaterThan(90.0))
        let slow = df.filter { row in (row.double("score") ?? 0) > 90.0 }

        XCTAssertEqual(fast.shape.rows, slow.shape.rows, "Row count mismatch")
        XCTAssertEqual(fast.shape.rows, 2) // 94.0 and 96.5
    }

    func testFilterFastDoubleGreaterThanOrEqual() throws {
        let scores = TypedColumn<Double>(name: "score", values: [85.0, 94.0, 72.0, 85.0, 100.0])
        let df     = try DataFrame(columns: [scores])

        let fast = try df.filterFast(column: "score", where: .greaterThanOrEqual(85.0))
        let slow = df.filter { row in (row.double("score") ?? 0) >= 85.0 }

        XCTAssertEqual(fast.shape.rows, slow.shape.rows)
        XCTAssertEqual(fast.shape.rows, 4)
    }

    func testFilterFastDoubleLessThan() throws {
        let scores = TypedColumn<Double>(name: "score", values: [10.0, 50.0, 99.0, 5.0])
        let df     = try DataFrame(columns: [scores])

        let fast = try df.filterFast(column: "score", where: .lessThan(50.0))
        XCTAssertEqual(fast.shape.rows, 2) // 10.0, 5.0
    }

    func testFilterFastDoubleEquals() throws {
        let scores = TypedColumn<Double>(name: "score", values: [1.0, 2.0, 2.0, 3.0])
        let df     = try DataFrame(columns: [scores])

        let fast = try df.filterFast(column: "score", where: .equals(2.0))
        XCTAssertEqual(fast.shape.rows, 2)
    }

    // MARK: - Int64 column

    func testFilterFastInt64GreaterThan() throws {
        let ids = TypedColumn<Int64>(name: "id", values: [1, 5, 3, 10, 2])
        let df  = try DataFrame(columns: [ids])

        let fast = try df.filterFast(column: "id", where: .greaterThan(4))
        XCTAssertEqual(fast.shape.rows, 2) // 5, 10
    }

    // MARK: - Sugar overload

    func testFilterFastSugarOverload() throws {
        let scores = TypedColumn<Double>(name: "score", values: [10.0, 20.0, 30.0, 40.0])
        let df     = try DataFrame(columns: [scores])

        let fast = try df.filterFast(column: "score", op: .greaterThanOrEqual, threshold: 25.0)
        XCTAssertEqual(fast.shape.rows, 2) // 30.0, 40.0
    }

    // MARK: - Large column parity test (100k rows)

    func testFilterFastLargeColumnMatchesClosurePath() throws {
        let n = 100_000
        var vals = [Double](repeating: 0.0, count: n)
        for i in 0..<n { vals[i] = Double(i % 1000) }

        let col = TypedColumn<Double>(name: "v", values: vals)
        let df  = try DataFrame(columns: [col])

        let threshold = 500.0
        let fast = try df.filterFast(column: "v", where: .greaterThanOrEqual(threshold))
        let slow = df.filter { row in (row.double("v") ?? 0) >= threshold }

        XCTAssertEqual(fast.shape.rows, slow.shape.rows,
                       "filterFast must produce identical row count to closure filter on 100k rows")
    }

    // MARK: - Error: column not found

    func testFilterFastThrowsOnMissingColumn() throws {
        let col = TypedColumn<Double>(name: "score", values: [1.0, 2.0])
        let df  = try DataFrame(columns: [col])

        XCTAssertThrowsError(try df.filterFast(column: "nonexistent", where: .greaterThan(0.0)))
    }
}
