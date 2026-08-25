import XCTest
@testable import SwiftDataFrame

final class SIMDJoinTests: XCTestCase {

    func testSIMDInnerJoinInt64() throws {
        let left = try DataFrame(columns: [
            TypedColumn<Int64>(name: "user_id", values: [1, 2, 3, 4]),
            TypedColumn<String>(name: "name", values: ["Alice", "Bob", "Charlie", "David"])
        ])

        let right = try DataFrame(columns: [
            TypedColumn<Int64>(name: "user_id", values: [2, 4, 5]),
            TypedColumn<Double>(name: "balance", values: [250.0, 450.0, 550.0])
        ])

        let joined = try left.join(right, on: "user_id", how: .inner)

        XCTAssertEqual(joined.rowCount, 2)
        XCTAssertEqual(joined.columnNames, ["user_id", "name", "balance"])

        let userIds = (joined[column: "user_id"] as? TypedColumn<Int64>)?.values
        XCTAssertEqual(userIds, [2, 4])

        let names = (joined[column: "name"] as? TypedColumn<String>)?.values
        XCTAssertEqual(names, ["Bob", "David"])

        let balances = (joined[column: "balance"] as? TypedColumn<Double>)?.values
        XCTAssertEqual(balances, [250.0, 450.0])
    }

    func testSIMDLeftJoinDouble() throws {
        let left = try DataFrame(columns: [
            TypedColumn<Double>(name: "key", values: [1.1, 2.2, 3.3]),
            TypedColumn<String>(name: "l_val", values: ["L1", "L2", "L3"])
        ])

        let right = try DataFrame(columns: [
            TypedColumn<Double>(name: "key", values: [2.2, 4.4]),
            TypedColumn<String>(name: "r_val", values: ["R2", "R4"])
        ])

        let joined = try left.joinSIMD(right, on: "key", how: .left)

        XCTAssertEqual(joined.rowCount, 3)
        let keys = (joined[column: "key"] as? TypedColumn<Double>)?.values
        XCTAssertEqual(keys, [1.1, 2.2, 3.3])

        let rVals = (joined[column: "r_val"] as? TypedColumn<String>)?.values
        XCTAssertEqual(rVals, [nil, "R2", nil])
    }

    func testSIMDOuterJoinString() throws {
        let left = try DataFrame(columns: [
            TypedColumn<String>(name: "country", values: ["UA", "US"]),
            TypedColumn<Int64>(name: "pop_m", values: [40, 330])
        ])

        let right = try DataFrame(columns: [
            TypedColumn<String>(name: "country", values: ["US", "DE"]),
            TypedColumn<String>(name: "capital", values: ["Washington", "Berlin"])
        ])

        let joined = try left.join(right, on: "country", how: .outer)

        XCTAssertEqual(joined.rowCount, 3)
        XCTAssertEqual(joined.columnNames, ["country", "pop_m", "capital"])
    }
}
