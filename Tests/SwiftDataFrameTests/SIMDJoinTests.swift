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

    func testSIMDRightJoinInt32() throws {
        let left = try DataFrame(columns: [
            TypedColumn<Int32>(name: "id", values: [1, 2]),
            TypedColumn<String>(name: "value", values: ["A", "B"])
        ])

        let right = try DataFrame(columns: [
            TypedColumn<Int32>(name: "id", values: [2, 3]),
            TypedColumn<String>(name: "value", values: ["B2", "C2"])
        ])

        let joined = try left.joinSIMD(right, on: "id", how: .right)
        XCTAssertEqual(joined.rowCount, 2)
        XCTAssertEqual(joined.columnNames, ["id", "value_x", "value_y"])
    }

    func testSIMDJoinFloat32AndBool() throws {
        let left = try DataFrame(columns: [
            TypedColumn<Float>(name: "f_key", values: [1.0, 2.0]),
            TypedColumn<Bool>(name: "b_key", values: [true, false])
        ])

        let right = try DataFrame(columns: [
            TypedColumn<Float>(name: "f_key", values: [2.0, 3.0]),
            TypedColumn<Bool>(name: "b_key", values: [false, true])
        ])

        let floatJoined = try left.joinSIMD(right, on: "f_key", how: .inner)
        XCTAssertEqual(floatJoined.rowCount, 1)

        let boolJoined = try left.joinSIMD(right, on: "b_key", how: .inner)
        XCTAssertEqual(boolJoined.rowCount, 2)
    }

    func testSIMDJoinErrorHandlingAndEmpty() throws {
        let df1 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2])
        ])
        let df2 = try DataFrame(columns: [
            TypedColumn<Double>(name: "id", values: [1.0, 2.0])
        ])
        let df3 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "other_id", values: [1, 2])
        ])

        // Column not found
        XCTAssertThrowsError(try df1.joinSIMD(df3, on: "id", how: .inner))
        XCTAssertThrowsError(try df3.joinSIMD(df1, on: "id", how: .inner))

        // Type mismatch
        XCTAssertThrowsError(try df1.joinSIMD(df2, on: "id", how: .inner))

        // Empty DataFrames
        let emptyDF = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [])
        ])
        let innerEmpty = try emptyDF.joinSIMD(df1, on: "id", how: .inner)
        XCTAssertEqual(innerEmpty.rowCount, 0)

        let leftEmpty = try emptyDF.joinSIMD(df1, on: "id", how: .left)
        XCTAssertEqual(leftEmpty.rowCount, 0)
    }
}
