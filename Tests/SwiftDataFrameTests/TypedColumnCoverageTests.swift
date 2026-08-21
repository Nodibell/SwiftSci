import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("TypedColumn Full Coverage Tests")
struct TypedColumnCoverageTests {

    @Test("TypedColumn initializers with nulls")
    func testInitWithNulls() {
        let col = TypedColumn<Int64>(name: "id", values: [1, nil, 3, nil, 5])
        #expect(col.count == 5)
        #expect(col.nullCount == 2)
        #expect(col[0] == 1)
        #expect(col[1] == nil)
        #expect(col.name == "id")
        #expect(col.dtype == .int64)
    }

    @Test("TypedColumn value(at:) reflection")
    func testValueAt() {
        let col = TypedColumn<Double>(name: "val", values: [1.5, nil, 3.5])
        #expect(col.value(at: 0) as? Double == 1.5)
        #expect(col.value(at: 1) == nil)
        #expect(col.value(at: 2) as? Double == 3.5)
    }

    @Test("TypedColumn toDoubles and toStrings conversions")
    func testToDoublesAndToStrings() {
        let colInt = TypedColumn<Int64>(name: "nums", values: [10, 20, 30])
        let dbls = colInt.toDoubles()
        #expect(dbls == [10.0, 20.0, 30.0])

        let strCols = colInt.toStrings()
        #expect(strCols == ["10", "20", "30"])

        let colWithNull = TypedColumn<Double>(name: "d", values: [1.0, nil, 3.0])
        #expect(colWithNull.toDoubles() == [1.0, 3.0])
        #expect(colWithNull.toStrings() == ["1.0", "null", "3.0"])

        let colStr = TypedColumn<String>(name: "s", values: ["a", "b"])
        #expect(colStr.toDoubles() == nil)
    }

    @Test("TypedColumn renamed, dropNulls, fillNull")
    func testRenamedDropNullsFillNull() {
        let col = TypedColumn<Double>(name: "orig", values: [1.0, nil, 3.0])
        let renamed = col.renamed(to: "new_name")
        #expect(renamed.name == "new_name")
        #expect(renamed.count == 3)

        let dropped = col.dropNulls()
        #expect(dropped.count == 2)
        #expect(dropped.nullCount == 0)
        #expect(dropped.values == [1.0, 3.0])

        let filled = col.fillNull(with: 0.0)
        #expect(filled.nullCount == 0)
        #expect(filled.values == [1.0, 0.0, 3.0])
    }

    @Test("TypedColumn map and compactMap")
    func testMapAndCompactMap() {
        let col = TypedColumn<Int64>(name: "x", values: [1, nil, 3])
        let mapped = col.map { val in val.map { Double($0 * 10) } }
        #expect(mapped.values == [10.0, nil, 30.0])

        let compactMapped = col.compactMap { val -> Double? in
            return Double(val * 2)
        }
        #expect(compactMapped.count == 3)
        #expect(compactMapped.values == [2.0, nil, 6.0])
    }

    @Test("TypedColumn lagged")
    func testLagged() {
        let col = TypedColumn<Int64>(name: "t", values: [10, 20, 30, 40])
        guard let lag1 = col.lagged(by: 1) as? TypedColumn<Int64> else {
            #expect(Bool(false), "Lagged cast failed")
            return
        }
        #expect(lag1.values == [nil, 10, 20, 30])

        guard let lagNeg1 = col.lagged(by: -1) as? TypedColumn<Int64> else {
            #expect(Bool(false), "Lagged cast failed")
            return
        }
        #expect(lagNeg1.values == [20, 30, 40, nil])
    }

    @Test("TypedColumn sortedIndices")
    func testSortedIndices() {
        let col = TypedColumn<Double>(name: "val", values: [30.0, 10.0, nil, 20.0])
        let asc = col.sortedIndices(ascending: true)
        #expect(asc == [1, 3, 0, 2])

        let desc = col.sortedIndices(ascending: false)
        #expect(desc == [0, 3, 1, 2])
    }

    @Test("TypedColumn gathered and filtered with mask length mismatch")
    func testGatheredAndFilteredMismatch() {
        let col = TypedColumn<Int64>(name: "id", values: [100, 200, 300])
        guard let gathered = col.gathered(at: [2, 0]) as? TypedColumn<Int64> else {
            #expect(Bool(false))
            return
        }
        #expect(gathered.values == [300, 100])

        #expect(throws: SwiftMLError.self) {
            _ = try col.filtered(by: [true, false]) // Count mismatch (2 != 3)
        }
    }

    @Test("TypedColumn Double statistics: mean, variance, stdDev, vGather")
    func testDoubleStatistics() {
        let col = TypedColumn<Double>(name: "scores", values: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        #expect(abs(col.mean() - 5.0) < 1e-6)
        #expect(col.variance() > 0.0)
        #expect(col.stdDev() > 0.0)

        let gathered = col.vGather(at: [0, 7])
        #expect(gathered.values == [2.0, 9.0])
    }
}

@Suite("SwiftMLError Description & Error Handling Tests")
struct SwiftMLErrorCoverageTests {

    @Test("SwiftMLError description properties output human-readable strings")
    func testErrorDescriptions() {
        let errors: [SwiftMLError] = [
            .invalidInput("bad input"),
            .emptyInput,
            .dimensionMismatch(expected: 10, got: 5),
            .modelNotFitted,
            .trainingFailed("OOM"),
            .convergenceFailed(iterations: 100),
            .exportFailed("Protobuf encoding error"),
            .invalidParameter("learningRate <= 0"),
            .columnNotFound("missing_col"),
            .duplicateColumnName("dup_col"),
            .typeMismatch(column: "age", expected: "Int64", got: "String"),
            .fileNotFound(URL(fileURLWithPath: "/tmp/missing.csv")),
            .insufficientData(minimum: 10, got: 2),
            .emptyTimeSeries,
            .invalidAlpha(1.5)
        ]

        for err in errors {
            #expect(!err.description.isEmpty)
            #expect(err.errorDescription != nil)
        }
    }
}
