import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Experimental compact Double CSV storage")
struct CSVCompactDoublePrototypeTests {
    @Test("Word boundaries preserve nil, NaN payloads, signed zero and infinities")
    func boundaries() {
        for count in [0, 1, 63, 64, 65, 127, 128, 129] {
            for pattern in 0..<3 {
                var builder = CSVCompactDoubleBuilder(capacity: count)
                let expected: [Double?] = (0..<count).map { row in
                    if pattern == 0 { return nil }
                    if pattern == 1 { return Double(row) }
                    return fixture(row)
                }
                for value in expected { builder.append(value) }
                let column = builder.column(named: "x")
                #expect(column.count == count)
                #expect(column.nullCount == expected.filter { $0 == nil }.count)
                compare(column.values, expected)
                for row in expected.indices { compare([column[row]], [expected[row]]) }
            }
        }
    }

    @Test("A finalized builder releases ownership and can grow from empty capacity")
    func reuseAndSnapshots() {
        var builder = CSVCompactDoubleBuilder(capacity: 1)
        builder.append(-0.0)
        let first = builder.column(named: "first")
        var expected = [Double?]()
        for row in 0..<130 {
            let value: Double? = row % 7 == 0 ? nil : Double(row)
            builder.append(value)
            expected.append(value)
        }
        let second = builder.column(named: "second")
        compare(first.values, [-0.0])
        compare(second.values, expected)
        var exported = second.values
        exported[1] = 999
        compare(second.values, expected)
    }

    @Test("Existing typed operations agree with optional-backed frames")
    func publicOperations() throws {
        var builder = CSVCompactDoubleBuilder(capacity: 130)
        let expected = (0..<130).map(fixture)
        for value in expected { builder.append(value) }
        let keys = TypedColumn<Int64>(name: "id", values: (0..<130).map(Int64.init))
        let compact = try DataFrame(columns: [keys, builder.column(named: "x")])
        let optional = try DataFrame(columns: [keys, TypedColumn(name: "x", values: expected)])
        compare(try #require(compact[column: "x", as: Double.self]).values, expected)
        compareFrames(compact.gathered(at: [129, 0, 64, 64, 63]), optional.gathered(at: [129, 0, 64, 64, 63]))
        compareFrames(try compact.filter(column: "x", where: .greaterThan(0.0)),
                      try optional.filter(column: "x", where: .greaterThan(0.0)))
        compareFrames(try compact.sortBy("x"), try optional.sortBy("x"))
        compareFrames(try compact.sortBy("x", ascending: false), try optional.sortBy("x", ascending: false))
        let right = try DataFrame(columns: [TypedColumn<Int64>(name: "id", values: [0, 63, 64, 129]),
                                           TypedColumn<String>(name: "label", values: ["a", "b", "c", "d"])])
        compareFrames(try compact.join(right, on: "id", how: .left), try optional.join(right, on: "id", how: .left))
        let actualMatrix = try compact.prepareNumericMatrix(["x"])
        let expectedMatrix = try optional.prepareNumericMatrix(["x"])
        compare(actualMatrix.values.map(Optional.some), expectedMatrix.values.map(Optional.some))
        let batch = try compact.prepareNumericBatch(["x"])
        compare(batch.columnValues(at: 0), expected)
    }

    @Test("Scalar row access, renaming and lags preserve the compact values")
    func scalarAndLag() throws {
        var builder = CSVCompactDoubleBuilder(capacity: 130)
        let expected = (0..<130).map(fixture)
        for value in expected { builder.append(value) }
        let column = builder.column(named: "x")
        let frame = try DataFrame(columns: [column])
        #expect(column.value(at: -1) == nil)
        #expect(column.value(at: 130) == nil)
        for row in expected.indices {
            let direct = column.value(at: row) as? Double
            let boxed = frame.row(at: row)["x"] as? Double
            #expect(direct?.bitPattern == expected[row]?.bitPattern)
            #expect(boxed?.bitPattern == expected[row]?.bitPattern)
        }
        let renamed = column.renamed(to: "renamed") as! TypedColumn<Double>
        compare(renamed.values, expected)
        for offset in [-3, 0, 3] {
            let lagged = column.lagged(by: offset) as! TypedColumn<Double>
            let wanted = expected.indices.map { row -> Double? in
                let source = row - offset
                return expected.indices.contains(source) ? expected[source] : nil
            }
            compare(lagged.values, wanted)
        }
    }

    @Test("Public CSV retains native types in both conversion policies", arguments: [CSVColumnConversionMode.parallel, .serial])
    func publicCSV(_ mode: CSVColumnConversionMode) async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("compact-double-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("id,x,label\n9007199254740993,NaN,valid\n2,NULL,missing\n3,-0,zero\n4,inf,infinity\n".utf8).write(to: url)
        var options = CSVReadOptions()
        options.columnConversionMode = mode
        options.columnTypeOverrides = ["id": .int64, "x": .float64, "label": .utf8]
        options.nullValues = ["NULL"]
        let frame = try await DataFrame(csv: url, options: options)
        #expect(frame[column: "id", as: Int64.self]?[0] == 9_007_199_254_740_993)
        let x = try #require(frame[column: "x", as: Double.self])
        #expect(x[0]?.isNaN == true)
        #expect(x[1] == nil)
        #expect(x[2]?.bitPattern == (-0.0).bitPattern)
        #expect(x[3] == .infinity)
        #expect(x.nullCount == 1)
    }

    private func fixture(_ row: Int) -> Double? {
        switch row % 7 {
        case 0: return nil
        case 1: return Double(bitPattern: 0x7ff8000000000042)
        case 2: return -0.0
        case 3: return .infinity
        case 4: return -.infinity
        default: return Double(row) - 65
        }
    }

    private func compareFrames(_ actual: DataFrame, _ expected: DataFrame) {
        #expect(actual.columnNames == expected.columnNames)
        #expect(actual.rowCount == expected.rowCount)
        for name in expected.columnNames {
            if let source = expected[column: name, as: Double.self] {
                compare(actual[column: name, as: Double.self]!.values, source.values)
            } else if let source = expected[column: name, as: Int64.self] {
                #expect(actual[column: name, as: Int64.self]?.values == source.values)
            } else if let source = expected[column: name, as: String.self] {
                #expect(actual[column: name, as: String.self]?.values == source.values)
            }
        }
    }

    private func compare(_ actual: [Double?], _ expected: [Double?]) {
        #expect(actual.count == expected.count)
        for (actual, expected) in zip(actual, expected) {
            #expect(actual?.bitPattern == expected?.bitPattern)
        }
    }
}
