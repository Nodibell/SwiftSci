import Testing
import SwiftDataFrame

@Suite("Gathering large narrow frames")
struct GatherWorkTests {
    private let count = 131_101

    private func reorderedIndices() -> [Int] {
        let permutation = (0..<count).map { ($0 * 7_919) % count }
        return permutation + Array(permutation.reversed())
    }

    private func checkColumn<T: SupportedType>(
        _ result: DataFrame,
        name: String,
        original: [T?],
        indices: [Int]
    ) throws {
        let column = try #require(result[column: name, as: T.self])
        let expected = indices.map { original[$0] }
        #expect(column.values == expected)
        #expect(column.count == indices.count)
        #expect(column.nullCount == expected.filter { $0 == nil }.count)
        #expect(column.dtype == TypedColumn<T>(name: name, values: [T?]()).dtype)
    }

    @Test("Two numeric columns retain duplicates, requested order and null counts", arguments: [0, 17])
    func twoNumericColumns(nullEvery: Int) throws {
        let floats: [Float?] = (0..<count).map { index in
            nullEvery > 0 && index % nullEvery == 0 ? nil : Float(index % 8_192) * 0.25
        }
        let integers: [Int32?] = (0..<count).map { index in
            nullEvery > 0 && index % (nullEvery + 2) == 0 ? nil : Int32(index - count / 2)
        }
        let frame = try DataFrame(columns: [
            TypedColumn<Float>(name: "score", values: floats),
            TypedColumn<Int32>(name: "identifier", values: integers)
        ])
        let indices = reorderedIndices()
        let result = frame.gathered(at: indices)
        #expect(result.rowCount == indices.count)
        #expect(result.columnNames == ["score", "identifier"])
        try checkColumn(result, name: "score", original: floats, indices: indices)
        try checkColumn(result, name: "identifier", original: integers, indices: indices)
        #expect(frame[column: "score", as: Float.self]?.values == floats)
        #expect(frame[column: "identifier", as: Int32.self]?.values == integers)
    }

    @Test("Three mixed columns preserve string storage, Boolean values and all-null counts")
    func threeMixedColumns() throws {
        let strings: [String?] = (0..<count).map { index in
            index % 11 == 0 ? nil : "long retained string value \(index % 257)"
        }
        let flags: [Bool?] = (0..<count).map { index in
            index % 13 == 0 ? nil : index % 3 == 0
        }
        let missing = [Int32?](repeating: nil, count: count)
        let frame = try DataFrame(columns: [
            TypedColumn<String>(name: "label", values: strings),
            TypedColumn<Bool>(name: "flag", values: flags),
            TypedColumn<Int32>(name: "missing", values: missing)
        ])
        let indices = reorderedIndices()
        let result = frame.gathered(at: indices)
        #expect(result.rowCount == indices.count)
        #expect(result.columnNames == ["label", "flag", "missing"])
        try checkColumn(result, name: "label", original: strings, indices: indices)
        try checkColumn(result, name: "flag", original: flags, indices: indices)
        try checkColumn(result, name: "missing", original: missing, indices: indices)
        #expect(frame[column: "label", as: String.self]?.values == strings)
        #expect(frame[column: "flag", as: Bool.self]?.values == flags)
        #expect(frame[column: "missing"]?.nullCount == count)
    }

    @Test("Empty row selections return the existing empty-frame result")
    func emptySelection() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Float>(name: "score", values: [Float?](repeating: 1, count: count)),
            TypedColumn<Int32>(name: "identifier", values: [Int32?](repeating: nil, count: count)),
            TypedColumn<String>(name: "label", values: [String?](repeating: "retained", count: count))
        ])
        let result = frame.gathered(at: [])
        #expect(result.rowCount == 0)
        #expect(result.columnNames.isEmpty)
        #expect(frame.rowCount == count)
        #expect(frame[column: "identifier"]?.nullCount == count)
        let empty = DataFrame.empty.gathered(at: [])
        #expect(empty.rowCount == 0)
        #expect(empty.columnNames.isEmpty)
    }
}
