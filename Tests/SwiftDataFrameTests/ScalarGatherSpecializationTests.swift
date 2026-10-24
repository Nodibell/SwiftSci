import Foundation
import Testing
import SwiftDataFrame

@Suite("Scalar gather specialization")
struct ScalarGatherSpecializationTests {
    private func check<T: SupportedType>(_ values: [T?]) throws {
        let source = TypedColumn<T>(name: "payload", values: values)
        for count in [0, 1, 129, 32769, 65539] {
            let indices = (0..<count).map { ($0 * 5) % values.count }
            let expected = indices.map { values[$0] }
            let result = try #require((source as any AnyColumn).gathered(at: indices) as? TypedColumn<T>)
            #expect(result.values == expected)
            #expect(result.nullCount == expected.filter { $0 == nil }.count)
            #expect(result.name == source.name && result.dtype == source.dtype)
            #expect(source.values == values)
        }
    }

    @Test("String, Boolean and Date preserve duplicate selections and missing values")
    func standardTypes() throws {
        try check(["retained text longer than inline storage", nil, "", "日本語", nil, "null", "tail"])
        try check([true, nil, false, false, nil, true, false])
        try check([Date(timeIntervalSince1970: -0.25), nil, Date(timeIntervalSince1970: 1.125)])
    }

    @Test("Cached null counts remain correct with uniform validity")
    func uniformValidity() throws {
        try check(["first", "second", "third"])
        try check([nil, nil, nil] as [String?])
        try check([true, false, true])
        try check([nil, nil, nil] as [Bool?])
        try check([nil, nil, nil] as [Date?])
    }

    @Test("User-defined SupportedType columns retain the generic fallback")
    func customType() throws {
        try check([GatherToken(text: "kept"), nil, GatherToken(text: "other")])
    }
}

private struct GatherToken: SupportedType {
    let text: String
    static var columnDType: ColumnDType { .utf8 }
    static func parse(from string: String) -> Self? { Self(text: string) }
    var doubleValue: Double? { nil }
}
