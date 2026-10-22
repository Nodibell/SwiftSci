import Testing
@testable import SwiftDataFrame

@Suite("CSV batch builder state")
struct CSVBatchBuilderStateTests {
    @Test("Int64 batches preserve values, null counts, draining, and rejection")
    func integerLifecycle() {
        let bytes: [UInt8] = Array("1\n?\n2\nbad\n".utf8)
        let expected: [Int64?] = [1, nil, 2]
        let one: [Int64?] = [1]
        bytes.withUnsafeBufferPointer { buffer in
            let records = SystemsCSVParser().parseIndex(buffer: buffer)
            let decoder = CSVIntegerDecoder(nulls: CSVNullMatcher(["?"]))
            let attempt = CSVColumnAttempt(decoder: decoder, capacity: 3, validates: true)
            attempt.append(buffer, records, 0, 2, 0, sourceOffset: 0)
            attempt.append(buffer, records, 2, 1, 0, sourceOffset: 0)
            let prefix = attempt.column(named: "value")
            #expect(prefix.values == expected)
            #expect(prefix.nullCount == 1)
            attempt.append(buffer, records, 0, 1, 0, sourceOffset: 0)
            let next = attempt.column(named: "value")
            #expect(next.values == one)
            #expect(next.nullCount == 0)
            attempt.append(buffer, records, 0, 4, 0, sourceOffset: 100)
            #expect(attempt.rejectedField?.startOffset == 106)
            attempt.append(buffer, records, 0, 1, 0, sourceOffset: 0)
            let rejected = attempt.column(named: "value")
            #expect(rejected.values.isEmpty)
            #expect(rejected.nullCount == 0)
            #expect(prefix.values == expected)
        }
    }

    @Test("Double batches preserve values, null counts, draining, and rejection")
    func doubleLifecycle() {
        let bytes: [UInt8] = Array("1\n?\n2\nbad\n".utf8)
        let expected: [Double?] = [1, nil, 2]
        let one: [Double?] = [1]
        bytes.withUnsafeBufferPointer { buffer in
            let records = SystemsCSVParser().parseIndex(buffer: buffer)
            let decoder = CSVDoubleDecoder(nulls: CSVNullMatcher(["?"]))
            let attempt = CSVColumnAttempt(decoder: decoder, capacity: 3, validates: true)
            attempt.appendDouble(buffer, records, 0, 2, 0, sourceOffset: 0)
            attempt.appendDouble(buffer, records, 2, 1, 0, sourceOffset: 0)
            let prefix = attempt.column(named: "value")
            #expect(prefix.values == expected)
            #expect(prefix.nullCount == 1)
            attempt.appendDouble(buffer, records, 0, 1, 0, sourceOffset: 0)
            let next = attempt.column(named: "value")
            #expect(next.values == one)
            #expect(next.nullCount == 0)
            attempt.appendDouble(buffer, records, 0, 4, 0, sourceOffset: 100)
            #expect(attempt.rejectedField?.startOffset == 106)
            attempt.appendDouble(buffer, records, 0, 1, 0, sourceOffset: 0)
            let rejected = attempt.column(named: "value")
            #expect(rejected.values.isEmpty)
            #expect(rejected.nullCount == 0)
            #expect(prefix.values == expected)
        }
    }

}
