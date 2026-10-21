import Darwin
import Testing
@testable import SwiftDataFrame

@Suite("CSV absolute offset selection")
struct CSVLargeOffsetTests {
    @Test("Automatic selection preserves coordinates beyond two and four GiB")
    func wideRanges() throws {
        let payload = Array("\"x\",\"a\"\"b\"\n".utf8)
        let starts = [Int(Int32.max) - 32, Int(Int32.max) + 1, Int(UInt32.max) + 1]
        for start in starts {
            let size = start + payload.count
            // Reserve virtual address space; only the payload pages are touched.
            let mapping = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
            #expect(mapping != MAP_FAILED)
            guard mapping != MAP_FAILED else { return }
            let allocation = try #require(mapping)
            defer { #expect(munmap(allocation, size) == 0) }
            let bytes = allocation.assumingMemoryBound(to: UInt8.self)
            payload.withUnsafeBufferPointer { source in
                bytes.advanced(by: start).update(from: source.baseAddress!, count: source.count)
            }
            let buffer = UnsafeBufferPointer(start: bytes, count: size)
            payload.withUnsafeBufferPointer { source in
                bytes.update(from: source.baseAddress!, count: source.count)
            }
            let batch = try SystemsCSVParser().scanBatch(buffer: buffer, maximumRows: 1)
            #expect(batch.records.count == 1)
            #expect(batch.nextOffset == payload.count)
            #expect(batch.records.withCompactFields { _ in true } == true)
            #expect(batch.records.field(row: 0, column: 1)?.startOffset == 4)
            #expect(batch.records.field(row: 0, column: 1)?.escapedQuotesPresent == true)
            let index = SystemsCSVParser().scanIndex(buffer: buffer, range: start..<size)
            #expect(index.count == 1)
            #expect(index.field(row: 0, column: 0)?.startOffset == start)
            #expect(index.field(row: 0, column: 0)?.length == 3)
            #expect(index.field(row: 0, column: 1)?.startOffset == start + 4)
            #expect(index.field(row: 0, column: 1)?.length == 6)
            #expect(index.field(row: 0, column: 1)?.escapedQuotesPresent == true)
            let compact = index.withCompactFields { _ in true } == true
            #expect(compact == (start < Int(Int32.max)))
        }
    }
}
