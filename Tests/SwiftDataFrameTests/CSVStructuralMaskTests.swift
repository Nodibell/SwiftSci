import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV structural mask packing")
struct CSVStructuralMaskTests {
    @Test("Every 16-byte delimiter mask preserves ordered field coordinates")
    func allDelimiterMasks() {
        let parser = SystemsCSVParser()
        for mask in 0..<65_536 {
            var bytes = [UInt8](repeating: 120, count: 17)
            bytes[16] = 10
            var boundaries: [Int] = [-1]
            for lane in 0..<16 {
                if mask & (1 << lane) != 0 {
                    bytes[lane] = 44
                    boundaries.append(lane)
                }
            }
            boundaries.append(16)
            let rows = bytes.withUnsafeBufferPointer { parser.parse(buffer: $0) }
            guard rows.count == 1, rows[0].count == boundaries.count - 1 else {
                Issue.record("Incorrect field count for mask \(mask)")
                return
            }
            for index in rows[0].indices {
                let field = rows[0][index]
                let start = boundaries[index] + 1
                let length = boundaries[index + 1] - start
                guard field.startOffset == start, field.length == length,
                      !field.escapedQuotesPresent else {
                    Issue.record("Incorrect coordinate for mask \(mask), field \(index)")
                    return
                }
            }
        }
    }
}
