import Testing
@testable import SwiftDataFrame

@Suite("Compact CSV validity growth")
struct CSVCompactValidityGrowthTests {
    @Test("Late missing values preserve earlier values while growing across words",
          arguments: [0, 1, 64, 512], [0, 63, 64, 65, 127, 128])
    func lateFirstMissing(capacity: Int, firstMissing: Int) {
        var builder = CSVCompactDoubleBuilder(capacity: capacity)
        var expected = [Double?]()
        let missingRows = Set([firstMissing, firstMissing + 64, 256])
        for row in 0..<257 {
            let value: Double?
            if missingRows.contains(row) {
                value = nil
            } else if row % 7 == 0 {
                value = Double(bitPattern: 0x7ff8000000000042)
            } else {
                value = Double(row)
            }
            expected.append(value)
            builder.append(value)
        }
        let column = builder.column(named: "value")
        #expect(column.count == expected.count)
        #expect(column.nullCount == missingRows.count)
        for row in expected.indices {
            #expect(column[row]?.bitPattern == expected[row]?.bitPattern)
        }
    }
}
