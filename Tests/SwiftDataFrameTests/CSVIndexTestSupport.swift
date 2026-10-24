@testable import SwiftDataFrame

extension CSVRecordIndex {
    var testFields: [CSVFieldOffset] {
        (0..<count).flatMap { row(at: $0) }
    }
}
