import Foundation

internal struct PackedCSVField: Sendable {
    let startOffset: Int
    private let encodedLength: Int

    init(startOffset: Int, length: Int, escapedQuotesPresent: Bool) {
        self.startOffset = startOffset
        self.encodedLength = escapedQuotesPresent ? ~length : length
    }

    var length: Int { encodedLength >= 0 ? encodedLength : ~encodedLength }
    var escapedQuotesPresent: Bool { encodedLength < 0 }

    var offset: CSVFieldOffset {
        CSVFieldOffset(
            startOffset: startOffset,
            length: length,
            escapedQuotesPresent: escapedQuotesPresent
        )
    }
}

internal struct CSVRecordIndex: Sendable {
    let fields: [PackedCSVField]
    let rowStarts: [Int]

    var count: Int { rowStarts.count - 1 }
    var isEmpty: Bool { count == 0 }

    func row(at index: Int) -> [CSVFieldOffset] {
        fields[rowStarts[index]..<rowStarts[index + 1]].map(\.offset)
    }

    func field(row: Int, column: Int) -> CSVFieldOffset? {
        guard row < count else { return nil }
        let start = rowStarts[row]
        guard column < rowStarts[row + 1] - start else { return nil }
        return fields[start + column].offset
    }
}
