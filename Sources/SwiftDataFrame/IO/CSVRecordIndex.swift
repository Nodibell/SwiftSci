import Foundation

internal struct CSVRecordBatch: Sendable {
    let records: CSVRecordIndex
    let nextOffset: Int
}

internal struct PackedCSVField: CSVIndexField {
    static var maximumEndOffset: Int { Int.max }
    let startOffset: Int
    private let encodedLength: Int

    init(startOffset: Int, length: Int, escapedQuotesPresent: Bool) {
        self.startOffset = startOffset
        self.encodedLength = escapedQuotesPresent ? ~length : length
    }

    init(startOffset: Int, encodedLength: Int) {
        self.startOffset = startOffset
        self.encodedLength = encodedLength
    }

    static func recordIndex(blocks: CSVFieldBlocks<Self>, rowStarts: [Int]) -> CSVRecordIndex {
        CSVRecordIndex(blocks: blocks, rowStarts: rowStarts)
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

internal struct CSVRecordIndex: Sendable, CSVRecordAccess {
    private let fields: CSVFieldStorage

    init(fields: [PackedCSVField], rowStarts: [Int]) {
        self.fields = .flat(fields)
        self.rowStarts = rowStarts
    }

    init(blocks: CSVFieldBlocks<PackedCSVField>, rowStarts: [Int]) {
        self.fields = .blocked(blocks.finished())
        self.rowStarts = rowStarts
    }

    init(blocks: CSVFieldBlocks<CompactCSVField>, rowStarts: [Int]) {
        self.fields = .compact(blocks.finished())
        self.rowStarts = rowStarts
    }

    let rowStarts: [Int]

    var count: Int { rowStarts.count - 1 }
    var isEmpty: Bool { count == 0 }

    // The borrowed view is valid only during this synchronous closure.
    // It must not be returned, retained, or passed to asynchronous work.
    func withFlatFields<Result>(_ body: (CSVFlatRecordView) -> Result) -> Result? {
        guard case .flat(let values) = fields else { return nil }
        return values.withUnsafeBufferPointer { fields in
            rowStarts.withUnsafeBufferPointer { starts in
                body(CSVFlatRecordView(fields: fields, rowStarts: starts))
            }
        }
    }

    // The borrowed block view has the same synchronous lifetime as withFlatFields.
    func withCompactFields<Result>(_ body: (CSVCompactRecordView) -> Result) -> Result? {
        guard case .compact(let values) = fields else { return nil }
        return values.withUnsafeBufferPointer { blocks in
            rowStarts.withUnsafeBufferPointer { starts in
                body(CSVCompactRecordView(blocks: blocks, rowStarts: starts))
            }
        }
    }

    func row(at index: Int) -> [CSVFieldOffset] {
        (rowStarts[index]..<rowStarts[index + 1]).map { fields[$0].offset }
    }

    @inline(__always)
    func field(row: Int, column: Int) -> CSVFieldOffset? {
        guard row < count else { return nil }
        let start = rowStarts[row]
        guard column < rowStarts[row + 1] - start else { return nil }
        return fields[start + column].offset
    }
}

private enum CSVFieldBlockLayout {
    static var shift: Int { 14 }
    static var capacity: Int { 1 << shift }
    static var mask: Int { capacity - 1 }
}

private enum CSVFieldStorage: Sendable {
    case flat([PackedCSVField])
    case blocked([[PackedCSVField]])
    case compact([[CompactCSVField]])

    subscript(index: Int) -> PackedCSVField {
        switch self {
        case .flat(let values): return values[index]
        case .blocked(let blocks): return blocks[index >> CSVFieldBlockLayout.shift][index & CSVFieldBlockLayout.mask]
        case .compact(let blocks): return blocks[index >> CSVFieldBlockLayout.shift][index & CSVFieldBlockLayout.mask].expanded
        }
    }
}

internal struct CSVFieldBlocks<Field: CSVIndexField> {
    private var completed = [[Field]]()
    private var current = [Field]()
    private(set) var count = 0

    init(byteCount: Int) {
        current.reserveCapacity(min(byteCount / 8, CSVFieldBlockLayout.capacity))
    }

    mutating func append(_ field: Field) {
        if current.count == CSVFieldBlockLayout.capacity {
            completed.append(current)
            current = []
            current.reserveCapacity(CSVFieldBlockLayout.capacity)
        }
        current.append(field)
        count += 1
    }

    mutating func truncate(to newCount: Int) {
        while count > newCount {
            if current.isEmpty { current = completed.removeLast() }
            let removed = min(count - newCount, current.count)
            current.removeLast(removed)
            count -= removed
        }
    }

    func finished() -> [[Field]] {
        var result = completed
        if !current.isEmpty { result.append(current) }
        return result
    }
}

internal protocol CSVIndexField: Sendable {
    static var maximumEndOffset: Int { get }
    init(startOffset: Int, length: Int, escapedQuotesPresent: Bool)
    static func recordIndex(blocks: CSVFieldBlocks<Self>, rowStarts: [Int]) -> CSVRecordIndex
}

internal struct CompactCSVField: CSVIndexField {
    static var maximumEndOffset: Int { Int(Int32.max) }
    private let startOffset: UInt32
    private let encodedLength: Int32

    static func canRepresent(endOffset: Int) -> Bool {
        endOffset >= 0 && endOffset <= Int(Int32.max)
    }

    init(startOffset: Int, length: Int, escapedQuotesPresent: Bool) {
        self.startOffset = UInt32(startOffset)
        let length = Int32(length)
        self.encodedLength = escapedQuotesPresent ? ~length : length
    }

    var expanded: PackedCSVField {
        PackedCSVField(startOffset: Int(startOffset), encodedLength: Int(encodedLength))
    }

    var offset: CSVFieldOffset { expanded.offset }

    static func recordIndex(blocks: CSVFieldBlocks<Self>, rowStarts: [Int]) -> CSVRecordIndex {
        CSVRecordIndex(blocks: blocks, rowStarts: rowStarts)
    }
}

internal protocol CSVRecordAccess {
    func field(row: Int, column: Int) -> CSVFieldOffset?
}

internal struct CSVFlatRecordView: CSVRecordAccess {
    let fields: UnsafeBufferPointer<PackedCSVField>
    let rowStarts: UnsafeBufferPointer<Int>

    @inline(__always)
    func field(row: Int, column: Int) -> CSVFieldOffset? {
        guard row < rowStarts.count - 1 else { return nil }
        let start = rowStarts[row]
        guard column < rowStarts[row + 1] - start else { return nil }
        return fields[start + column].offset
    }
}

internal struct CSVCompactRecordView: CSVRecordAccess {
    let blocks: UnsafeBufferPointer<[CompactCSVField]>
    let rowStarts: UnsafeBufferPointer<Int>

    @inline(__always)
    func field(row: Int, column: Int) -> CSVFieldOffset? {
        guard row < rowStarts.count - 1 else { return nil }
        let start = rowStarts[row]
        guard column < rowStarts[row + 1] - start else { return nil }
        let index = start + column
        return blocks[index >> CSVFieldBlockLayout.shift][index & CSVFieldBlockLayout.mask].offset
    }
}
