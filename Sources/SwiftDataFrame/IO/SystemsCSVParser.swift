import Foundation

/// Coordinates of a CSV field within an un-copied byte buffer.
public struct CSVFieldOffset: Sendable {
    /// The start offset.
    public let startOffset: Int
    /// The length.
    public let length: Int
    /// The escaped quotes present.
    public let escapedQuotesPresent: Bool
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - startOffset: The start offset.
    ///   - length: The length.
    ///   - escapedQuotesPresent: The escaped quotes present.
    public init(startOffset: Int, length: Int, escapedQuotesPresent: Bool) {
        self.startOffset = startOffset
        self.length = length
        self.escapedQuotesPresent = escapedQuotesPresent
    }
}

/// Zero-copy, RFC 4180 compliant CSV byte-level parser.
///
/// Uses a Deterministic Finite Automaton (DFA) on an `UnsafeBufferPointer<UInt8>`
/// to scan CSV byte blocks without string heap allocations. Correctly handles embedded
/// commas, newlines, and escaped double quotes (`""`).
public final class SystemsCSVParser: Sendable {
    private let delimiterByte: UInt8
    private let quoteByte: UInt8
    private let lfByte: UInt8 = 10         // ASCII '\n'
    private let crByte: UInt8 = 13         // ASCII '\r'

    /// Creates a new instance.
    /// - Parameters:
    ///   - delimiterByte: The delimiter byte.
    ///   - quoteByte: The quote byte.
    public init(delimiterByte: UInt8 = 44, quoteByte: UInt8 = 34) {
        self.delimiterByte = delimiterByte
        self.quoteByte = quoteByte
    }

    /// Parses an un-copied raw byte buffer into a grid of field offsets.
    /// - Parameters:
    ///   - buffer: Underlying byte buffer or contiguous memory storage.
    /// - Returns: The computed [[CSVFieldOffset]] result instance.
    public func parse(buffer: UnsafeBufferPointer<UInt8>) -> [[CSVFieldOffset]] {
        let index = parseIndex(buffer: buffer)
        return (0..<index.count).map { Array(index.row(at: $0)) }
    }

    internal func parseIndex(
        buffer: UnsafeBufferPointer<UInt8>, minimumParallelBytes: Int = 1024 * 1024
    ) -> CSVRecordIndex {
        if buffer.count >= minimumParallelBytes,
           let records = parseParallelIndex(buffer: buffer) {
            return records
        }
        return scanIndex(buffer: buffer, range: 0..<buffer.count)
    }

    // The fast path accepts only unquoted records. A quote in any chunk discards
    // all chunk results and runs the full DFA, including embedded-newline handling.
    internal func parseParallelIndex(buffer: UnsafeBufferPointer<UInt8>) -> CSVRecordIndex? {
        let workers = min(8, ProcessInfo.processInfo.activeProcessorCount)
        guard workers > 1, buffer.count > workers, delimiterByte != lfByte, quoteByte != lfByte else { return nil }
        var boundaries = [0]
        for worker in 1..<workers {
            var position = buffer.count / workers * worker
            let limit = worker == workers - 1 ? buffer.count : buffer.count / workers * (worker + 1)
            // A later window finds any newline beyond this one, without rescanning its bytes.
            while position < limit && buffer[position - 1] != lfByte { position += 1 }
            if position < buffer.count && buffer[position - 1] == lfByte && position > boundaries.last! {
                boundaries.append(position)
            }
        }
        boundaries.append(buffer.count)
        let ranges = zip(boundaries, boundaries.dropFirst()).map { $0..<$1 }
        guard ranges.count > 1 else { return nil }
        let input = CSVScanBuffer(pointer: buffer)
        var counts = [CSVChunkCount?](repeating: nil, count: ranges.count)
        counts.withUnsafeMutableBufferPointer { output in
            let slots = CSVCountSlots(pointer: output.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: ranges.count) { chunk in
                slots.pointer[chunk] = countUnquoted(buffer: input.pointer, range: ranges[chunk])
            }
        }
        guard counts.allSatisfy({ $0 != nil }) else { return nil }
        let chunks = counts.map { $0! }
        var fieldBases = [0]
        var rowBases = [0]
        for chunk in chunks {
            fieldBases.append(fieldBases.last! + chunk.fields)
            rowBases.append(rowBases.last! + chunk.rows)
        }
        let starts = fieldBases
        let rows = rowBases
        var rowStarts = [Int](unsafeUninitializedCapacity: rows.last! + 1) { output, initialized in
            // Row starts are initialized alongside the fields below, before publication.
            output.initialize(repeating: 0)
            initialized = output.count
        }
        let fields = rowStarts.withUnsafeMutableBufferPointer { rowBuffer in
            let rowOutput = CSVRowSlots(pointer: rowBuffer.baseAddress!)
            return [PackedCSVField](unsafeUninitializedCapacity: starts.last!) { output, initialized in
                let destination = CSVFieldSlots(pointer: output.baseAddress!)
                DispatchQueue.concurrentPerform(iterations: chunks.count) { chunk in
                    fillUnquoted(
                        buffer: input.pointer, range: ranges[chunk].lowerBound..<chunks[chunk].end,
                        count: chunks[chunk], fields: destination.pointer, rows: rowOutput.pointer,
                        fieldBase: starts[chunk], rowBase: rows[chunk]
                    )
                }
                initialized = output.count
            }
        }
        rowStarts[rows.last!] = starts.last!
        return CSVRecordIndex(fields: fields, rowStarts: rowStarts)
    }

    private func countUnquoted(buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>) -> CSVChunkCount? {
        let bytes = buffer.baseAddress!
        var fields = 0
        var rows = 0
        var pendingFields = 0
        var fieldStart = range.lowerBound
        var recordEnd = range.lowerBound
        var scanner = CSVStructuralScanner(bytes: bytes, range: range, delimiter: delimiterByte, quote: quoteByte)
        while let index = scanner.next() {
            let byte = bytes[index]
            if byte == quoteByte { return nil }
            if byte == delimiterByte {
                pendingFields += 1
                fieldStart = index + 1
            } else if byte == lfByte {
                fields += pendingFields + 1
                rows += 1
                pendingFields = 0
                fieldStart = index + 1
                recordEnd = index + 1
            }
        }
        if fieldStart < range.upperBound {
            fields += pendingFields + 1
            rows += 1
            recordEnd = range.upperBound
        }
        return CSVChunkCount(fields: fields, rows: rows, end: recordEnd)
    }

    private func fillUnquoted(
        buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>, count: CSVChunkCount,
        fields: UnsafeMutablePointer<PackedCSVField>, rows: UnsafeMutablePointer<Int>,
        fieldBase: Int, rowBase: Int
    ) {
        guard count.rows > 0 else { return }
        let bytes = buffer.baseAddress!
        var field = fieldBase
        var row = rowBase
        var fieldStart = range.lowerBound
        rows[row] = field
        var scanner = CSVStructuralScanner(bytes: bytes, range: range, delimiter: delimiterByte, quote: nil)
        while let index = scanner.next() {
            let byte = bytes[index]
            if byte == delimiterByte || byte == lfByte {
                let isNewline = byte == lfByte
                let end = isNewline && index > 0 && bytes[index - 1] == crByte ? index - 1 : index
                fields.advanced(by: field).initialize(to: PackedCSVField(
                    startOffset: fieldStart, length: max(0, end - fieldStart), escapedQuotesPresent: false
                ))
                field += 1
                fieldStart = index + 1
                if isNewline {
                    row += 1
                    if row < rowBase + count.rows { rows[row] = field }
                }
            }
        }
        if fieldStart < range.upperBound {
            fields.advanced(by: field).initialize(to: PackedCSVField(
                startOffset: fieldStart, length: range.upperBound - fieldStart, escapedQuotesPresent: false
            ))
            field += 1
            row += 1
        }
        assert(field == fieldBase + count.fields && row == rowBase + count.rows)
    }

    internal func scanIndex(
        buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>
    ) -> CSVRecordIndex {
        if CompactCSVField.canRepresent(endOffset: range.upperBound) {
            return scanIndex(buffer: buffer, range: range, fieldType: CompactCSVField.self)
        }
        return scanIndex(buffer: buffer, range: range, fieldType: PackedCSVField.self)
    }

    internal func scanIndex<Field: CSVIndexField>(
        buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>, fieldType: Field.Type
    ) -> CSVRecordIndex {
        var fields = CSVFieldBlocks<Field>(byteCount: range.count)
        var rowStarts = [0]
        rowStarts.reserveCapacity(min(range.count / 16, 100_000) + 1)

        let count = range.upperBound
        var index = range.lowerBound
        var fieldStart = range.lowerBound
        var insideQuotes = false
        var escapedQuotesFound = false

        while index < count {
            let byte = buffer[index]

            if insideQuotes {
                if byte == quoteByte {
                    // Check for escaped double quote ("")
                    if index + 1 < count && buffer[index + 1] == quoteByte {
                        escapedQuotesFound = true
                        index += 2
                        continue
                    } else {
                        // End of quoted block
                        insideQuotes = false
                    }
                }
            } else {
                if byte == quoteByte {
                    insideQuotes = true
                } else if byte == delimiterByte {
                    let len = index - fieldStart
                    fields.append(Field(
                        startOffset: fieldStart,
                        length: max(0, len),
                        escapedQuotesPresent: escapedQuotesFound
                    ))
                    fieldStart = index + 1
                    escapedQuotesFound = false
                } else if byte == lfByte {
                    var endPosition = index
                    if endPosition > 0 && buffer[endPosition - 1] == crByte {
                        endPosition -= 1
                    }
                    let len = endPosition - fieldStart
                    fields.append(Field(
                        startOffset: fieldStart,
                        length: max(0, len),
                        escapedQuotesPresent: escapedQuotesFound
                    ))

                    rowStarts.append(fields.count)

                    fieldStart = index + 1
                    escapedQuotesFound = false
                }
            }
            index += 1
        }

        // Handle trailing line without newline
        if fieldStart < count {
            let len = count - fieldStart
            fields.append(Field(
                startOffset: fieldStart,
                length: max(0, len),
                escapedQuotesPresent: escapedQuotesFound
            ))
            rowStarts.append(fields.count)
        } else {
            // Keep the public parser's existing EOF behavior for an unfinished row.
            fields.truncate(to: rowStarts.last!)
        }

        return Field.recordIndex(blocks: fields, rowStarts: rowStarts)
    }
}

// Buffers are borrowed until concurrentPerform joins. Each worker owns a
// disjoint output range; the mapped input is immutable throughout the scan.
private struct CSVScanBuffer: @unchecked Sendable { let pointer: UnsafeBufferPointer<UInt8> }
private struct CSVChunkCount: Sendable { let fields: Int; let rows: Int; let end: Int }
private struct CSVCountSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<CSVChunkCount?> }
private struct CSVFieldSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<PackedCSVField> }
private struct CSVRowSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<Int> }

/// Visits structural bytes without branching over every ordinary character.
private struct CSVStructuralScanner {
    let bytes: UnsafePointer<UInt8>
    let end: Int
    let delimiter: UInt8
    let quote: UInt8?
    var cursor: Int
    var blockStart = 0
    var matches: UInt16 = 0

    init(bytes: UnsafePointer<UInt8>, range: Range<Int>, delimiter: UInt8, quote: UInt8?) {
        self.bytes = bytes
        self.end = range.upperBound
        self.cursor = range.lowerBound
        self.delimiter = delimiter
        self.quote = quote
    }

    @inline(__always)
    mutating func next() -> Int? {
        while matches == 0 {
            guard cursor < end else { return nil }
            blockStart = cursor
            if end - cursor >= 16 {
                // The complete load stays inside this chunk, even at an unaligned row boundary.
                let block = UnsafeRawPointer(bytes.advanced(by: cursor)).loadUnaligned(as: SIMD16<UInt8>.self)
                var structural = (block .== SIMD16(repeating: delimiter)) .| (block .== SIMD16(repeating: 10))
                if let quote { structural = structural .| (block .== SIMD16(repeating: quote)) }
                let weights = SIMD16<UInt8>(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)
                let selected = SIMD16<UInt8>.zero.replacing(with: weights, where: structural)
                matches = UInt16(selected.lowHalf.wrappedSum()) | (UInt16(selected.highHalf.wrappedSum()) << 8)
                cursor += 16
            } else {
                while cursor < end {
                    let byte = bytes[cursor]
                    if byte == delimiter || byte == 10 || byte == quote {
                        matches |= UInt16(1) << (cursor - blockStart)
                    }
                    cursor += 1
                }
            }
        }
        let position = matches.trailingZeroBitCount
        matches &= matches - 1
        return blockStart + position
    }
}
