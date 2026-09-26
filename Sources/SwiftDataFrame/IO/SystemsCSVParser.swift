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

internal struct CSVRecordIndex: Sendable {
    let fields: [CSVFieldOffset]
    let rowStarts: [Int]

    var count: Int { rowStarts.count - 1 }
    var isEmpty: Bool { count == 0 }

    func row(at index: Int) -> ArraySlice<CSVFieldOffset> {
        fields[rowStarts[index]..<rowStarts[index + 1]]
    }

    func field(row: Int, column: Int) -> CSVFieldOffset? {
        guard row < count else { return nil }
        let start = rowStarts[row]
        guard column < rowStarts[row + 1] - start else { return nil }
        return fields[start + column]
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
        buffer: UnsafeBufferPointer<UInt8>, minimumParallelBytes: Int = 4 * 1024 * 1024
    ) -> CSVRecordIndex {
        if buffer.count >= minimumParallelBytes,
           let records = parseParallelIndex(buffer: buffer) {
            return records
        }
        return scanIndex(buffer: buffer, range: 0..<buffer.count, rejectQuotes: false)!
    }

    // The fast path accepts only unquoted records. A quote in any chunk discards
    // all chunk results and runs the full DFA, including embedded-newline handling.
    internal func parseParallelIndex(buffer: UnsafeBufferPointer<UInt8>) -> CSVRecordIndex? {
        let workers = min(8, ProcessInfo.processInfo.activeProcessorCount)
        guard workers > 1, buffer.count > workers else { return nil }
        var boundaries = [0]
        for worker in 1..<workers {
            var position = buffer.count / workers * worker
            while position < buffer.count && buffer[position - 1] != lfByte { position += 1 }
            if position < buffer.count && position > boundaries.last! { boundaries.append(position) }
        }
        boundaries.append(buffer.count)
        let ranges = zip(boundaries, boundaries.dropFirst()).map { $0..<$1 }
        guard ranges.count > 1 else { return nil }
        let input = CSVScanBuffer(pointer: buffer)
        var chunks = [CSVRecordIndex?](repeating: nil, count: ranges.count)
        chunks.withUnsafeMutableBufferPointer { output in
            let slots = CSVChunkSlots(pointer: output.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: ranges.count) { chunk in
                slots.pointer[chunk] = scanIndex(buffer: input.pointer, range: ranges[chunk], rejectQuotes: true)
            }
        }
        guard chunks.allSatisfy({ $0 != nil }) else { return nil }
        let records = chunks.map { $0! }
        var fieldBases = [0]
        var rowBases = [0]
        for record in records {
            fieldBases.append(fieldBases.last! + record.fields.count)
            rowBases.append(rowBases.last! + record.count)
        }
        let starts = fieldBases
        let rows = rowBases
        let fields = [CSVFieldOffset](unsafeUninitializedCapacity: starts.last!) { output, initialized in
            let destination = CSVFieldSlots(pointer: output.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: records.count) { chunk in
                records[chunk].fields.withUnsafeBufferPointer { source in
                    if !source.isEmpty {
                        destination.pointer.advanced(by: starts[chunk]).initialize(from: source.baseAddress!, count: source.count)
                    }
                }
            }
            initialized = output.count
        }
        let rowStarts = [Int](unsafeUninitializedCapacity: rows.last! + 1) { output, initialized in
            let destination = CSVRowSlots(pointer: output.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: records.count) { chunk in
                for row in 0..<records[chunk].count {
                    destination.pointer.advanced(by: rows[chunk] + row).initialize(to: starts[chunk] + records[chunk].rowStarts[row])
                }
            }
            output.baseAddress!.advanced(by: rows.last!).initialize(to: starts.last!)
            initialized = output.count
        }
        return CSVRecordIndex(fields: fields, rowStarts: rowStarts)
    }

    private func scanIndex(
        buffer: UnsafeBufferPointer<UInt8>, range: Range<Int>, rejectQuotes: Bool
    ) -> CSVRecordIndex? {
        var fields = [CSVFieldOffset]()
        fields.reserveCapacity(min(range.count / 8, 100_000))
        var rowStarts = [0]
        rowStarts.reserveCapacity(min(range.count / 16, 100_000) + 1)

        let count = range.upperBound
        var index = range.lowerBound
        var fieldStart = range.lowerBound
        var insideQuotes = false
        var escapedQuotesFound = false

        while index < count {
            let byte = buffer[index]
            if rejectQuotes && byte == quoteByte { return nil }

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
                    fields.append(CSVFieldOffset(
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
                    fields.append(CSVFieldOffset(
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
            fields.append(CSVFieldOffset(
                startOffset: fieldStart,
                length: max(0, len),
                escapedQuotesPresent: escapedQuotesFound
            ))
            rowStarts.append(fields.count)
        } else {
            // Keep the public parser's existing EOF behavior for an unfinished row.
            fields.removeSubrange(rowStarts.last!..<fields.count)
        }

        return CSVRecordIndex(fields: fields, rowStarts: rowStarts)
    }
}

// Buffers are borrowed until concurrentPerform joins. Each worker owns a
// disjoint output range; the mapped input is immutable throughout the scan.
private struct CSVScanBuffer: @unchecked Sendable { let pointer: UnsafeBufferPointer<UInt8> }
private struct CSVChunkSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<CSVRecordIndex?> }
private struct CSVFieldSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<CSVFieldOffset> }
private struct CSVRowSlots: @unchecked Sendable { let pointer: UnsafeMutablePointer<Int> }
