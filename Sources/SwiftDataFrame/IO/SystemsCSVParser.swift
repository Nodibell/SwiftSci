import Foundation

/// Coordinates of a CSV field within an un-copied byte buffer.
public struct CSVFieldOffset: Sendable {
    public let startOffset: Int
    public let length: Int
    public let escapedQuotesPresent: Bool
    
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

    public init(delimiterByte: UInt8 = 44, quoteByte: UInt8 = 34) {
        self.delimiterByte = delimiterByte
        self.quoteByte = quoteByte
    }

    /// Parses an un-copied raw byte buffer into a grid of field offsets.
    public func parse(buffer: UnsafeBufferPointer<UInt8>) -> [[CSVFieldOffset]] {
        var records = [[CSVFieldOffset]]()
        records.reserveCapacity(100_000)

        var currentRecord = [CSVFieldOffset]()
        currentRecord.reserveCapacity(16)

        let count = buffer.count
        var index = 0
        var fieldStart = 0
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
                    currentRecord.append(CSVFieldOffset(
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
                    currentRecord.append(CSVFieldOffset(
                        startOffset: fieldStart,
                        length: max(0, len),
                        escapedQuotesPresent: escapedQuotesFound
                    ))

                    records.append(currentRecord)
                    currentRecord = []
                    currentRecord.reserveCapacity(16)

                    fieldStart = index + 1
                    escapedQuotesFound = false
                }
            }
            index += 1
        }

        // Handle trailing line without newline
        if fieldStart < count {
            let len = count - fieldStart
            currentRecord.append(CSVFieldOffset(
                startOffset: fieldStart,
                length: max(0, len),
                escapedQuotesPresent: escapedQuotesFound
            ))
            records.append(currentRecord)
        }

        return records
    }
}
