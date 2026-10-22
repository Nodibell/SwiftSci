import Foundation

/// Rejects impossible ASCII starts before retaining the original String equality check.
internal struct CSVASCIIPrefixStringDecoder: CSVValueDecoder, Sendable {
    private let nullValues: Set<String>
    private let lowStarts: UInt64
    private let highStarts: UInt64

    init?(nullValues: Set<String>, selectedRowCount: Int) {
        guard nullValues.count > 2, nullValues.count <= selectedRowCount else { return nil }
        // Bound token visits plus inspected bytes by the work selected for this attempt.
        var remaining = selectedRowCount - nullValues.count
        var low: UInt64 = 0
        var high: UInt64 = 0
        for token in nullValues {
            var first = true
            for byte in token.utf8 {
                guard remaining > 0, byte < 128 else { return nil }
                remaining -= 1
                if first {
                    if byte < 64 { low |= UInt64(1) << byte }
                    else { high |= UInt64(1) << (byte - 64) }
                    first = false
                }
            }
        }
        guard low != UInt64.max || high != UInt64.max else { return nil }
        self.nullValues = nullValues
        lowStarts = low
        highStarts = high
    }

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<String> {
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        if let first = text.utf8.first, first < 128 {
            let mask = first < 64 ? lowStarts : highStarts
            if mask & (UInt64(1) << (first & 63)) == 0 { return .value(text) }
        }
        // Empty and non-ASCII-leading strings retain canonical String equality.
        return nullValues.contains(text) ? .missing : .value(text)
    }
}
