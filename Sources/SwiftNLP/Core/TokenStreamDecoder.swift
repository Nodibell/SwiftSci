import Foundation

/// Decodes generated tokens incrementally. Create one decoder per output stream.
public struct TokenStreamDecoder: Sendable {
    private enum Mode: Sendable {
        case text(@Sendable (Int) -> String)
        case bytes(@Sendable (Int) -> [UInt8])
    }
    private let mode: Mode
    private var pending: [UInt8] = []

    /// Preserves per-token text decoding for tokenizers without a byte representation.
    public init(decodeToken: @escaping @Sendable (Int) -> String) {
        mode = .text(decodeToken)
    }

    /// Creates a decoder for UTF-8 bytes that may span token boundaries.
    public init(decodeBytes: @escaping @Sendable (Int) -> [UInt8]) {
        mode = .bytes(decodeBytes)
    }

    /// Returns available text, or `nil` while waiting for a complete UTF-8 scalar.
    /// An empty token returns `""`; pending bytes remain available to ``finish()``.
    public mutating func append(_ token: Int) -> String? {
        switch mode {
        case .text(let decode): return decode(token)
        case .bytes(let decode):
            let tokenBytes = decode(token)
            guard !tokenBytes.isEmpty else { return "" }
            var bytes = pending
            bytes.append(contentsOf: tokenBytes)
            let end = Self.completePrefixLength(bytes)
            pending = Array(bytes[end...])
            if end == 0 && !pending.isEmpty { return nil }
            return String(decoding: bytes[..<end], as: UTF8.self)
        }
    }

    /// Flushes any incomplete final UTF-8 sequence using standard replacement decoding.
    public mutating func finish() -> String {
        let text = String(decoding: pending, as: UTF8.self)
        pending.removeAll(keepingCapacity: true)
        return text
    }

    private static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        // Only a valid, incomplete scalar can change when more bytes arrive.
        guard let start = bytes.indices.reversed().prefix(4).first(where: {
            bytes[$0] & 0xc0 != 0x80
        }) else { return bytes.count }
        let lead = bytes[start]
        let width: Int
        switch lead {
        case 0xc2...0xdf: width = 2
        case 0xe0...0xef: width = 3
        case 0xf0...0xf4: width = 4
        default: return bytes.count
        }
        guard bytes.count - start < width else { return bytes.count }
        for index in (start + 1)..<bytes.count {
            let byte = bytes[index]
            guard byte >= 0x80 && byte <= 0xbf else { return bytes.count }
            if index == start + 1 {
                if lead == 0xe0 && byte < 0xa0 { return bytes.count }
                if lead == 0xed && byte > 0x9f { return bytes.count }
                if lead == 0xf0 && byte < 0x90 { return bytes.count }
                if lead == 0xf4 && byte > 0x8f { return bytes.count }
            }
        }
        return start
    }
}
