import Foundation

// MARK: - Shared Binary Protobuf Serialization Utility

/// Lightweight binary Protocol Buffer (protobuf) wire-format encoder.
///
/// Encodes primitive fields (varint, 64-bit fixed, length-delimited string/bytes) following
/// the protobuf binary wire specification. Used internally by both ``ONNXExporter`` and
/// ``CoreMLExporter`` to produce binary `.onnx` and `.mlmodel` payloads without depending
/// on a generated protobuf runtime.
///
/// ### Wire Types
/// | Type | ID | Used for |
/// |------|----|---------|
/// | Varint | 0 | `int32`, `int64`, `uint32`, `uint64`, `bool`, `enum` |
/// | 64-bit | 1 | `double`, `fixed64` |
/// | Length-delimited | 2 | `string`, `bytes`, embedded messages, packed repeated |
///
/// - Note: All methods mutate the writer in place. Call ``data`` to retrieve the
///   accumulated encoded bytes.
internal struct ProtobufWriter {

    // MARK: - State

    /// The accumulated encoded bytes.
    private(set) var data = Data()

    // MARK: - Primitive Encoding

    /// Encodes an unsigned 64-bit integer as a base-128 varint.
    mutating func writeVarint(_ value: UInt64) {
        var v = value
        while v >= 0x80 {
            data.append(UInt8((v & 0x7F) | 0x80))
            v >>= 7
        }
        data.append(UInt8(v & 0x7F))
    }

    /// Writes a protobuf field tag (field number + wire type).
    mutating func writeTag(fieldNumber: Int, wireType: Int) {
        writeVarint(UInt64((fieldNumber << 3) | wireType))
    }

    // MARK: - Field Writers

    /// Encodes a varint field (wire type 0).
    mutating func writeVarintField(fieldNumber: Int, value: UInt64) {
        writeTag(fieldNumber: fieldNumber, wireType: 0)
        writeVarint(value)
    }

    /// Encodes a 64-bit fixed-width `Double` field (wire type 1).
    mutating func writeDoubleField(fieldNumber: Int, value: Double) {
        writeTag(fieldNumber: fieldNumber, wireType: 1)
        var bitPattern = value.bitPattern
        withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
    }

    /// Encodes a length-delimited UTF-8 string field (wire type 2).
    mutating func writeStringField(fieldNumber: Int, value: String) {
        let utf8 = Data(value.utf8)
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(utf8.count))
        data.append(utf8)
    }

    /// Encodes a length-delimited bytes / embedded-message field (wire type 2).
    mutating func writeBytesField(fieldNumber: Int, bytes: Data) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(bytes.count))
        data.append(bytes)
    }

    /// Encodes a packed repeated `double` field (wire type 2).
    ///
    /// Writes all `Double` values consecutively as 8-byte little-endian fixed-width fields
    /// inside a single length-delimited block, as required by protobuf packed encoding.
    ///
    /// - Parameters:
    ///   - fieldNumber: The protobuf field number for the repeated field.
    ///   - values: The array of `Double` values to encode.
    mutating func writePackedDoublesField(fieldNumber: Int, values: [Double]) {
        let byteCount = values.count * 8
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(byteCount))
        for v in values {
            var bitPattern = v.bitPattern
            withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
        }
    }

    /// Encodes a 32-bit fixed-width `Float` field (wire type 5).
    mutating func writeFloatField(fieldNumber: Int, value: Float) {
        writeTag(fieldNumber: fieldNumber, wireType: 5)
        var bitPattern = value.bitPattern
        withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
    }

    /// Encodes a packed repeated `Float` field (wire type 2).
    mutating func writePackedFloatsField(fieldNumber: Int, values: [Float]) {
        let byteCount = values.count * 4
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeVarint(UInt64(byteCount))
        for v in values {
            var bitPattern = v.bitPattern
            withUnsafeBytes(of: &bitPattern) { data.append(contentsOf: $0) }
        }
    }

    /// Encodes a packed repeated `UInt64` (varint) field (wire type 2).
    ///
    /// - Parameters:
    ///   - fieldNumber: The protobuf field number for the repeated field.
    ///   - values: The array of `UInt64` values to encode as varints.
    mutating func writePackedVarintsField(fieldNumber: Int, values: [UInt64]) {
        var inner = ProtobufWriter()
        for v in values { inner.writeVarint(v) }
        writeBytesField(fieldNumber: fieldNumber, bytes: inner.data)
    }
}
