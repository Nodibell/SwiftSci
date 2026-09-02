import Foundation
#if canImport(Compression)
import Compression
#endif

/// A high-performance, zero-dependency pure-Swift reader for NumPy NPZ archives (`.npz`).
///
/// NPZ files are zip archives containing named `.npy` tensor array files.
/// `NPZReader` parses the zip container format, extracts uncompressed and deflate-compressed
/// tensor arrays, and provides direct conversions into `DataFrame` tables.
public enum NPZReader: Sendable {

    /// Reads an NPZ archive from a local URL, returning a dictionary of named `NPYArray` instances.
    ///
    /// - Parameter url: The file URL of the `.npz` archive.
    /// - Returns: A dictionary mapping array names (without `.npy` suffix) to `NPYArray` objects.
    public static func read(url: URL) throws -> [String: NPYArray] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SwiftMLError.fileNotFound(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try read(data: data)
    }

    /// Reads an NPZ archive from raw data bytes.
    public static func read(data: Data) throws -> [String: NPYArray] {
        var arrays: [String: NPYArray] = [:]
        let byteCount = data.count
        var offset = 0

        while offset + 30 <= byteCount {
            // Check for Local File Header signature: PK\x03\x04 (0x04034B50)
            let sig0 = data[offset]
            let sig1 = data[offset + 1]
            let sig2 = data[offset + 2]
            let sig3 = data[offset + 3]

            if sig0 == 0x50 && sig1 == 0x4B && sig2 == 0x03 && sig3 == 0x04 {
                let compMethod = UInt16(data[offset + 8]) | (UInt16(data[offset + 9]) << 8)
                var compSize = Int(UInt32(data[offset + 18]) | (UInt32(data[offset + 19]) << 8) | (UInt32(data[offset + 20]) << 16) | (UInt32(data[offset + 21]) << 24))
                var uncompSize = Int(UInt32(data[offset + 22]) | (UInt32(data[offset + 23]) << 8) | (UInt32(data[offset + 24]) << 16) | (UInt32(data[offset + 25]) << 24))
                let fileNameLen = Int(UInt16(data[offset + 26]) | (UInt16(data[offset + 27]) << 8))
                let extraLen = Int(UInt16(data[offset + 28]) | (UInt16(data[offset + 29]) << 8))

                offset += 30
                guard offset + fileNameLen <= byteCount else { break }

                let nameBytes = data.subdata(in: offset..<(offset + fileNameLen))
                let rawName = String(decoding: nameBytes, as: UTF8.self)
                let cleanName = rawName.replacingOccurrences(of: ".npy", with: "")
                offset += fileNameLen

                // Parse ZIP64 extra field if sizes are 0xFFFFFFFF
                if extraLen >= 20 && offset + extraLen <= byteCount {
                    let extraData = data.subdata(in: offset..<(offset + extraLen))
                    var extraIdx = 0
                    while extraIdx + 4 <= extraData.count {
                        let tag = UInt16(extraData[extraIdx]) | (UInt16(extraData[extraIdx + 1]) << 8)
                        let blockSize = Int(UInt16(extraData[extraIdx + 2]) | (UInt16(extraData[extraIdx + 3]) << 8))
                        extraIdx += 4
                        if tag == 0x0001 && extraIdx + blockSize <= extraData.count {
                            // ZIP64 extended information
                            var pos = extraIdx
                            if uncompSize == -1 || UInt32(uncompSize) == 0xFFFFFFFF {
                                var val: UInt64 = 0
                                for b in 0..<8 { val |= UInt64(extraData[pos + b]) << (b * 8) }
                                uncompSize = Int(val)
                                pos += 8
                            }
                            if compSize == -1 || UInt32(compSize) == 0xFFFFFFFF {
                                var val: UInt64 = 0
                                for b in 0..<8 { val |= UInt64(extraData[pos + b]) << (b * 8) }
                                compSize = Int(val)
                                pos += 8
                            }
                            break
                        }
                        extraIdx += blockSize
                    }
                }
                offset += extraLen

                guard offset + compSize <= byteCount else { break }
                let entryPayload = data.subdata(in: offset..<(offset + compSize))
                offset += compSize

                // Decompress payload
                let npyBytes: Data
                if compMethod == 0 {
                    // Stored (uncompressed)
                    npyBytes = entryPayload
                } else if compMethod == 8 {
                    // Deflate compressed
                    npyBytes = try decompressDeflate(data: entryPayload, expectedSize: uncompSize)
                } else {
                    continue
                }

                if let arr = try? NPYReader.read(data: npyBytes) {
                    arrays[cleanName] = arr
                }
            } else if sig0 == 0x50 && sig1 == 0x4B && (sig2 == 0x01 || sig2 == 0x05) {
                // Central Directory or End of Central Directory reached
                break
            } else {
                offset += 1
            }
        }

        return arrays
    }

    /// Reads an NPZ archive directly into a `DataFrame`.
    ///
    /// If an array named `"x_train"` or `"features"` is present along with `"y_train"` or `"label"`,
    /// they are combined into features and target columns automatically.
    public static func readDataFrame(url: URL, preferredArray: String? = nil) throws -> DataFrame {
        let arrays = try read(url: url)
        guard !arrays.isEmpty else { return DataFrame.empty }

        // 1. If explicit array requested
        if let name = preferredArray, let arr = arrays[name] {
            return try arr.toDataFrame(columnPrefix: name)
        }

        // 2. Check for typical feature + label pairs (e.g. mnist: x_train + y_train)
        if let xArr = arrays["x_train"] ?? arrays["features"] ?? arrays["X"] {
            var df = try xArr.toDataFrame(columnPrefix: "pixel")
            if let yArr = arrays["y_train"] ?? arrays["labels"] ?? arrays["label"] ?? arrays["y"] {
                let labels = yArr.toDoubles()
                if labels.count == df.rowCount {
                    let lblCol = TypedColumn(name: "label", values: labels)
                    var cols = df.columns
                    cols.append(lblCol)
                    df = try DataFrame(columns: cols)
                }
            }
            return df
        }

        // 3. Fallback to first available array
        if let first = arrays.first {
            return try first.value.toDataFrame(columnPrefix: first.key)
        }

        return DataFrame.empty
    }

    private static func decompressDeflate(data: Data, expectedSize: Int) throws -> Data {
        #if canImport(Compression)
        var destination = [UInt8](repeating: 0, count: expectedSize)
        let decompressedSize = data.withUnsafeBytes { srcBuf in
            destination.withUnsafeMutableBytes { dstBuf in
                guard let src = srcBuf.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let dst = dstBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
                return compression_decode_buffer(
                    dst,
                    expectedSize,
                    src,
                    data.count,
                    nil,
                    COMPRESSION_ZLIB
                )
            }
        }
        guard decompressedSize > 0 else {
            throw SwiftMLError.parseError(line: 0, description: "Deflate decompression failed for NPZ entry")
        }
        return Data(destination.prefix(decompressedSize))
        #else
        throw SwiftMLError.parseError(line: 0, description: "Compression framework unavailable for Deflate NPZ")
        #endif
    }
}

// MARK: - DataFrame Initializers

extension DataFrame {
    /// Reads a NumPy binary array (`.npy`) into a `DataFrame`.
    public init(npy url: URL) throws {
        let arr = try NPYReader.read(url: url)
        self = try arr.toDataFrame()
    }

    /// Reads a NumPy tensor archive (`.npz`) into a `DataFrame`.
    public init(npz url: URL, arrayName: String? = nil) throws {
        self = try NPZReader.readDataFrame(url: url, preferredArray: arrayName)
    }
}
