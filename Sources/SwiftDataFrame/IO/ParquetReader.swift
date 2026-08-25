import Foundation

/// A high-performance, pure-Swift zero-dependency reader for Apache Parquet (`.parquet`) columnar files.
///
/// `ParquetReader` reads Parquet metadata footers using Thrift Compact Protocol,
/// parses data pages with optional Snappy decompression, and constructs strongly-typed `DataFrame` instances.
public enum ParquetReader: Sendable {

    /// Reads a Parquet file from a local URL into a `DataFrame`.
    ///
    /// - Parameter url: The file URL of the `.parquet` file.
    /// - Returns: A `DataFrame` containing the columnar data.
    /// - Throws: `SwiftMLError.fileNotFound`, `SwiftMLError.parseError`, or `SwiftMLError.ioError`.
    public static func read(url: URL) async throws -> DataFrame {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SwiftMLError.fileNotFound(url)
        }

        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 12 else {
            throw SwiftMLError.parseError(line: 0, description: "Parquet file too small (< 12 bytes)")
        }

        // 1. Verify magic bytes 'PAR1' at start and end
        let magicStart = data.subdata(in: 0..<4)
        let magicEnd = data.subdata(in: (data.count - 4)..<data.count)
        let par1 = Data([0x50, 0x41, 0x52, 0x31]) // "PAR1"

        guard magicStart == par1 && magicEnd == par1 else {
            throw SwiftMLError.parseError(line: 0, description: "Invalid Parquet magic bytes; expected 'PAR1'")
        }

        // 2. Read 4-byte little-endian footer length from end
        let n = data.count
        let footerLength = UInt32(data[n - 8]) |
                           (UInt32(data[n - 7]) << 8) |
                           (UInt32(data[n - 6]) << 16) |
                           (UInt32(data[n - 5]) << 24)

        let footerStart = n - 8 - Int(footerLength)
        guard footerStart >= 4 else {
            throw SwiftMLError.parseError(line: 0, description: "Invalid Parquet footer length: \(footerLength)")
        }

        let footerData = data.subdata(in: footerStart..<(n - 8))

        // 3. Parse FileMetaData Thrift struct
        let reader = ThriftCompact.Reader(data: footerData)
        let meta = try parseFileMetaData(reader: reader)

        guard meta.numRows > 0, !meta.schema.isEmpty else {
            return DataFrame.empty
        }

        // Schema elements: element 0 is root group, subsequent elements 1..<count are columns
        let leafSchema = Array(meta.schema.dropFirst())
        var columns: [any AnyColumn] = []

        for (colIdx, schemaElem) in leafSchema.enumerated() {
            var colValues: [Any?] = []
            colValues.reserveCapacity(Int(meta.numRows))

            for rowGroup in meta.rowGroups {
                guard colIdx < rowGroup.columns.count else { continue }
                let colChunk = rowGroup.columns[colIdx]
                let pageData = data.subdata(in: Int(colChunk.dataPageOffset)..<(Int(colChunk.dataPageOffset) + Int(colChunk.totalCompressedSize)))

                let chunkValues = try decodeColumnChunk(
                    pageData: pageData,
                    schema: schemaElem,
                    codec: colChunk.codec,
                    numValues: Int(colChunk.numValues)
                )
                colValues.append(contentsOf: chunkValues)
            }

            let dtype = schemaElem.columnDType
            columns.append(makeColumn(name: schemaElem.name, dtype: dtype, rawValues: colValues))
        }

        return try DataFrame(columns: columns)
    }

    // MARK: – Internal Thrift Metadata Models

    struct SchemaElem {
        let name: String
        let type: Int32?
        let repetitionType: Int32?
        let numChildren: Int32?

        var columnDType: ColumnDType {
            switch type {
            case 1: // INT32
                return .int32
            case 2: // INT64
                return .int64
            case 4: // FLOAT
                return .float32
            case 5: // DOUBLE
                return .float64
            case 0: // BOOLEAN
                return .boolean
            case 6: // BYTE_ARRAY
                return .utf8
            default:
                return .utf8
            }
        }
    }

    struct ColChunkInfo {
        let codec: Int32
        let numValues: Int64
        let totalUncompressedSize: Int64
        let totalCompressedSize: Int64
        let dataPageOffset: Int64
    }

    struct RowGroupInfo {
        let columns: [ColChunkInfo]
        let numRows: Int64
    }

    struct FileMetaDataInfo {
        let version: Int32
        let schema: [SchemaElem]
        let numRows: Int64
        let rowGroups: [RowGroupInfo]
    }

    // MARK: – Thrift Parsers

    private static func parseFileMetaData(reader: ThriftCompact.Reader) throws -> FileMetaDataInfo {
        reader.pushStruct()
        var version: Int32 = 1
        var schema: [SchemaElem] = []
        var numRows: Int64 = 0
        var rowGroups: [RowGroupInfo] = []

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 1: // version (i32)
                version = try reader.readZigZagI32()
            case 2: // schema (list<SchemaElement>)
                let listHead = try reader.readByte()
                var size = Int((listHead >> 4) & 0x0F)
                if size == 0x0F {
                    size = Int(try reader.readVarint())
                }
                for _ in 0..<size {
                    schema.append(try parseSchemaElement(reader: reader))
                }
            case 3: // num_rows (i64)
                numRows = try reader.readZigZagI64()
            case 4: // row_groups (list<RowGroup>)
                let listHead = try reader.readByte()
                var size = Int((listHead >> 4) & 0x0F)
                if size == 0x0F {
                    size = Int(try reader.readVarint())
                }
                for _ in 0..<size {
                    rowGroups.append(try parseRowGroup(reader: reader))
                }
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()

        return FileMetaDataInfo(version: version, schema: schema, numRows: numRows, rowGroups: rowGroups)
    }

    private static func parseSchemaElement(reader: ThriftCompact.Reader) throws -> SchemaElem {
        reader.pushStruct()
        var type: Int32?
        var rep: Int32?
        var name = ""
        var numChildren: Int32?

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 1: // type
                type = try reader.readZigZagI32()
            case 3: // repetition_type
                rep = try reader.readZigZagI32()
            case 4: // name
                name = try reader.readString()
            case 5: // num_children
                numChildren = try reader.readZigZagI32()
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()
        return SchemaElem(name: name, type: type, repetitionType: rep, numChildren: numChildren)
    }

    private static func parseRowGroup(reader: ThriftCompact.Reader) throws -> RowGroupInfo {
        reader.pushStruct()
        var columns: [ColChunkInfo] = []
        var numRows: Int64 = 0

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 1: // columns (list<ColumnChunk>)
                let listHead = try reader.readByte()
                var size = Int((listHead >> 4) & 0x0F)
                if size == 0x0F {
                    size = Int(try reader.readVarint())
                }
                for _ in 0..<size {
                    columns.append(try parseColumnChunk(reader: reader))
                }
            case 3: // num_rows
                numRows = try reader.readZigZagI64()
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()
        return RowGroupInfo(columns: columns, numRows: numRows)
    }

    private static func parseColumnChunk(reader: ThriftCompact.Reader) throws -> ColChunkInfo {
        reader.pushStruct()
        var chunk = ColChunkInfo(codec: 0, numValues: 0, totalUncompressedSize: 0, totalCompressedSize: 0, dataPageOffset: 0)

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 3: // meta_data (ColumnMetaData struct)
                chunk = try parseColumnMetaData(reader: reader)
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()
        return chunk
    }

    private static func parseColumnMetaData(reader: ThriftCompact.Reader) throws -> ColChunkInfo {
        reader.pushStruct()
        var codec: Int32 = 0
        var numValues: Int64 = 0
        var uncompressedSize: Int64 = 0
        var compressedSize: Int64 = 0
        var dataPageOffset: Int64 = 0

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 4: // codec
                codec = try reader.readZigZagI32()
            case 5: // num_values
                numValues = try reader.readZigZagI64()
            case 6: // total_uncompressed_size
                uncompressedSize = try reader.readZigZagI64()
            case 7: // total_compressed_size
                compressedSize = try reader.readZigZagI64()
            case 8: // data_page_offset
                dataPageOffset = try reader.readZigZagI64()
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()
        return ColChunkInfo(
            codec: codec,
            numValues: numValues,
            totalUncompressedSize: uncompressedSize,
            totalCompressedSize: compressedSize,
            dataPageOffset: dataPageOffset
        )
    }

    // MARK: – Page Decoding

    private static func decodeColumnChunk(
        pageData: Data,
        schema: SchemaElem,
        codec: Int32,
        numValues: Int
    ) throws -> [Any?] {
        guard !pageData.isEmpty else { return [] }

        // Decompress if codec == 1 (Snappy)
        let rawData: Data
        if codec == 1 {
            rawData = try SnappyDecompressor.decompress(data: pageData)
        } else {
            rawData = pageData
        }

        // Decode Page Header
        let pageReader = ThriftCompact.Reader(data: rawData)
        pageReader.pushStruct()
        while true {
            let field = try pageReader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 1: // type
                _ = try pageReader.readZigZagI32()
            case 2: // uncompressed_page_size
                _ = try pageReader.readZigZagI32()
            case 3: // compressed_page_size
                _ = try pageReader.readZigZagI32()
            case 5: // data_page_header
                pageReader.pushStruct()
                while true {
                    let subField = try pageReader.readFieldBegin()
                    if subField.type == .stop { break }
                    try pageReader.skip(type: subField.type)
                }
                pageReader.popStruct()
            default:
                try pageReader.skip(type: field.type)
            }
        }
        pageReader.popStruct()

        let payloadOffset = pageReader.offset
        let payload = rawData.subdata(in: payloadOffset..<rawData.count)

        return try decodeValues(payload: payload, schema: schema, count: numValues)
    }

    private static func decodeValues(payload: Data, schema: SchemaElem, count: Int) throws -> [Any?] {
        guard !payload.isEmpty && count > 0 else { return [] }

        return try payload.withUnsafeBytes { rawBuf in
            guard let ptr = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return []
            }
            let byteCount = rawBuf.count
            var offset = 0

            // 1. Read Definition Levels (if optional)
            var defLevels = [UInt8]()
            defLevels.reserveCapacity(count)

            if schema.repetitionType == 1 { // OPTIONAL
                guard offset + 4 <= byteCount else {
                    throw SwiftMLError.parseError(line: 0, description: "Corrupted def levels length header")
                }
                let defLen = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8) | (Int(ptr[offset + 2]) << 16) | (Int(ptr[offset + 3]) << 24)
                offset += 4

                guard offset + defLen <= byteCount else {
                    throw SwiftMLError.parseError(line: 0, description: "Def levels exceed payload")
                }
                let defEnd = offset + defLen

                // RLE/bit-packed decoding of 1-bit definition levels
                while offset < defEnd && defLevels.count < count {
                    let header = ptr[offset]
                    offset += 1
                    if (header & 1) == 0 {
                        // RLE run
                        let runCount = Int(header >> 1)
                        guard offset < defEnd else { break }
                        let val = ptr[offset]
                        offset += 1
                        for _ in 0..<runCount {
                            if defLevels.count < count {
                                defLevels.append(val)
                            }
                        }
                    } else {
                        // Bit-packed group
                        let numGroups = Int(header >> 1)
                        let numBytes = numGroups // 1-bit values: 1 byte per 8 values
                        for _ in 0..<numBytes {
                            guard offset < defEnd else { break }
                            let byteVal = ptr[offset]
                            offset += 1
                            for bit in 0..<8 {
                                if defLevels.count < count {
                                    defLevels.append((byteVal >> bit) & 1)
                                }
                            }
                        }
                    }
                }
                offset = defEnd
            } else {
                // All values are required (non-null)
                defLevels = [UInt8](repeating: 1, count: count)
            }

            // 2. Decode Plain Values
            var results: [Any?] = []
            results.reserveCapacity(count)

            for i in 0..<count {
                let isPresent = i < defLevels.count ? defLevels[i] == 1 : true
                if !isPresent {
                    results.append(nil)
                    continue
                }

                switch schema.type {
                case 1: // INT32
                    guard offset + 4 <= byteCount else { results.append(nil); continue }
                    let val = Int32(ptr[offset]) | (Int32(ptr[offset + 1]) << 8) | (Int32(ptr[offset + 2]) << 16) | (Int32(ptr[offset + 3]) << 24)
                    offset += 4
                    results.append(val)

                case 2: // INT64
                    guard offset + 8 <= byteCount else { results.append(nil); continue }
                    var raw: UInt64 = 0
                    for b in 0..<8 {
                        raw |= UInt64(ptr[offset + b]) << (b * 8)
                    }
                    offset += 8
                    results.append(Int64(bitPattern: raw))

                case 4: // FLOAT
                    guard offset + 4 <= byteCount else { results.append(nil); continue }
                    let raw = UInt32(ptr[offset]) | (UInt32(ptr[offset + 1]) << 8) | (UInt32(ptr[offset + 2]) << 16) | (UInt32(ptr[offset + 3]) << 24)
                    offset += 4
                    results.append(Float(bitPattern: raw))

                case 5: // DOUBLE
                    guard offset + 8 <= byteCount else { results.append(nil); continue }
                    var raw: UInt64 = 0
                    for b in 0..<8 {
                        raw |= UInt64(ptr[offset + b]) << (b * 8)
                    }
                    offset += 8
                    results.append(Double(bitPattern: raw))

                case 0: // BOOLEAN
                    guard offset < byteCount else { results.append(nil); continue }
                    let val = ptr[offset] != 0
                    offset += 1
                    results.append(val)

                case 6: // BYTE_ARRAY (string / binary)
                    guard offset + 4 <= byteCount else { results.append(nil); continue }
                    let strLen = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8) | (Int(ptr[offset + 2]) << 16) | (Int(ptr[offset + 3]) << 24)
                    offset += 4
                    guard offset + strLen <= byteCount else { results.append(nil); continue }
                    let strBytes = UnsafeBufferPointer(start: ptr + offset, count: strLen)
                    let str = String(decoding: strBytes, as: UTF8.self)
                    offset += strLen
                    results.append(str)

                default:
                    results.append(nil)
                }
            }

            return results
        }
    }
}
