import Foundation

/// A high-performance, pure-Swift zero-dependency reader for Apache Parquet (`.parquet`) columnar files.
///
/// `ParquetReader` reads Parquet metadata footers using Thrift Compact Protocol,
/// parses data and dictionary pages with optional Snappy decompression,
/// unpacks `RLE_DICTIONARY` and `PLAIN` encodings, and constructs strongly-typed `DataFrame` instances.
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

        // Leaf schema elements (primitive columns with type != nil)
        let leafSchema = meta.schema.filter { $0.type != nil }
        var columns: [any AnyColumn] = []

        for (colIdx, schemaElem) in leafSchema.enumerated() {
            var colValues: [Any?] = []
            colValues.reserveCapacity(Int(meta.numRows))

            for rowGroup in meta.rowGroups {
                guard colIdx < rowGroup.columns.count else { continue }
                let colChunk = rowGroup.columns[colIdx]

                let chunkValues = try decodeColumnChunk(
                    fileData: data,
                    colChunk: colChunk,
                    schema: schemaElem,
                    numRows: Int(rowGroup.numRows)
                )
                colValues.append(contentsOf: chunkValues)
            }

            let colName = resolveColumnName(schemaElem: schemaElem, colChunk: meta.rowGroups.first?.columns[safe: colIdx])
            let dtype = schemaElem.isNestedList ? .utf8 : schemaElem.columnDType
            columns.append(makeColumn(name: colName, dtype: dtype, rawValues: colValues))
        }

        return try DataFrame(columns: columns)
    }

    private static func resolveColumnName(schemaElem: SchemaElem, colChunk: ColChunkInfo?) -> String {
        if let path = colChunk?.pathInSchema, !path.isEmpty {
            return path.first ?? schemaElem.name
        }
        return schemaElem.name
    }

    // MARK: – Internal Thrift Metadata Models

    /// Representation of a Parquet schema element.
    public struct SchemaElem: Sendable {
        /// The element name.
        public let name: String
        /// Primitive type tag.
        public let type: Int32?
        /// Repetition type: 0: REQUIRED, 1: OPTIONAL, 2: REPEATED.
        public let repetitionType: Int32?
        /// Number of child elements if this is a group.
        public let numChildren: Int32?

        /// Whether this element represents a repeated or nested list item.
        public var isNestedList: Bool {
            return repetitionType == 2 || name == "item" || name == "element"
        }

        /// Inferred `ColumnDType`.
        public var columnDType: ColumnDType {
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

    /// Column chunk metadata extracted from row groups.
    public struct ColChunkInfo: Sendable {
        /// Compression codec tag: 0: UNCOMPRESSED, 1: SNAPPY, 2: GZIP.
        public let codec: Int32
        /// Number of values.
        public let numValues: Int64
        /// Uncompressed byte size.
        public let totalUncompressedSize: Int64
        /// Compressed byte size.
        public let totalCompressedSize: Int64
        /// Offset of the first data page.
        public let dataPageOffset: Int64
        /// Offset of the dictionary page, if present.
        public let dictionaryPageOffset: Int64?
        /// Path in schema.
        public let pathInSchema: [String]

        /// Creates a new column chunk info instance.
        public init(
            codec: Int32,
            numValues: Int64,
            totalUncompressedSize: Int64,
            totalCompressedSize: Int64,
            dataPageOffset: Int64,
            dictionaryPageOffset: Int64? = nil,
            pathInSchema: [String] = []
        ) {
            self.codec = codec
            self.numValues = numValues
            self.totalUncompressedSize = totalUncompressedSize
            self.totalCompressedSize = totalCompressedSize
            self.dataPageOffset = dataPageOffset
            self.dictionaryPageOffset = dictionaryPageOffset
            self.pathInSchema = pathInSchema
        }
    }

    /// Row group metadata.
    public struct RowGroupInfo: Sendable {
        /// Column chunks in this row group.
        public let columns: [ColChunkInfo]
        /// Total number of rows in this row group.
        public let numRows: Int64

        /// Creates a new row group info instance.
        public init(columns: [ColChunkInfo], numRows: Int64) {
            self.columns = columns
            self.numRows = numRows
        }
    }

    /// File metadata.
    public struct FileMetaDataInfo: Sendable {
        /// File format version.
        public let version: Int32
        /// Schema elements.
        public let schema: [SchemaElem]
        /// Total rows.
        public let numRows: Int64
        /// Row groups.
        public let rowGroups: [RowGroupInfo]

        /// Creates a new file metadata info instance.
        public init(version: Int32, schema: [SchemaElem], numRows: Int64, rowGroups: [RowGroupInfo]) {
            self.version = version
            self.schema = schema
            self.numRows = numRows
            self.rowGroups = rowGroups
        }
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
        var dictionaryPageOffset: Int64? = nil
        var pathInSchema: [String] = []

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 3: // path_in_schema (list<string>)
                let listHead = try reader.readByte()
                var size = Int((listHead >> 4) & 0x0F)
                if size == 0x0F {
                    size = Int(try reader.readVarint())
                }
                for _ in 0..<size {
                    pathInSchema.append(try reader.readString())
                }
            case 4: // codec
                codec = try reader.readZigZagI32()
            case 5: // num_values
                numValues = try reader.readZigZagI64()
            case 6: // total_uncompressed_size
                uncompressedSize = try reader.readZigZagI64()
            case 7: // total_compressed_size
                compressedSize = try reader.readZigZagI64()
            case 8, 9: // data_page_offset (field 9 in official Apache Parquet spec, field 8 in legacy)
                dataPageOffset = try reader.readZigZagI64()
            case 11: // dictionary_page_offset
                dictionaryPageOffset = try reader.readZigZagI64()
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
            dataPageOffset: dataPageOffset,
            dictionaryPageOffset: dictionaryPageOffset,
            pathInSchema: pathInSchema
        )
    }

    // MARK: – Page Decoding

    private struct PageHeaderInfo {
        var type: Int32 // 0: DATA_PAGE, 2: DICTIONARY_PAGE, 3: DATA_PAGE_V2
        var uncompressedPageSize: Int32
        var compressedPageSize: Int32
        var numValues: Int = 0
        var encoding: Int32 = 0
        var defLevelEncoding: Int32 = 0
        var repLevelEncoding: Int32 = 0
    }

    private static func parsePageHeader(reader: ThriftCompact.Reader) throws -> PageHeaderInfo {
        reader.pushStruct()
        var type: Int32 = 0
        var uncomp: Int32 = 0
        var comp: Int32 = 0
        var numValues: Int = 0
        var encoding: Int32 = 0
        var defLevelEncoding: Int32 = 0
        var repLevelEncoding: Int32 = 0

        while true {
            let field = try reader.readFieldBegin()
            if field.type == .stop { break }

            switch field.fieldId {
            case 1: // type
                type = try reader.readZigZagI32()
            case 2: // uncompressed_page_size
                uncomp = try reader.readZigZagI32()
            case 3: // compressed_page_size
                comp = try reader.readZigZagI32()
            case 5: // data_page_header
                reader.pushStruct()
                while true {
                    let subField = try reader.readFieldBegin()
                    if subField.type == .stop { break }
                    switch subField.fieldId {
                    case 1: // num_values
                        numValues = Int(try reader.readZigZagI32())
                    case 2: // encoding
                        encoding = try reader.readZigZagI32()
                    case 3: // definition_level_encoding
                        defLevelEncoding = try reader.readZigZagI32()
                    case 4: // repetition_level_encoding
                        repLevelEncoding = try reader.readZigZagI32()
                    default:
                        try reader.skip(type: subField.type)
                    }
                }
                reader.popStruct()
            case 7: // dictionary_page_header
                reader.pushStruct()
                while true {
                    let subField = try reader.readFieldBegin()
                    if subField.type == .stop { break }
                    switch subField.fieldId {
                    case 1: // num_values
                        numValues = Int(try reader.readZigZagI32())
                    case 2: // encoding
                        encoding = try reader.readZigZagI32()
                    default:
                        try reader.skip(type: subField.type)
                    }
                }
                reader.popStruct()
            default:
                try reader.skip(type: field.type)
            }
        }
        reader.popStruct()
        return PageHeaderInfo(
            type: type,
            uncompressedPageSize: uncomp,
            compressedPageSize: comp,
            numValues: numValues,
            encoding: encoding,
            defLevelEncoding: defLevelEncoding,
            repLevelEncoding: repLevelEncoding
        )
    }

    private static func decodeColumnChunk(
        fileData: Data,
        colChunk: ColChunkInfo,
        schema: SchemaElem,
        numRows: Int
    ) throws -> [Any?] {
        guard !fileData.isEmpty else { return [] }

        // 1. Read Dictionary Page if present
        var dictionary: [Any?] = []
        if let dictOffset = colChunk.dictionaryPageOffset, dictOffset > 0 && dictOffset < fileData.count {
            let dictReader = ThriftCompact.Reader(data: fileData.subdata(in: Int(dictOffset)..<fileData.count))
            let header = try parsePageHeader(reader: dictReader)
            let payloadOffset = Int(dictOffset) + dictReader.offset
            let payloadEnd = payloadOffset + Int(header.compressedPageSize)
            guard payloadEnd <= fileData.count else {
                throw SwiftMLError.parseError(line: 0, description: "Dictionary page exceeds file buffer")
            }
            let rawBytes = fileData.subdata(in: payloadOffset..<payloadEnd)
            let decompressed = colChunk.codec == 1 ? try SnappyDecompressor.decompress(data: rawBytes) : rawBytes
            dictionary = try decodePlainValues(payload: decompressed, schema: schema, count: header.numValues)
        }

        // 2. Read Data Page
        let dataOffset = Int(colChunk.dataPageOffset)
        guard dataOffset < fileData.count else { return [] }

        // Standard Apache Parquet has uncompressed Thrift PageHeader starting with fieldId 1 (byte 0x15)
        if fileData[dataOffset] == 0x15 {
            let pageReader = ThriftCompact.Reader(data: fileData.subdata(in: dataOffset..<fileData.count))
            let header = try parsePageHeader(reader: pageReader)
            let payloadOffset = dataOffset + pageReader.offset
            let payloadEnd = payloadOffset + Int(header.compressedPageSize)
            guard payloadEnd <= fileData.count else {
                throw SwiftMLError.parseError(line: 0, description: "Data page payload exceeds file buffer")
            }
            let rawBytes = fileData.subdata(in: payloadOffset..<payloadEnd)
            let decompressed = colChunk.codec == 1 ? try SnappyDecompressor.decompress(data: rawBytes) : rawBytes

            return try decodeDataPagePayload(
                payload: decompressed,
                schema: schema,
                numValues: header.numValues,
                encoding: header.encoding,
                dictionary: dictionary,
                expectedRowCount: numRows
            )
        } else {
            // Fallback for legacy SwiftSci files where the entire page was Snappy-compressed
            let chunkEnd = min(fileData.count, dataOffset + Int(colChunk.totalCompressedSize))
            let rawChunk = fileData.subdata(in: dataOffset..<chunkEnd)
            let decompressed = colChunk.codec == 1 ? try SnappyDecompressor.decompress(data: rawChunk) : rawChunk
            let pageReader = ThriftCompact.Reader(data: decompressed)
            let header = try parsePageHeader(reader: pageReader)
            let payload = decompressed.subdata(in: pageReader.offset..<decompressed.count)

            return try decodeDataPagePayload(
                payload: payload,
                schema: schema,
                numValues: header.numValues,
                encoding: header.encoding,
                dictionary: dictionary,
                expectedRowCount: numRows
            )
        }
    }

    private static func decodeDataPagePayload(
        payload: Data,
        schema: SchemaElem,
        numValues: Int,
        encoding: Int32,
        dictionary: [Any?],
        expectedRowCount: Int
    ) throws -> [Any?] {
        guard !payload.isEmpty && numValues > 0 else { return [] }

        return try payload.withUnsafeBytes { rawBuf in
            guard let ptr = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return []
            }
            let byteCount = rawBuf.count
            var offset = 0

            // 1. Repetition levels (if nested/repeated list)
            var repLevels: [UInt8] = []
            if schema.isNestedList {
                guard offset + 4 <= byteCount else { return [] }
                let repLen = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8) | (Int(ptr[offset + 2]) << 16) | (Int(ptr[offset + 3]) << 24)
                offset += 4
                guard offset + repLen <= byteCount else { return [] }
                repLevels = decodeBitLevels(ptr: ptr + offset, length: repLen, maxLevel: 1, count: numValues)
                offset += repLen
            }

            // 2. Definition levels (if optional or nested)
            var defLevels: [UInt8] = []
            if schema.repetitionType == 1 || schema.isNestedList { // OPTIONAL or LIST
                guard offset + 4 <= byteCount else { return [] }
                let defLen = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8) | (Int(ptr[offset + 2]) << 16) | (Int(ptr[offset + 3]) << 24)
                offset += 4
                guard offset + defLen <= byteCount else { return [] }
                let maxLevel = schema.isNestedList ? 3 : 1
                defLevels = decodeBitLevels(ptr: ptr + offset, length: defLen, maxLevel: maxLevel, count: numValues)
                offset += defLen
            } else {
                defLevels = [UInt8](repeating: 1, count: numValues)
            }

            // 3. Values
            if (encoding == 8 || encoding == 7) && !dictionary.isEmpty {
                // RLE_DICTIONARY or PLAIN_DICTIONARY
                guard offset < byteCount else { return [] }
                let bitWidth = Int(ptr[offset])
                offset += 1

                let indices = decodeRLEBitPackedIndices(
                    ptr: ptr,
                    offset: &offset,
                    count: byteCount,
                    totalValues: numValues,
                    bitWidth: bitWidth
                )

                var mappedValues: [Any?] = []
                mappedValues.reserveCapacity(numValues)
                var valIdx = 0

                for i in 0..<numValues {
                    let isPresent = i < defLevels.count ? (schema.isNestedList ? defLevels[i] >= 3 : defLevels[i] == 1) : true
                    if !isPresent {
                        mappedValues.append(nil)
                        continue
                    }
                    if valIdx < indices.count {
                        let idx = indices[valIdx]
                        valIdx += 1
                        mappedValues.append(idx < dictionary.count ? dictionary[idx] : nil)
                    } else {
                        mappedValues.append(nil)
                    }
                }

                // If this is a nested repeated list, aggregate values by repetition level
                if schema.isNestedList {
                    return aggregateNestedList(values: mappedValues, repLevels: repLevels, expectedRows: expectedRowCount)
                }

                return mappedValues
            } else {
                // PLAIN values
                let plainSlice = payload.subdata(in: offset..<payload.count)
                return try decodePlainValues(payload: plainSlice, schema: schema, count: numValues, defLevels: defLevels)
            }
        }
    }

    private static func aggregateNestedList(values: [Any?], repLevels: [UInt8], expectedRows: Int) -> [Any?] {
        var rows: [String] = []
        var current: [String] = []

        for (i, val) in values.enumerated() {
            let rep = i < repLevels.count ? repLevels[i] : 0
            if rep == 0 && i > 0 {
                rows.append("[\(current.joined(separator: ", "))]")
                current = []
            }
            if let v = val {
                current.append(String(describing: v))
            }
        }
        rows.append("[\(current.joined(separator: ", "))]")

        // Pad if needed to expected row count
        while rows.count < expectedRows {
            rows.append("[]")
        }
        return rows
    }

    private static func decodeBitLevels(ptr: UnsafePointer<UInt8>, length: Int, maxLevel: Int, count: Int) -> [UInt8] {
        var levels: [UInt8] = []
        levels.reserveCapacity(count)
        var offset = 0
        let bitWidth = maxLevel <= 1 ? 1 : 2

        while offset < length && levels.count < count {
            var header: UInt32 = 0
            var shift: UInt32 = 0
            while offset < length {
                let b = ptr[offset]
                offset += 1
                header |= UInt32(b & 0x7F) << shift
                if (b & 0x80) == 0 { break }
                shift += 7
            }

            if (header & 1) == 0 {
                // RLE run
                let runCount = Int(header >> 1)
                guard offset < length else { break }
                let val = ptr[offset]
                offset += 1
                for _ in 0..<runCount {
                    if levels.count < count { levels.append(val) }
                }
            } else {
                // Bit-packed run
                let numGroups = Int(header >> 1)
                let numBytes = (numGroups * 8 * bitWidth + 7) / 8
                for _ in 0..<numBytes {
                    guard offset < length else { break }
                    let byteVal = ptr[offset]
                    offset += 1
                    let itemsPerByte = 8 / bitWidth
                    let mask: UInt8 = (1 << bitWidth) - 1
                    for item in 0..<itemsPerByte {
                        if levels.count < count {
                            levels.append((byteVal >> (item * bitWidth)) & mask)
                        }
                    }
                }
            }
        }
        return levels
    }

    private static func decodeRLEBitPackedIndices(
        ptr: UnsafePointer<UInt8>,
        offset: inout Int,
        count: Int,
        totalValues: Int,
        bitWidth: Int
    ) -> [Int] {
        guard bitWidth > 0 else {
            return [Int](repeating: 0, count: totalValues)
        }

        var indices: [Int] = []
        indices.reserveCapacity(totalValues)

        while offset < count && indices.count < totalValues {
            // Read varint header
            var header: UInt32 = 0
            var shift: UInt32 = 0
            while offset < count {
                let b = ptr[offset]
                offset += 1
                header |= UInt32(b & 0x7F) << shift
                if (b & 0x80) == 0 { break }
                shift += 7
            }

            if (header & 1) == 0 {
                // RLE run
                let runCount = Int(header >> 1)
                let bytesForVal = (bitWidth + 7) / 8
                guard offset + bytesForVal <= count else { break }
                var val = 0
                for b in 0..<bytesForVal {
                    val |= Int(ptr[offset + b]) << (b * 8)
                }
                offset += bytesForVal
                for _ in 0..<runCount {
                    if indices.count < totalValues { indices.append(val) }
                }
            } else {
                // Bit-packed run
                let numGroups = Int(header >> 1)
                let numValuesInRun = numGroups * 8
                var bitPos = offset * 8
                let mask = (1 << bitWidth) - 1

                for _ in 0..<numValuesInRun {
                    guard indices.count < totalValues else { break }
                    let byteIdx = bitPos / 8
                    let bitOffset = bitPos % 8
                    var raw: UInt64 = 0
                    for b in 0..<min(8, count - byteIdx) {
                        raw |= UInt64(ptr[byteIdx + b]) << (b * 8)
                    }
                    indices.append(Int((raw >> bitOffset) & UInt64(mask)))
                    bitPos += bitWidth
                }
                offset += (numGroups * bitWidth)
            }
        }

        return indices
    }

    private static func decodePlainValues(
        payload: Data,
        schema: SchemaElem,
        count: Int,
        defLevels: [UInt8]? = nil
    ) throws -> [Any?] {
        guard !payload.isEmpty && count > 0 else { return [] }

        return payload.withUnsafeBytes { rawBuf in
            guard let ptr = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return []
            }
            let byteCount = rawBuf.count
            var offset = 0
            var results: [Any?] = []
            results.reserveCapacity(count)

            for i in 0..<count {
                if let defs = defLevels, i < defs.count && defs[i] == 0 {
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

                case 6: // BYTE_ARRAY
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

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
