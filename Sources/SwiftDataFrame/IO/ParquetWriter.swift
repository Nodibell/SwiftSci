import Foundation

/// A high-performance, pure-Swift zero-dependency writer for Apache Parquet (`.parquet`) columnar files.
///
/// `ParquetWriter` encodes `DataFrame` columns into columnar data pages with Snappy compression
/// and generates compliant Thrift Compact Protocol metadata footers.
public enum ParquetWriter: Sendable {

    /// Writes a `DataFrame` to a local file URL in Apache Parquet format.
    ///
    /// - Parameters:
    ///   - dataFrame: The `DataFrame` to serialize.
    ///   - url: The target file URL.
    /// - Throws: `SwiftMLError.emptySchema` or `SwiftMLError.ioError`.
    public static func write(dataFrame: DataFrame, to url: URL) async throws {
        guard !dataFrame.columns.isEmpty else {
            throw SwiftMLError.emptySchema
        }

        var fileBytes = [UInt8]()
        // 1. Magic bytes 'PAR1'
        let par1: [UInt8] = [0x50, 0x41, 0x52, 0x31]
        fileBytes.append(contentsOf: par1)

        let rowCount = dataFrame.rowCount
        var colChunksMeta: [ParquetReader.ColChunkInfo] = []
        var schemaElems: [ParquetReader.SchemaElem] = []

        // Root schema group element
        schemaElems.append(ParquetReader.SchemaElem(
            name: "schema",
            type: nil,
            repetitionType: nil,
            numChildren: Int32(dataFrame.columns.count)
        ))

        // 2. Write Column Chunks
        for col in dataFrame.columns {
            let colStartOffset = Int64(fileBytes.count)
            let (colType, repType) = parquetType(for: col.dtype)

            schemaElems.append(ParquetReader.SchemaElem(
                name: col.name,
                type: colType,
                repetitionType: repType,
                numChildren: nil
            ))

            // Build Page Data
            let pageData = try encodeDataPage(column: col, rowCount: rowCount)
            let uncompressedSize = Int64(pageData.count)

            // Snappy compress
            let compressedData = SnappyDecompressor.compress(data: pageData)
            let compressedSize = Int64(compressedData.count)

            fileBytes.append(contentsOf: [UInt8](compressedData))

            colChunksMeta.append(ParquetReader.ColChunkInfo(
                codec: 1, // SNAPPY
                numValues: Int64(rowCount),
                totalUncompressedSize: uncompressedSize,
                totalCompressedSize: compressedSize,
                dataPageOffset: colStartOffset
            ))
        }

        // 3. Encode FileMetaData via Thrift Compact
        let rowGroup = ParquetReader.RowGroupInfo(
            columns: colChunksMeta,
            numRows: Int64(rowCount)
        )
        let fileMeta = ParquetReader.FileMetaDataInfo(
            version: 1,
            schema: schemaElems,
            numRows: Int64(rowCount),
            rowGroups: [rowGroup]
        )

        let metaDataBytes = try encodeFileMetaData(fileMeta: fileMeta)
        fileBytes.append(contentsOf: metaDataBytes)

        // 4. Write 4-byte little endian footer length
        var footerLength = UInt32(metaDataBytes.count).littleEndian
        withUnsafeBytes(of: &footerLength) { raw in
            fileBytes.append(contentsOf: raw)
        }

        // 5. Write trailing magic bytes 'PAR1'
        fileBytes.append(contentsOf: par1)

        // Write to destination URL
        let finalData = Data(fileBytes)
        try finalData.write(to: url)
    }

    // MARK: – Page Encoding

    private static func parquetType(for dtype: ColumnDType) -> (type: Int32, repType: Int32) {
        let repType: Int32 = 1 // OPTIONAL
        switch dtype {
        case .int32:
            return (1, repType) // INT32
        case .int64:
            return (2, repType) // INT64
        case .float32:
            return (4, repType) // FLOAT
        case .float64:
            return (5, repType) // DOUBLE
        case .boolean:
            return (0, repType) // BOOLEAN
        case .utf8, .date32:
            return (6, repType) // BYTE_ARRAY
        }
    }

    private static func encodeDataPage(column: any AnyColumn, rowCount: Int) throws -> Data {
        var pageBytes = [UInt8]()

        // 1. Definition Levels (1 for present, 0 for null)
        let numGroups = (rowCount + 7) / 8
        var bitPackedBytes: [UInt8]

        if column.nullCount == 0 {
            // Fast-path: all values present (all bits set to 1)
            bitPackedBytes = [UInt8](repeating: 0xFF, count: numGroups)
            let remainder = rowCount % 8
            if remainder != 0 {
                bitPackedBytes[numGroups - 1] = UInt8((1 << remainder) - 1)
            }
        } else {
            bitPackedBytes = [UInt8]()
            bitPackedBytes.reserveCapacity(numGroups)
            var currentByte: UInt8 = 0
            var bitCount = 0

            for r in 0..<rowCount {
                if column.value(at: r) != nil {
                    currentByte |= (1 << bitCount)
                }
                bitCount += 1
                if bitCount == 8 {
                    bitPackedBytes.append(currentByte)
                    currentByte = 0
                    bitCount = 0
                }
            }
            if bitCount > 0 {
                bitPackedBytes.append(currentByte)
            }
        }

        // Def level header: bit-packed run header = (num_groups << 1) | 1
        var defPayload = [UInt8]()
        defPayload.reserveCapacity(1 + bitPackedBytes.count)
        let headerByte = UInt8((numGroups << 1) | 1)
        defPayload.append(headerByte)
        defPayload.append(contentsOf: bitPackedBytes)

        // Prefix with 4-byte little endian definition level byte length
        var defLen = UInt32(defPayload.count).littleEndian
        withUnsafeBytes(of: &defLen) { pageBytes.append(contentsOf: $0) }
        pageBytes.append(contentsOf: defPayload)

        // 2. Plain Encoded Values (Direct typed fast-path)
        if let c = column as? TypedColumn<Int64> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount * 8)
            for vOpt in c.values {
                if let val = vOpt {
                    var v = val.littleEndian
                    withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                }
            }
        } else if let c = column as? TypedColumn<Int32> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount * 4)
            for vOpt in c.values {
                if let val = vOpt {
                    var v = val.littleEndian
                    withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                }
            }
        } else if let c = column as? TypedColumn<Double> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount * 8)
            for vOpt in c.values {
                if let val = vOpt {
                    var v = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                }
            }
        } else if let c = column as? TypedColumn<Float> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount * 4)
            for vOpt in c.values {
                if let val = vOpt {
                    var v = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                }
            }
        } else if let c = column as? TypedColumn<Bool> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount)
            for vOpt in c.values {
                if let val = vOpt {
                    pageBytes.append(val ? 1 : 0)
                }
            }
        } else if let c = column as? TypedColumn<String> {
            pageBytes.reserveCapacity(pageBytes.count + rowCount * 12)
            for vOpt in c.values {
                if let val = vOpt {
                    let written = val.utf8.withContiguousStorageIfAvailable { strBuf in
                        var strLen = UInt32(strBuf.count).littleEndian
                        withUnsafeBytes(of: &strLen) { pageBytes.append(contentsOf: $0) }
                        pageBytes.append(contentsOf: strBuf)
                        return true
                    } ?? false
                    if !written {
                        let utf8Bytes = Array(val.utf8)
                        var strLen = UInt32(utf8Bytes.count).littleEndian
                        withUnsafeBytes(of: &strLen) { pageBytes.append(contentsOf: $0) }
                        pageBytes.append(contentsOf: utf8Bytes)
                    }
                }
            }
        } else if let c = column as? TypedColumn<Date> {
            for vOpt in c.values {
                if let val = vOpt {
                    let days = Int32(val.timeIntervalSince1970 / 86400.0)
                    var v = days.littleEndian
                    withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                }
            }
        } else {
            // Generic fallback
            for r in 0..<rowCount {
                guard let val = column.value(at: r) else { continue }
                switch column.dtype {
                case .int32:
                    if let i = val as? Int32 {
                        var v = i.littleEndian
                        withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                    }
                case .int64:
                    if let i = val as? Int64 {
                        var v = i.littleEndian
                        withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                    }
                case .float32:
                    if let f = val as? Float {
                        var v = f.bitPattern.littleEndian
                        withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                    }
                case .float64:
                    if let d = val as? Double {
                        var v = d.bitPattern.littleEndian
                        withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                    }
                case .boolean:
                    if let b = val as? Bool {
                        pageBytes.append(b ? 1 : 0)
                    }
                case .utf8:
                    let s = "\(val)"
                    let utf8Bytes = [UInt8](s.utf8)
                    var strLen = UInt32(utf8Bytes.count).littleEndian
                    withUnsafeBytes(of: &strLen) { pageBytes.append(contentsOf: $0) }
                    pageBytes.append(contentsOf: utf8Bytes)
                case .date32:
                    if let date = val as? Date {
                        let days = Int32(date.timeIntervalSince1970 / 86400.0)
                        var v = days.littleEndian
                        withUnsafeBytes(of: &v) { pageBytes.append(contentsOf: $0) }
                    }
                }
            }
        }

        // Prepend PageHeader
        let uncompressedPageSize = Int32(pageBytes.count)
        let pageHeaderBytes = try encodePageHeader(
            uncompressedSize: uncompressedPageSize,
            compressedSize: uncompressedPageSize,
            numValues: Int32(rowCount)
        )

        var fullPage = [UInt8]()
        fullPage.reserveCapacity(pageHeaderBytes.count + pageBytes.count)
        fullPage.append(contentsOf: pageHeaderBytes)
        fullPage.append(contentsOf: pageBytes)

        return Data(fullPage)
    }

    private static func encodePageHeader(
        uncompressedSize: Int32,
        compressedSize: Int32,
        numValues: Int32
    ) throws -> [UInt8] {
        let writer = ThriftCompact.Writer()
        writer.pushStruct()

        // field 1: type (PageType: DATA_PAGE = 0)
        writer.writeFieldBegin(fieldId: 1, type: .i32)
        writer.writeZigZagI32(0)

        // field 2: uncompressed_page_size
        writer.writeFieldBegin(fieldId: 2, type: .i32)
        writer.writeZigZagI32(uncompressedSize)

        // field 3: compressed_page_size
        writer.writeFieldBegin(fieldId: 3, type: .i32)
        writer.writeZigZagI32(compressedSize)

        // field 5: data_page_header
        writer.writeFieldBegin(fieldId: 5, type: .struct)
        writer.pushStruct()
        // data_page_header.num_values (field 1)
        writer.writeFieldBegin(fieldId: 1, type: .i32)
        writer.writeZigZagI32(numValues)
        // data_page_header.encoding (field 2: PLAIN = 0)
        writer.writeFieldBegin(fieldId: 2, type: .i32)
        writer.writeZigZagI32(0)
        // data_page_header.definition_level_encoding (field 3: RLE = 3)
        writer.writeFieldBegin(fieldId: 3, type: .i32)
        writer.writeZigZagI32(3)
        // data_page_header.repetition_level_encoding (field 4: RLE = 3)
        writer.writeFieldBegin(fieldId: 4, type: .i32)
        writer.writeZigZagI32(3)
        writer.writeFieldStop()
        writer.popStruct()

        writer.writeFieldStop()
        writer.popStruct()

        return writer.bytes
    }

    private static func encodeFileMetaData(fileMeta: ParquetReader.FileMetaDataInfo) throws -> [UInt8] {
        let writer = ThriftCompact.Writer()
        writer.pushStruct()

        // field 1: version (i32)
        writer.writeFieldBegin(fieldId: 1, type: .i32)
        writer.writeZigZagI32(fileMeta.version)

        // field 2: schema (list<SchemaElement>)
        writer.writeFieldBegin(fieldId: 2, type: .list)
        writer.writeListBegin(elemType: .struct, size: fileMeta.schema.count)
        for elem in fileMeta.schema {
            writer.pushStruct()
            if let t = elem.type {
                writer.writeFieldBegin(fieldId: 1, type: .i32)
                writer.writeZigZagI32(t)
            }
            if let rep = elem.repetitionType {
                writer.writeFieldBegin(fieldId: 3, type: .i32)
                writer.writeZigZagI32(rep)
            }
            writer.writeFieldBegin(fieldId: 4, type: .binary)
            writer.writeString(elem.name)
            if let numCh = elem.numChildren {
                writer.writeFieldBegin(fieldId: 5, type: .i32)
                writer.writeZigZagI32(numCh)
            }
            writer.writeFieldStop()
            writer.popStruct()
        }

        // field 3: num_rows (i64)
        writer.writeFieldBegin(fieldId: 3, type: .i64)
        writer.writeZigZagI64(fileMeta.numRows)

        // field 4: row_groups (list<RowGroup>)
        writer.writeFieldBegin(fieldId: 4, type: .list)
        writer.writeListBegin(elemType: .struct, size: fileMeta.rowGroups.count)
        for rg in fileMeta.rowGroups {
            writer.pushStruct()
            // columns (field 1: list<ColumnChunk>)
            writer.writeFieldBegin(fieldId: 1, type: .list)
            writer.writeListBegin(elemType: .struct, size: rg.columns.count)
            for colChunk in rg.columns {
                writer.pushStruct()
                // meta_data (field 3: ColumnMetaData)
                writer.writeFieldBegin(fieldId: 3, type: .struct)
                writer.pushStruct()
                // codec (field 4)
                writer.writeFieldBegin(fieldId: 4, type: .i32)
                writer.writeZigZagI32(colChunk.codec)
                // num_values (field 5)
                writer.writeFieldBegin(fieldId: 5, type: .i64)
                writer.writeZigZagI64(colChunk.numValues)
                // total_uncompressed_size (field 6)
                writer.writeFieldBegin(fieldId: 6, type: .i64)
                writer.writeZigZagI64(colChunk.totalUncompressedSize)
                // total_compressed_size (field 7)
                writer.writeFieldBegin(fieldId: 7, type: .i64)
                writer.writeZigZagI64(colChunk.totalCompressedSize)
                // data_page_offset (field 8)
                writer.writeFieldBegin(fieldId: 8, type: .i64)
                writer.writeZigZagI64(colChunk.dataPageOffset)
                writer.writeFieldStop()
                writer.popStruct()

                writer.writeFieldStop()
                writer.popStruct()
            }
            // num_rows (field 3)
            writer.writeFieldBegin(fieldId: 3, type: .i64)
            writer.writeZigZagI64(rg.numRows)
            writer.writeFieldStop()
            writer.popStruct()
        }

        // field 6: created_by (string)
        writer.writeFieldBegin(fieldId: 6, type: .binary)
        writer.writeString("SwiftSci v3.4.0 (Pure Swift)")

        writer.writeFieldStop()
        writer.popStruct()

        return writer.bytes
    }
}
