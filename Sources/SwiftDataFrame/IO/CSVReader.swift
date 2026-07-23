import Foundation

private struct UnsafeSendableBuffer: @unchecked Sendable {
    let pointer: UnsafeBufferPointer<UInt8>
}

private struct RecordsBox: @unchecked Sendable {
    let records: [[CSVFieldOffset]]
}

private struct SendableColumnPointer: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<any AnyColumn>
}

private struct OptionsBox: @unchecked Sendable {
    let options: CSVReadOptions
}

private struct HeadersBox: @unchecked Sendable {
    let headers: [String]
}

/// Reads CSV files into a `DataFrame`.
/// Uses Foundation for file I/O and performs Swift-native type inference.
internal enum CSVReader {

    static func read(url: URL, options: CSVReadOptions) async throws -> DataFrame {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DataFrameError.fileNotFound(url)
        }

        let mappedData = try Data(contentsOf: url, options: .alwaysMapped)
        var records: [[CSVFieldOffset]] = []
        let delimByte = UInt8(options.delimiter.utf8.first ?? 44)

        mappedData.withUnsafeBytes { rawBuffer in
            guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
            let parser = SystemsCSVParser(delimiterByte: delimByte)
            records = parser.parse(buffer: bufferPointer)
        }

        guard !records.isEmpty else { return DataFrame.empty }

        var headers: [String] = []
        var startRowIdx = 0

        mappedData.withUnsafeBytes { rawBuffer in
            guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
            if options.hasHeader {
                headers = records[0].map { VectorizedByteParsers.parseString(buffer: bufferPointer, offset: $0) }
                startRowIdx = 1
            } else {
                headers = records[0].indices.map { "col\($0)" }
                startRowIdx = 0
            }
        }

        let dataRowCount = records.count - startRowIdx
        guard dataRowCount > 0, !headers.isEmpty else { return DataFrame.empty }

        let dataRowsToRead: Int
        if let maxRows = options.maxRows {
            dataRowsToRead = min(dataRowCount, maxRows)
        } else {
            dataRowsToRead = dataRowCount
        }

        let colCount = headers.count
        var columns: [any AnyColumn] = Array(repeating: TypedColumn<String>(name: "", values: []), count: colCount)

        guard let basePtr = mappedData.withUnsafeBytes({ $0.baseAddress?.assumingMemoryBound(to: UInt8.self) }) else {
            return DataFrame.empty
        }
        let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
        let sendableBuf = UnsafeSendableBuffer(pointer: bufferPointer)
        let recordsBox = RecordsBox(records: records)
        let optionsBox = OptionsBox(options: options)
        let headersBox = HeadersBox(headers: headers)

        let readStartRowIdx = startRowIdx
        columns.withUnsafeMutableBufferPointer { colBuf in
            guard let baseColPtr = colBuf.baseAddress else { return }
            let sendableColPtr = SendableColumnPointer(pointer: baseColPtr)

            DispatchQueue.concurrentPerform(iterations: colCount) { c in
                let colName = headersBox.headers[c]
                let col = buildColumn(
                    buffer: sendableBuf.pointer,
                    records: recordsBox.records,
                    startRowIdx: readStartRowIdx,
                    dataRowsToRead: dataRowsToRead,
                    colIndex: c,
                    name: colName,
                    options: optionsBox.options
                )
                sendableColPtr.pointer[c] = col
            }
        }

        return try DataFrame(columns: columns)
    }

    // MARK: – Core parser

    static func parse(_ raw: String, options: CSVReadOptions) throws -> DataFrame {
        var lines = splitLines(raw)

        // Remove trailing empty line
        if lines.last?.trimmingCharacters(in: .whitespaces).isEmpty == true {
            lines.removeLast()
        }

        guard !lines.isEmpty else { return DataFrame.empty }

        let delim = options.delimiter

        // ── Parse header ─────────────────────────────────────────────────
        var headers: [String]
        var dataStartLine: Int

        if options.hasHeader {
            headers       = parseRow(lines[0], delimiter: delim)
            dataStartLine = 1
        } else {
            let firstRow = parseRow(lines[0], delimiter: delim)
            headers       = firstRow.indices.map { "col\($0)" }
            dataStartLine = 0
        }

        guard !headers.isEmpty else {
            throw DataFrameError.parseError(line: 1, description: "Empty header row.")
        }

        // ── Limit rows ───────────────────────────────────────────────────
        let dataLines: [String]
        if let max = options.maxRows {
            dataLines = Array(lines[dataStartLine...].prefix(max))
        } else {
            dataLines = Array(lines[dataStartLine...])
        }

        // ── Parse cells ──────────────────────────────────────────────────
        let colCount = headers.count
        // rawCells[col][row]
        var rawCells: [[String?]] = Array(repeating: [], count: colCount)

        for (lineOffset, line) in dataLines.enumerated() {
            let row = parseRow(line, delimiter: delim)
            let lineNumber = dataStartLine + lineOffset + 1

            guard row.count == colCount else {
                throw DataFrameError.parseError(
                    line: lineNumber,
                    description: "Expected \(colCount) columns, found \(row.count)."
                )
            }

            for (ci, cell) in row.enumerated() {
                let trimmed = cell.trimmingCharacters(in: .whitespaces)
                rawCells[ci].append(options.nullValues.contains(trimmed) ? nil : trimmed)
            }
        }

        // ── Build columns ────────────────────────────────────────────────
        var columns: [any AnyColumn] = []
        for (ci, name) in headers.enumerated() {
            let cells = rawCells[ci]
            let col   = options.inferTypes
                ? inferColumn(name: name, cells: cells, options: options)
                : TypedColumn<String>(name: name, values: cells)
            columns.append(col)
        }

        return try DataFrame(columns: columns)
    }

    // MARK: – Type inference

    /// Infers the best ColumnDType by trying types in priority order or respecting columnTypeOverrides.
    internal static func inferColumn(name: String, cells: [String?], options: CSVReadOptions = .default) -> any AnyColumn {
        if let overrideType = options.columnTypeOverrides[name] {
            switch overrideType {
            case .int32, .int64:
                return TypedColumn<Int64>(name: name, values: cells.map { $0.flatMap(Int64.parse) })
            case .float32, .float64:
                return TypedColumn<Double>(name: name, values: cells.map { $0.flatMap(Double.parse) })
            case .boolean:
                return TypedColumn<Bool>(name: name, values: cells.map { $0.flatMap(Bool.parse) })
            case .utf8:
                return TypedColumn<String>(name: name, values: cells)
            case .date32:
                return TypedColumn<Date>(name: name, values: cells.map { $0.flatMap(Date.parse) })
            }
        }

        let nonNull = cells.compactMap { $0 }
        guard !nonNull.isEmpty else {
            return TypedColumn<String>(name: name, values: cells)
        }

        if nonNull.allSatisfy({ Bool.parse(from: $0) != nil }) {
            return TypedColumn<Bool>(name: name, values: cells.map { $0.flatMap(Bool.parse) })
        }

        if nonNull.allSatisfy({ Int64.parse(from: $0) != nil }) {
            return TypedColumn<Int64>(name: name, values: cells.map { $0.flatMap(Int64.parse) })
        }

        if nonNull.allSatisfy({ Double.parse(from: $0) != nil }) {
            return TypedColumn<Double>(name: name, values: cells.map { $0.flatMap(Double.parse) })
        }

        if nonNull.allSatisfy({ Date.parse(from: $0) != nil }) {
            return TypedColumn<Date>(name: name, values: cells.map { $0.flatMap(Date.parse) })
        }

        return TypedColumn<String>(name: name, values: cells)
    }

    // MARK: – Column Builder (Column-Parallel)

    internal static func buildColumn(
        buffer: UnsafeBufferPointer<UInt8>,
        records: [[CSVFieldOffset]],
        startRowIdx: Int,
        dataRowsToRead: Int,
        colIndex: Int,
        name: String,
        options: CSVReadOptions
    ) -> any AnyColumn {
        if let overrideType = options.columnTypeOverrides[name] {
            switch overrideType {
            case .int32, .int64:
                var values = [Int64?]()
                values.reserveCapacity(dataRowsToRead)
                for r in 0..<dataRowsToRead {
                    let rowIdx = startRowIdx + r
                    if rowIdx < records.count && colIndex < records[rowIdx].count {
                        let offset = records[rowIdx][colIndex]
                        let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                        if options.nullValues.contains(str) {
                            values.append(nil)
                        } else if let val = VectorizedByteParsers.parseInt(buffer: buffer, offset: offset) {
                            values.append(Int64(val))
                        } else {
                            values.append(Int64.parse(from: str))
                        }
                    } else {
                        values.append(nil)
                    }
                }
                return TypedColumn<Int64>(name: name, values: values)
            case .float32, .float64:
                var values = [Double?]()
                values.reserveCapacity(dataRowsToRead)
                for r in 0..<dataRowsToRead {
                    let rowIdx = startRowIdx + r
                    if rowIdx < records.count && colIndex < records[rowIdx].count {
                        let offset = records[rowIdx][colIndex]
                        let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                        if options.nullValues.contains(str) {
                            values.append(nil)
                        } else if let val = VectorizedByteParsers.parseDouble(buffer: buffer, offset: offset) {
                            values.append(val)
                        } else {
                            values.append(Double.parse(from: str))
                        }
                    } else {
                        values.append(nil)
                    }
                }
                return TypedColumn<Double>(name: name, values: values)
            case .boolean:
                var values = [Bool?]()
                values.reserveCapacity(dataRowsToRead)
                for r in 0..<dataRowsToRead {
                    let rowIdx = startRowIdx + r
                    if rowIdx < records.count && colIndex < records[rowIdx].count {
                        let offset = records[rowIdx][colIndex]
                        let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                        values.append(options.nullValues.contains(str) ? nil : Bool.parse(from: str))
                    } else {
                        values.append(nil)
                    }
                }
                return TypedColumn<Bool>(name: name, values: values)
            case .utf8:
                var values = [String?]()
                values.reserveCapacity(dataRowsToRead)
                for r in 0..<dataRowsToRead {
                    let rowIdx = startRowIdx + r
                    if rowIdx < records.count && colIndex < records[rowIdx].count {
                        let offset = records[rowIdx][colIndex]
                        let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                        values.append(options.nullValues.contains(str) ? nil : str)
                    } else {
                        values.append(nil)
                    }
                }
                return TypedColumn<String>(name: name, values: values)
            case .date32:
                var values = [Date?]()
                values.reserveCapacity(dataRowsToRead)
                for r in 0..<dataRowsToRead {
                    let rowIdx = startRowIdx + r
                    if rowIdx < records.count && colIndex < records[rowIdx].count {
                        let offset = records[rowIdx][colIndex]
                        let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                        values.append(options.nullValues.contains(str) ? nil : Date.parse(from: str))
                    } else {
                        values.append(nil)
                    }
                }
                return TypedColumn<Date>(name: name, values: values)
            }
        }

        if !options.inferTypes {
            var values = [String?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    values.append(options.nullValues.contains(str) ? nil : str)
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<String>(name: name, values: values)
        }

        var isBool = true
        var isInt64 = true
        var isDouble = true
        var isDate = true
        var nonNullCount = 0

        for r in 0..<dataRowsToRead {
            let rowIdx = startRowIdx + r
            guard rowIdx < records.count && colIndex < records[rowIdx].count else { continue }
            let offset = records[rowIdx][colIndex]
            if offset.length == 0 { continue }
            let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
            if options.nullValues.contains(str) { continue }

            nonNullCount += 1

            if isBool && Bool.parse(from: str) == nil { isBool = false }
            if isInt64 && Int64.parse(from: str) == nil { isInt64 = false }
            if isDouble && VectorizedByteParsers.parseDouble(buffer: buffer, offset: offset) == nil && Double.parse(from: str) == nil { isDouble = false }
            if isDate && Date.parse(from: str) == nil { isDate = false }

            if !isBool && !isInt64 && !isDouble && !isDate {
                break
            }
        }

        if nonNullCount == 0 {
            var values = [String?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    values.append(options.nullValues.contains(str) ? nil : str)
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<String>(name: name, values: values)
        }

        if isBool {
            var values = [Bool?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    values.append(options.nullValues.contains(str) ? nil : Bool.parse(from: str))
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<Bool>(name: name, values: values)
        }

        if isInt64 {
            var values = [Int64?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    if options.nullValues.contains(str) {
                        values.append(nil)
                    } else if let val = VectorizedByteParsers.parseInt(buffer: buffer, offset: offset) {
                        values.append(Int64(val))
                    } else {
                        values.append(Int64.parse(from: str))
                    }
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<Int64>(name: name, values: values)
        }

        if isDouble {
            var values = [Double?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    if options.nullValues.contains(str) {
                        values.append(nil)
                    } else if let val = VectorizedByteParsers.parseDouble(buffer: buffer, offset: offset) {
                        values.append(val)
                    } else {
                        values.append(Double.parse(from: str))
                    }
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<Double>(name: name, values: values)
        }

        if isDate {
            var values = [Date?]()
            values.reserveCapacity(dataRowsToRead)
            for r in 0..<dataRowsToRead {
                let rowIdx = startRowIdx + r
                if rowIdx < records.count && colIndex < records[rowIdx].count {
                    let offset = records[rowIdx][colIndex]
                    let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                    values.append(options.nullValues.contains(str) ? nil : Date.parse(from: str))
                } else {
                    values.append(nil)
                }
            }
            return TypedColumn<Date>(name: name, values: values)
        }

        var values = [String?]()
        values.reserveCapacity(dataRowsToRead)
        for r in 0..<dataRowsToRead {
            let rowIdx = startRowIdx + r
            if rowIdx < records.count && colIndex < records[rowIdx].count {
                let offset = records[rowIdx][colIndex]
                let str = VectorizedByteParsers.parseString(buffer: buffer, offset: offset).trimmingCharacters(in: .whitespaces)
                values.append(options.nullValues.contains(str) ? nil : str)
            } else {
                values.append(nil)
            }
        }
        return TypedColumn<String>(name: name, values: values)
    }

    // MARK: – Streaming CSV Reader (v1.5 mmap-backed)

    static func readStream(url: URL, options: CSVReadOptions, chunkSize: Int) -> AsyncThrowingStream<DataFrame, any Error> {
        AsyncThrowingStream(DataFrame.self) { continuation in
            let task = Task {
                do {
                    guard FileManager.default.fileExists(atPath: url.path) else {
                        throw DataFrameError.fileNotFound(url)
                    }

                    let mappedData = try Data(contentsOf: url, options: .alwaysMapped)
                    let delimByte = UInt8(options.delimiter.utf8.first ?? 44)

                    var records: [[CSVFieldOffset]] = []
                    mappedData.withUnsafeBytes { rawBuffer in
                        guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                        let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
                        let parser = SystemsCSVParser(delimiterByte: delimByte)
                        records = parser.parse(buffer: bufferPointer)
                    }

                    guard !records.isEmpty else {
                        continuation.finish()
                        return
                    }

                    var headers: [String] = []
                    var startRowIdx = 0

                    mappedData.withUnsafeBytes { rawBuffer in
                        guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                        let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
                        if options.hasHeader {
                            headers = records[0].map { VectorizedByteParsers.parseString(buffer: bufferPointer, offset: $0) }
                            startRowIdx = 1
                        } else {
                            headers = records[0].indices.map { "col\($0)" }
                            startRowIdx = 0
                        }
                    }

                    let totalDataRows = records.count - startRowIdx
                    guard totalDataRows > 0, !headers.isEmpty else {
                        continuation.finish()
                        return
                    }

                    let colCount = headers.count
                    var currentOffset = startRowIdx

                    while currentOffset < records.count && !Task.isCancelled {
                        let rowsInChunk = min(chunkSize, records.count - currentOffset)
                        let offset = currentOffset

                        var columns: [any AnyColumn] = Array(repeating: TypedColumn<String>(name: "", values: []), count: colCount)

                        guard let basePtr = mappedData.withUnsafeBytes({ $0.baseAddress?.assumingMemoryBound(to: UInt8.self) }) else { break }
                        let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)
                        let sendableBuf = UnsafeSendableBuffer(pointer: bufferPointer)
                        let recordsBox = RecordsBox(records: records)
                        let optionsBox = OptionsBox(options: options)
                        let headersBox = HeadersBox(headers: headers)

                        columns.withUnsafeMutableBufferPointer { colBuf in
                            guard let baseColPtr = colBuf.baseAddress else { return }
                            let sendableColPtr = SendableColumnPointer(pointer: baseColPtr)

                            DispatchQueue.concurrentPerform(iterations: colCount) { c in
                                let colName = headersBox.headers[c]
                                let col = buildColumn(
                                    buffer: sendableBuf.pointer,
                                    records: recordsBox.records,
                                    startRowIdx: offset,
                                    dataRowsToRead: rowsInChunk,
                                    colIndex: c,
                                    name: colName,
                                    options: optionsBox.options
                                )
                                sendableColPtr.pointer[c] = col
                            }
                        }

                        let chunkDF = try DataFrame(columns: columns)
                        continuation.yield(chunkDF)

                        currentOffset += rowsInChunk
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { termination in
                if case .cancelled = termination {
                    task.cancel()
                }
            }
        }
    }

    // MARK: – CSV row parser (RFC 4180 compliant)

    /// Splits a single CSV line respecting quoted fields.
    static func parseRow(_ line: String, delimiter: Character) -> [String] {
        var fields: [String] = []
        var current    = ""
        var inQuotes   = false
        var i          = line.startIndex

        while i < line.endIndex {
            let c = line[i]

            if c == "\"" {
                if inQuotes {
                    let next = line.index(after: i)
                    if next < line.endIndex && line[next] == "\"" {
                        current.append("\"")
                        i = line.index(after: next)
                        continue
                    } else {
                        inQuotes = false
                    }
                } else {
                    inQuotes = true
                }
            } else if c == delimiter && !inQuotes {
                fields.append(current)
                current = ""
            } else {
                current.append(c)
            }

            i = line.index(after: i)
        }

        fields.append(current)
        return fields
    }

    // MARK: – Line splitter (handles \r\n and \n respecting quoted fields)

    private static func splitLines(_ raw: String) -> [String] {
        var lines: [String] = []
        var currentLine = ""
        var inQuotes = false
        var i = raw.startIndex

        while i < raw.endIndex {
            let c = raw[i]

            if c == "\"" {
                inQuotes.toggle()
                currentLine.append(c)
            } else if (c == "\n" || c == "\r") && !inQuotes {
                if c == "\r" {
                    let next = raw.index(after: i)
                    if next < raw.endIndex && raw[next] == "\n" {
                        i = next
                    }
                }
                lines.append(currentLine)
                currentLine = ""
            } else {
                currentLine.append(c)
            }

            i = raw.index(after: i)
        }

        if !currentLine.isEmpty || raw.hasSuffix("\n") || raw.hasSuffix("\r") {
            lines.append(currentLine)
        }

        return lines
    }
}
