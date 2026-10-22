import Foundation

private struct UnsafeSendableBuffer: @unchecked Sendable {
    let pointer: UnsafeBufferPointer<UInt8>
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
            throw SwiftMLError.fileNotFound(url)
        }

        let mappedData = try Data(contentsOf: url, options: .alwaysMapped)
        return try mappedData.withUnsafeBytes { rawBuffer in
            let buffer = rawBuffer.bindMemory(to: UInt8.self)
            return try CSVEagerReader.read(buffer: buffer, options: options)
        }
    }

    internal static func buildFrame(
        buffer: UnsafeBufferPointer<UInt8>, records: CSVRecordIndex, startRow: Int,
        rowCount: Int, headers: [String], options: CSVReadOptions
    ) throws -> DataFrame {
        var columns: [any AnyColumn] = Array(
            repeating: TypedColumn<String>(name: "", values: []), count: headers.count
        )
        let sendableBuffer = UnsafeSendableBuffer(pointer: buffer)
        let optionsBox = OptionsBox(options: options)
        let headersBox = HeadersBox(headers: headers)
        columns.withUnsafeMutableBufferPointer { columnBuffer in
            guard let base = columnBuffer.baseAddress else { return }
            let output = SendableColumnPointer(pointer: base)
            options.columnConversionMode.forEachColumn(count: headers.count) { column in
                output.pointer[column] = buildColumn(
                    buffer: sendableBuffer.pointer, records: records, startRowIdx: startRow,
                    dataRowsToRead: rowCount, colIndex: column,
                    name: headersBox.headers[column], options: optionsBox.options)
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
            throw SwiftMLError.parseError(line: 1, description: "Empty header row.")
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
                throw SwiftMLError.parseError(
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
                let parser = CSVDateParser(format: options.columnDateFormats[name])
                return TypedColumn<Date>(name: name, values: cells.map { $0.flatMap(parser.parse) })
            }
        }

        let nonNull = cells.compactMap { $0 }
        guard !nonNull.isEmpty else {
            return TypedColumn<String>(name: name, values: cells)
        }

        if nonNull.allSatisfy({ Int64.parse(from: $0) != nil }) {
            return TypedColumn<Int64>(name: name, values: cells.map { $0.flatMap(Int64.parse) })
        }

        if nonNull.allSatisfy({ Double.parse(from: $0) != nil }) {
            return TypedColumn<Double>(name: name, values: cells.map { $0.flatMap(Double.parse) })
        }

        if nonNull.allSatisfy({ Bool.parse(from: $0) != nil }) {
            return TypedColumn<Bool>(name: name, values: cells.map { $0.flatMap(Bool.parse) })
        }

        if nonNull.allSatisfy({ Date.parse(from: $0) != nil }) {
            return TypedColumn<Date>(name: name, values: cells.map { $0.flatMap(Date.parse) })
        }

        return TypedColumn<String>(name: name, values: cells)
    }

    // MARK: – Column Builder (Column-Parallel)

    internal static func buildColumn(
        buffer: UnsafeBufferPointer<UInt8>,
        records: CSVRecordIndex,
        startRowIdx: Int,
        dataRowsToRead: Int,
        colIndex: Int,
        name: String,
        options: CSVReadOptions
    ) -> any AnyColumn {
        if let column = records.withFlatFields({ view in
            buildIndexedColumn(buffer: buffer, records: view,
                startRowIdx: startRowIdx, dataRowsToRead: dataRowsToRead,
                colIndex: colIndex, name: name, options: options)
        }) { return column }
        if let column = records.withCompactFields({ view in
            buildIndexedColumn(buffer: buffer, records: view,
                startRowIdx: startRowIdx, dataRowsToRead: dataRowsToRead,
                colIndex: colIndex, name: name, options: options)
        }) { return column }
        return buildIndexedColumn(buffer: buffer, records: records,
            startRowIdx: startRowIdx, dataRowsToRead: dataRowsToRead,
            colIndex: colIndex, name: name, options: options)
    }

    private static func buildIndexedColumn<Records: CSVRecordAccess>(
        buffer: UnsafeBufferPointer<UInt8>, records: Records, startRowIdx: Int,
        dataRowsToRead: Int, colIndex: Int, name: String, options: CSVReadOptions
    ) -> any AnyColumn {
        let plan = CSVColumnPlan(buffer: buffer, records: records, startRow: startRowIdx,
            rowCount: dataRowsToRead, column: colIndex, name: name, options: options)
        var candidate = 0
        while true {
            let conversion = plan.conversion(type: plan.candidates[candidate], capacity: dataRowsToRead)
            conversion.append(buffer: buffer, records: records, startRow: startRowIdx,
                rowCount: dataRowsToRead, column: colIndex)
            guard let field = conversion.rejectedField else { return conversion.column(named: name) }
            candidate = plan.nextCandidate(after: candidate, buffer: buffer, rejectedField: field)
        }
    }

    // MARK: – Demand-driven streaming CSV reader

    static func readStream(url: URL, options: CSVReadOptions, chunkSize: Int) -> AsyncThrowingStream<DataFrame, any Error> {
        let cursor = CSVReadCursor(url: url, options: options, chunkSize: chunkSize)
        return AsyncThrowingStream(unfolding: { try cursor.next() })
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

    internal static func deduplicateHeaders(_ rawHeaders: [String]) -> [String] {
        var counts: [String: Int] = [:]
        var used = Set<String>()
        var result: [String] = []

        for (i, h) in rawHeaders.enumerated() {
            let trimmed = h.trimmingCharacters(in: .whitespacesAndNewlines)
            let baseName = trimmed.isEmpty ? "col_\(i)" : trimmed
            var name = baseName
            var count = counts[baseName] ?? 0
            while used.contains(name) {
                count += 1
                name = "\(baseName)_\(count)"
            }
            counts[baseName] = count
            used.insert(name)
            result.append(name)
        }
        return result
    }
}
