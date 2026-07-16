import Foundation

/// Reads CSV files into a `DataFrame`.
/// Uses Foundation for file I/O and performs Swift-native type inference.
internal enum CSVReader {

    static func read(url: URL, options: CSVReadOptions) async throws -> DataFrame {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DataFrameError.fileNotFound(url)
        }

        let raw = try String(contentsOf: url, encoding: .utf8)
        return try parse(raw, options: options)
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
                ? inferColumn(name: name, cells: cells)
                : TypedColumn<String>(name: name, values: cells)
            columns.append(col)
        }

        return try DataFrame(columns: columns)
    }

    // MARK: – Type inference

    /// Infers the best ColumnDType by trying types in priority order.
    internal static func inferColumn(name: String, cells: [String?]) -> any AnyColumn {
        let nonNull = cells.compactMap { $0 }
        guard !nonNull.isEmpty else {
            // All null → default to String
            return TypedColumn<String>(name: name, values: cells)
        }

        // Try Bool first (before Int, since "1"/"0" parse as both)
        if nonNull.allSatisfy({ Bool.parse(from: $0) != nil }) {
            return TypedColumn<Bool>(name: name, values: cells.map { $0.flatMap(Bool.parse) })
        }

        // Int64
        if nonNull.allSatisfy({ Int64.parse(from: $0) != nil }) {
            return TypedColumn<Int64>(name: name, values: cells.map { $0.flatMap(Int64.parse) })
        }

        // Double
        if nonNull.allSatisfy({ Double.parse(from: $0) != nil }) {
            return TypedColumn<Double>(name: name, values: cells.map { $0.flatMap(Double.parse) })
        }

        // Date (ISO 8601)
        if nonNull.allSatisfy({ Date.parse(from: $0) != nil }) {
            return TypedColumn<Date>(name: name, values: cells.map { $0.flatMap(Date.parse) })
        }

        // Fallback: String
        return TypedColumn<String>(name: name, values: cells)
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
                        // Escaped quote ""
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
