import Foundation

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
        var rawCells: [[String?]] = Array(repeating: Array(repeating: nil, count: dataRowsToRead), count: colCount)

        mappedData.withUnsafeBytes { rawBuffer in
            guard let basePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let bufferPointer = UnsafeBufferPointer(start: basePtr, count: mappedData.count)

            for r in 0..<dataRowsToRead {
                let rec = records[startRowIdx + r]
                for c in 0..<min(colCount, rec.count) {
                    let str = VectorizedByteParsers.parseString(buffer: bufferPointer, offset: rec[c]).trimmingCharacters(in: .whitespaces)
                    rawCells[c][r] = options.nullValues.contains(str) ? nil : str
                }
            }
        }

        var columns: [any AnyColumn] = []
        for (ci, name) in headers.enumerated() {
            let cells = rawCells[ci]
            let col = options.inferTypes
                ? inferColumn(name: name, cells: cells)
                : TypedColumn<String>(name: name, values: cells)
            columns.append(col)
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

    // MARK: – Streaming CSV Reader (v1.1)

    static func readStream(url: URL, options: CSVReadOptions, chunkSize: Int) -> AsyncThrowingStream<DataFrame, any Error> {
        AsyncThrowingStream(DataFrame.self) { continuation in
            let task = Task {
                do {
                    guard FileManager.default.fileExists(atPath: url.path) else {
                        throw DataFrameError.fileNotFound(url)
                    }
                    
                    let fileHandle = try FileHandle(forReadingFrom: url)
                    defer { try? fileHandle.close() }
                    
                    var remainder = ""
                    var headers: [String]? = nil
                    var isFirstChunk = true
                    let delim = options.delimiter
                    
                    var pendingRows: [String] = []
                    let bufferSize = 256 * 1024
                    
                    while !Task.isCancelled {
                        let data: Data
                        if #available(macOS 10.15.4, iOS 13.4, watchOS 6.2, tvOS 13.4, *) {
                            guard let chunkData = try fileHandle.read(upToCount: bufferSize) else { break }
                            data = chunkData
                        } else {
                            data = fileHandle.readData(ofLength: bufferSize)
                        }
                        if data.isEmpty { break }
                        
                        guard let str = String(data: data, encoding: .utf8) else {
                            throw DataFrameError.parseError(line: 0, description: "Invalid UTF-8 encoding in chunk")
                        }
                        
                        let combined = remainder + str
                        var lines = splitLines(combined)
                        
                        if !combined.hasSuffix("\n") && !combined.hasSuffix("\r") {
                            if let last = lines.popLast() {
                                remainder = last
                            } else {
                                remainder = ""
                            }
                        } else {
                            remainder = ""
                        }
                        
                        for line in lines {
                            let trimmed = line.trimmingCharacters(in: .whitespaces)
                            if trimmed.isEmpty { continue }
                            
                            if isFirstChunk && options.hasHeader {
                                headers = parseRow(line, delimiter: delim)
                                isFirstChunk = false
                                continue
                            } else if isFirstChunk {
                                let firstRow = parseRow(line, delimiter: delim)
                                headers = firstRow.indices.map { "col\($0)" }
                                isFirstChunk = false
                                pendingRows.append(line)
                                continue
                            }
                            
                            pendingRows.append(line)
                            
                            if pendingRows.count >= chunkSize {
                                if let h = headers {
                                    let chunkDF = try await parseLines(pendingRows, headers: h, options: options)
                                    continuation.yield(chunkDF)
                                }
                                pendingRows.removeAll(keepingCapacity: true)
                            }
                        }
                    }
                    
                    if !remainder.isEmpty && !Task.isCancelled {
                        let trimmed = remainder.trimmingCharacters(in: .whitespaces)
                        if !trimmed.isEmpty {
                            pendingRows.append(remainder)
                        }
                    }
                    
                    if !pendingRows.isEmpty && !Task.isCancelled {
                        if let h = headers {
                            let chunkDF = try await parseLines(pendingRows, headers: h, options: options)
                            continuation.yield(chunkDF)
                        }
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

    private static func parseLines(_ lines: [String], headers: [String], options: CSVReadOptions) async throws -> DataFrame {
        let colCount = headers.count
        let rowCount = lines.count
        let delim = options.delimiter
        
        let numPartitions = min(8, max(1, ProcessInfo.processInfo.activeProcessorCount))
        let partitionSize = (rowCount + numPartitions - 1) / numPartitions
        
        var rawCells: [[String?]] = Array(repeating: Array(repeating: nil, count: rowCount), count: colCount)
        
        try await withThrowingTaskGroup(of: (Int, [[String?]]).self) { group in
            for part in 0..<numPartitions {
                let startIdx = part * partitionSize
                let endIdx = min(rowCount, startIdx + partitionSize)
                if startIdx >= endIdx { continue }
                
                group.addTask {
                    var segmentCells: [[String?]] = Array(repeating: Array(repeating: nil, count: endIdx - startIdx), count: colCount)
                    for rIdx in startIdx..<endIdx {
                        let row = parseRow(lines[rIdx], delimiter: delim)
                        guard row.count == colCount else {
                            throw DataFrameError.parseError(
                                line: rIdx + 1,
                                description: "Expected \(colCount) columns, found \(row.count)."
                            )
                        }
                        let localIdx = rIdx - startIdx
                        for (cIdx, cell) in row.enumerated() {
                            let trimmed = cell.trimmingCharacters(in: .whitespaces)
                            segmentCells[cIdx][localIdx] = options.nullValues.contains(trimmed) ? nil : trimmed
                        }
                    }
                    return (startIdx, segmentCells)
                }
            }
            
            for try await (startIdx, segmentCells) in group {
                let segmentLen = segmentCells[0].count
                for cIdx in 0..<colCount {
                    for localIdx in 0..<segmentLen {
                        rawCells[cIdx][startIdx + localIdx] = segmentCells[cIdx][localIdx]
                    }
                }
            }
        }
        
        var columns: [any AnyColumn] = []
        for (ci, name) in headers.enumerated() {
            let cells = rawCells[ci]
            let col = options.inferTypes
                ? inferColumn(name: name, cells: cells)
                : TypedColumn<String>(name: name, values: cells)
            columns.append(col)
        }
        
        return try DataFrame(columns: columns)
    }
}
