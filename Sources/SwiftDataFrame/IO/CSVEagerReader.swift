import Foundation

/// Retains final values, but releases record coordinates after each joined batch.
internal enum CSVEagerReader {
    private static let batchRows = 16_384

    static func read(buffer: UnsafeBufferPointer<UInt8>, options: CSVReadOptions) throws -> DataFrame {
        let parser = SystemsCSVParser(delimiterByte: UInt8(options.delimiter.utf8.first ?? 44))
        let first = try parser.scanBatch(buffer: buffer, maximumRows: 1)
        guard !first.records.isEmpty else { return DataFrame.empty }
        let headers: [String]
        let dataStart: Int
        if options.hasHeader {
            headers = CSVReader.deduplicateHeaders(first.records.row(at: 0).map {
                VectorizedByteParsers.parseString(buffer: buffer, offset: $0)
            })
            dataStart = first.nextOffset
        } else {
            headers = (0..<first.records.row(at: 0).count).map { "col\($0)" }
            dataStart = 0
        }
        guard !headers.isEmpty, dataStart < buffer.count else { return DataFrame.empty }
        let source = UnsafeBufferPointer(rebasing: buffer[dataStart...])
        guard let initial = try begin(source: source, parser: parser, headers: headers, options: options) else {
            return DataFrame.empty
        }
        let plans = initial.plans
        var attempts = initial.attempts
        var candidate = [Int](repeating: 0, count: headers.count)
        var completed = [ (any AnyColumn)? ](repeating: nil, count: headers.count)
        var active = Array(headers.indices)
        var offset = initial.nextOffset
        var rows = initial.rows
        repeat {
            while offset < source.count && options.maxRows.map({ rows < $0 }) != false {
                try Task.checkCancellation()
                if active.allSatisfy({ attempts[$0]?.rejected == true }) { break }
                let remaining = options.maxRows.map { $0 - rows } ?? batchRows
                let bytes = UnsafeBufferPointer(rebasing: source[offset...])
                let batch = try parser.scanBatch(buffer: bytes, maximumRows: min(batchRows, remaining))
                guard !batch.records.isEmpty else { break }
                append(buffer: bytes, records: batch.records, attempts: attempts, columns: active, sourceOffset: offset)
                offset += batch.nextOffset
                rows += batch.records.count
            }
            var retry = [Int]()
            for column in active {
                guard let attempt = attempts[column] else { preconditionFailure("Active CSV column has no attempt") }
                if let field = attempt.rejectedField {
                    candidate[column] = plans[column].nextCandidate(after: candidate[column],
                        buffer: source, rejectedField: field)
                    attempts[column] = plans[column].conversion(
                        type: plans[column].candidates[candidate[column]], capacity: initial.capacity)
                    retry.append(column)
                } else {
                    completed[column] = attempt.column(named: headers[column])
                    attempts[column] = nil
                }
            }
            active = retry
            offset = 0
            rows = 0
        } while !active.isEmpty
        try Task.checkCancellation()
        return try DataFrame(columns: completed.map {
            guard let column = $0 else { preconditionFailure("CSV column was not completed") }
            return column
        })
    }

    private struct Initial {
        let plans: [CSVColumnPlan]
        let capacity: Int
        let attempts: [CSVColumnConversion?]
        let nextOffset: Int
        let rows: Int
    }

    /// Sample and append the same batch before its coordinates leave this scope.
    private static func begin(
        source: UnsafeBufferPointer<UInt8>, parser: SystemsCSVParser,
        headers: [String], options: CSVReadOptions
    ) throws -> Initial? {
        let limit = options.maxRows.map { min(batchRows, max(1, $0)) } ?? batchRows
        let batch = try parser.scanBatch(buffer: source, maximumRows: limit)
        guard !batch.records.isEmpty else { return nil }
        let rows = options.maxRows == 0 ? 0 : batch.records.count
        let remainder = UnsafeBufferPointer(rebasing: source[batch.nextOffset...])
        let remaining = options.maxRows.map { max(0, $0 - rows) }
        let capacity = rows + (try parser.countRecords(buffer: remainder, maximumRows: remaining))
        let plans = headers.enumerated().map { column, name in
            CSVColumnPlan(buffer: source, records: batch.records, startRow: 0,
                rowCount: rows, column: column, name: name, options: options)
        }
        let attempts: [CSVColumnConversion?] = plans.map {
            $0.conversion(type: $0.candidates[0], capacity: capacity)
        }
        append(buffer: source, records: batch.records, attempts: attempts, columns: Array(headers.indices), rowCount: rows)
        return Initial(plans: plans, capacity: capacity, attempts: attempts, nextOffset: batch.nextOffset, rows: rows)
    }

    private static func append(
        buffer: UnsafeBufferPointer<UInt8>, records: CSVRecordIndex,
        attempts: [CSVColumnConversion?], columns: [Int], rowCount: Int? = nil, sourceOffset: Int = 0
    ) {
        let input = BatchInput(buffer: buffer, attempts: attempts)
        let count = rowCount ?? records.count
        DispatchQueue.concurrentPerform(iterations: columns.count) { index in
            let column = columns[index]
            input.attempts[column]?.appendIndexed(buffer: input.buffer, records: records,
                startRow: 0, rowCount: count, column: column, sourceOffset: sourceOffset)
        }
    }

    // Every column is owned by exactly one joined worker. The borrowed bytes and
    // immutable coordinates remain valid until concurrentPerform returns.
    private struct BatchInput: @unchecked Sendable {
        let buffer: UnsafeBufferPointer<UInt8>
        let attempts: [CSVColumnConversion?]
    }
}
