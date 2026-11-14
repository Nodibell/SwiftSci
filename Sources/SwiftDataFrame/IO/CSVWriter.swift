import Foundation

/// Writes a DataFrame to CSV format.
internal enum CSVWriter {

    private enum ColumnAccessor {
        case double([Double?])
        case doubleDense([Double], validity: [UInt64]?)
        case int64([Int64?])
        case int64Dense([Int64], validity: [UInt64]?)
        case int([Int?])
        case intDense([Int], validity: [UInt64]?)
        case int32([Int32?])
        case int32Dense([Int32], validity: [UInt64]?)
        case float([Float?])
        case floatDense([Float], validity: [UInt64]?)
        case string([String?])
        case stringDense([String], validity: [UInt64]?)
        case bool([Bool?])
        case boolDense([Bool], validity: [UInt64]?)
        case date([Date?])
        case dateDense([Date], validity: [UInt64]?)
        case generic(any AnyColumn)

        init(column: any AnyColumn) {
            if let col = column as? TypedColumn<Double> {
                switch col.storage {
                case .optional(let vals): self = .double(vals)
                case .dense(let vals, let validity): self = .doubleDense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Int64> {
                switch col.storage {
                case .optional(let vals): self = .int64(vals)
                case .dense(let vals, let validity): self = .int64Dense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Int> {
                switch col.storage {
                case .optional(let vals): self = .int(vals)
                case .dense(let vals, let validity): self = .intDense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Int32> {
                switch col.storage {
                case .optional(let vals): self = .int32(vals)
                case .dense(let vals, let validity): self = .int32Dense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Float> {
                switch col.storage {
                case .optional(let vals): self = .float(vals)
                case .dense(let vals, let validity): self = .floatDense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<String> {
                switch col.storage {
                case .optional(let vals): self = .string(vals)
                case .dense(let vals, let validity): self = .stringDense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Bool> {
                switch col.storage {
                case .optional(let vals): self = .bool(vals)
                case .dense(let vals, let validity): self = .boolDense(vals, validity: validity)
                }
            } else if let col = column as? TypedColumn<Date> {
                switch col.storage {
                case .optional(let vals): self = .date(vals)
                case .dense(let vals, let validity): self = .dateDense(vals, validity: validity)
                }
            } else {
                self = .generic(column)
            }
        }
    }

    @inline(__always)
    private static func isValidRow(_ validity: [UInt64]?, _ row: Int) -> Bool {
        guard let validity else { return true }
        return validity[row >> 6] & (UInt64(1) << (row & 63)) != 0
    }

    static func write(
        _ df: DataFrame,
        to url: URL,
        delimiter: Character = ",",
        nullValues: Set<String> = CSVReadOptions.default.nullValues
    ) async throws {
        let rowCount = df.shape.rows
        let colCount = df.columns.count
        let delimStr = String(delimiter)
        let delimASCII = delimiter.asciiValue
        let safeNumericDelimiter = !"0123456789.+-eEnafiNFAIturles\"\n\r".contains(delimiter)

        var output = String()
        output.reserveCapacity(max(64, (rowCount + 1) * max(colCount, 1) * 12))

        // Header
        for c in 0..<colCount {
            if c > 0 { output.append(delimStr) }
            appendEscaped(
                df.columnNames[c],
                to: &output,
                delimiter: delimiter,
                delimASCII: delimASCII,
                isStringField: false,
                nullValues: nullValues
            )
        }

        let accessors = df.columns.map(ColumnAccessor.init(column:))

        // Rows
        for row in 0..<rowCount {
            output.append("\n")
            for c in 0..<colCount {
                if c > 0 { output.append(delimStr) }
                appendCell(
                    accessor: accessors[c],
                    row: row,
                    to: &output,
                    delimiter: delimiter,
                    delimASCII: delimASCII,
                    safeNumericDelimiter: safeNumericDelimiter,
                    nullValues: nullValues
                )
            }
        }

        guard let data = output.data(using: .utf8) else {
            throw SwiftMLError.writeError("Failed to encode CSV as UTF-8.")
        }

        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw SwiftMLError.writeError(error.localizedDescription)
        }
    }

    @inline(__always)
    private static func appendCell(
        accessor: ColumnAccessor,
        row: Int,
        to output: inout String,
        delimiter: Character,
        delimASCII: UInt8?,
        safeNumericDelimiter: Bool,
        nullValues: Set<String>
    ) {
        switch accessor {
        case .double(let vals):
            guard let v = vals[row] else { return }
            appendScalar(String(v), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .doubleDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(String(vals[row]), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .int64(let vals):
            guard let v = vals[row] else { return }
            appendScalar(String(v), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .int64Dense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(String(vals[row]), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .int(let vals):
            guard let v = vals[row] else { return }
            appendScalar(String(v), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .intDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(String(vals[row]), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .int32(let vals):
            guard let v = vals[row] else { return }
            appendScalar(String(v), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .int32Dense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(String(vals[row]), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .float(let vals):
            guard let v = vals[row] else { return }
            appendScalar(String(v), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .floatDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(String(vals[row]), to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .string(let vals):
            guard let v = vals[row] else { return }
            appendEscaped(v, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: true, nullValues: nullValues)
        case .stringDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendEscaped(vals[row], to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: true, nullValues: nullValues)
        case .bool(let vals):
            guard let v = vals[row] else { return }
            appendScalar(v ? "true" : "false", to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .boolDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendScalar(vals[row] ? "true" : "false", to: &output, delimiter: delimiter, delimASCII: delimASCII, safeNumericDelimiter: safeNumericDelimiter, nullValues: nullValues)
        case .date(let vals):
            guard let d = vals[row] else { return }
            appendEscaped(d.formattedDateOrDateTimeString, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: false, nullValues: nullValues)
        case .dateDense(let vals, let validity):
            guard isValidRow(validity, row) else { return }
            appendEscaped(vals[row].formattedDateOrDateTimeString, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: false, nullValues: nullValues)
        case .generic(let col):
            guard let val = col.value(at: row) else { return }
            if let d = val as? Date {
                appendEscaped(d.formattedDateOrDateTimeString, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: false, nullValues: nullValues)
            } else if let str = val as? String {
                appendEscaped(str, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: true, nullValues: nullValues)
            } else {
                appendEscaped("\(val)", to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: false, nullValues: nullValues)
            }
        }
    }

    @inline(__always)
    private static func appendScalar(
        _ str: String,
        to output: inout String,
        delimiter: Character,
        delimASCII: UInt8?,
        safeNumericDelimiter: Bool,
        nullValues: Set<String>
    ) {
        if safeNumericDelimiter {
            output.append(str)
        } else {
            appendEscaped(str, to: &output, delimiter: delimiter, delimASCII: delimASCII, isStringField: false, nullValues: nullValues)
        }
    }

    @inline(__always)
    private static func isASCIIWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || (byte >= 0x09 && byte <= 0x0D)
    }

    @inline(__always)
    private static func collidesWithNullToken(_ field: String, nullValues: Set<String>) -> Bool {
        if field.isEmpty { return true }
        if nullValues.contains(field) { return true }
        let utf8 = field.utf8
        if let first = utf8.first, let last = utf8.last,
           first < 0x80, last < 0x80,
           !isASCIIWhitespace(first), !isASCIIWhitespace(last) {
            return false
        }
        return nullValues.contains(field.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Appends a CSV field to `output`, quoting when required by delimiters/quotes/newlines or null-token collisions.
    private static func appendEscaped(
        _ field: String,
        to output: inout String,
        delimiter: Character,
        delimASCII: UInt8?,
        isStringField: Bool,
        nullValues: Set<String>
    ) {
        let collides = isStringField && collidesWithNullToken(field, nullValues: nullValues)
        var hasSpecial = false
        var hasQuote = false

        if let delimByte = delimASCII {
            for b in field.utf8 {
                if b == 0x22 {
                    hasSpecial = true
                    hasQuote = true
                    break
                }
                if b == delimByte || b == 0x0A || b == 0x0D {
                    hasSpecial = true
                }
            }
        } else {
            hasQuote = field.contains("\"")
            hasSpecial = hasQuote || field.contains(delimiter) || field.contains("\n") || field.contains("\r")
        }

        if !collides && !hasSpecial {
            output.append(field)
            return
        }

        output.append("\"")
        if !hasQuote {
            output.append(field)
        } else {
            output.append(field.replacingOccurrences(of: "\"", with: "\"\""))
        }
        output.append("\"")
    }
}
