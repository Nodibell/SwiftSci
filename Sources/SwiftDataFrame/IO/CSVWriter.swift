import Foundation

/// Writes a DataFrame to CSV format.
internal enum CSVWriter {

    static func write(_ df: DataFrame, to url: URL, delimiter: Character = ",") async throws {
        var lines: [String] = []

        // Header
        lines.append(df.columnNames.map { escape($0, delimiter: delimiter) }.joined(separator: String(delimiter)))

        // Rows
        for row in 0..<df.shape.rows {
            let fields = df.columns.map { col -> String in
                guard let val = col.value(at: row) else { return "" }
                return escape("\(val)", delimiter: delimiter)
            }
            lines.append(fields.joined(separator: String(delimiter)))
        }

        let content = lines.joined(separator: "\n")
        guard let data = content.data(using: .utf8) else {
            throw DataFrameError.writeError("Failed to encode CSV as UTF-8.")
        }

        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw DataFrameError.writeError(error.localizedDescription)
        }
    }

    /// Wraps a field in quotes if it contains delimiter, quote, or newline.
    private static func escape(_ field: String, delimiter: Character) -> String {
        let needsQuoting = field.contains(delimiter) || field.contains("\"") || field.contains("\n") || field.contains("\r")
        if needsQuoting {
            let escaped = field.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }
        return field
    }
}
