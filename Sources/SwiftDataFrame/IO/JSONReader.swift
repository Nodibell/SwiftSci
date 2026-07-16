import Foundation

/// Reads a JSON file (array of objects) into a `DataFrame`.
/// Each object's keys become column names; missing keys become null values.
internal enum JSONReader {

    static func read(url: URL) async throws -> DataFrame {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw DataFrameError.fileNotFound(url)
        }

        let data = try Data(contentsOf: url)

        guard let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw DataFrameError.parseError(line: 0,
                description: "JSON file must contain an array of objects at the top level.")
        }

        guard !jsonArray.isEmpty else { return DataFrame.empty }

        // Collect all unique keys preserving first-seen order
        var allKeys: [String] = []
        var seen = Set<String>()
        for obj in jsonArray {
            for key in obj.keys where !seen.contains(key) {
                allKeys.append(key)
                seen.insert(key)
            }
        }

        // Build raw string cells per column
        var rawCells: [String: [String?]] = [:]
        for key in allKeys { rawCells[key] = [] }

        for obj in jsonArray {
            for key in allKeys {
                if let value = obj[key] {
                    rawCells[key]?.append(jsonValueToString(value))
                } else {
                    rawCells[key]?.append(nil)
                }
            }
        }

        // Build columns with type inference
        var columns: [any AnyColumn] = []
        for key in allKeys {
            let cells = rawCells[key] ?? []
            columns.append(CSVReader.inferColumn(name: key, cells: cells))
        }

        return try DataFrame(columns: columns)
    }

    private static func jsonValueToString(_ value: Any) -> String? {
        switch value {
        case is NSNull:           return nil
        case let n as NSNumber:   return n.stringValue
        case let s as String:     return s
        default:                  return "\(value)"
        }
    }
}

// (inferColumn is internal on CSVReader — callable directly from JSONReader)
