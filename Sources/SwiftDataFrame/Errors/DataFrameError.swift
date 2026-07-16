import Foundation

/// Errors thrown by SwiftDataFrame operations.
public enum DataFrameError: Error, Sendable, Equatable {

    // MARK: – Schema
    case emptySchema
    case columnLengthMismatch(expected: Int, got: Int, column: String)
    case columnNotFound(String)
    case duplicateColumnName(String)
    case typeMismatch(column: String, expected: String, got: String)

    // MARK: – I/O
    case fileNotFound(URL)
    case parseError(line: Int, description: String)
    case unsupportedFormat(String)
    case writeError(String)

    // MARK: – Operations
    case indexOutOfRange(index: Int, count: Int)
    case invalidSampleSize(requested: Int, available: Int)
    case emptyDataFrame(operation: String)
    case castFailed(column: String, targetType: String)
}

extension DataFrameError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .emptySchema:
            return "Cannot create DataFrame from empty list of columns."
        case let .columnLengthMismatch(expected, got, column):
            return "Column '\(column)' has \(got) elements, expected \(expected)."
        case let .columnNotFound(name):
            return "Column '\(name)' not found in DataFrame."
        case let .duplicateColumnName(name):
            return "Duplicate column name '\(name)'."
        case let .typeMismatch(col, expected, got):
            return "Type mismatch in '\(col)': expected \(expected), got \(got)."
        case let .fileNotFound(url):
            return "File not found: \(url.path)."
        case let .parseError(line, desc):
            return "Parse error at line \(line): \(desc)."
        case let .unsupportedFormat(fmt):
            return "Unsupported format: \(fmt)."
        case let .writeError(msg):
            return "Write error: \(msg)."
        case let .indexOutOfRange(index, count):
            return "Index \(index) is out of range (count: \(count))."
        case let .invalidSampleSize(req, avail):
            return "Requested sample size \(req) exceeds available rows \(avail)."
        case let .emptyDataFrame(op):
            return "Cannot perform '\(op)' on an empty DataFrame."
        case let .castFailed(col, type):
            return "Cannot cast column '\(col)' to \(type)."
        }
    }
}
