import Foundation
import SQLite3
import SwiftDataFrame

/// Errors specific to database operations.
public enum DatabaseError: Error, LocalizedError {
    case connectionFailed(String)
    case queryFailed(String)
    case notImplemented(String)

    /// The error description.
    public var errorDescription: String? {
        switch self {
        case .connectionFailed(let msg):
            return "Database connection failed: \(msg)"
        case .queryFailed(let msg):
            return "SQL query execution failed: \(msg)"
        case .notImplemented(let feature):
            return "Feature not implemented: \(feature)"
        }
    }
}

/// Protocol for relational database drivers.
public protocol DatabaseConnection: Sendable {
    func executeQuery(_ sql: String) async throws -> SQLQueryResult
}

/// Structure representing SQL query result tabular data.
public struct SQLQueryResult: Sendable {
    /// The columns.
    public let columns: [String]
    /// The rows.
    public let rows: [[AnySendableValue]]

    /// Creates a new instance.
    /// - Parameters:
    ///   - columns: The columns.
    ///   - rows: The rows.
    public init(columns: [String], rows: [[AnySendableValue]]) {
        self.columns = columns
        self.rows = rows
    }
}

/// Type-safe wrapper for SQL query values.
public enum AnySendableValue: Sendable, CustomStringConvertible {
    case double(Double)
    case int(Int)
    case string(String)
    case null

    /// The description.
    public var description: String {
        switch self {
        case .double(let v): return "\(v)"
        case .int(let v): return "\(v)"
        case .string(let v): return v
        case .null: return "NULL"
        }
    }
}

/// Embedded SQLite database driver executing real SQL statements using SQLite C library.
public final class SQLiteConnection: DatabaseConnection, @unchecked Sendable {
    /// The database path.
    public let databasePath: String
    private let actualPath: String

    /// Creates a new instance.
    /// - Parameters:
    ///   - databasePath: The database path.
    public init(databasePath: String) {
        self.databasePath = databasePath
        if databasePath == ":memory:" {
            self.actualPath = NSTemporaryDirectory() + "swiftsci_\(UUID().uuidString).sqlite"
        } else if databasePath.contains("mode=memory") {
            let hashStr = String(abs(databasePath.hashValue))
            self.actualPath = NSTemporaryDirectory() + "swiftsci_\(hashStr).sqlite"
        } else {
            self.actualPath = databasePath
        }
    }

    deinit {
        if databasePath == ":memory:" || databasePath.contains("mode=memory") {
            try? FileManager.default.removeItem(atPath: actualPath)
        }
    }

    /// Execute query.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `SQLQueryResult` result.
    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI
        if sqlite3_open_v2(actualPath, &db, flags, nil) != SQLITE_OK {
            let msg = db != nil ? String(cString: sqlite3_errmsg(db)) : "Unknown error"
            if let db { sqlite3_close(db) }
            throw DatabaseError.connectionFailed("\(databasePath): \(msg)")
        }
        defer { sqlite3_close(db) }

        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) != SQLITE_OK {
            let msg = String(cString: sqlite3_errmsg(db))
            throw DatabaseError.queryFailed(msg)
        }
        defer { sqlite3_finalize(stmt) }

        let colCount = sqlite3_column_count(stmt)
        var columns: [String] = []
        columns.reserveCapacity(Int(colCount))
        for i in 0..<colCount {
            if let namePtr = sqlite3_column_name(stmt, i) {
                columns.append(String(cString: namePtr))
            } else {
                columns.append("col_\(i)")
            }
        }

        var rows: [[AnySendableValue]] = []
        while true {
            let stepResult = sqlite3_step(stmt)
            if stepResult == SQLITE_ROW {
                var row: [AnySendableValue] = []
                row.reserveCapacity(Int(colCount))
                for i in 0..<colCount {
                    let colType = sqlite3_column_type(stmt, i)
                    switch colType {
                    case SQLITE_INTEGER:
                        let val = sqlite3_column_int64(stmt, i)
                        row.append(.int(Int(val)))
                    case SQLITE_FLOAT:
                        let val = sqlite3_column_double(stmt, i)
                        row.append(.double(val))
                    case SQLITE_TEXT:
                        if let textPtr = sqlite3_column_text(stmt, i) {
                            row.append(.string(String(cString: textPtr)))
                        } else {
                            row.append(.string(""))
                        }
                    case SQLITE_NULL:
                        row.append(.null)
                    default:
                        if let textPtr = sqlite3_column_text(stmt, i) {
                            row.append(.string(String(cString: textPtr)))
                        } else {
                            row.append(.null)
                        }
                    }
                }
                rows.append(row)
            } else if stepResult == SQLITE_DONE {
                break
            } else {
                let msg = String(cString: sqlite3_errmsg(db))
                throw DatabaseError.queryFailed(msg)
            }
        }

        return SQLQueryResult(columns: columns, rows: rows)
    }
}

/// PostgreSQL database connection driver.
public final class PostgreSQLConnection: DatabaseConnection, @unchecked Sendable {
    /// The connection URL.
    public let connectionURL: String

    /// Creates a new instance.
    /// - Parameters:
    ///   - connectionURL: The connection URL.
    public init(connectionURL: String) {
        self.connectionURL = connectionURL
    }

    /// Executes a SQL query against PostgreSQL database connection.
    /// - Parameter sql: The SQL query statement.
    /// - Throws: DatabaseError.notImplemented for PostgreSQL driver integration.
    /// - Returns: A SQLQueryResult tabular result.
    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        guard !connectionURL.isEmpty else {
            throw DatabaseError.connectionFailed("Empty PostgreSQL connection URL")
        }
        guard !sql.isEmpty else {
            throw DatabaseError.queryFailed("SQL query cannot be empty")
        }
        throw DatabaseError.notImplemented("PostgreSQL native wire protocol driver requires libpq integration. Use SQLiteConnection for local embedded SQL databases or export models via CoreML/ONNX exporters.")
    }
}

extension DataFrame {
    /// Ingests data from a SQL database connection directly into a DataFrame.
    public static func fromSQL(_ query: String, connection: any DatabaseConnection) async throws -> DataFrame {
        let result = try await connection.executeQuery(query)
        var cols: [any AnyColumn] = []
        for (colIdx, colName) in result.columns.enumerated() {
            var hasString = false
            for row in result.rows {
                if colIdx < row.count, case .string = row[colIdx] {
                    hasString = true
                    break
                }
            }

            if hasString {
                var colValues: [String?] = []
                colValues.reserveCapacity(result.rows.count)
                for row in result.rows {
                    if colIdx < row.count {
                        switch row[colIdx] {
                        case .string(let s): colValues.append(s)
                        case .double(let d): colValues.append("\(d)")
                        case .int(let i): colValues.append("\(i)")
                        case .null: colValues.append(nil)
                        }
                    } else {
                        colValues.append(nil)
                    }
                }
                cols.append(TypedColumn(name: colName, values: colValues))
            } else {
                var colValues: [Double?] = []
                colValues.reserveCapacity(result.rows.count)
                for row in result.rows {
                    if colIdx < row.count {
                        switch row[colIdx] {
                        case .double(let d): colValues.append(d)
                        case .int(let i): colValues.append(Double(i))
                        case .string(let s): colValues.append(Double(s))
                        case .null: colValues.append(nil)
                        }
                    } else {
                        colValues.append(nil)
                    }
                }
                cols.append(TypedColumn(name: colName, values: colValues))
            }
        }
        return try DataFrame(columns: cols)
    }
}
