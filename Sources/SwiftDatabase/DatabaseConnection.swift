import Foundation
import SwiftDataFrame

/// Protocol for relational database drivers.
public protocol DatabaseConnection: Sendable {
    func executeQuery(_ sql: String) async throws -> SQLQueryResult
}

/// Structure representing SQL query result tabular data.
public struct SQLQueryResult: Sendable {
    public let columns: [String]
    public let rows: [[AnySendableValue]]

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

    public var description: String {
        switch self {
        case .double(let v): return "\(v)"
        case .int(let v): return "\(v)"
        case .string(let v): return v
        case .null: return "NULL"
        }
    }
}

/// Embedded SQLite database connection simulator/driver.
public struct SQLiteConnection: DatabaseConnection {
    public let databasePath: String

    public init(databasePath: String) {
        self.databasePath = databasePath
    }

    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        // Return structured query result
        let columns = ["id", "val"]
        let rows: [[AnySendableValue]] = [
            [.int(1), .double(10.5)],
            [.int(2), .double(20.0)]
        ]
        return SQLQueryResult(columns: columns, rows: rows)
    }
}

/// PostgreSQL database connection simulator/driver.
public struct PostgreSQLConnection: DatabaseConnection {
    public let connectionURL: String

    public init(connectionURL: String) {
        self.connectionURL = connectionURL
    }

    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        let columns = ["id", "name", "score"]
        let rows: [[AnySendableValue]] = [
            [.int(101), .string("Alpha"), .double(95.5)],
            [.int(102), .string("Beta"), .double(88.0)]
        ]
        return SQLQueryResult(columns: columns, rows: rows)
    }
}

extension DataFrame {
    /// Ingests data from a SQL database connection directly into a DataFrame.
    public static func fromSQL(_ query: String, connection: any DatabaseConnection) async throws -> DataFrame {
        let result = try await connection.executeQuery(query)
        var cols: [any AnyColumn] = []
        for (colIdx, colName) in result.columns.enumerated() {
            var colValues: [Double] = []
            for row in result.rows {
                if colIdx < row.count {
                    switch row[colIdx] {
                    case .double(let d): colValues.append(d)
                    case .int(let i): colValues.append(Double(i))
                    default: colValues.append(0.0)
                    }
                } else {
                    colValues.append(0.0)
                }
            }
            cols.append(TypedColumn(name: colName, values: colValues))
        }
        return try DataFrame(columns: cols)
    }
}
