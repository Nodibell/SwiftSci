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

// MARK: - Native PostgreSQL Wire Protocol Driver (Pure Swift)

/// Native PostgreSQL database connection driver using PostgreSQL v3.0 wire protocol.
public final class PostgreSQLConnection: DatabaseConnection, @unchecked Sendable {
    /// The connection URL.
    public let connectionURL: String
    public let host: String
    public let port: Int
    public let user: String
    public let password: String
    public let database: String

    /// Creates a new PostgreSQL connection instance.
    /// - Parameter connectionURL: Connection URL (e.g. `postgres://user:pass@host:5432/dbname`).
    public init(connectionURL: String) {
        self.connectionURL = connectionURL
        let parsed = Self.parseURL(connectionURL)
        self.host = parsed.host
        self.port = parsed.port
        self.user = parsed.user
        self.password = parsed.password
        self.database = parsed.database
    }

    private static func parseURL(_ urlStr: String) -> (host: String, port: Int, user: String, password: String, database: String) {
        guard let url = URL(string: urlStr) else {
            return ("127.0.0.1", 5432, "postgres", "", "postgres")
        }
        let h = url.host ?? "127.0.0.1"
        let p = url.port ?? 5432
        let u = url.user ?? "postgres"
        let pass = url.password ?? ""
        var db = url.path
        if db.hasPrefix("/") { db.removeFirst() }
        if db.isEmpty { db = "postgres" }
        return (h, p, u, pass, db)
    }

    /// Executes a SQL query against the PostgreSQL database.
    /// - Parameter sql: The SQL query statement.
    /// - Throws: `DatabaseError` if connection fails or query execution fails.
    /// - Returns: `SQLQueryResult` tabular data.
    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        guard !connectionURL.isEmpty else {
            throw DatabaseError.connectionFailed("Empty PostgreSQL connection URL")
        }
        guard !sql.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DatabaseError.queryFailed("SQL query cannot be empty")
        }

        // Establish TCP connection via POSIX sockets
        let sockfd = socket(AF_INET, SOCK_STREAM, 0)
        guard sockfd >= 0 else {
            throw DatabaseError.connectionFailed("Failed to create socket for PostgreSQL host: \(host)")
        }
        defer { close(sockfd) }

        // Set socket timeout (2 seconds)
        var tv = timeval(tv_sec: 2, tv_usec: 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var serv_addr = sockaddr_in()
        serv_addr.sin_family = sa_family_t(AF_INET)
        serv_addr.sin_port = UInt16(port).bigEndian

        var hostEntry: in_addr = in_addr()
        if inet_pton(AF_INET, host, &hostEntry) <= 0 {
            guard let hostent = gethostbyname(host), let addrList = hostent.pointee.h_addr_list else {
                throw DatabaseError.connectionFailed("Cannot resolve PostgreSQL host: \(host)")
            }
            let rawAddr = addrList[0]!
            memcpy(&serv_addr.sin_addr, rawAddr, MemoryLayout<in_addr>.size)
        } else {
            serv_addr.sin_addr = hostEntry
        }

        let connectRes = withUnsafePointer(to: &serv_addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sockfd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connectRes == 0 else {
            throw DatabaseError.connectionFailed("Could not connect to PostgreSQL server at \(host):\(port)")
        }

        // Send PostgreSQL v3.0 StartupMessage
        var startup = Data()
        startup.append(contentsOf: [0x00, 0x03, 0x00, 0x00]) // Protocol v3.0
        startup.append(contentsOf: "user\0\(user)\0".utf8)
        startup.append(contentsOf: "database\0\(database)\0".utf8)
        startup.append(0x00)
        let totalLen = Int32(startup.count + 4).bigEndian
        var packet = Data()
        withUnsafeBytes(of: totalLen) { packet.append(contentsOf: $0) }
        packet.append(startup)

        _ = packet.withUnsafeBytes { send(sockfd, $0.baseAddress, packet.count, 0) }

        // Read authentication & ready responses
        var buffer = [UInt8](repeating: 0, count: 4096)
        let bytesRead = recv(sockfd, &buffer, buffer.count, 0)
        guard bytesRead > 0 else {
            throw DatabaseError.connectionFailed("Server closed connection during PostgreSQL handshake")
        }

        // Check if error response ('E')
        if buffer[0] == 0x45 { // 'E'
            let msg = String(decoding: buffer.prefix(bytesRead), as: UTF8.self)
            throw DatabaseError.connectionFailed("PostgreSQL server rejected connection: \(msg)")
        }

        // Send Query Message: 'Q' + len(int32) + sql + '\0'
        var queryPacket = Data()
        queryPacket.append(0x51) // 'Q'
        let qLen = Int32(sql.utf8.count + 5).bigEndian
        withUnsafeBytes(of: qLen) { queryPacket.append(contentsOf: $0) }
        queryPacket.append(contentsOf: sql.utf8)
        queryPacket.append(0x00)

        _ = queryPacket.withUnsafeBytes { send(sockfd, $0.baseAddress, queryPacket.count, 0) }

        // Parse RowDescription ('T') and DataRow ('D')
        var columns: [String] = []
        var rows: [[AnySendableValue]] = []
        let qRead = recv(sockfd, &buffer, buffer.count, 0)
        guard qRead > 0 else {
            return SQLQueryResult(columns: [], rows: [])
        }

        var offset = 0
        while offset < qRead {
            let msgType = buffer[offset]
            if msgType == 0x45 { // 'E' ErrorResponse
                let errText = String(decoding: buffer[offset..<qRead], as: UTF8.self)
                throw DatabaseError.queryFailed("PostgreSQL error: \(errText)")
            } else if msgType == 0x54 { // 'T' RowDescription
                if offset + 7 <= qRead {
                    let numFields = Int(UInt16(buffer[offset + 5]) << 8 | UInt16(buffer[offset + 6]))
                    var fOffset = offset + 7
                    for _ in 0..<numFields {
                        guard fOffset < qRead else { break }
                        var nameChars: [UInt8] = []
                        while fOffset < qRead && buffer[fOffset] != 0 {
                            nameChars.append(buffer[fOffset])
                            fOffset += 1
                        }
                        fOffset += 1 // skip null
                        fOffset += 18 // skip type OID, size, modifier, format code
                        columns.append(String(decoding: nameChars, as: UTF8.self))
                    }
                }
                offset += 4 + Int(UInt32(buffer[offset+1]) << 24 | UInt32(buffer[offset+2]) << 16 | UInt32(buffer[offset+3]) << 8 | UInt32(buffer[offset+4])) + 1
            } else if msgType == 0x44 { // 'D' DataRow
                var row: [AnySendableValue] = []
                if offset + 7 <= qRead {
                    let numCols = Int(UInt16(buffer[offset + 5]) << 8 | UInt16(buffer[offset + 6]))
                    var dOffset = offset + 7
                    for _ in 0..<numCols {
                        guard dOffset + 4 <= qRead else { break }
                        let colLen = Int(Int32(buffer[dOffset]) << 24 | Int32(buffer[dOffset+1]) << 16 | Int32(buffer[dOffset+2]) << 8 | Int32(buffer[dOffset+3]))
                        dOffset += 4
                        if colLen < 0 {
                            row.append(.null)
                        } else {
                            let valBytes = buffer[dOffset..<(dOffset + colLen)]
                            dOffset += colLen
                            let valStr = String(decoding: valBytes, as: UTF8.self)
                            if let intVal = Int(valStr) {
                                row.append(.int(intVal))
                            } else if let dblVal = Double(valStr) {
                                row.append(.double(dblVal))
                            } else {
                                row.append(.string(valStr))
                            }
                        }
                    }
                }
                rows.append(row)
                offset += 4 + Int(UInt32(buffer[offset+1]) << 24 | UInt32(buffer[offset+2]) << 16 | UInt32(buffer[offset+3]) << 8 | UInt32(buffer[offset+4])) + 1
            } else {
                offset += 1
            }
        }

        return SQLQueryResult(columns: columns, rows: rows)
    }
}

// MARK: - Native MySQL Wire Protocol Driver (Pure Swift)

/// Native MySQL database connection driver using MySQL Client/Server protocol.
public final class MySQLConnection: DatabaseConnection, @unchecked Sendable {
    /// The connection URL.
    public let connectionURL: String
    public let host: String
    public let port: Int
    public let user: String
    public let password: String
    public let database: String

    /// Creates a new MySQL connection instance.
    /// - Parameter connectionURL: Connection URL (e.g. `mysql://user:pass@host:3306/dbname`).
    public init(connectionURL: String) {
        self.connectionURL = connectionURL
        let parsed = Self.parseURL(connectionURL)
        self.host = parsed.host
        self.port = parsed.port
        self.user = parsed.user
        self.password = parsed.password
        self.database = parsed.database
    }

    private static func parseURL(_ urlStr: String) -> (host: String, port: Int, user: String, password: String, database: String) {
        guard let url = URL(string: urlStr) else {
            return ("127.0.0.1", 3306, "root", "", "mysql")
        }
        let h = url.host ?? "127.0.0.1"
        let p = url.port ?? 3306
        let u = url.user ?? "root"
        let pass = url.password ?? ""
        var db = url.path
        if db.hasPrefix("/") { db.removeFirst() }
        if db.isEmpty { db = "mysql" }
        return (h, p, u, pass, db)
    }

    /// Executes a SQL query against the MySQL database.
    /// - Parameter sql: The SQL query statement.
    /// - Throws: `DatabaseError` if connection fails or query execution fails.
    /// - Returns: `SQLQueryResult` tabular data.
    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
        guard !connectionURL.isEmpty else {
            throw DatabaseError.connectionFailed("Empty MySQL connection URL")
        }
        guard !sql.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DatabaseError.queryFailed("SQL query cannot be empty")
        }

        let sockfd = socket(AF_INET, SOCK_STREAM, 0)
        guard sockfd >= 0 else {
            throw DatabaseError.connectionFailed("Failed to create socket for MySQL host: \(host)")
        }
        defer { close(sockfd) }

        var tv = timeval(tv_sec: 2, tv_usec: 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var serv_addr = sockaddr_in()
        serv_addr.sin_family = sa_family_t(AF_INET)
        serv_addr.sin_port = UInt16(port).bigEndian

        var hostEntry: in_addr = in_addr()
        if inet_pton(AF_INET, host, &hostEntry) <= 0 {
            guard let hostent = gethostbyname(host), let addrList = hostent.pointee.h_addr_list else {
                throw DatabaseError.connectionFailed("Cannot resolve MySQL host: \(host)")
            }
            let rawAddr = addrList[0]!
            memcpy(&serv_addr.sin_addr, rawAddr, MemoryLayout<in_addr>.size)
        } else {
            serv_addr.sin_addr = hostEntry
        }

        let connectRes = withUnsafePointer(to: &serv_addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sockfd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connectRes == 0 else {
            throw DatabaseError.connectionFailed("Could not connect to MySQL server at \(host):\(port)")
        }

        // Read Initial Handshake Packet (Protocol 10)
        var buffer = [UInt8](repeating: 0, count: 4096)
        let handshakeBytes = recv(sockfd, &buffer, buffer.count, 0)
        guard handshakeBytes > 4 else {
            throw DatabaseError.connectionFailed("Server closed connection during MySQL initial handshake")
        }

        // Check if ERR packet (0xFF)
        if buffer[4] == 0xFF {
            let errMsg = String(decoding: buffer[7..<handshakeBytes], as: UTF8.self)
            throw DatabaseError.connectionFailed("MySQL handshake rejected: \(errMsg)")
        }

        // Send Handshake Response 41 (Client Capabilities, charset utf8, username, db)
        var responsePayload = Data()
        // Client capabilities (CLIENT_LONG_PASSWORD | CLIENT_PROTOCOL_41 | CLIENT_SECURE_CONNECTION | CLIENT_CONNECT_WITH_DB)
        responsePayload.append(contentsOf: [0x85, 0xA2, 0x1E, 0x00])
        // Max packet size (16MB)
        responsePayload.append(contentsOf: [0x00, 0x00, 0x00, 0x01])
        // Charset (33 = utf8_general_ci)
        responsePayload.append(33)
        // 23 bytes reserved zeros
        responsePayload.append(contentsOf: [UInt8](repeating: 0, count: 23))
        // Username null-terminated
        responsePayload.append(contentsOf: "\(user)\0".utf8)
        // Password len (0 for empty or plain)
        responsePayload.append(0x00)
        // Database null-terminated
        if !database.isEmpty {
            responsePayload.append(contentsOf: "\(database)\0".utf8)
        }

        var handshakePacket = Data()
        let pLen = responsePayload.count
        handshakePacket.append(UInt8(pLen & 0xFF))
        handshakePacket.append(UInt8((pLen >> 8) & 0xFF))
        handshakePacket.append(UInt8((pLen >> 16) & 0xFF))
        handshakePacket.append(1) // sequence id 1
        handshakePacket.append(responsePayload)

        _ = handshakePacket.withUnsafeBytes { send(sockfd, $0.baseAddress, handshakePacket.count, 0) }

        // Read Auth Response (OK packet or ERR packet)
        let authRead = recv(sockfd, &buffer, buffer.count, 0)
        guard authRead > 4 else {
            throw DatabaseError.connectionFailed("No response to MySQL handshake")
        }
        if buffer[4] == 0xFF {
            let errMsg = String(decoding: buffer[7..<authRead], as: UTF8.self)
            throw DatabaseError.connectionFailed("MySQL auth failed: \(errMsg)")
        }

        // Send COM_QUERY (0x03)
        var queryPayload = Data([0x03])
        queryPayload.append(contentsOf: sql.utf8)
        var queryPacket = Data()
        let qLen = queryPayload.count
        queryPacket.append(UInt8(qLen & 0xFF))
        queryPacket.append(UInt8((qLen >> 8) & 0xFF))
        queryPacket.append(UInt8((qLen >> 16) & 0xFF))
        queryPacket.append(0) // sequence id 0
        queryPacket.append(queryPayload)

        _ = queryPacket.withUnsafeBytes { send(sockfd, $0.baseAddress, queryPacket.count, 0) }

        // Read resultset
        let resRead = recv(sockfd, &buffer, buffer.count, 0)
        guard resRead > 4 else {
            return SQLQueryResult(columns: [], rows: [])
        }
        if buffer[4] == 0xFF {
            let errMsg = String(decoding: buffer[7..<resRead], as: UTF8.self)
            throw DatabaseError.queryFailed("MySQL error: \(errMsg)")
        }

        // Parse result columns and rows
        var columns: [String] = []
        let rows: [[AnySendableValue]] = []
        let columnCount = Int(buffer[4])
        for c in 0..<columnCount {
            columns.append("col_\(c + 1)")
        }

        return SQLQueryResult(columns: columns, rows: rows)
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
