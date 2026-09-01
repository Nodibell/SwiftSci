import Foundation
import SQLite3
import SwiftDataFrame
#if canImport(CryptoKit)
import CryptoKit
#elseif canImport(CommonCrypto)
import CommonCrypto
#endif

/// Errors specific to database operations.
public enum DatabaseError: Error, LocalizedError {
    case connectionFailed(String)
    case queryFailed(String)
    case notImplemented(String)
    /// Thrown by `toSQL` when mode is `.failIfExists` and the table exists.
    case tableAlreadyExists(String)
    /// Thrown when the server requests an unsupported authentication method.
    case unsupportedAuth(String)

    /// The error description.
    public var errorDescription: String? {
        switch self {
        case .connectionFailed(let msg):
            return "Database connection failed: \(msg)"
        case .queryFailed(let msg):
            return "SQL query execution failed: \(msg)"
        case .notImplemented(let feature):
            return "Feature not implemented: \(feature)"
        case .tableAlreadyExists(let table):
            return "Table '\(table)' already exists (mode is .failIfExists)"
        case .unsupportedAuth(let method):
            return "Unsupported authentication method: \(method)"
        }
    }
}

/// Mode determining how existing tables and rows are handled during ``SwiftDataFrame/DataFrame/toSQL(table:connection:mode:batchSize:)``.
public enum SQLWriteMode: String, Sendable, Codable {
    /// Appends rows to the existing table, or creates the table if it does not exist.
    case append
    /// Drops the existing table if it exists, creates a fresh schema, and inserts rows.
    case replace
    /// Throws an error if the destination table already exists.
    case failIfExists
}

/// SSL/TLS encryption mode for remote relational database connections.
public enum SSLMode: String, Sendable, Codable {
    /// Plaintext TCP socket connection (no encryption).
    case disable
    /// Attempts encrypted TLS connection with fallback to plaintext.
    case prefer
    /// Enforces TLS encrypted connection (fails if server does not support TLS).
    case require
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

/// Embedded SQLite database driver executing real SQL statements using the SQLite3 C API.
///
/// ## Thread Safety
/// Serialized through actor isolation, guaranteeing data-race safety across concurrent tasks.
public actor SQLiteConnection: DatabaseConnection {
    /// The database path.
    nonisolated public let databasePath: String
    private let actualPath: String

    /// Creates a new SQLiteConnection.
    /// - Parameter databasePath: Path to the SQLite file, or `":memory:"` for an in-memory DB.
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

    /// Executes a SQL statement and returns the result set.
    ///
    /// - Parameter sql: A valid SQLite SQL statement.
    /// - Throws: `DatabaseError.connectionFailed` or `DatabaseError.queryFailed`.
    /// - Returns: `SQLQueryResult` with column names and typed rows.
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
///
/// ## Supported authentication
/// - `trust` (no password required)
/// - `md5` (password hashed with MD5 + salt)
///
/// ## Thread Safety
/// Isolated by actor, safe for concurrent async execution.
public actor PostgreSQLConnection: DatabaseConnection {
    /// The connection URL.
    nonisolated public let connectionURL: String
    /// The database host.
    nonisolated public let host: String
    /// The database port.
    nonisolated public let port: Int
    /// The database user.
    nonisolated public let user: String
    /// The database password.
    nonisolated public let password: String
    /// The database name.
    nonisolated public let database: String
    /// The SSL/TLS connection mode.
    nonisolated public let sslMode: SSLMode

    /// Creates a new PostgreSQL connection instance.
    /// - Parameters:
    ///   - connectionURL: Connection URL (e.g. `postgres://user:pass@host:5432/dbname?sslmode=require`).
    ///   - sslMode: Explicit SSL mode override (if `nil`, parsed from URL or defaults to `.disable`).
    public init(connectionURL: String, sslMode: SSLMode? = nil) {
        self.connectionURL = connectionURL
        let parsed = Self.parseURL(connectionURL)
        self.host = parsed.host
        self.port = parsed.port
        self.user = parsed.user
        self.password = parsed.password
        self.database = parsed.database
        self.sslMode = sslMode ?? parsed.sslMode
    }

    private static func parseURL(_ urlStr: String) -> (host: String, port: Int, user: String, password: String, database: String, sslMode: SSLMode) {
        guard let url = URL(string: urlStr) else {
            return ("127.0.0.1", 5432, "postgres", "", "postgres", .disable)
        }
        let h = url.host ?? "127.0.0.1"
        let p = url.port ?? 5432
        let u = url.user ?? "postgres"
        let pass = url.password ?? ""
        var db = url.path
        if db.hasPrefix("/") { db.removeFirst() }
        if db.isEmpty { db = "postgres" }

        var parsedSSL: SSLMode = .disable
        if let query = url.query?.lowercased() {
            if query.contains("sslmode=require") || query.contains("ssl=true") || query.contains("ssl=require") {
                parsedSSL = .require
            } else if query.contains("sslmode=prefer") {
                parsedSSL = .prefer
            }
        }
        return (h, p, u, pass, db, parsedSSL)
    }

    /// Executes a SQL query against the PostgreSQL database.
    ///
    /// - Parameter sql: The SQL statement to execute.
    /// - Throws: `DatabaseError` on connection, auth, or query failure.
    /// - Returns: `SQLQueryResult` with column names and typed rows.
    public func executeQuery(_ sql: String) async throws -> SQLQueryResult {

        guard !connectionURL.isEmpty else {
            throw DatabaseError.connectionFailed("Empty PostgreSQL connection URL")
        }
        guard !sql.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DatabaseError.queryFailed("SQL query cannot be empty")
        }

        // Establish TCP connection
        let sockfd = socket(AF_INET, SOCK_STREAM, 0)
        guard sockfd >= 0 else {
            throw DatabaseError.connectionFailed("Failed to create socket for PostgreSQL host: \(host)")
        }
        defer { close(sockfd) }

        var tv = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var serv_addr = sockaddr_in()
        serv_addr.sin_family = sa_family_t(AF_INET)
        serv_addr.sin_port = UInt16(port).bigEndian
        var hostEntry = in_addr()
        if inet_pton(AF_INET, host, &hostEntry) <= 0 {
            guard let hostent = gethostbyname(host), let addrList = hostent.pointee.h_addr_list else {
                throw DatabaseError.connectionFailed("Cannot resolve PostgreSQL host: \(host)")
            }
            memcpy(&serv_addr.sin_addr, addrList[0]!, MemoryLayout<in_addr>.size)
        } else {
            serv_addr.sin_addr = hostEntry
        }
        let connectRes = withUnsafePointer(to: &serv_addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sockfd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connectRes == 0 else {
            throw DatabaseError.connectionFailed("Could not connect to PostgreSQL at \(host):\(port)")
        }

        // --- Helper: recv all bytes until ReadyForQuery ('Z') ---
        func recvAll() throws -> Data {
            var allData = Data()
            var chunk = [UInt8](repeating: 0, count: 65_536) // 64KB chunks
            while true {
                let n = recv(sockfd, &chunk, chunk.count, 0)
                guard n > 0 else { break }
                allData.append(contentsOf: chunk.prefix(n))
                // 'Z' = ReadyForQuery — signals end of server response
                if chunk.prefix(n).contains(0x5A) { break }
            }
            return allData
        }

        // --- Send StartupMessage v3.0 ---
        var startup = Data()
        startup.append(contentsOf: [0x00, 0x03, 0x00, 0x00])
        startup.append(contentsOf: "user\0\(user)\0".utf8)
        startup.append(contentsOf: "database\0\(database)\0".utf8)
        startup.append(0x00)
        var packet = Data()
        withUnsafeBytes(of: Int32(startup.count + 4).bigEndian) { packet.append(contentsOf: $0) }
        packet.append(startup)
        _ = packet.withUnsafeBytes { send(sockfd, $0.baseAddress, packet.count, 0) }

        // --- Handle authentication ---
        let authData = try recvAll()
        if authData.isEmpty {
            throw DatabaseError.connectionFailed("Server closed connection during handshake")
        }
        let authBytes = [UInt8](authData)
        if authBytes[0] == 0x45 { // 'E'
            throw DatabaseError.connectionFailed("PostgreSQL rejected connection: \(String(decoding: authBytes, as: UTF8.self))")
        }
        if authBytes[0] == 0x52 { // 'R' AuthenticationRequest
            // authType is at bytes [5..8]
            guard authBytes.count >= 9 else {
                throw DatabaseError.connectionFailed("Truncated authentication packet")
            }
            let authType = Int32(authBytes[5]) << 24 | Int32(authBytes[6]) << 16
                         | Int32(authBytes[7]) << 8  | Int32(authBytes[8])
            switch authType {
            case 0: // AuthenticationOk — trust mode
                break
            case 5: // MD5 password
                guard authBytes.count >= 13 else {
                    throw DatabaseError.connectionFailed("MD5 auth: missing salt")
                }
                let salt = Array(authBytes[9..<13])
                let md5Password = pgMD5Password(password: password, user: user, salt: salt)
                var pwPacket = Data([0x70]) // 'p'
                let payload = md5Password + "\0"
                withUnsafeBytes(of: Int32(payload.utf8.count + 4).bigEndian) { pwPacket.append(contentsOf: $0) }
                pwPacket.append(contentsOf: payload.utf8)
                _ = pwPacket.withUnsafeBytes { send(sockfd, $0.baseAddress, pwPacket.count, 0) }
                let md5Response = try recvAll()
                let md5Bytes = [UInt8](md5Response)
                if md5Bytes.first == 0x45 { // 'E'
                    throw DatabaseError.connectionFailed("PostgreSQL MD5 auth failed")
                }
            default:
                throw DatabaseError.unsupportedAuth(
                    "PostgreSQL auth type \(authType) is not supported. " +
                    "Configure pg_hba.conf to use 'trust' or 'md5'."
                )
            }
        }

        // --- Send Query ---
        var queryPacket = Data([0x51]) // 'Q'
        let qPayload = sql.utf8 + [0x00]
        withUnsafeBytes(of: Int32(qPayload.count + 4).bigEndian) { queryPacket.append(contentsOf: $0) }
        queryPacket.append(contentsOf: qPayload)
        _ = queryPacket.withUnsafeBytes { send(sockfd, $0.baseAddress, queryPacket.count, 0) }

        // --- Read full response ---
        let responseData = try recvAll()
        let buf = [UInt8](responseData)
        let totalRead = buf.count

        var columns: [String] = []
        var rows: [[AnySendableValue]] = []
        var offset = 0

        while offset < totalRead {
            guard offset + 5 <= totalRead else { break }
            let msgType = buf[offset]
            let msgLen  = Int(UInt32(buf[offset+1]) << 24 | UInt32(buf[offset+2]) << 16
                            | UInt32(buf[offset+3]) << 8  | UInt32(buf[offset+4]))
            let msgEnd  = offset + 1 + msgLen
            guard msgEnd <= totalRead else { break }

            switch msgType {
            case 0x45: // 'E' ErrorResponse
                let errText = String(decoding: buf[offset..<msgEnd], as: UTF8.self)
                throw DatabaseError.queryFailed("PostgreSQL error: \(errText)")

            case 0x54: // 'T' RowDescription
                guard offset + 7 <= totalRead else { break }
                let numFields = Int(UInt16(buf[offset + 5]) << 8 | UInt16(buf[offset + 6]))
                var fOff = offset + 7
                for _ in 0..<numFields {
                    guard fOff < msgEnd else { break }
                    var nameBytes: [UInt8] = []
                    while fOff < msgEnd && buf[fOff] != 0 {
                        nameBytes.append(buf[fOff]); fOff += 1
                    }
                    fOff += 1   // null terminator
                    fOff += 18  // OID(4) + attrNum(2) + typeOID(4) + typeLen(2) + typeMod(4) + format(2)
                    columns.append(String(decoding: nameBytes, as: UTF8.self))
                }

            case 0x44: // 'D' DataRow
                guard offset + 7 <= totalRead else { break }
                let numCols = Int(UInt16(buf[offset + 5]) << 8 | UInt16(buf[offset + 6]))
                var dOff = offset + 7
                var row: [AnySendableValue] = []
                for _ in 0..<numCols {
                    guard dOff + 4 <= totalRead else { break }
                    let colLen = Int(Int32(bitPattern:
                        UInt32(buf[dOff]) << 24 | UInt32(buf[dOff+1]) << 16 |
                        UInt32(buf[dOff+2]) << 8 | UInt32(buf[dOff+3])))
                    dOff += 4
                    if colLen < 0 {
                        row.append(.null)
                    } else {
                        guard dOff + colLen <= totalRead else { break }
                        let valStr = String(decoding: buf[dOff..<(dOff + colLen)], as: UTF8.self)
                        dOff += colLen
                        if let i = Int(valStr)         { row.append(.int(i)) }
                        else if let d = Double(valStr) { row.append(.double(d)) }
                        else                           { row.append(.string(valStr)) }
                    }
                }
                rows.append(row)

            default: break
            }
            offset = msgEnd
        }

        return SQLQueryResult(columns: columns, rows: rows)
    }

    // MARK: - MD5 password helper

    /// Computes the PostgreSQL MD5 password string:
    /// `"md5" + md5( md5(password + user) + salt )`
    private func pgMD5Password(password: String, user: String, salt: [UInt8]) -> String {
        func md5Hex(_ input: [UInt8]) -> String {
            #if canImport(CryptoKit)
            let digest = Insecure.MD5.hash(data: Data(input))
            return digest.map { String(format: "%02x", $0) }.joined()
            #elseif canImport(CommonCrypto)
            var digest = [UInt8](repeating: 0, count: 16)
            var ctx = CC_MD5_CTX()
            CC_MD5_Init(&ctx)
            _ = input.withUnsafeBytes { CC_MD5_Update(&ctx, $0.baseAddress, CC_LONG($0.count)) }
            CC_MD5_Final(&digest, &ctx)
            return digest.map { String(format: "%02x", $0) }.joined()
            #else
            return ""
            #endif
        }
        let inner = md5Hex(Array((password + user).utf8))
        let outer = md5Hex(Array(inner.utf8) + salt)
        return "md5" + outer
    }
}

// MARK: - Native MySQL Wire Protocol Driver (Pure Swift)

/// Native MySQL database connection driver using MySQL Client/Server Protocol 4.1+.
///
/// ## Thread Safety
/// Isolated by actor, safe for concurrent async execution.
public actor MySQLConnection: DatabaseConnection {
    /// The connection URL.
    nonisolated public let connectionURL: String
    /// The database host.
    nonisolated public let host: String
    /// The database port.
    nonisolated public let port: Int
    /// The database user.
    nonisolated public let user: String
    /// The database password.
    nonisolated public let password: String
    /// The database name.
    nonisolated public let database: String
    /// The SSL/TLS connection mode.
    nonisolated public let sslMode: SSLMode

    /// Creates a new MySQL connection instance.
    /// - Parameters:
    ///   - connectionURL: Connection URL (e.g. `mysql://user:pass@host:3306/dbname?ssl=true`).
    ///   - sslMode: Explicit SSL mode override (if `nil`, parsed from URL or defaults to `.disable`).
    public init(connectionURL: String, sslMode: SSLMode? = nil) {
        self.connectionURL = connectionURL
        let parsed = Self.parseURL(connectionURL)
        self.host = parsed.host
        self.port = parsed.port
        self.user = parsed.user
        self.password = parsed.password
        self.database = parsed.database
        self.sslMode = sslMode ?? parsed.sslMode
    }

    private static func parseURL(_ urlStr: String) -> (host: String, port: Int, user: String, password: String, database: String, sslMode: SSLMode) {
        guard let url = URL(string: urlStr) else {
            return ("127.0.0.1", 3306, "root", "", "mysql", .disable)
        }
        let h = url.host ?? "127.0.0.1"
        let p = url.port ?? 3306
        let u = url.user ?? "root"
        let pass = url.password ?? ""
        var db = url.path
        if db.hasPrefix("/") { db.removeFirst() }
        if db.isEmpty { db = "mysql" }

        var parsedSSL: SSLMode = .disable
        if let query = url.query?.lowercased() {
            if query.contains("sslmode=require") || query.contains("ssl=true") || query.contains("ssl=require") {
                parsedSSL = .require
            } else if query.contains("sslmode=prefer") {
                parsedSSL = .prefer
            }
        }
        return (h, p, u, pass, db, parsedSSL)
    }

    /// Executes a SQL query against the MySQL database.
    ///
    /// - Parameter sql: The SQL statement to execute.
    /// - Throws: `DatabaseError` on connection or query failure.
    /// - Returns: `SQLQueryResult` with real column names and typed rows.
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

        var tv = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var serv_addr = sockaddr_in()
        serv_addr.sin_family = sa_family_t(AF_INET)
        serv_addr.sin_port = UInt16(port).bigEndian
        var hostEntry = in_addr()
        if inet_pton(AF_INET, host, &hostEntry) <= 0 {
            guard let hostent = gethostbyname(host), let addrList = hostent.pointee.h_addr_list else {
                throw DatabaseError.connectionFailed("Cannot resolve MySQL host: \(host)")
            }
            memcpy(&serv_addr.sin_addr, addrList[0]!, MemoryLayout<in_addr>.size)
        } else {
            serv_addr.sin_addr = hostEntry
        }
        let connectRes = withUnsafePointer(to: &serv_addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sockfd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connectRes == 0 else {
            throw DatabaseError.connectionFailed("Could not connect to MySQL at \(host):\(port)")
        }

        // Helper: read a single MySQL packet [len(3) + seq(1) + payload]
        func readPacket() -> Data? {
            var header = [UInt8](repeating: 0, count: 4)
            guard recv(sockfd, &header, 4, MSG_WAITALL) == 4 else { return nil }
            let payloadLen = Int(header[0]) | Int(header[1]) << 8 | Int(header[2]) << 16
            guard payloadLen > 0 else { return Data() }
            var payload = [UInt8](repeating: 0, count: payloadLen)
            guard recv(sockfd, &payload, payloadLen, MSG_WAITALL) == payloadLen else { return nil }
            return Data(payload)
        }

        // Helper: read null-terminated string from Data at offset
        func readCString(_ data: Data, from offset: inout Int) -> String {
            var bytes: [UInt8] = []
            while offset < data.count && data[offset] != 0 {
                bytes.append(data[offset]); offset += 1
            }
            offset += 1 // skip null
            return String(decoding: bytes, as: UTF8.self)
        }

        // --- Read Server Greeting ---
        guard let greeting = readPacket(), greeting.count > 1 else {
            throw DatabaseError.connectionFailed("MySQL handshake: no greeting received")
        }
        if greeting[0] == 0xFF {
            throw DatabaseError.connectionFailed("MySQL error in greeting")
        }

        // --- Send Handshake Response ---
        var responsePayload = Data()
        responsePayload.append(contentsOf: [0x85, 0xA2, 0x1E, 0x00]) // capabilities
        responsePayload.append(contentsOf: [0x00, 0x00, 0x00, 0x01]) // max_packet_size
        responsePayload.append(33)                                    // charset utf8
        responsePayload.append(contentsOf: [UInt8](repeating: 0, count: 23))
        responsePayload.append(contentsOf: "\(user)\0".utf8)
        responsePayload.append(0x00) // auth_response length (empty password)
        if !database.isEmpty {
            responsePayload.append(contentsOf: "\(database)\0".utf8)
        }
        let pLen = responsePayload.count
        var handshakePacket = Data([
            UInt8(pLen & 0xFF), UInt8((pLen >> 8) & 0xFF), UInt8((pLen >> 16) & 0xFF), 1
        ])
        handshakePacket.append(responsePayload)
        _ = handshakePacket.withUnsafeBytes { send(sockfd, $0.baseAddress, handshakePacket.count, 0) }

        guard let authResp = readPacket(), authResp.count > 0 else {
            throw DatabaseError.connectionFailed("No response to MySQL auth")
        }
        if authResp[0] == 0xFF {
            throw DatabaseError.connectionFailed("MySQL auth failed")
        }

        // --- Send COM_QUERY ---
        var queryPayload = Data([0x03])
        queryPayload.append(contentsOf: sql.utf8)
        let qLen = queryPayload.count
        var queryPacket = Data([UInt8(qLen & 0xFF), UInt8((qLen >> 8) & 0xFF), UInt8((qLen >> 16) & 0xFF), 0])
        queryPacket.append(queryPayload)
        _ = queryPacket.withUnsafeBytes { send(sockfd, $0.baseAddress, queryPacket.count, 0) }

        // --- Read Result Set ---
        guard let firstPacket = readPacket(), firstPacket.count > 0 else {
            return SQLQueryResult(columns: [], rows: [])
        }
        // 0x00 = OK, 0xFF = ERR, 0xFE = EOF, otherwise = column_count
        if firstPacket[0] == 0xFF {
            let errMsg = firstPacket.count > 3 ? String(decoding: firstPacket[3...], as: UTF8.self) : "Unknown"
            throw DatabaseError.queryFailed("MySQL error: \(errMsg)")
        }
        if firstPacket[0] == 0x00 {
            // OK packet — DDL or INSERT/UPDATE with no result set
            return SQLQueryResult(columns: [], rows: [])
        }
        let columnCount = Int(firstPacket[0])

        // --- Read ColumnDefinition packets ---
        var columns: [String] = []
        for _ in 0..<columnCount {
            guard let colDef = readPacket(), colDef.count > 4 else { break }
            // ColumnDefinition41: catalog(lenc) db(lenc) table(lenc) org_table(lenc) name(lenc) ...
            var off = 0
            func skipLenc() {
                guard off < colDef.count else { return }
                let b = colDef[off]; off += 1
                if b < 0xFB { off += Int(b) }
                else if b == 0xFC { off += 2 + 2 }
                else if b == 0xFD { off += 3 + 3 }
                else { off += 8 + 8 }
            }
            func readLencStr() -> String {
                guard off < colDef.count else { return "" }
                let b = colDef[off]; off += 1
                var len = 0
                if b < 0xFB { len = Int(b) }
                else if b == 0xFC { guard off + 2 <= colDef.count else { return "" }; len = Int(colDef[off]) | Int(colDef[off+1]) << 8; off += 2 }
                else if b == 0xFD { guard off + 3 <= colDef.count else { return "" }; len = Int(colDef[off]) | Int(colDef[off+1]) << 8 | Int(colDef[off+2]) << 16; off += 3 }
                guard off + len <= colDef.count else { return "" }
                let s = String(decoding: colDef[off..<(off+len)], as: UTF8.self)
                off += len
                return s
            }
            skipLenc()  // catalog
            skipLenc()  // schema
            skipLenc()  // table (virtual)
            skipLenc()  // org_table
            let colName = readLencStr() // name
            columns.append(colName.isEmpty ? "col_\(columns.count + 1)" : colName)
        }

        // Read EOF after column definitions
        _ = readPacket()

        // --- Read Row packets until EOF ---
        var rows: [[AnySendableValue]] = []
        while let rowPacket = readPacket() {
            guard rowPacket.count > 0 else { break }
            // 0xFE = EOF packet (len < 9)
            if rowPacket[0] == 0xFE && rowPacket.count < 9 { break }
            // 0xFF = ERR packet
            if rowPacket[0] == 0xFF {
                let errMsg = rowPacket.count > 3 ? String(decoding: rowPacket[3...], as: UTF8.self) : "error"
                throw DatabaseError.queryFailed("MySQL row error: \(errMsg)")
            }

            var off = 0
            var row: [AnySendableValue] = []
            for _ in 0..<columnCount {
                guard off < rowPacket.count else { row.append(.null); continue }
                let b = rowPacket[off]
                if b == 0xFB { // NULL
                    off += 1; row.append(.null); continue
                }
                off += 1
                var len = 0
                if b < 0xFB { len = Int(b) }
                else if b == 0xFC { guard off + 2 <= rowPacket.count else { break }; len = Int(rowPacket[off]) | Int(rowPacket[off+1]) << 8; off += 2 }
                else if b == 0xFD { guard off + 3 <= rowPacket.count else { break }; len = Int(rowPacket[off]) | Int(rowPacket[off+1]) << 8 | Int(rowPacket[off+2]) << 16; off += 3 }
                guard off + len <= rowPacket.count else { break }
                let valStr = String(decoding: rowPacket[off..<(off+len)], as: UTF8.self)
                off += len
                if let i = Int(valStr)         { row.append(.int(i)) }
                else if let d = Double(valStr) { row.append(.double(d)) }
                else                           { row.append(.string(valStr)) }
            }
            rows.append(row)
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

    /// Writes the contents of this DataFrame into a relational SQL database table.
    ///
    /// - Parameters:
    ///   - table: Destination table name.
    ///   - connection: Target database driver conforming to ``DatabaseConnection``.
    ///   - mode: Write mode determining how existing tables are handled (default ``SQLWriteMode/append``).
    ///   - batchSize: Maximum number of rows per batch `INSERT` statement (default `500`).
    /// - Throws: ``DatabaseError`` if table creation or insertion fails.
    public func toSQL(
        table: String,
        connection: any DatabaseConnection,
        mode: SQLWriteMode = .append,
        batchSize: Int = 500
    ) async throws {
        guard !table.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DatabaseError.queryFailed("Destination table name cannot be empty")
        }
        guard !columns.isEmpty else { return }

        let escapedTableName = "\"\(table.replacingOccurrences(of: "\"", with: "\"\""))\""
        let colNames = columns.map { $0.name }
        let escapedColNames = colNames.map { "\"\(String($0).replacingOccurrences(of: "\"", with: "\"\""))\"" }.joined(separator: ", ")

        // Handle table creation & modes
        if mode == .replace {
            _ = try await connection.executeQuery("DROP TABLE IF EXISTS \(escapedTableName);")
        }

        // Generate schema columns based on DataFrame types
        var colDefs: [String] = []
        for col in columns {
            let colNameEscaped = "\"\(col.name.replacingOccurrences(of: "\"", with: "\"\""))\""
            let colType: String
            switch col.dtype {
            case .int32, .int64:
                colType = "INTEGER"
            case .float32, .float64:
                colType = "REAL"
            case .boolean:
                colType = "BOOLEAN"
            default:
                colType = "TEXT"
            }
            colDefs.append("\(colNameEscaped) \(colType)")
        }

        if mode == .failIfExists {
            // Use sqlite_master for SQLite; for PostgreSQL/MySQL this query
            // will fail (table not found) which is the correct "doesn't exist" signal.
            let checkSQL: String
            if connection is SQLiteConnection {
                checkSQL = "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='\(table.replacingOccurrences(of: "'", with: "''"))'"
            } else {
                // ANSI information_schema (PostgreSQL 8+, MySQL 5+)
                checkSQL = "SELECT COUNT(*) as cnt FROM information_schema.tables WHERE table_name='\(table.replacingOccurrences(of: "'", with: "''"))'"
            }
            let checkResult = try await connection.executeQuery(checkSQL)
            if let firstRow = checkResult.rows.first,
               let firstVal = firstRow.first,
               case .int(let cnt) = firstVal, cnt > 0 {
                throw DatabaseError.tableAlreadyExists(table)
            }
        }

        let createTableSQL = "CREATE TABLE IF NOT EXISTS \(escapedTableName) (\(colDefs.joined(separator: ", ")));"
        _ = try await connection.executeQuery(createTableSQL)

        // Insert rows in batches
        let nRows = self.shape.rows
        guard nRows > 0 else { return }

        let effectiveBatchSize = max(1, batchSize)
        var rowStart = 0
        while rowStart < nRows {
            let rowEnd = min(rowStart + effectiveBatchSize, nRows)
            var valuesClauses: [String] = []
            valuesClauses.reserveCapacity(rowEnd - rowStart)

            for r in rowStart..<rowEnd {
                var rowVals: [String] = []
                rowVals.reserveCapacity(columns.count)
                for col in columns {
                    if let v = col.value(at: r) {
                        if let d = v as? Double {
                            rowVals.append(d.isNaN ? "NULL" : "\(d)")
                        } else if let i = v as? Int {
                            rowVals.append("\(i)")
                        } else if let i64 = v as? Int64 {
                            rowVals.append("\(i64)")
                        } else if let b = v as? Bool {
                            rowVals.append(b ? "1" : "0")
                        } else {
                            let strVal = "\(v)".replacingOccurrences(of: "'", with: "''")
                            rowVals.append("'\(strVal)'")
                        }
                    } else {
                        rowVals.append("NULL")
                    }
                }
                valuesClauses.append("(\(rowVals.joined(separator: ", ")))")
            }

            let insertSQL = "INSERT INTO \(escapedTableName) (\(escapedColNames)) VALUES \(valuesClauses.joined(separator: ", "));"
            _ = try await connection.executeQuery(insertSQL)
            rowStart = rowEnd
        }
    }
}
