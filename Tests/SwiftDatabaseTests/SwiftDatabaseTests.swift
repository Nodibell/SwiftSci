import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftDatabase

@Suite("SwiftDatabase Tests (Phase 5)")
struct SwiftDatabaseTests {

    @Test("Test real SQLite query execution and DataFrame ingestion")
    func testSQLiteDataFrameIngestion() async throws {
        let dbURI = "file:test_\(UUID().uuidString)?mode=memory&cache=shared"
        let conn = SQLiteConnection(databasePath: dbURI)
        _ = try await conn.executeQuery("CREATE TABLE test (id INTEGER PRIMARY KEY, val REAL, name TEXT);")
        _ = try await conn.executeQuery("INSERT INTO test (id, val, name) VALUES (1, 10.5, 'Alpha');")
        _ = try await conn.executeQuery("INSERT INTO test (id, val, name) VALUES (2, 20.0, 'Beta');")

        let df = try await DataFrame.fromSQL("SELECT id, val, name FROM test ORDER BY id ASC", connection: conn)

        #expect(df.rowCount == 2)
        #expect(df.columnNames.contains("id"))
        #expect(df.columnNames.contains("val"))
        #expect(df.columnNames.contains("name"))

        guard let nameCol = df[column: "name", as: String.self] else {
            Issue.record("Missing name column")
            return
        }
        #expect(nameCol[0] == "Alpha")
        #expect(nameCol[1] == "Beta")
    }

    @Test("Test SQLite query error handling")
    func testSQLiteQueryError() async {
        let conn = SQLiteConnection(databasePath: ":memory:")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("SELECT * FROM non_existent_table;")
        }
    }

    @Test("Test PostgreSQL wire connection error handling on unreachable host")
    func testPostgreSQLIngestion() async throws {
        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@127.0.0.1:5432/testdb")
        #expect(conn.host == "127.0.0.1")
        #expect(conn.port == 5432)
        #expect(conn.user == "user")
        #expect(conn.password == "pass")
        #expect(conn.database == "testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("CREATE TABLE users (id INTEGER PRIMARY KEY, score REAL);")
        }
    }

    @Test("Test PostgreSQL query validation error handling")
    func testPostgreSQLErrorHandling() async {
        let connEmptyURL = PostgreSQLConnection(connectionURL: "")
        await #expect(throws: DatabaseError.self) {
            _ = try await connEmptyURL.executeQuery("SELECT 1;")
        }

        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@127.0.0.1:5432/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("")
        }
    }

    @Test("Test MySQL wire connection error handling on unreachable host")
    func testMySQLIngestion() async throws {
        let conn = MySQLConnection(connectionURL: "mysql://user:pass@127.0.0.1:3306/testdb")
        #expect(conn.host == "127.0.0.1")
        #expect(conn.port == 3306)
        #expect(conn.user == "user")
        #expect(conn.password == "pass")
        #expect(conn.database == "testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("CREATE TABLE orders (id INT PRIMARY KEY, amount DOUBLE);")
        }
    }

    @Test("Test MySQL query validation error handling")
    func testMySQLErrorHandling() async {
        let connEmptyURL = MySQLConnection(connectionURL: "")
        await #expect(throws: DatabaseError.self) {
            _ = try await connEmptyURL.executeQuery("SELECT 1;")
        }

        let conn = MySQLConnection(connectionURL: "mysql://user:pass@127.0.0.1:3306/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("")
        }
    }

    @Test("Test AnySendableValue description and custom conversion")
    func testAnySendableValue() {
        let d: AnySendableValue = .double(3.14)
        let i: AnySendableValue = .int(42)
        let s: AnySendableValue = .string("hello")
        let n: AnySendableValue = .null

        #expect(d.description == "3.14")
        #expect(i.description == "42")
        #expect(s.description == "hello")
        #expect(n.description == "NULL")
    }

    @Test("Test DatabaseError errorDescription messages")
    func testDatabaseErrorMessages() {
        let err1 = DatabaseError.connectionFailed("Host unreachable")
        let err2 = DatabaseError.queryFailed("Syntax error")
        let err3 = DatabaseError.notImplemented("Feature Z")

        #expect(err1.errorDescription?.contains("Host unreachable") == true)
        #expect(err2.errorDescription?.contains("Syntax error") == true)
        #expect(err3.errorDescription?.contains("Feature Z") == true)
    }

    @Test("Test DataFrame fromSQL with mock query results")
    func testDataFrameFromSQLMock() async throws {
        struct MockConnection: DatabaseConnection {
            func executeQuery(_ sql: String) async throws -> SQLQueryResult {
                return SQLQueryResult(
                    columns: ["col_d", "col_i", "col_s", "col_n"],
                    rows: [
                        [.double(1.5), .int(10), .string("first"), .null],
                        [.double(2.5), .int(20), .string("second"), .null]
                    ]
                )
            }
        }

        let mockConn = MockConnection()
        let df = try await DataFrame.fromSQL("SELECT * FROM mock", connection: mockConn)
        #expect(df.rowCount == 2)
        #expect(df.columnNames.count == 4)
        #expect(df.columnNames == ["col_d", "col_i", "col_s", "col_n"])
    }

    @Test("Test DataFrame fromSQL with empty query results")
    func testDataFrameFromSQLEmpty() async throws {
        struct EmptyMockConnection: DatabaseConnection {
            func executeQuery(_ sql: String) async throws -> SQLQueryResult {
                return SQLQueryResult(columns: ["a", "b"], rows: [])
            }
        }

        let df = try await DataFrame.fromSQL("SELECT a, b FROM empty", connection: EmptyMockConnection())
        #expect(df.rowCount == 0)
    }

    @Test("Test expanded AnySendableValue types: int64, bool, date, data")
    func testExpandedAnySendableValueTypes() {
        let i64: AnySendableValue = .int64(9_000_000_000_000_000_000)
        let bTrue: AnySendableValue = .bool(true)
        let bFalse: AnySendableValue = .bool(false)
        let now = Date(timeIntervalSince1970: 1700000000)
        let d: AnySendableValue = .date(now)
        let rawData = Data([0xDE, 0xAD, 0xBE, 0xEF])
        let blob: AnySendableValue = .data(rawData)

        #expect(i64.description == "9000000000000000000")
        #expect(bTrue.description == "true")
        #expect(bFalse.description == "false")
        #expect(d.description.contains("2023"))
        #expect(blob.description == "4 bytes")

        #expect(i64 == .int64(9_000_000_000_000_000_000))
        #expect(bTrue == .bool(true))
        #expect(blob == .data(rawData))
        #expect(i64 != .int(0))
    }

    @Test("Test SQLite BLOB, large INT64, and DataFrame integration")
    func testSQLiteBlobAndInt64Ingestion() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        _ = try await conn.executeQuery("CREATE TABLE binary_data (id INTEGER PRIMARY KEY, payload BLOB, flag INTEGER);")
        
        let sampleBytes = Data([1, 2, 3, 4, 5])
        let hex = sampleBytes.map { String(format: "%02X", $0) }.joined()
        _ = try await conn.executeQuery("INSERT INTO binary_data (id, payload, flag) VALUES (1, X'\(hex)', 1);")

        let res = try await conn.executeQuery("SELECT id, payload, flag FROM binary_data;")
        #expect(res.rows.count == 1)
        #expect(res.rows[0].count == 3)
        
        let payloadVal = res.rows[0][1]
        if case .data(let d) = payloadVal {
            #expect(d == sampleBytes)
        } else {
            Issue.record("Expected .data payload, got \(payloadVal)")
        }

        let df = try await DataFrame.fromSQL("SELECT id, payload, flag FROM binary_data", connection: conn)
        #expect(df.rowCount == 1)
        #expect(df.columnNames.contains("payload"))
    }
}
