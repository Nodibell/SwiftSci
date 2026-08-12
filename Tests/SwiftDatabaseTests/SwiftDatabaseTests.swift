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

    @Test("Test PostgreSQL query throws notImplemented error")
    func testPostgreSQLIngestion() async throws {
        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@localhost:5432/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("CREATE TABLE users (id INTEGER PRIMARY KEY, score REAL);")
        }
    }

    @Test("Test PostgreSQL query error handling")
    func testPostgreSQLErrorHandling() async {
        let connEmptyURL = PostgreSQLConnection(connectionURL: "")
        await #expect(throws: DatabaseError.self) {
            _ = try await connEmptyURL.executeQuery("SELECT 1;")
        }

        let conn = PostgreSQLConnection(connectionURL: "postgres://user:pass@localhost:5432/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("")
        }
    }

    @Test("Test MySQL query throws notImplemented error")
    func testMySQLQueryThrowsNotImplemented() async {
        let conn = MySQLConnection(connectionURL: "mysql://user:pass@localhost:3306/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("SELECT 1;")
        }
    }

    @Test("Test MySQL query error handling")
    func testMySQLErrorHandling() async {
        let connEmptyURL = MySQLConnection(connectionURL: "")
        await #expect(throws: DatabaseError.self) {
            _ = try await connEmptyURL.executeQuery("SELECT 1;")
        }

        let conn = MySQLConnection(connectionURL: "mysql://user:pass@localhost:3306/testdb")
        await #expect(throws: DatabaseError.self) {
            _ = try await conn.executeQuery("")
        }
    }
}
