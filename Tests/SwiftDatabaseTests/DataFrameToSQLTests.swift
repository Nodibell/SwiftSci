import XCTest
import SwiftDataFrame
@testable import SwiftDatabase

final class DataFrameToSQLTests: XCTestCase {

    func testDataFrameToSQLAppendAndFromSQLRoundtrip() async throws {
        let dbPath = ":memory:"
        let conn = SQLiteConnection(databasePath: dbPath)

        let col1 = TypedColumn(name: "id", values: [1, 2, 3])
        let col2 = TypedColumn(name: "name", values: ["Alice", "Bob", "Charlie"])
        let col3 = TypedColumn(name: "score", values: [95.5, 88.0, 72.3])
        let df = try DataFrame(columns: [col1, col2, col3])

        // Write DataFrame to SQL
        try await df.toSQL(table: "users", connection: conn, mode: .append)

        // Read back
        let dfRead = try await DataFrame.fromSQL("SELECT id, name, score FROM users ORDER BY id ASC;", connection: conn)
        XCTAssertEqual(dfRead.shape.rows, 3)
        XCTAssertEqual(dfRead.shape.columns, 3)
        XCTAssertEqual(dfRead.columns.map { $0.name }, ["id", "name", "score"])

        // Test appending another batch
        let col1b = TypedColumn(name: "id", values: [4])
        let col2b = TypedColumn(name: "name", values: ["David"])
        let col3b = TypedColumn(name: "score", values: [99.9])
        let dfAppend = try DataFrame(columns: [col1b, col2b, col3b])

        try await dfAppend.toSQL(table: "users", connection: conn, mode: .append)
        let dfAfterAppend = try await DataFrame.fromSQL("SELECT COUNT(*) as count FROM users;", connection: conn)
        XCTAssertEqual(dfAfterAppend.shape.rows, 1)
    }

    func testDataFrameToSQLReplaceMode() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")

        let colA = TypedColumn(name: "val", values: [10.0, 20.0])
        let df1 = try DataFrame(columns: [colA])
        try await df1.toSQL(table: "metrics", connection: conn, mode: .append)

        let colB = TypedColumn(name: "val", values: [100.0, 200.0, 300.0])
        let df2 = try DataFrame(columns: [colB])
        try await df2.toSQL(table: "metrics", connection: conn, mode: .replace)

        let dfRead = try await DataFrame.fromSQL("SELECT val FROM metrics;", connection: conn)
        XCTAssertEqual(dfRead.shape.rows, 3)
    }

    func testSSLModeParsing() {
        let pgConn1 = PostgreSQLConnection(connectionURL: "postgres://user:pass@remote.db.com:5432/mydb?sslmode=require")
        XCTAssertEqual(pgConn1.sslMode, .require)

        let pgConn2 = PostgreSQLConnection(connectionURL: "postgres://user:pass@localhost:5432/mydb")
        XCTAssertEqual(pgConn2.sslMode, .disable)

        let mysqlConn1 = MySQLConnection(connectionURL: "mysql://root:pass@remote.mysql.com:3306/mydb?ssl=true")
        XCTAssertEqual(mysqlConn1.sslMode, .require)

        let mysqlConn2 = MySQLConnection(connectionURL: "mysql://root:pass@localhost:3306/mydb", sslMode: .prefer)
        XCTAssertEqual(mysqlConn2.sslMode, .prefer)
    }
}
