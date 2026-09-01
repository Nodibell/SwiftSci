import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftDatabase

@Suite("SwiftDatabase Complete Coverage Tests")
struct DatabaseCoverageTests {

    @Test("DatabaseError all enum cases and error descriptions")
    func testAllDatabaseErrors() {
        let errs: [DatabaseError] = [
            .connectionFailed("Connection refused"),
            .queryFailed("Table does not exist"),
            .notImplemented("Async cursor"),
            .tableAlreadyExists("customers"),
            .unsupportedAuth("scram-sha-256")
        ]

        for err in errs {
            let desc = err.errorDescription
            #expect(desc != nil && !desc!.isEmpty)
        }

        #expect(DatabaseError.tableAlreadyExists("orders").errorDescription?.contains("orders") == true)
        #expect(DatabaseError.unsupportedAuth("scram").errorDescription?.contains("scram") == true)
    }

    @Test("SQLWriteMode and SSLMode Codable serialization")
    func testEnumCodable() throws {
        let modes: [SQLWriteMode] = [.append, .replace, .failIfExists]
        for mode in modes {
            let data = try JSONEncoder().encode(mode)
            let decoded = try JSONDecoder().decode(SQLWriteMode.self, from: data)
            #expect(decoded == mode)
        }

        let sslModes: [SSLMode] = [.disable, .prefer, .require]
        for ssl in sslModes {
            let data = try JSONEncoder().encode(ssl)
            let decoded = try JSONDecoder().decode(SSLMode.self, from: data)
            #expect(decoded == ssl)
        }
    }

    @Test("DataFrame.toSQL mode .failIfExists throws when table exists")
    func testToSQLFailIfExists() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        let colA = TypedColumn(name: "val", values: [1.0, 2.0])
        let df = try DataFrame(columns: [colA])

        // First write creates table
        try await df.toSQL(table: "items", connection: conn, mode: .replace)

        // Second write with .failIfExists should throw tableAlreadyExists
        await #expect(throws: DatabaseError.self) {
            try await df.toSQL(table: "items", connection: conn, mode: .failIfExists)
        }
    }

    @Test("DataFrame.toSQL mode .failIfExists succeeds when table does not exist")
    func testToSQLFailIfExistsFresh() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        let colA = TypedColumn(name: "score", values: [100, 200])
        let df = try DataFrame(columns: [colA])

        try await df.toSQL(table: "new_table", connection: conn, mode: .failIfExists)
        let readDf = try await DataFrame.fromSQL("SELECT score FROM new_table ORDER BY score ASC", connection: conn)
        #expect(readDf.rowCount == 2)
    }

    @Test("DataFrame.toSQL handles multiple batches and various column data types")
    func testToSQLDataTypesAndBatching() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")

        let colInt = TypedColumn(name: "c_int", values: [1, 2, 3, 4, 5])
        let colDbl = TypedColumn(name: "c_dbl", values: [1.1, 2.2, 3.3, 4.4, 5.5])
        let colStr = TypedColumn(name: "c_str", values: ["a", "b", "c", "d", "e"])
        let colBool = TypedColumn(name: "c_bool", values: [true, false, true, false, true])
        let colDate = TypedColumn(name: "c_date", values: [Date(timeIntervalSince1970: 1000), Date(timeIntervalSince1970: 2000), Date(timeIntervalSince1970: 3000), Date(timeIntervalSince1970: 4000), Date(timeIntervalSince1970: 5000)])

        let df = try DataFrame(columns: [colInt, colDbl, colStr, colBool, colDate])

        // Write with batchSize: 2 to test multi-batch chunking
        try await df.toSQL(table: "all_types", connection: conn, mode: .replace, batchSize: 2)

        let readDf = try await DataFrame.fromSQL("SELECT c_int, c_dbl, c_str FROM all_types ORDER BY c_int ASC", connection: conn)
        #expect(readDf.rowCount == 5)
        #expect(readDf.columnNames == ["c_int", "c_dbl", "c_str"])
    }

    @Test("DataFrame.toSQL handles nullable columns with NULLs")
    func testToSQLNullable() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")

        let col1 = TypedColumn(name: "id", values: [1, 2, 3])
        let col2 = TypedColumn(name: "opt_val", values: [Optional(10.0), nil, Optional(30.0)])
        let col3 = TypedColumn(name: "opt_txt", values: [Optional("first"), Optional("second"), nil])

        let df = try DataFrame(columns: [col1, col2, col3])
        try await df.toSQL(table: "nullables", connection: conn, mode: .replace)

        let readDf = try await DataFrame.fromSQL("SELECT id, opt_val, opt_txt FROM nullables WHERE opt_val IS NULL", connection: conn)
        #expect(readDf.rowCount == 1)
    }

    @Test("SQLiteConnection file-based database lifecycle")
    func testSQLiteFileDatabase() async throws {
        let tempFile = NSTemporaryDirectory() + "test_sqlite_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempFile) }

        let conn = SQLiteConnection(databasePath: tempFile)
        _ = try await conn.executeQuery("CREATE TABLE t (id INT, v TEXT);")
        _ = try await conn.executeQuery("INSERT INTO t VALUES (1, 'hello');")

        let res = try await conn.executeQuery("SELECT * FROM t;")
        #expect(res.columns == ["id", "v"])
        #expect(res.rows.count == 1)
        #expect(res.rows[0][0].description == "1")
        #expect(res.rows[0][1].description == "hello")
    }

    @Test("PostgreSQL and MySQL connection URL parsing variants")
    func testConnectionURLParsing() {
        // PostgreSQL URLs
        let pg1 = PostgreSQLConnection(connectionURL: "postgres://alice:secret@db.host.com:5433/production?sslmode=require")
        #expect(pg1.host == "db.host.com")
        #expect(pg1.port == 5433)
        #expect(pg1.user == "alice")
        #expect(pg1.password == "secret")
        #expect(pg1.database == "production")
        #expect(pg1.sslMode == .require)

        let pg2 = PostgreSQLConnection(connectionURL: "postgres://db.host.com/mydb?sslmode=prefer")
        #expect(pg2.user == "postgres")
        #expect(pg2.password == "")
        #expect(pg2.database == "mydb")
        #expect(pg2.sslMode == .prefer)

        let pg3 = PostgreSQLConnection(connectionURL: "invalid-url", sslMode: .require)
        #expect(pg3.sslMode == .require)

        // MySQL URLs
        let my1 = MySQLConnection(connectionURL: "mysql://admin:pw123@sql.host.com:3307/shop?ssl=true")
        #expect(my1.host == "sql.host.com")
        #expect(my1.port == 3307)
        #expect(my1.user == "admin")
        #expect(my1.password == "pw123")
        #expect(my1.database == "shop")
        #expect(my1.sslMode == .require)

        let my2 = MySQLConnection(connectionURL: "mysql://sql.host.com/db", sslMode: .prefer)
        #expect(my2.user == "root")
        #expect(my2.sslMode == .prefer)
    }
}
