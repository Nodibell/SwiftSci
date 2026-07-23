import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftDatabase

@Suite("SwiftDatabase Tests")
struct SwiftDatabaseTests {
    @Test("Test SQLite query to DataFrame ingestion")
    func testSQLiteDataFrameIngestion() async throws {
        let conn = SQLiteConnection(databasePath: ":memory:")
        let df = try await DataFrame.fromSQL("SELECT * FROM test", connection: conn)

        #expect(df.rowCount == 2)
        #expect(df.columnNames.contains("id"))
        #expect(df.columnNames.contains("val"))
    }
}
