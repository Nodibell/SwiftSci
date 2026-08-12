# Relational Database Connectors

Query relational databases directly into zero-copy SwiftDataFrames.

## Overview

Execute SQL queries over native database connections without intermediate string parsing overhead.

**Driver status:** SQLite is fully implemented via `SQLiteConnection`. `PostgreSQLConnection` and
`MySQLConnection` are present as typed connection stubs and throw `DatabaseError.notImplemented` until
native wire-protocol drivers are integrated.

### 1. SQLite Database Querying

```swift
import SwiftDatabase
import SwiftDataFrame

let db = try DatabaseConnection(driver: .sqlite(path: "/path/to/database.db"))
let df = try await db.query("SELECT age, fare, survived FROM passengers WHERE fare > 50.0")

print("Fetched \(df.rowCount) rows into DataFrame")
df.head(5).debugPrint()
```
