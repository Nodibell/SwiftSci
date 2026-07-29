# Relational Database Connectors

Query relational databases (SQLite, PostgreSQL, MySQL) directly into zero-copy SwiftDataFrames.

## Overview

Execute SQL queries over native database connections without intermediate string parsing overhead.

### 1. SQLite Database Querying

```swift
import SwiftDatabase
import SwiftDataFrame

let db = try DatabaseConnection(driver: .sqlite(path: "/path/to/database.db"))
let df = try await db.query("SELECT age, fare, survived FROM passengers WHERE fare > 50.0")

print("Fetched \(df.rowCount) rows into DataFrame")
df.head(5).debugPrint()
```
