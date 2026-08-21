# Relational Database Connectors

Query relational databases directly into zero-copy SwiftDataFrames.

## Overview

Execute SQL queries and bulk insert tabular data over native database connections with zero intermediate string parsing overhead.

**Driver support:** Full pure-Swift wire-protocol drivers for SQLite, PostgreSQL (v3.0 wire protocol), and MySQL (Client/Server binary protocol) with optional TLS/SSL encryption.

### 1. Database Querying (SQLite, PostgreSQL, MySQL)

```swift
import SwiftDatabase
import SwiftDataFrame

let db = try DatabaseConnection(driver: .postgres(host: "localhost", port: 5432, database: "analytics", user: "postgres", password: "secret", sslMode: .require))
let df = try await db.query("SELECT age, fare, survived FROM passengers WHERE fare > 50.0")

print("Fetched \(df.rowCount) rows into DataFrame")
df.head(5).debugPrint()
```

### 2. High-Throughput Bulk Ingestion (`toSQL`)

```swift
// Bulk export DataFrame to relational database table
try await df.toSQL(
    table: "processed_metrics",
    connection: db,
    mode: .replace,
    batchSize: 1000
)
```
