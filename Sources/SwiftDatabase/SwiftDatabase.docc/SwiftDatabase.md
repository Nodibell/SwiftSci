# ``SwiftDatabase``

Relational Database Connectors to DataFrames.

## Overview

`SwiftDatabase` connects directly to relational SQL engines, executing queries straight into zero-copy SwiftDataFrames.

### Key Capabilities

- **Supported Drivers**: Full native pure-Swift implementations for SQLite (`SQLiteConnection`), PostgreSQL v3.0 wire protocol with SCRAM-SHA-256 and TLS (`PostgreSQLConnection`), and MySQL Client/Server Protocol 4.1+ with TLS (`MySQLConnection`).
- **Direct Querying**: Ingest SQL queries directly into strongly-typed `DataFrame` instances with zero intermediate CSV conversion.
- **Memory Efficiency**: Columnar stream parsing with 1024-row chunk buffering to eliminate allocation bottlenecks.
- **Connectivity Verification**: Built-in async `ping()` verification across all database driver types.

### Example Usage

```swift
import SwiftDatabase
import SwiftDataFrame

// SQLite (in-memory or file-backed)
let sqlite = SQLiteConnection(databasePath: "/path/sales.db")
let df = try await DataFrame.fromSQL("SELECT id, amount, customer FROM orders WHERE amount > 100", connection: sqlite)

// Remote PostgreSQL or MySQL via native wire protocol actors
let pg = PostgreSQLConnection(host: "db.internal", user: "analyst", database: "analytics", sslMode: .require)
let isAlive = await pg.ping()
```

## Topics

### Guides & Tutorials
- <doc:DatabaseConnectors>
- <doc:HighThroughputIngestion>
- <doc:TypeSystem>
- <doc:SecurityAndPerformance>
