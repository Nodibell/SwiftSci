# Relational Database Drivers & Zero-Copy SQL Ingestion

Direct high-throughput database connectivity for SQLite, PostgreSQL, and MySQL into `SwiftDataFrame` under Swift 6 Actor Concurrency.

## Overview

`SwiftDatabase` provides native, zero-dependency, pure-Swift network wire-protocol drivers and embedded SQLite connectors. All connection types are implemented as Swift 6 `public actor` types, guaranteeing complete thread safety and preventing data races across concurrent tasks.

---

## 🔒 Driver Architecture & Concurrency Model

```
Swift 6 Concurrency Layer
┌─────────────────────────────────────────────────────────────┐
│ actor SQLiteConnection        (C-API Prepared Statements)   │
│ actor PostgreSQLConnection    (Pure-Swift v3.0 Wire Recv)   │
│ actor MySQLConnection         (Client/Server Binary Packet) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│       Zero-Copy Column Ingestion into SwiftDataFrame        │
└─────────────────────────────────────────────────────────────┘
```

* **Actor Isolation:** Concurrent queries to the same database connection are automatically serialized without external lock primitives.
* **Streaming `recv` Buffer:** Network drivers use a 64KB receive buffer reading continuously until `ReadyForQuery ('Z')` is received.
* **Authentication:** Built-in support for Cleartext and MD5 password authentication via `CommonCrypto`.

---

## 1. Querying SQLite Databases & C-Pointer Lifecycle Safety

`SQLiteConnection` executes SQL statements through native `sqlite3_prepare_v2` and directly streams result rows into columnar `TypedColumn` buffers with full C-pointer lifecycle safety:

- **Statement Finalization:** All prepared statements are guaranteed to be finalized via `defer { if let stmt { sqlite3_finalize(stmt) } }`, preventing statement pointer leaks during both normal termination and thrown query errors.
- **Resource Cleanup:** In-memory or file-backed database handles are automatically closed via `sqlite3_close_v2` upon deallocation. Callers may also invoke `await sqlite.close()` to immediately release operating system locks and memory.

```swift
import Foundation
import SwiftDatabase
import SwiftDataFrame

// 1. Initialize SQLite connection to file or in-memory
let sqlite = SQLiteConnection(databasePath: "analytics.sqlite")

// 2. Execute SQL query into a DataFrame
let df = try await DataFrame.fromSQL(
    "SELECT id, temperature, humidity, timestamp FROM sensor_logs WHERE temperature > 25.0",
    connection: sqlite
)

print("Ingested \(df.shape.rows) rows with \(df.shape.columns) columns:")
df.debugPrint(maxRows: 5)

// 3. Explicitly release SQLite database handle when done
await sqlite.close()
```

---

## 2. PostgreSQL Wire Connection (v3.0 Protocol)

Connect to remote PostgreSQL clusters over TCP/TLS without external C libraries (`libpq`):

```swift
import SwiftDatabase
import SwiftDataFrame

let pgConn = PostgreSQLConnection(
    host: "localhost",
    port: 5432,
    database: "production_db",
    user: "postgres",
    password: "secure_password",
    sslMode: .disable
)

// Ingest query results asynchronously
let salesDF = try await DataFrame.fromSQL(
    "SELECT customer_id, total_amount, order_date FROM orders ORDER BY order_date DESC LIMIT 1000",
    connection: pgConn
)

print("Fetched \(salesDF.shape.rows) orders from PostgreSQL.")
```

---

## 3. High-Throughput Bulk Writeback (`toSQL`)

Bulk export in-memory `DataFrame` objects back into database tables with configurable append, replace, or fail-if-exists modes:

```swift
// Write transformed DataFrame to database
try await salesDF.toSQL(
    table: "aggregated_metrics",
    connection: sqlite,
    mode: .replace,      // .append, .replace, or .failIfExists
    batchSize: 500
)

print("Successfully written DataFrame to 'aggregated_metrics' table.")
```
