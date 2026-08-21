# Bulk SQL Ingestion & Database Wire Protocols

Ingest millions of tabular rows into SQLite, PostgreSQL, and MySQL using zero-dependency wire protocols and batch writeback.

## Overview

`SwiftDatabase` eliminates third-party C driver dependencies by implementing native pure-Swift network wire protocols (PostgreSQL v3.0 protocol, MySQL Client/Server binary protocol) and high-throughput batch writeback via `DataFrame.toSQL`.

## 1. Bulk Ingestion with `toSQL`

```swift
import SwiftDatabase
import SwiftDataFrame

let df = try DataFrame.fromCSV(url: URL(fileURLWithPath: "transactions.csv"))

let db = try DatabaseConnection(driver: .postgres(
    host: "db.internal.net",
    port: 5432,
    database: "analytics",
    user: "app_worker",
    password: "secure_token",
    sslMode: .require
))

// Ingest in batches of 5,000 rows
try await df.toSQL(
    table: "fact_transactions",
    connection: db,
    mode: .append,
    batchSize: 5000
)
```

## 2. Ingestion Modes

* **`.append`**: Appends rows to an existing table (creates table if not present).
* **`.replace`**: Drops the existing table and reconstructs it with the DataFrame's inferred schema.
* **`.failIfExists`**: Throws an error if the table already exists, ensuring schema immutability.

## Topics

### Database APIs
- ``DatabaseConnection``
- ``SQLWriteMode``
- ``SSLMode``
