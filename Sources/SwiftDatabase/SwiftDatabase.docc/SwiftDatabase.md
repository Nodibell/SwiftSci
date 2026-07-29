# ``SwiftDatabase``

Relational Database Connectors to DataFrames.

## Overview

`SwiftDatabase` connects directly to relational SQL engines, executing queries straight into zero-copy SwiftDataFrames.

### Key Capabilities

- **Supported Drivers**: SQLite, PostgreSQL, and MySQL database engines.
- **Direct Querying**: `DatabaseConnection.query(...)` loading results into typed `DataFrame` instances.
- **Memory Efficiency**: Low-allocation columnar byte parsing from database cursor streams.

### Example Usage

```swift
import SwiftDatabase

let db = try DatabaseConnection(driver: .sqlite(path: "/path/db.sqlite"))
let df = try await db.query("SELECT * FROM sales WHERE amount > 100")
```

## Topics

### Guides & Tutorials
- <doc:DatabaseConnectors>
