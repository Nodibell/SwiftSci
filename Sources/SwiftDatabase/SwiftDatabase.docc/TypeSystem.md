# Database Type System & Value Representation

Explore the strongly typed, thread-safe value representations supporting relational database columns and zero-copy conversions.

## Overview

The `SwiftDatabase` module bridges the gap between relational SQL storage engines and Swift's type-safe, actor-isolated data ecosystem. Traditional database client libraries often convert SQL fields to intermediate strings or untyped objects, imposing significant heap allocation overhead and potential loss of numeric precision.

`AnySendableValue` provides a unified, zero-copy, enum-based abstraction covering primary relational types:
- 64-bit IEEE 754 floating point numbers (`.double`)
- Standard host-width integers (`.int`)
- Explicit 64-bit signed integers (`.int64`)
- UTF-8 text strings (`.string`)
- Boolean flags (`.bool`)
- High-precision timestamps (`.date`)
- Raw byte buffers (`.data`)
- Nullable representations (`.null`)

## Memory & Execution Architecture

```
                  ┌───────────────────────────────┐
                  │      Relational SQL Engine    │
                  │   (SQLite / Postgres / MySQL) │
                  └───────────────┬───────────────┘
                                  │ Direct C API / Wire Protocol
                                  ▼
                  ┌───────────────────────────────┐
                  │       AnySendableValue        │
                  │ (.int64, .double, .data, etc) │
                  └───────┬───────────────┬───────┘
                          │               │
     Columnar Zero-Copy   ▼               ▼   Fast Tabular Interop
                  ┌───────────────┐ ┌───────────────┐
                  │   DataFrame   │ │SQLQueryResult │
                  │    Series     │ │ Row Iteration │
                  └───────────────┘ └───────────────┘
```

### Direct SQLite C API Mapping

When ingesting rows from SQLite:
- `SQLITE_INTEGER`: Stored as `.int` if within the host pointer range `Int.min...Int.max`, or widened to `.int64` without truncation.
- `SQLITE_FLOAT`: Read directly using `sqlite3_column_double` as `.double`.
- `SQLITE_TEXT`: Ingested directly as UTF-8 `.string`.
- `SQLITE_BLOB`: Read directly via `sqlite3_column_blob` and `sqlite3_column_bytes` into `Data` without string conversion.
- `SQLITE_NULL`: Mapped to `.null`.

### Thread Safety & Swift 6 Strict Concurrency

All instances of `AnySendableValue` and `SQLQueryResult` conform unconditionally to `Sendable`. Value semantics ensure that data rows fetched across actor boundaries (e.g. from `SQLiteConnection` to background compute tasks) never introduce data races or retain cycles.

## Topics

### Value Types
- ``AnySendableValue``
- ``SQLQueryResult``
