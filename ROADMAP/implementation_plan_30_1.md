# SwiftSci v3.0.1 — Compatibility Patch (G-001–G-004)

*Context:* Post-3.0.0 patch release closing verified ecosystem compatibility gaps (2026-08-12).
This plan addresses four open gap items. Phases 1–4 are patch-scale and safe for
3.0.1. Binary Core ML export (G-001 full resolution) is explicitly deferred — see **Deferred** below.

**Upstream note (G-004):** `arrow-swift` `main` already pins `FlatBuffers` to `exact: "25.2.10"` (same
version confirmed empirically for a build-safe graph). SwiftSci still declares only `arrow-swift from: "21.0.0"` with no
direct FlatBuffers constraint, so fresh downstream resolution can still float to broken `25.12.19` until
Phase 1 lands.

**Agent instructions:** Execute phases in order. After each phase: `swift build`, run the listed test
targets, commit with the message given, then proceed. Do not implement binary `.mlmodel` export in this
release — doc honesty only (Phase 4).

---

## Phase 1 — FlatBuffers resolution pin (G-004)

**Problem.** Resolving `arrow-swift 21.0.0` with `FlatBuffers 25.12.19` fails compilation (`FlatBufferObject`,
`Table.directRead` not found). Pinning `FlatBuffers 25.2.10` restores a build-safe graph. SwiftPM merges
constraints from all packages — a direct declaration in SwiftSci prevents float to broken versions regardless
of which `arrow-swift` tag resolves.

**Steps:**

1. Edit `Package.swift`, add to `dependencies:` (alongside existing entries):
   ```swift
   .package(url: "https://github.com/google/flatbuffers.git", exact: "25.2.10"),
   ```
   No target needs to `import FlatBuffers` — the dependency entry alone participates in resolution.
2. Run `swift package resolve` and confirm `Package.resolved` pins `flatbuffers` to `25.2.10`.
3. Run `swift build` (debug + release).

**Commit:** `fix(deps): pin FlatBuffers 25.2.10 for arrow-swift compatibility (G-004)`

---

## Phase 2 — MySQL honest stub + documentation (G-003)

**Problem.** PostgreSQL has `PostgreSQLConnection` and throws `DatabaseError.notImplemented` honestly.
MySQL has **zero code** — only two `.docc` files claim "Supported Drivers: SQLite, PostgreSQL, and MySQL".
This is worse than the PostgreSQL stub: documented capability with no type at all.

**Steps:**

1. Add `MySQLConnection` to `Sources/SwiftDatabase/DatabaseConnection.swift`, symmetric to
   `PostgreSQLConnection` (same file, after the PostgreSQL class):
   ```swift
   /// MySQL database connection driver.
   public final class MySQLConnection: DatabaseConnection, @unchecked Sendable {
       /// The connection URL.
       public let connectionURL: String

       /// Creates a new instance.
       /// - Parameter connectionURL: The connection URL (e.g. `mysql://user:pass@host:3306/db`).
       public init(connectionURL: String) {
           self.connectionURL = connectionURL
       }

       /// Executes a SQL query against a MySQL database connection.
       /// - Throws: `DatabaseError.notImplemented` — native MySQL driver not yet integrated.
       public func executeQuery(_ sql: String) async throws -> SQLQueryResult {
           guard !connectionURL.isEmpty else {
               throw DatabaseError.connectionFailed("Empty MySQL connection URL")
           }
           guard !sql.isEmpty else {
               throw DatabaseError.queryFailed("SQL query cannot be empty")
           }
           throw DatabaseError.notImplemented(
               "MySQL native driver integration not yet implemented. Use SQLiteConnection for local embedded SQL databases."
           )
       }
   }
   ```

2. Update documentation honesty in both files:
   - `Sources/SwiftDatabase/SwiftDatabase.docc/SwiftDatabase.md` — change "Supported Drivers" to:
     `SQLite (implemented), PostgreSQL and MySQL (connection types present; native drivers not yet implemented)`.
   - `Sources/SwiftDatabase/SwiftDatabase.docc/DatabaseConnectors.md` — same wording in Overview; add a
     short **Driver status** note listing which engines are runnable vs stub.

3. Add tests in `Tests/SwiftDatabaseTests/SwiftDatabaseTests.swift` (mirror PostgreSQL tests):
   - `testMySQLQueryThrowsNotImplemented` — valid URL + query → `DatabaseError.notImplemented`
   - `testMySQLErrorHandling` — empty URL and empty SQL → `DatabaseError`

**Commit:** `feat(database): add MySQLConnection stub and honest driver docs (G-003)`

---

## Phase 3 — SwiftAgent command subset expansion (G-002)

**Problem.** `AgentCommand` accepts only `filter`, `sample`, `select`, `head`, `tail`. Downstream client apps need a
narrower superset for cleaning workflows — not the full preprocessing/join/calculated-column surface.

**Scope (in):** `rename`, `dropnulls`, `fillnulls`, `groupby` (single aggregation).
**Scope (out — separate cycle):** imputation strategies beyond scalar fill, encoding, outlier treatment,
joins, calculated columns, arbitrary Swift source execution.

**API mapping** — use existing `SwiftDataFrame` types; do **not** introduce `AggregationFunction`:

| Command | Maps to |
| :--- | :--- |
| `rename OLD to NEW` | `DataFrame.renameColumn(_:to:)` |
| `dropnulls` / `dropnulls COL1,COL2` | Row filter: drop rows where any listed column (or all columns if omitted) is null |
| `fillnulls COL VALUE` | Per-column `TypedColumn.fillNull(with:)` for `Double` columns only on 3.0.1 |
| `groupby KEY mean` | `DataFrame.groupBy(KEY).mean()` |
| `groupby KEY sum COL` | `DataFrame.groupBy(KEY).agg([COL: .sum])` (and `mean`/`min`/`max`/`count`) |

Use existing `Aggregation` enum from `GroupedDataFrame.swift` (`sum`, `mean`, `min`, `max`, `count`,
`first`, `last`).

**Steps:**

1. Extend `AgentCommand` in `Sources/SwiftAgent/SwiftAgentEvaluator.swift`:
   ```swift
   case rename(from: String, to: String)
   case dropNulls(columns: [String]?)   // nil = all columns
   case fillNulls(column: String, value: Double)
   case groupBy(column: String, aggregation: Aggregation, targetColumn: String?)
   ```
   `targetColumn` is `nil` for whole-frame aggregations like `groupby age mean`; set for
   `groupby age sum score`.

2. Add `evaluate` switch branches calling the mapped APIs above.

3. Implement row-based `dropNulls` as a private helper on `DataFrame` inside `SwiftAgentEvaluator.swift`
   (keep scope in SwiftAgent — do not add `DataFrame.dropNulls()` to SwiftDataFrame in 3.0.1):
   ```swift
   private func dropNullRows(in df: DataFrame, columns: [String]?) throws -> DataFrame {
       let cols = columns ?? df.columnNames
       let indices = (0..<df.rowCount).filter { i in
           !cols.contains { col in df[column: col]?.value(at: i) == nil }
       }
       return df.gathered(at: indices)  // or equivalent existing row-subset API
   }
   ```
   Verify `gathered(at:)` or use `filter`/`evaluatedIndices` if index gather differs — match existing
   DataFrame subset idioms in the codebase.

4. Extend `parseCommand(_:)` using the same prefix/regex style as existing commands:

   | Input pattern | Parsed case |
   | :--- | :--- |
   | `rename age to years` | `.rename(from: "age", to: "years")` |
   | `dropnulls` | `.dropNulls(columns: nil)` |
   | `dropnulls age,score` | `.dropNulls(columns: ["age", "score"])` |
   | `fillnulls age 0.0` | `.fillNulls(column: "age", value: 0.0)` |
   | `groupby age mean` | `.groupBy(column: "age", aggregation: .mean, targetColumn: nil)` |
   | `groupby age sum score` | `.groupBy(column: "age", aggregation: .sum, targetColumn: "score")` |

   Parser is case-insensitive on command verbs (match existing `lower.hasPrefix` pattern).

5. Add tests in `Tests/SwiftAgentTests/SwiftAgentTests.swift`:
   - `rename age to years` → column renamed
   - `dropnulls` on DataFrame with one null row → row removed
   - `fillnulls score 0.0` → null replaced
   - `groupby category mean` → grouped result row count correct
   - unparseable `groupby x bogus y` → `AgentError.unparseable`

**Commit:** `feat(agent): add rename, dropnulls, fillnulls, groupby commands (G-002)`

---

## Phase 4 — CoreMLExporter documentation honesty (G-001 partial)

**Problem.** `CoreMLExporter` module doc comment claims "linear, tree, and ensemble models" but only
`exportLinearModel` exists, returning JSON — not a binary `.mlmodel`/`.mlpackage`. Doc overstates capability;
Gap G-001 stays **open** for binary export but documentation must match reality (same honesty principle
as ONNX JSON vs `exportBinaryONNX`, and PostgreSQL/MySQL stubs).

**Steps:**

1. Edit `Sources/SwiftML/Serialization/CoreMLExporter.swift` module doc comment:
   ```swift
   /// Exporter for serializing SwiftML linear regression weights into a CoreML JSON specification payload.
   /// Note: Produces a JSON representation of model metadata and coefficients, not a binary .mlmodel or
   /// .mlpackage bundle. Tree, forest, logistic, and binary Core ML artifact export are not yet implemented.
   ```

2. Add `[3.0.1]` section to `CHANGELOG.md`:
   - **Fixed:** FlatBuffers pin (G-004)
   - **Added:** `MySQLConnection` stub (G-003); SwiftAgent `rename`/`dropnulls`/`fillnulls`/`groupby` (G-002)
   - **Changed:** CoreMLExporter and SwiftDatabase driver documentation aligned with actual capabilities (G-001, G-003)

3. Do **not** change `exportLinearModel` signature or add protobuf writers in this release.

**Commit:** `docs(ml): align CoreMLExporter documentation with implemented scope (G-001)`

---

## Deferred — Binary Core ML export (G-001 full resolution)

Real binary `.mlmodel` export requires Apple `Model.proto` protobuf encoding with correct message selection
(`GLMRegressor`, `GLMClassifier`, `NeuralNetwork`, etc.) per model family — same class of work as
`ONNXExporter.exportBinaryONNX` (`Sources/SwiftML/Serialization/ONNXExporter.swift`, `ProtobufWriter`).
This is **not** patch-scale; track as a separate implementation plan (`implementation_plan_coreml_binary.md`)
before closing G-001 in the compatibility gap log.

---

## Definition of Done

- [x] `swift build -c debug` and `swift build -c release` both clean
- [x] `swift test --filter SwiftDatabaseTests` green (including new MySQL tests)
- [x] `swift test --filter SwiftAgentTests` green (including new command tests)
- [x] `Package.resolved` shows `flatbuffers` at `25.2.10`
- [x] `grep -rl "MySQL" Sources/SwiftDatabase/` finds `DatabaseConnection.swift` plus updated `.docc` files (not docc-only)
- [x] `CoreMLExporter.swift` module doc no longer mentions "tree, and ensemble" as implemented
- [x] Downstream consumers can remove a local FlatBuffers pin once consuming SwiftSci 3.0.1
- [x] Gap log: G-004 → Resolved; G-003 → partial (stub + honesty, native driver still open);
      G-002 → partial (four commands, full workflow still open); G-001 → open (doc fixed, binary deferred)
