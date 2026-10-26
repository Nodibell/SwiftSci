# Data and memory foundation contract

This reference describes SwiftSci's supported data representations and numerical handoffs on Apple silicon. It complements [compact numerical batches](CompactNumericBatches.md), which contains preparation and mutation examples. Representation preservation does not imply that an operation performs no allocation.

## Dataframe storage and operations

| Boundary | Representation and behavior |
| --- | --- |
| CSV ingestion | `CSVReadOptions.doubleStorage` selects legacy optional Double arrays or compact Double values with separate validity bits. Optional storage remains the default. Explicit schema, inference, eager reading, and streaming retain their validation rules. |
| Missing values | A missing entry differs from a valid NaN. Compact validity tracks only missing entries. Native integer columns retain integer identity until a caller requests conversion to Double. |
| Dataframe filtering and gathering | Predicate selection and indexed gathering preserve compact Double storage. Gathered rows follow the requested order, including duplicates. Empty results retain column names and types. |
| Sorting | Sorting gathers complete associated rows using the selected ordering. It preserves compact Double columns and existing null, NaN, stability, and signed-zero rules. It does not select a processor or promise a fixed allocation size. |
| Joins | Typed hash joins read compact keys directly. Joined Double keys and payloads preserve compact storage when their gathered source requires it. Empty or disjoint joins retain their relational schema, including payload suffixes. Missing keys do not match. IEEE NaNs do not compare equal in joins. |
| Grouping | Floating keys group signed zeros together and normalize NaN payloads into one group. Group order follows first appearance. Compact keys and numeric payloads use indexed reads without optional-array expansion. Group keys and aggregate outputs retain the existing public output representation. |
| Compatibility exports | `TypedColumn.values`, `PreparedNumericBatch.columnValues(at:)`, generic map and fill operations, and nested matrix exports may construct arrays. `TypedColumn.filtered(by:)` returns optional storage. These APIs do not promise compact output. |

Grouping and joining intentionally have different NaN equality rules. Neither changes the stored input values. Grouped floating sums retain their compensated reduction order. Integer aggregation and conversion retain the documented [checked integer sum rules](../Sources/SwiftDataFrame/SwiftDataFrame.docc/CheckedIntegerSums.md).

## Numerical preparation and mutation

`PreparedNumericBatch` owns contiguous Double columns and optional validity bitmaps. Construction and copies use Swift array copy-on-write. Indexed mutation validates the complete edit request before changing storage. A mutation detaches shared affected buffers and preserves previous batch and matrix snapshots. Separate copied values can be edited concurrently. Concurrent access to the same mutable variable requires caller synchronization.

Preparation converts supported native integers to Double, with binary64 precision limits. Bool maps to zero or one. A row selection records positions relative to the original prepared input, including repeated rows. Identity provenance stays implicit until an array export requests it. Row positions are not persistent application identifiers.

`PreparedNumericMatrix` records shape, column order, row provenance, and row-major or column-major layout. Multi-column packing constructs one output buffer. A single-column matrix can share its input values through copy-on-write. The default missing-value policy exports NaNs. `.reject` rejects missing entries while permitting valid NaNs.

Concrete batch preprocessing overloads avoid nested row arrays. Consuming transforms reuse uniquely held buffers where supported and detach shared storage. Generic preprocessing adapters keep their legacy fallback behavior. Preprocessing does not imply imputation or preservation of the input validity mask.

## CPU, GPU, and Core ML handoffs

| Consumer | Ownership and conversion boundary |
| --- | --- |
| CPU numerical APIs | Owned arrays or scoped buffer pointers feed Accelerate operations. Borrowed pointers remain valid only within their access closure. Precision selection applies to the operation that exposes it. Double storage alone does not guarantee an exact result. |
| Supported MLX GPU paths | Scoped reads retain their input owners and synchronize submitted work before releasing them. Results become owned CPU values within the supported scope. Float GPU paths convert Double values to Float with the existing rounding, underflow, and overflow behavior. PCA fitting validates finite inputs and rejects values that overflow Float conversion before entering its native solver. |
| Prepared Core ML input | Preparation owns model-compatible storage and row provenance. Float16, Float32, and Double conversion obey the model's declared type, including finite-value and overflow checks. This is a separate contract from legacy GPU conversion. |
| Core ML pool and batch pipeline | Slots remain exclusive through native completion. The pipeline bounds loaded but unconsumed batches and consumes predictions in source order. Cancellation and errors drain outstanding structured work and native callbacks before releasing owned capacity. Earlier consumer effects remain committed. |

Core ML compute-unit settings permit devices. They do not guarantee Neural Engine placement, cache residency, a particular internal arithmetic precision, or zero-copy input. Shared CPU and GPU memory still requires synchronization. Framework-managed allocations remain outside SwiftSci's allocator control.

## Memory accounting and completion

`payloadByteCount` measures logical values and validity bytes. It excludes spare capacity, metadata, row provenance, retained snapshots, and framework storage.

`MemoryBudget` admits declared byte estimates. It is not a process RSS limit. Reservations cover the lifetime defined by the operation, including awaited consumers where documented. Caller-held data, escaped results, loader working memory, and consumer retention require caller accounting. Known stored capacities count toward pipeline admission. Implicit identity row mappings have no retained index buffer to charge.

Swift reference counting and value ownership release storage. There is no separate garbage collector. GPU completion, task completion, pool shutdown, and reservation completion have distinct lifetimes. Returning a framework call does not by itself prove all asynchronous use has ended.

## Supported limits

The package keeps its declared Swift tools and macOS minimum versions. Local validation on a newer compiler does not certify every older compiler or device. Hosted CI and device-specific tests provide separate evidence.

Fixed-size batch mutation is supported. General borrowed mutable foreign arrays, arbitrary dtype expansion, automatic device routing, custom Metal staging, and guarantees about Neural Engine memory placement are outside this contract. The [benchmark suite](../Benchmarks/README.md) records correctness and resource evidence for declared workloads. A passing numerical regression policy does not mean every numerical conformance case passes, and variable timings do not establish a certified speed baseline.
