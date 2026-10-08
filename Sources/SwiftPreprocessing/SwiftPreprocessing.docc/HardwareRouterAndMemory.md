# Hardware routing and memory concurrency

SwiftSci selects CPU or GPU execution using the requested device and algorithm-specific size thresholds. Explicit `.cpu` and `.gpu` requests override automatic routing. The current training router maps `.ane` to `.gpu`; Core ML deployment handles ANE execution separately.

## Automatic routing

`HardwareRouter.resolveDevice` currently uses these rules:

| Algorithm | CPU condition for `.auto` |
| --- | --- |
| KMeans | Fewer than 500,000 input cells |
| PCA | Fewer than 2,000 samples and fewer than 500 features |
| LinearSVC | Fewer than 50,000 input cells |
| LinearRegression and LogisticRegression | Fewer than 1,000 samples |
| MLP variants | Fewer than 5,000 samples |
| Other current algorithms | CPU |

These are heuristics, not measured guarantees that a device is faster. A route can also change numerical representation and algorithm. For example, LinearRegression uses Double inputs for its CPU solver and converts inputs to Float for GPU gradient descent. Select the device explicitly when a workflow requires a particular numerical path.

## Estimated-byte admission

Use ``MemoryBudget`` to admit whole ``MemoryEstimate`` reservations under a caller-selected byte limit. It queues requests in order, rejects oversized estimates, and releases reservations after scoped operations finish or throw. Include retained inputs, copies, numerical workspace, outputs, and a runtime allowance in each estimate.

Use `withReservation` for a general operation, or the budgeted prepared regression overloads in SwiftML. This helper accepts the caller's estimate and returns an independent CPU array:

```swift
import SwiftPreprocessing
import MLX

func doubled(
    _ values: [Float],
    budget: MemoryBudget,
    estimate: MemoryEstimate
) async throws -> [Float] {
    try await budget.withReservation(estimate) {
        defer { Stream.gpu.synchronize() }
        let input = MLXArray(values)
        let result = multiply(input, Float(2), stream: .gpu)
        result.eval()
        return result.asArray(Float.self)
    }
}
```

Complete submitted device work before leaving a reservation scope, including on error and cancellation paths. Persistent model state and returned arrays can outlive the reservation and must be accounted for separately. The budget does not intercept allocations or enforce a process-wide memory cap.

## MLX allocation and wired-memory controls

For MLX-owned buffers, use MLX's existing `Memory` controls for cache limits, allocation limits, and memory statistics. Coordinate changes to these process-wide settings across callers. Runtime allocator counters can include imported buffers already counted by a host owner; adding those counters can double-count shared storage.

For MLX wired-memory coordination, use the module-qualified `MLX.WiredMemoryManager` and `MLX.WiredMemoryTicket`. They are separate from SwiftSci's deprecated types with the same names. A wired limit controls memory that can remain resident; it is not SwiftSci's estimated-byte reservation budget.

MLX 0.32.3's built-in sum and max policies do not impose an admission cap. Admission is grouped by policy identity and does not provide SwiftSci's strict FIFO ordering. If using MLX's `withWiredLimit` helper, check task cancellation inside the body before starting work: the helper can invoke the body after admission is cancelled. Preserve oversized-request validation when defining a bounded policy. See [MLX's wired-memory implementation](https://github.com/ml-explore/mlx-swift/blob/0.32.3/Source/MLX/WiredMemory.swift) for its lifecycle contract.

Cache clearing is not a completion fence. Keep resources alive until their consumers finish. Choose cache policy independently from ticket cleanup; MLX already recycles released buffers.

## Legacy task-count tickets

``WiredMemoryManager`` and ``WiredMemoryTicket`` are deprecated and retained for source compatibility. They only limit concurrent ticket holders. They do not reserve bytes, set an MLX wired-memory limit, or guarantee that allocations fit available memory. Their shared instance derives its task-count limit from the active processor count.

Existing callers retain the same behavior. `withTicket` calls `finish()` on success and error. Explicit `finish()` clears the global MLX cache and releases the slot once; deinitialization releases the slot without clearing the cache. Callers must complete submitted work before finishing a ticket.

Migration depends on the requirement:

- For estimated-byte admission, use ``MemoryBudget`` and ``MemoryReservation`` with a complete operation estimate.
- For MLX wired-memory coordination, use MLX's manager and tickets with an appropriate shared policy.
- For a task-count limit, retain the legacy implementation until the caller has selected and tested a replacement that preserves its scheduling contract.

These interfaces are not interchangeable. Switching to a byte budget or MLX ticket changes policy and may require different cleanup and cancellation handling.

## Topics

### Routing and byte admission

- ``HardwareRouter``
- ``MemoryBudget``
- ``MemoryEstimate``
- ``MemoryReservation``
- ``MemoryAdmissionError``

### Deprecated task-count APIs

- ``WiredMemoryManager``
- ``WiredMemoryTicket``
