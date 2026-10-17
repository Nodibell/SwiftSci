import Dispatch
import Foundation

extension PreparedNumericBatch {
    func gatheredColumns(at indices: [Int]) -> [CompactNumericColumn] {
        let source = columns
        // Require at least 2 MiB of values and 1,024 rows to amortize dispatch.
        let parallelRows = 262_143 / max(source.count, 1) + 1
        guard source.count >= 2, indices.count >= max(parallelRows, 1024) else {
            return source.map { $0.gathered(at: indices) }
        }
        let workers = min(source.count, 8, ProcessInfo.processInfo.activeProcessorCount)
        return [CompactNumericColumn](unsafeUninitializedCapacity: source.count) { output, initializedCount in
            let destination = CompactGatherDestination(base: output.baseAddress!)
            DispatchQueue.concurrentPerform(iterations: workers) { worker in
                for column in stride(from: worker, to: source.count, by: workers) {
                    destination.base.advanced(by: column).initialize(to: source[column].gathered(at: indices))
                }
            }
            initializedCount = source.count
        }
    }
}

// concurrentPerform joins before the borrowed array leaves scope. Workers
// initialize disjoint column slots, and each slot is initialized exactly once.
private struct CompactGatherDestination: @unchecked Sendable {
    let base: UnsafeMutablePointer<CompactNumericColumn>
}
