import Foundation
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

struct MixedWorkloadRequest: Codable, Sendable {
    let model: String
    let rawFixture: String
    let rows: Int
    let policy: String
    let batches: Int
    let quantum: Int
}
struct MixedLaneResult: Codable, Sendable {
    let lane: String
    let completed: Int
    let admissionSeconds: [Double]
    let serviceSeconds: [Double]
    let completionSeconds: [Double]
    let cancelled: Bool
    let hashes: [String]
}
struct MixedSample: Codable {
    let mode: String
    let iteration: Int
    let wallSeconds: Double
    let budgetBytes: Int
    let peakReservedBytes: Int
    let finalReservedBytes: Int
    let finalQueued: Int
    let peakQueued: Int
    let peakResidentBytes: UInt64
    let finalResidentBytes: UInt64
    let thermalState: Int
    let lanes: [MixedLaneResult]
}
actor MixedStartGate {
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private let count: Int
    init(_ count: Int) { self.count = count }
    func arrive() async {
        if waiters.count + 1 == count {
            let ready = waiters; waiters = []; ready.forEach { $0.resume() }
        } else { await withCheckedContinuation { waiters.append($0) } }
    }
}
actor MixedSampler {
    private(set) var peakResident: UInt64 = 0
    private(set) var peakQueued = 0
    func sample(_ budget: MemoryBudget) async throws {
        let rss = try residentBytes()
        peakResident = max(peakResident, rss)
        peakQueued = max(peakQueued, await budget.queuedCount)
        guard rss < 1_073_741_824 else { throw BenchmarkFailure("Mixed workload exceeded 1 GiB sampled RSS cutoff") }
    }
}
