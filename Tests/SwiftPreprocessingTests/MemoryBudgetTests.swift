import Testing
import SwiftPreprocessing

@Suite("Whole-operation memory admission")
struct MemoryBudgetTests {
    private enum Failure: Error, Equatable { case injected }

    // Bound observation waits so a broken queue fails instead of hanging the suite.
    private func waitUntil(_ predicate: () async -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while ContinuousClock.now < deadline {
            if await predicate() { return true }
            await Task.yield()
        }
        return false
    }

    @Test func rejectsInvalidEstimatesAndBudgets() throws {
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try MemoryEstimate(capacities: [-1]) }
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try MemoryEstimate(capacities: [Int.max, 1]) }
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try MemoryBudget(limit: 0) }
        #expect(throws: MemoryAdmissionError.invalidCapacity) { try MemoryBudget(limit: -1) }
        #expect(try MemoryEstimate(capacities: []).bytes == 0)
        #expect(try MemoryEstimate(capacities: [Int.max, 0]).bytes == Int.max)
    }

    @Test func scopedCompletionErrorAndOversizeReleaseAllBytes() async throws {
        let budget = try MemoryBudget(limit: 100)
        let plan = try MemoryEstimate(capacities: [40, 20])
        let result = try await budget.withReservation(plan) {
            #expect(await budget.reservedBytes == 60)
            return 42
        }
        #expect(result == 42)
        #expect(await budget.reservedBytes == 0)
        await #expect(throws: Failure.injected) {
            try await budget.withReservation(plan) { throw Failure.injected }
        }
        #expect(await budget.reservedBytes == 0)
        await #expect(throws: MemoryAdmissionError.exceedsLimit(required: 101, limit: 100)) {
            try await budget.acquire(MemoryEstimate(capacities: [101]))
        }
        #expect(await budget.queuedCount == 0)
        let reservation = try await budget.acquire(plan)
        await reservation.finish()
        await reservation.finish()
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.peak == 60)
    }

    @Test func fifoDoesNotPartiallyReserveOrBypassBlockedHead() async throws {
        let budget = try MemoryBudget(limit: 100)
        let first = try await budget.acquire(MemoryEstimate(capacities: [60]))
        let large = Task { try await budget.acquire(MemoryEstimate(capacities: [50])) }
        let largeQueued = await waitUntil { await budget.queuedCount == 1 }
        #expect(largeQueued)
        let small = Task { try await budget.acquire(MemoryEstimate(capacities: [20])) }
        let bothQueued = await waitUntil { await budget.queuedCount == 2 }
        #expect(bothQueued)
        #expect(await budget.reservedBytes == 60)
        // Removing the blocked head must admit the smaller request immediately.
        large.cancel()
        let drained = await waitUntil { await budget.queuedCount == 0 }
        #expect(drained)
        #expect(await budget.reservedBytes == 80)
        await first.finish()
        small.cancel()
        if let reservation = try? await small.value { await reservation.finish() }
        await #expect(throws: CancellationError.self) { try await large.value }
        #expect(await budget.reservedBytes == 0)
    }

    @Test func cancellationBeforeAcquiringDoesNotReserve() async throws {
        let budget = try MemoryBudget(limit: 100)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await budget.acquire(MemoryEstimate(capacities: [100]))
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }

    @Test func scopedCancellationReleasesGrant() async throws {
        let budget = try MemoryBudget(limit: 100)
        let task = Task {
            try await budget.withReservation(MemoryEstimate(capacities: [100])) {
                withUnsafeCurrentTask { $0?.cancel() }
                try Task.checkCancellation()
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await budget.reservedBytes == 0)
    }

    @Test func concurrentOperationsRespectLimit() async throws {
        let budget = try MemoryBudget(limit: 100)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<64 {
                group.addTask {
                    try await budget.withReservation(MemoryEstimate(capacities: [30])) {
                        #expect(await budget.reservedBytes <= 100)
                        await Task.yield()
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(await budget.peak <= 100)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }

    @Test func destroyingLastReferenceEventuallyReleasesReservation() async throws {
        let budget = try MemoryBudget(limit: 100)
        var reservation: MemoryReservation? = try await budget.acquire(MemoryEstimate(capacities: [100]))
        #expect(reservation != nil)
        #expect(await budget.reservedBytes == 100)
        reservation = nil
        let released = await waitUntil { await budget.reservedBytes == 0 }
        #expect(released)
    }
}
