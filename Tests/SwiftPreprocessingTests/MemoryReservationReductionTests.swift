import Testing
import SwiftPreprocessing

@Suite("Completed workspace release")
struct MemoryReservationReductionTests {
    @Test func reductionKeepsOutputAndAdmitsWaitingWork() async throws {
        let budget = try MemoryBudget(limit: 100)
        let owner = try await budget.acquire(MemoryEstimate(capacities: [100]))
        let waiter = Task { try await budget.acquire(MemoryEstimate(capacities: [40])) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while await budget.queuedCount == 0, ContinuousClock.now < deadline { await Task.yield() }
        #expect(await budget.queuedCount == 1)
        try await owner.reduce(to: MemoryEstimate(capacities: [60]))
        let next = try await waiter.value
        #expect(await budget.reservedBytes == 100)
        #expect(await budget.peak == 100)
        await next.finish()
        #expect(await budget.reservedBytes == 60)
        try await owner.reduce(to: MemoryEstimate(capacities: [60]))
        await #expect(throws: MemoryAdmissionError.self) {
            try await owner.reduce(to: MemoryEstimate(capacities: [61]))
        }
        #expect(await budget.reservedBytes == 60)
        await owner.finish()
        await owner.finish()
        #expect(await budget.reservedBytes == 0)
        await #expect(throws: MemoryAdmissionError.self) {
            try await owner.reduce(to: MemoryEstimate(capacities: [0]))
        }
    }
}
