import Testing
import Foundation
import os
@testable import SwiftML
@testable import SwiftPreprocessing

@Suite("Concurrency and Memory Management Tests")
struct ConcurrencyTests {
    
    @Test("Concurrent actor execution does not race")
    func testConcurrentActorExecution() async throws {
        let model = LinearRegression()
        
        let features: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0]
        ]
        let targets: [Double] = [5.0, 10.0, 15.0]
        
        // Spawn 10 concurrent tasks training the same actor model
        await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    try await model.fit(features: features, targets: targets, learningRate: 0.01, epochs: 10)
                    _ = try await model.predict(features: [[1.0, 2.0]])
                }
            }
        }
        
        let weights = await model.getWeights()
        #expect(weights != nil)
    }
    
    @Test("WiredMemoryManager rate limiting and ticket release")
    func testWiredMemoryManagerLimits() async throws {
        // Create a custom manager with maxConcurrentTasks = 2
        let manager = WiredMemoryManager(maxConcurrentTasks: 2)
        
        let t1 = try await manager.acquireTicket()
        let t2 = try await manager.acquireTicket()
        
        // Verify that 2 tasks are active
        let activeCount = await manager.activeTasks()
        #expect(activeCount == 2)
        
        let expBox = OSAllocatedUnfairLock<WiredMemoryTicket?>(initialState: nil)
        let expectation = expectation {
            // Task 3 will suspend because limit is 2
            let t3 = try await manager.acquireTicket()
            expBox.withLock { $0 = t3 }
        }
        
        // Verify task 3 is suspended (manager active count is still 2 because Task 3 is in queue)
        try await Task.sleep(for: .milliseconds(100))
        let activeCountAfterSuspend = await manager.activeTasks()
        #expect(activeCountAfterSuspend == 2)
        
        // Finish t1, releasing a ticket
        await t1.finish()
        
        // Now Task 3 should be able to acquire and manager active count remains 2
        await expectation.wait(timeout: 1.0)
        let activeCountAfterRelease = await manager.activeTasks()
        #expect(activeCountAfterRelease == 2)
        
        // Release remaining tickets
        await t2.finish()
        let t3 = expBox.withLock { $0 }
        if let t3 = t3 {
            await t3.finish()
        }
    }
    
    @Test("WiredMemoryManager withTicket scoped execution and cancellation")
    func testWiredMemoryManagerWithTicketAndCancellation() async throws {
        let manager = WiredMemoryManager(maxConcurrentTasks: 1)
        
        // test withTicket
        let result = try await manager.withTicket {
            let active = await manager.activeTasks()
            #expect(active == 1)
            return "success"
        }
        #expect(result == "success")
        let activeAfter = await manager.activeTasks()
        #expect(activeAfter == 0)
        
        // test task cancellation triggers cancellation error
        let t1 = try await manager.acquireTicket()
        
        let task = Task {
            try await manager.acquireTicket()
        }
        
        // Wait for it to queue up
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        
        do {
            _ = try await task.value
            Issue.record("Expected cancellation error but ticket was acquired")
        } catch is CancellationError {
            // expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        
        await t1.finish()
    }
}

// Simple helper for async expectations in Swift Testing
private final class AsyncExpectation: Sendable {
    private let continuation = OSAllocatedUnfairLock<CheckedContinuation<Void, Never>?>(initialState: nil)
    
    func fulfill() {
        continuation.withLock { cont in
            if let c = cont {
                c.resume()
                cont = nil
            }
        }
    }
    
    func wait(timeout: Double) async {
        let task = Task {
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
                self.continuation.withLock { cont in
                    cont = c
                }
            }
        }
        
        let timeoutTask = Task {
            try? await Task.sleep(for: .seconds(timeout))
            fulfill()
        }
        
        await task.value
        timeoutTask.cancel()
    }
}

private func expectation(block: @escaping @Sendable () async throws -> Void) -> AsyncExpectation {
    let exp = AsyncExpectation()
    Task {
        try? await block()
        exp.fulfill()
    }
    return exp
}
