import Testing
import Foundation
@testable import SwiftML

@Suite("LinearSVC Tests")
struct LinearSVCTests {

    @Test("LinearSVC CPU training and separable data classification")
    func testLinearSVCCPU() async throws {
        // Linearly separable data
        let features: [[Double]] = [
            [-2.0, -2.0],
            [-1.5, -1.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [1.5, 1.0],
            [2.0, 2.0]
        ]
        let targets: [Double] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        let svc = LinearSVC(C: 1.0, device: .cpu)
        try await svc.fit(features: features, targets: targets, learningRate: 0.1, epochs: 200)

        let preds = try await svc.predict(features: features)
        #expect(preds == [0, 0, 0, 1, 1, 1])

        let dec = try await svc.decisionFunction(features: features)
        #expect(dec.count == 6)
        #expect(dec[0] < 0.0)
        #expect(dec[5] > 0.0)
    }

    @Test("LinearSVC Auto device routing on small shape uses CPU")
    func testLinearSVCAutoSmallShape() async throws {
        let features: [[Double]] = [
            [-1.0, -1.0],
            [1.0, 1.0]
        ]
        let targets: [Double] = [0.0, 1.0]

        let svc = LinearSVC(C: 1.0, device: .auto)
        try await svc.fit(features: features, targets: targets)

        let resolved = await svc.resolvedDevice
        #expect(resolved == .cpu)
    }

    @Test("LinearSVC input validation and error handling")
    func testLinearSVCValidation() async throws {
        let svc = LinearSVC(C: 1.0)
        await #expect(throws: (any Error).self) {
            try await svc.fit(features: [], targets: [])
        }
        await #expect(throws: (any Error).self) {
            try await svc.fit(features: [[1.0, 2.0]], targets: [1.0, 0.0])
        }
    }
}
