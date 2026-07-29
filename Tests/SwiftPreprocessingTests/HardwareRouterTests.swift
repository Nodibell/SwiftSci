import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("HardwareRouter Heuristics Tests (Phase 7)")
struct HardwareRouterTests {

    @Test("HardwareRouter routes MLP correctly based on sample count")
    func testMLPRouting() async {
        let router = HardwareRouter.shared

        // Small sample count -> CPU
        let smallDev = await router.resolveDevice(for: "MLPClassifier", sampleCount: 100, featureCount: 10, requestedDevice: .auto)
        #expect(smallDev == .cpu)

        // Large sample count -> GPU
        let largeDev = await router.resolveDevice(for: "MLPRegressor", sampleCount: 10_000, featureCount: 50, requestedDevice: .auto)
        #expect(largeDev == .gpu)
    }

    @Test("HardwareRouter routes tree algorithms always to CPU")
    func testTreeAlgorithmsRouting() async {
        let router = HardwareRouter.shared

        let rfDev = await router.resolveDevice(for: "RandomForestClassifier", sampleCount: 100_000, featureCount: 100, requestedDevice: .auto)
        #expect(rfDev == .cpu)

        let gbdtDev = await router.resolveDevice(for: "GradientBoostedTreesRegressor", sampleCount: 100_000, featureCount: 100, requestedDevice: .auto)
        #expect(gbdtDev == .cpu)

        let dtDev = await router.resolveDevice(for: "DecisionTreeClassifier", sampleCount: 100_000, featureCount: 100, requestedDevice: .auto)
        #expect(dtDev == .cpu)
    }

    @Test("HardwareRouter honors explicit device requests")
    func testExplicitDeviceRouting() async {
        let router = HardwareRouter.shared

        let explicitCpu = await router.resolveDevice(for: "MLPClassifier", sampleCount: 50_000, featureCount: 100, requestedDevice: .cpu)
        #expect(explicitCpu == .cpu)

        let explicitGpu = await router.resolveDevice(for: "DecisionTreeClassifier", sampleCount: 10, featureCount: 2, requestedDevice: .gpu)
        #expect(explicitGpu == .gpu)
    }
}
