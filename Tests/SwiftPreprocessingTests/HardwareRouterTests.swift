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

    @Test("HardwareRouter thresholds sit exactly on documented boundaries")
    func testThresholdBoundaries() async {
        let r = HardwareRouter.shared
        // KMeans: n*p < 500_000
        #expect(await r.resolveDevice(for: "KMeans", sampleCount: 499_999, featureCount: 1, requestedDevice: .auto) == .cpu)
        #expect(await r.resolveDevice(for: "KMeans", sampleCount: 500_000, featureCount: 1, requestedDevice: .auto) == .gpu)
        // LinearSVC: n*p < 50_000
        #expect(await r.resolveDevice(for: "LinearSVC", sampleCount: 49_999, featureCount: 1, requestedDevice: .auto) == .cpu)
        #expect(await r.resolveDevice(for: "LinearSVC", sampleCount: 50_000, featureCount: 1, requestedDevice: .auto) == .gpu)
        // PCA: both n < 2000 and p < 500
        #expect(await r.resolveDevice(for: "PCA", sampleCount: 1_999, featureCount: 499, requestedDevice: .auto) == .cpu)
        #expect(await r.resolveDevice(for: "PCA", sampleCount: 2_000, featureCount: 499, requestedDevice: .auto) == .gpu)
        #expect(await r.resolveDevice(for: "PCA", sampleCount: 1_999, featureCount: 500, requestedDevice: .auto) == .gpu)
        // Regressions: n < 1000; MLP: n < 5000
        #expect(await r.resolveDevice(for: "LinearRegression", sampleCount: 999, featureCount: 5, requestedDevice: .auto) == .cpu)
        #expect(await r.resolveDevice(for: "LogisticRegression", sampleCount: 1_000, featureCount: 5, requestedDevice: .auto) == .gpu)
        #expect(await r.resolveDevice(for: "MLP", sampleCount: 4_999, featureCount: 5, requestedDevice: .auto) == .cpu)
        #expect(await r.resolveDevice(for: "MLP", sampleCount: 5_000, featureCount: 5, requestedDevice: .auto) == .gpu)
    }

    @Test("HardwareRouter does not trap on invalid or overflowing dimensions")
    func testInvalidDimensions() async {
        let r = HardwareRouter.shared
        for algo in ["KMeans", "LinearSVC", "PCA", "MLP", "LinearRegression", "Unknown"] {
            #expect(await r.resolveDevice(for: algo, sampleCount: 0, featureCount: 10, requestedDevice: .auto) == .cpu)
            #expect(await r.resolveDevice(for: algo, sampleCount: 10, featureCount: 0, requestedDevice: .auto) == .cpu)
            #expect(await r.resolveDevice(for: algo, sampleCount: -5, featureCount: -7, requestedDevice: .auto) == .cpu)
        }
        #expect(await r.resolveDevice(for: "KMeans", sampleCount: Int.max, featureCount: Int.max, requestedDevice: .auto) == .gpu)
        #expect(await r.resolveDevice(for: "LinearSVC", sampleCount: Int.max, featureCount: 2, requestedDevice: .auto) == .gpu)
        #expect(await r.resolveDevice(for: "RandomForest", sampleCount: Int.max, featureCount: Int.max, requestedDevice: .auto) == .cpu)
    }

    @Test("HardwareRouter maps ANE to GPU and honors .auto/.cpu/.gpu")
    func testAneAndAuto() async {
        let r = HardwareRouter.shared
        #expect(await r.resolveDevice(for: "KMeans", sampleCount: 1, featureCount: 1, requestedDevice: .ane) == .gpu)
        #expect(await r.resolveDevice(for: "KMeans", sampleCount: 10, featureCount: 10, requestedDevice: .auto) == .cpu)
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
