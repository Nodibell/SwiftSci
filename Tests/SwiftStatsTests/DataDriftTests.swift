import Testing
import SwiftStats
import SwiftDataFrame

@Suite("Data Drift & Distribution Shift Tests")
struct DataDriftTests {

    // MARK: - Wasserstein Distance Tests

    @Test("Wasserstein distance between identical distributions is 0.0")
    func testWassersteinIdentical() throws {
        let sample = [1.0, 2.0, 3.0, 4.0, 5.0]
        let distance = try Stats.wassersteinDistance(sample, sample)
        #expect(distance == 0.0)
    }

    @Test("Wasserstein distance under constant scalar shift matches absolute difference")
    func testWassersteinScalarShift() throws {
        let u = [1.0, 2.0, 3.0, 4.0, 5.0]
        let shift = 3.5
        let v = u.map { $0 + shift }
        
        let dist = try Stats.wassersteinDistance(u, v)
        #expect(abs(dist - shift) < 1e-6)
    }

    @Test("Wasserstein distance empty input throws")
    func testWassersteinEmptyInputThrows() throws {
        #expect(throws: StatsError.emptyInput) {
            _ = try Stats.wassersteinDistance([], [1.0, 2.0])
        }
        #expect(throws: StatsError.emptyInput) {
            _ = try Stats.wassersteinDistance([1.0, 2.0], [])
        }
    }

    @Test("Wasserstein distance with unequal sample sizes")
    func testWassersteinUnequalSampleSizes() throws {
        let u = [0.0, 1.0, 2.0]
        let v = [0.0, 0.5, 1.0, 1.5, 2.0]
        let dist = try Stats.wassersteinDistance(u, v)
        #expect(dist >= 0.0)
        #expect(dist < 1.0)
    }

    // MARK: - Population Stability Index Tests

    @Test("PSI between identical distributions indicates stable distribution")
    func testPSIIdentical() throws {
        let sample = Array(stride(from: 0.0, through: 100.0, by: 1.0))
        let result = try Stats.populationStabilityIndex(expected: sample, actual: sample, buckets: 10)
        
        #expect(result.psi < 0.01)
        #expect(result.severity == .stable)
        #expect(!result.buckets.isEmpty)
    }

    @Test("PSI detects moderate and significant drift under shifted distributions")
    func testPSIShift() throws {
        let baseline = Array(stride(from: 0.0, through: 100.0, by: 1.0))
        let shifted = baseline.map { $0 + 40.0 }
        
        let result = try Stats.populationStabilityIndex(expected: baseline, actual: shifted, buckets: 10)
        #expect(result.psi > 0.2)
        #expect(result.severity == .significant)
    }

    @Test("PSI error validation for empty input or invalid bucket counts")
    func testPSIValidationErrors() throws {
        #expect(throws: StatsError.emptyInput) {
            _ = try Stats.populationStabilityIndex(expected: [], actual: [1.0, 2.0])
        }
        #expect(throws: StatsError.invalidParameter("Number of buckets must be at least 2, got 1")) {
            _ = try Stats.populationStabilityIndex(expected: [1.0, 2.0], actual: [1.0, 2.0], buckets: 1)
        }
    }
}
