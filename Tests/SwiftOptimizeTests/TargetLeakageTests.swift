import Testing
import Foundation
import SwiftDataFrame
import SwiftML
@testable import SwiftOptimize

@Suite("Target Leakage Detector Sentry Tests")
struct TargetLeakageTests {

    @Test("Clean dataset without leakage yields clean report")
    func testCleanDataset() throws {
        let X = [
            [2.5, 9.1],
            [1.2, 3.4],
            [5.8, 8.2],
            [3.1, 1.5],
            [4.0, 7.3],
            [0.5, 4.2]
        ]
        let y = [10.0, 14.0, 9.0, 21.0, 18.0, 15.0]
        let report = try TargetLeakageDetector.audit(
            features: X,
            featureNames: ["f1", "f2"],
            targets: y
        )
        
        #expect(!report.hasSevereLeakage)
        #expect(report.violations.isEmpty)
        #expect(report.recommendedExclusions.isEmpty)
    }

    @Test("Duplicate target vector is flagged with severe leakage")
    func testDuplicateTargetLeakage() throws {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0]
        let X = [
            [5.0, 1.0],
            [1.0, 2.0],
            [8.0, 3.0],
            [2.0, 4.0],
            [9.0, 5.0]
        ]
        let report = try TargetLeakageDetector.audit(
            features: X,
            featureNames: ["feature_normal", "target_copy"],
            targets: y
        )
        
        #expect(report.hasSevereLeakage)
        #expect(report.recommendedExclusions.contains("target_copy"))
        #expect(!report.recommendedExclusions.contains("feature_normal"))
        
        let violation = report.violations.first { $0.featureName == "target_copy" }
        #expect(violation?.type == .duplicateTarget)
    }

    @Test("Extreme linear correlation is flagged and recommended for exclusion")
    func testExtremeCorrelationLeakage() throws {
        let y = [10.0, 20.0, 30.0, 40.0, 50.0]
        let xLeaked = y.map { $0 * 2.0 + 0.00001 }
        let xNormal = [5.0, 1.0, 8.0, 2.0, 9.0]
        
        var X: [[Double]] = []
        for i in 0..<y.count {
            X.append([xNormal[i], xLeaked[i]])
        }
        
        let report = try TargetLeakageDetector.audit(
            features: X,
            featureNames: ["normal_feat", "leaked_feat"],
            targets: y
        )
        
        #expect(report.hasSevereLeakage)
        #expect(report.recommendedExclusions.contains("leaked_feat"))
        #expect(!report.recommendedExclusions.contains("normal_feat"))
        
        let violation = report.violations.first { $0.featureName == "leaked_feat" }
        #expect(violation?.type == .highLinearCorrelation)
    }

    @Test("Monotonic sequential identifier index memorizing target order is flagged")
    func testIdentifierLeakage() throws {
        let n = 20
        var X: [[Double]] = []
        var y: [Double] = []
        
        for i in 0..<n {
            X.append([Double(i), Double(i % 3)])
            y.append(Double(i) * 10.0)
        }
        
        let report = try TargetLeakageDetector.audit(
            features: X,
            featureNames: ["row_id", "category"],
            targets: y
        )
        
        #expect(report.hasSevereLeakage)
        #expect(report.recommendedExclusions.contains("row_id"))
    }

    @Test("DataFrame auditing correctly detects leaked columns")
    func testDataFrameAudit() throws {
        let col1 = TypedColumn<Double>(name: "clean_feature", values: [5.2, 1.4, 8.7, 2.1])
        let col2 = TypedColumn<Double>(name: "cheating_feature", values: [10.0, 20.0, 30.0, 40.0])
        let target = TypedColumn<Double>(name: "label", values: [10.0, 20.0, 30.0, 40.0])
        
        let df = try DataFrame(columns: [col1, col2, target])
        let report = try TargetLeakageDetector.audit(dataFrame: df, targetColumn: "label")
        
        #expect(report.hasSevereLeakage)
        #expect(report.recommendedExclusions.contains("cheating_feature"))
        #expect(!report.recommendedExclusions.contains("clean_feature"))
    }
}
