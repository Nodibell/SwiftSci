import Testing
import Foundation
@testable import SwiftCluster

@Suite("Outlier Detection Tests")
struct OutlierDetectionTests {
    
    @Test("IsolationForest identifies injected anomalies")
    func testIsolationForestAnomalies() throws {
        // Normal data cluster around (0, 0)
        var data: [[Double]] = []
        for _ in 0..<100 {
            data.append([Double.random(in: -1.0...1.0), Double.random(in: -1.0...1.0)])
        }
        // Injected extreme outliers
        data.append([50.0, 50.0])
        data.append([-50.0, -50.0])
        
        let forest = try IsolationForest.fit(data: data, nEstimators: 50, contamination: 0.05, seed: 42)
        let pred = try forest.predict(data: data)
        
        #expect(pred.labels.count == 102)
        #expect(pred.labels[100] == -1) // Outlier 1
        #expect(pred.labels[101] == -1) // Outlier 2
    }
    
    @Test("LocalOutlierFactor identifies density outliers")
    func testLocalOutlierFactor() throws {
        var data: [[Double]] = []
        for _ in 0..<50 {
            data.append([Double.random(in: -0.5...0.5), Double.random(in: -0.5...0.5)])
        }
        data.append([20.0, 20.0])
        
        let lof = LocalOutlierFactor(k: 10, contamination: 0.05)
        let pred = try lof.fitPredict(data: data)
        
        #expect(pred.labels.count == 51)
        #expect(pred.labels.last == -1) // Outlier
    }
}
