import Testing
import Foundation
@testable import SwiftCluster

@Suite("Cluster & Anomaly Leaderboard Metrics Tests (G-024)")
struct LeaderboardMetricsTests {

    @Test("SilhouetteScore with ignoreNoise excludes noise points and applies proportional scaling")
    func testSilhouetteWithIgnoreNoise() throws {
        // Two distinct clusters: cluster 0 around (0, 0), cluster 1 around (10, 10), and one noise point (-1)
        let features: [[Double]] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
            [50.0, 50.0] // noise
        ]
        let labels = [0, 0, 0, 1, 1, 1, -1]

        let score = try SilhouetteScore.compute(features: features, labels: labels, ignoreNoise: true)
        #expect(score > 0.5)

        // Standard without ignoreNoise (treating -1 as single-element cluster)
        let standardScore = try SilhouetteScore.compute(features: features, labels: labels, ignoreNoise: false)
        #expect(standardScore >= 0.0)
    }

    @Test("evaluateClustering produces standardized leaderboard metrics record")
    func testEvaluateClustering() throws {
        let features: [[Double]] = [
            [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
            [8.0, 8.0], [8.1, 7.9], [7.9, 8.1]
        ]
        let labels = [0, 0, 0, 1, 1, 1]

        let metrics = try ClusteringMetrics.evaluateClustering(features: features, labels: labels)
        #expect(metrics.silhouette > 0.5)
        #expect(metrics.calinskiHarabasz > 0.0)
        #expect(metrics.daviesBouldin >= 0.0)
        #expect(metrics.noiseRatio == 0.0)
        #expect(metrics.compositeScore > 0.5)
    }

    @Test("anomalySeparationScore measures distribution contrast between inliers and outliers")
    func testAnomalySeparationScore() {
        let scores = [0.1, 0.12, 0.08, 0.15, 0.85, 0.92, 0.88]
        let labels = [1, 1, 1, 1, -1, -1, -1]

        let sep = ClusteringMetrics.anomalySeparationScore(scores: scores, labels: labels)
        #expect(sep > 2.0)

        // Edge case: empty or single class
        #expect(ClusteringMetrics.anomalySeparationScore(scores: [], labels: []) == 0.0)
        #expect(ClusteringMetrics.anomalySeparationScore(scores: [0.5, 0.6], labels: [1, 1]) == 0.0)
    }
}
