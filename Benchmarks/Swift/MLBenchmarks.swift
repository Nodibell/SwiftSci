// MLBenchmarks.swift
// Benchmarks for SwiftML vs Scikit-Learn:
//   • Linear Regression  (10k rows, 10 features, 100 epochs)
//   • Random Forest Classifier (1k rows, 4 features, 50 trees, maxDepth=4)
//   • GBDT Regressor (1k rows, 4 features, 50 estimators)
//   • K-Means (10k points, 3 clusters, via SwiftCluster)
//   • PCA SVD (1k rows × 100 cols, via SwiftCluster)

import Foundation
import SwiftML
import SwiftCluster

struct MLBenchmarks: BenchmarkSuite {
    let module = "SwiftML"

    // MARK: – Data generators

    private static func makeRegression(rows: Int, cols: Int, seed: UInt64 = 42) -> ([[Double]], [Double]) {
        var rng = LCGML(seed: seed)
        let weights = (0..<cols).map { Double($0 + 1) }   // w_i = i+1
        let X = (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 1000) / 100.0 - 5.0 } }
        let y = X.map { row in zip(row, weights).map(*).reduce(0, +) + 1.0 }
        return (X, y)
    }

    private static func makeClassification(rows: Int, cols: Int = 4, seed: UInt64 = 42) -> ([[Double]], [Double]) {
        var rng = LCGML(seed: seed)
        let X = (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 1000) / 100.0 - 5.0 } }
        let y = X.map { row in row[0] > 0 && row[1] > 0 ? 0.0 : 1.0 }
        return (X, y)
    }

    private static func makeCluster(rows: Int, cols: Int = 4, seed: UInt64 = 42) -> [[Double]] {
        var rng = LCGML(seed: seed)
        return (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 2000) / 100.0 - 10.0 } }
    }

    // MARK: – Run all benchmarks

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // ── 1. Linear Regression ──────────────────────────────────────────
        let (lrX, lrY) = MLBenchmarks.makeRegression(rows: 10_000, cols: 10)
        let linRegResult = await BenchmarkRunner.run(
            name: "LinearRegression fit (10k×10, 100 epochs)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let model = LinearRegression()
            try await model.fit(features: lrX, targets: lrY, learningRate: 0.01, epochs: 100)
        }
        results.append(linRegResult)

        // ── 2. Random Forest Classifier ───────────────────────────────────
        let (rfX, rfY) = MLBenchmarks.makeClassification(rows: 1_000, cols: 4)
        let rfResult = await BenchmarkRunner.run(
            name: "RandomForest fit (1k×4, 50 trees)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let rf = try RandomForestClassifier(nEstimators: 50, maxDepth: 4, criterion: .gini)
            try await rf.fit(features: rfX, targets: rfY)
        }
        results.append(rfResult)

        // ── 3. GBDT Regressor ─────────────────────────────────────────────
        let (gbX, gbY) = MLBenchmarks.makeRegression(rows: 1_000, cols: 4)
        let gbResult = await BenchmarkRunner.run(
            name: "GBDT Regressor fit (1k×4, 50 est.)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let gbdt = try GradientBoostedTreesRegressor(nEstimators: 50, learningRate: 0.1, maxDepth: 3)
            try await gbdt.fit(features: gbX, targets: gbY)
        }
        results.append(gbResult)

        // ── 4. K-Means (SwiftCluster) ─────────────────────────────────────
        let kmeansData = MLBenchmarks.makeCluster(rows: 10_000, cols: 4)
        let kmeansResult = await BenchmarkRunner.run(
            name: "KMeans fit (10k×4, 3 clusters)",
            module: "SwiftCluster",
            warmup: 1,
            iterations: 5
        ) {
            let km = try KMeans(nClusters: 3, maxIterations: 50)
            try await km.fit(features: kmeansData)
        }
        results.append(kmeansResult)

        // ── 5. PCA SVD (SwiftCluster) ─────────────────────────────────────
        let pcaData = MLBenchmarks.makeCluster(rows: 1_000, cols: 100, seed: 77)
        let pcaResult = await BenchmarkRunner.run(
            name: "PCA SVD fit (1k×100 → 10 comps)",
            module: "SwiftCluster",
            warmup: 1,
            iterations: 5
        ) {
            let pca = try PCA(nComponents: 10)
            _ = try await pca.fitTransform(pcaData)
        }
        results.append(pcaResult)

        return results
    }
}

private struct LCGML {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}
