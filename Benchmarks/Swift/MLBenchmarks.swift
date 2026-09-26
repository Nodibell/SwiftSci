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
        let xMat = BenchmarkDataLoader.load2DMatrix(filename: "regression_10k_10_X.bin", rows: rows, cols: cols) {
            var rng = BenchmarkLCG(seed: seed)
            return (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 1000) / 100.0 - 5.0 } }
        }
        let yVec = BenchmarkDataLoader.loadDoubleVector(filename: "regression_10k_10_y.bin", fallbackCount: rows, fallbackSeed: seed)
        return (xMat, yVec)
    }

    private static func makeClassification(rows: Int, cols: Int = 4, seed: UInt64 = 42) -> ([[Double]], [Double]) {
        let xMat = BenchmarkDataLoader.load2DMatrix(filename: "classification_1k_4_X.bin", rows: rows, cols: cols) {
            var rng = BenchmarkLCG(seed: seed)
            return (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 1000) / 100.0 - 5.0 } }
        }
        let yVec = BenchmarkDataLoader.loadDoubleVector(filename: "classification_1k_4_y.bin", fallbackCount: rows, fallbackSeed: seed)
        return (xMat, yVec)
    }

    private static func makeCluster(filename: String? = nil, rows: Int, cols: Int = 4, seed: UInt64 = 42) -> [[Double]] {
        if let filename {
            return BenchmarkDataLoader.load2DMatrix(filename: filename, rows: rows, cols: cols) {
                var rng = BenchmarkLCG(seed: seed)
                return (0..<rows).map { _ in (0..<cols).map { _ in Double(rng.next() % 2000) / 100.0 - 10.0 } }
            }
        }
        var rng = BenchmarkLCG(seed: seed)
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
        let gbX = rfX
        let gbY = rfX.map { $0[0] * 2.0 + sin($0[1]) }
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
        let kmeansData = MLBenchmarks.makeCluster(filename: "kmeans_10k_4.bin", rows: 10_000, cols: 4)
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
        let pcaData = MLBenchmarks.makeCluster(filename: "pca_1k_100.bin", rows: 1_000, cols: 100, seed: 77)
        let pcaResult = await BenchmarkRunner.run(
            name: "PCA SVD fitTransform (1k×100 → 10 comps)",
            module: "SwiftCluster",
            warmup: 1,
            iterations: 5
        ) {
            let pca = try PCA(nComponents: 10)
            _ = try await pca.fitTransform(pcaData)
        }
        results.append(pcaResult)

        // ── 6. Isolation Forest (SwiftCluster) ────────────────────────────
        let isoData = MLBenchmarks.makeCluster(rows: 1_000, cols: 10, seed: 99)
        let isoResult = await BenchmarkRunner.run(
            name: "IsolationForest fit (1k×10, 100 trees)",
            module: "SwiftCluster",
            warmup: 1,
            iterations: 5
        ) {
            _ = try IsolationForest.fit(data: isoData, nEstimators: 100)
        }
        results.append(isoResult)

        // ── 7. LinearSVC (SwiftML, Metal GPU) ────────────────────────────
        let (svcX, svcY) = MLBenchmarks.makeClassification(rows: 1_000, cols: 4)
        let svcResult = await BenchmarkRunner.run(
            name: "LinearSVC fit (1k×4, 100 epochs, Metal GPU)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let svc = LinearSVC(C: 1.0, device: .auto)
            try await svc.fit(features: svcX, targets: svcY, learningRate: 0.01, epochs: 100)
        }
        results.append(svcResult)

        // ── 8. VectorStore Similarity Search (5k vectors × 128 dim) ───────
        let store = VectorStore(metric: .cosineSimilarity)
        var vRng = BenchmarkLCG(seed: 777)
        let dim = 128
        let entries = (0..<5_000).map { i in
            VectorEntry(id: "doc_\(i)", vector: (0..<dim).map { _ in vRng.nextDouble(in: -1.0...1.0) })
        }
        store.addBatch(entries: entries)
        let queryVec = (0..<dim).map { _ in vRng.nextDouble(in: -1.0...1.0) }

        let vstoreResult = await BenchmarkRunner.run(
            name: "VectorStore Cosine Search (5k × 128d, top 10)",
            module: "SwiftCluster",
            warmup: 2,
            iterations: 10
        ) {
            _ = store.search(query: queryVec, topK: 10)
        }
        results.append(vstoreResult)

        return results
    }
}
