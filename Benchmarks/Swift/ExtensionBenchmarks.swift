import Foundation
import SwiftVision
import SwiftDatabase
import SwiftAgent
import SwiftExplain
import SwiftOptimize
import SwiftDataFrame
import SwiftML
import SwiftNLP

struct ExtensionBenchmarks: BenchmarkSuite {
    let module = "SwiftSci Extensions"

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // 1. SwiftVision CNN Feature Extraction & Vision Metrics benchmark
        let visionRes = await BenchmarkRunner.run(name: "CNN Feature Extraction & Vision Metrics", module: module, warmup: 2, iterations: 10) {
            let img = ImageDataset(width: 32, height: 32, channels: 3, data: Array(repeating: 0.8, count: 3072))
            let extractor = CNNFeatureExtractor()
            let feats = extractor.extractFeatures(image: img)
            _ = VisionMetrics.diceCoefficient(predicted: [feats], groundTruth: [feats])
        }
        results.append(visionRes)

        // 2. SwiftDatabase Ingestion benchmark
        let dbRes = await BenchmarkRunner.run(name: "SQLite Direct DataFrame Ingestion", module: module, warmup: 2, iterations: 10) {
            let dbURI = "file:bench_\(UUID().uuidString)?mode=memory&cache=shared"
            let conn = SQLiteConnection(databasePath: dbURI)
            _ = try await conn.executeQuery("CREATE TABLE test (id INT, val REAL);")
            _ = try await conn.executeQuery("INSERT INTO test VALUES (1, 10.5), (2, 20.0);")
            _ = try await DataFrame.fromSQL("SELECT * FROM test", connection: conn)
        }
        results.append(dbRes)

        // 3. SwiftAgent RAG Context generation benchmark
        let agentRes = await BenchmarkRunner.run(name: "RAG Context Summary Generation", module: module, warmup: 2, iterations: 10) {
            let df = DataFrame()
            let gen = RAGContextGenerator()
            _ = gen.generateSummary(df: df, name: "BenchDF")
        }
        results.append(agentRes)

        // 4. TreeSHAP benchmark
        let shapRes = await BenchmarkRunner.run(name: "TreeSHAP Explanation (100 samples)", module: module, warmup: 2, iterations: 10) {
            let shap = TreeSHAP()
            let features = Array(repeating: [1.0, 2.0, 3.0, 4.0, 5.0], count: 10)
            _ = await shap.explain(model: { $0.reduce(0, +) }, features: features, numCoalitions: 20)
        }
        results.append(shapRes)

        // 5. OneVsRestClassifier Multi-Class Fit benchmark
        let ovrRes = await BenchmarkRunner.run(name: "OneVsRestClassifier (5 classes, 100 samples)", module: module, warmup: 2, iterations: 10) {
            let ovr = OneVsRestClassifier(numClasses: 5)
            let feats = Array(repeating: [0.1, 0.5, 0.9, 0.2, 0.8], count: 100)
            let targets = Array(repeating: 1.0, count: 100)
            _ = try await ovr.fit(features: feats, targets: targets)
        }
        results.append(ovrRes)

        // 6. TF-IDF Text Vectorization benchmark
        let tfidfRes = await BenchmarkRunner.run(name: "TF-IDF Vectorizer (50 documents)", module: module, warmup: 2, iterations: 10) {
            let vec = TFIDFVectorizer()
            let docs = Array(repeating: "уряд ухвалив новий законопроект про бюджет на наступний рік", count: 50)
            _ = try await vec.fitTransform(docs)
        }
        results.append(tfidfRes)

        return results
    }
}
