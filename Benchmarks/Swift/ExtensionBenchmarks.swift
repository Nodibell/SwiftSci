import Foundation
import SwiftVision
import SwiftDatabase
import SwiftAgent
import SwiftExplain
import SwiftOptimize
import SwiftDataFrame
import SwiftML
import SwiftNLP
import SwiftPreprocessing

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
        var rng = BenchmarkLCG(seed: 42)
        let shapRes = await BenchmarkRunner.run(name: "TreeSHAP Explanation (100 samples)", module: module, warmup: 2, iterations: 10) {
            let shap = TreeSHAP()
            let features = (0..<10).map { _ in (0..<5).map { _ in rng.nextDouble(in: 0.0...10.0) } }
            _ = await shap.explain(model: { $0.reduce(0, +) }, features: features, numCoalitions: 20)
        }
        results.append(shapRes)

        // 5. OneVsRestClassifier Multi-Class Fit benchmark
        let ovrFeats = (0..<100).map { _ in (0..<5).map { _ in rng.nextDouble(in: 0.0...1.0) } }
        let ovrTargets = (0..<100).map { Double($0 % 5) }
        let ovrRes = await BenchmarkRunner.run(name: "OneVsRestClassifier (5 classes, 100 samples)", module: module, warmup: 2, iterations: 10) {
            let ovr = OneVsRestClassifier(numClasses: 5)
            _ = try await ovr.fit(features: ovrFeats, targets: ovrTargets)
        }
        results.append(ovrRes)

        // 6. TF-IDF Text Vectorization benchmark
        let tfidfRes = await BenchmarkRunner.run(name: "TF-IDF Vectorizer (50 documents)", module: module, warmup: 2, iterations: 10) {
            let vec = TFIDFVectorizer()
            let docs = Array(repeating: "уряд ухвалив новий законопроект про бюджет на наступний рік", count: 50)
            _ = try await vec.fitTransform(docs)
        }
        results.append(tfidfRes)

        // 7. Forecast & Regression Quality Error Metrics (100k predictions)
        let nMetrics = 100_000
        let yTrue = (0..<nMetrics).map { _ in rng.nextDouble(in: 10.0...100.0) }
        let yPred = yTrue.map { $0 + rng.nextDouble(in: -5.0...5.0) }
        let metricsRes = await BenchmarkRunner.run(name: "Forecast Errors Suite (RMSE, MAE, MAPE, R² 100k)", module: module, warmup: 2, iterations: 10) {
            _ = Metrics.rootMeanSquaredError(yTrue: yTrue, yPred: yPred)
            _ = Metrics.meanAbsoluteError(yTrue: yTrue, yPred: yPred)
            _ = Metrics.mape(yTrue: yTrue, yPred: yPred)
            _ = Metrics.r2Score(yTrue: yTrue, yPred: yPred)
        }
        results.append(metricsRes)

        // 8. Classification ROC-AUC Metric (50k predictions)
        let yTrueBin = (0..<50_000).map { _ in rng.next() % 2 == 0 ? 1 : 0 }
        let yScore = (0..<50_000).map { _ in rng.nextDouble(in: 0.0...1.0) }
        let rocAucRes = await BenchmarkRunner.run(name: "Classification ROC-AUC (50k predictions)", module: module, warmup: 2, iterations: 5) {
            _ = Metrics.rocAUC(yTrue: yTrueBin, yScore: yScore)
        }
        results.append(rocAucRes)

        // 9. OneHotEncoder (50k rows, 2 categorical cols)
        let catData = (0..<50_000).map { ["dept_\($0 % 8)", "region_\($0 % 4)"] }
        let oheRes = await BenchmarkRunner.run(name: "OneHotEncoder fitTransform (50k rows)", module: module, warmup: 2, iterations: 5) {
            let ohe = OneHotEncoder()
            ohe.fit(catData)
            _ = try ohe.transform(catData)
        }
        results.append(oheRes)

        // 10. VADER Sentiment Analyzer (1k sentences)
        let sentences = (0..<1_000).map { i in
            i % 2 == 0 ? "This scientific framework is absolutely fantastic and super fast!" : "The model prediction was awful, terrible error rate and disappointing results."
        }
        let vaderRes = await BenchmarkRunner.run(name: "VADER Sentiment Analysis (1k sentences)", module: module, warmup: 2, iterations: 10) {
            let vader = VADERSentimentAnalyzer()
            for s in sentences {
                _ = vader.polarityScores(text: s)
            }
        }
        results.append(vaderRes)

        // 11. NaiveBayesClassifier (1k docs × 100 features, 3 classes)
        let nbX = (0..<1_000).map { _ in (0..<100).map { _ in Double(rng.next() % 10) } }
        let nbY = (0..<1_000).map { Double($0 % 3) }
        let nbRes = await BenchmarkRunner.run(name: "NaiveBayesClassifier fit (1k×100, 3 classes)", module: module, warmup: 2, iterations: 10) {
            let nb = NaiveBayesClassifier()
            try await nb.fit(features: nbX, targets: nbY)
        }
        results.append(nbRes)

        return results
    }
}
