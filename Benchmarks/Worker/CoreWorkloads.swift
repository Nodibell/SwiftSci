import Foundation
import SwiftAgent
import SwiftDataFrame
import SwiftDatabase
import SwiftNLP
import SwiftOptimize
import SwiftPreprocessing
import SwiftSciBenchmarkSupport
import SwiftStats
import SwiftVision

struct CoreInputs {
  let labels: [Int]
  let categories: [[String]]
  let documents: [String]
  init(operation: String, rows: Int) {
    labels = operation == "roc-auc" ? (0..<rows).map { $0 % 2 } : []
    categories = operation == "onehot"
      ? (0..<rows).map { ["dept_\($0 % 8)", "region_\($0 % 4)"] } : []
    documents = operation == "tfidf"
      ? (0..<rows).map { $0 % 2 == 0 ? "alpha beta beta" : "beta gamma" } : []
  }
}

extension Worker {
  @inline(never) static func executeCore(_ op: String, x: [Double], y: [Double], inputs: CoreInputs)
    throws -> Output?
  {
    switch op {
    case "welch", "student", "paired":
      let result = op == "paired"
        ? try Stats.pairedTTest(before: x, after: y)
        : try Stats.tTest(sample1: x, sample2: y, equalVariances: op == "student")
      return .values([result.statistic, result.pValue, result.degreesOfFreedom,
        result.confidenceInterval.lower, result.confidenceInterval.upper, result.effectSize])
    case "anova":
      let result = try Stats.oneWayANOVA(groups: [x, y])
      return .values([result.fStatistic, result.pValue, Double(result.dfBetween), Double(result.dfWithin), result.etaSquared])
    case "regression-metrics":
      return .values([
        Metrics.rootMeanSquaredError(yTrue: x, yPred: y),
        Metrics.meanAbsoluteError(yTrue: x, yPred: y),
        Metrics.mape(yTrue: x, yPred: y), Metrics.r2Score(yTrue: x, yPred: y),
      ])
    case "roc-auc": return .values([Metrics.rocAUC(yTrue: inputs.labels, yScore: x)])
    case "onehot": return .values(try OneHotEncoder().fitTransform(inputs.categories).flatMap { $0 })
    case "tfidf":
      return .values(try TFIDFVectorizer(removeStopWords: false).fitTransform(inputs.documents).flatMap { $0 })
    case "rag-summary":
      return .text(RAGContextGenerator().generateSummary(df: DataFrame(), name: "BenchDF"))
    case "pool-dice":
      let image = ImageDataset(width: 32, height: 32, channels: 3, data: Array(repeating: 0.8, count: 3072))
      let features = CNNFeatureExtractor().extractFeatures(image: image)
      return .values(features + [VisionMetrics.diceCoefficient(predicted: [features], groundTruth: [features])])
    default: return nil
    }
  }

  static func sqliteIngest() async throws -> DataFrame {
    let connection = SQLiteConnection(databasePath: ":memory:")
    do {
      _ = try await connection.executeQuery("CREATE TABLE sample (id INT, val REAL)")
      _ = try await connection.executeQuery("INSERT INTO sample VALUES (1, 10.5), (2, 20.0)")
      let frame = try await DataFrame.fromSQL("SELECT id, val FROM sample ORDER BY id", connection: connection)
      await connection.close()
      return frame
    } catch {
      await connection.close()
      throw error
    }
  }
}
