import Foundation
import SwiftML
import SwiftOptimize
import SwiftSciBenchmarkSupport

extension Worker {
  @inline(never) static func executeSupervisedClassifier(
    input: SupervisedFixtureInput,
    means: [Double],
    scales: [Double],
    scaled: [[[Double]]],
    targets: [[Double]]
  ) async throws -> Output {
    let columns = input.feature_names.count
    guard let training = input.training,
      training.epochs >= 0,
      training.learning_rate.isFinite, training.learning_rate > 0,
      Float(training.learning_rate).isFinite, Float(training.learning_rate) > 0,
      means.count == columns, means.allSatisfy(\.isFinite),
      scales.count == columns, scales.allSatisfy({ $0.isFinite && $0 > 0 }),
      scaled.count == 3, targets.count == 3,
      zip(scaled, targets).allSatisfy({ matrix, actual in
        !actual.isEmpty && matrix.count == actual.count
          && matrix.allSatisfy({ $0.count == columns && $0.allSatisfy(\.isFinite) })
          && actual.allSatisfy({ $0 == 0 || $0 == 1 })
          && actual.contains(0) && actual.contains(1)
      })
    else { throw BenchmarkFailure("Invalid supervised classifier input") }

    let model = LogisticRegression(device: .cpu)
    try await model.fit(
      features: scaled[0], targets: targets[0],
      learningRate: Float(training.learning_rate), epochs: training.epochs)
    guard await model.resolvedDevice == .cpu else {
      throw BenchmarkFailure("Supervised classifier did not resolve to CPU")
    }

    var probabilities = [[[Double]]]()
    var labels = [[Int]]()
    for matrix in scaled {
      probabilities.append(try await model.predictProbability(features: matrix))
      labels.append(try await model.predict(features: matrix))
    }
    let parameters = await model.getWeightsAndBias()
    guard let weights = parameters.weights, let bias = parameters.bias,
      weights.count == columns, weights.allSatisfy(\.isFinite), bias.isFinite
    else { throw BenchmarkFailure("Invalid supervised classifier parameters") }
    for split in 0..<3 {
      guard probabilities[split].count == targets[split].count,
        labels[split].count == targets[split].count,
        probabilities[split].allSatisfy({ row in
          row.count == 2 && row.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 })
            && abs(row[0] + row[1] - 1) <= 1e-12
        }),
        zip(probabilities[split], labels[split]).allSatisfy({ row, label in
          (label == 0 || label == 1) && label == (row[1] > 0.5 ? 1 : 0)
        })
      else { throw BenchmarkFailure("Invalid supervised classifier predictions") }
    }

    let trainingPrevalence = targets[0].reduce(0, +) / Double(targets[0].count)
    var scores = [Double]()
    for split in 1...2 {
      let actual = targets[split].map(Int.init)
      let positiveProbabilities = probabilities[split].map { $0[1] }
      scores.append(contentsOf: supervisedClassificationScores(
        actual: actual, predicted: labels[split], probabilities: positiveProbabilities))
      scores.append(contentsOf: supervisedClassificationScores(
        actual: actual,
        predicted: [Int](repeating: trainingPrevalence > 0.5 ? 1 : 0, count: actual.count),
        probabilities: [Double](repeating: trainingPrevalence, count: actual.count)))
    }
    let values = means + scales + [bias] + weights
      + probabilities.flatMap { $0.flatMap { $0 } }
      + labels.flatMap { $0.map(Double.init) } + [trainingPrevalence] + scores
    guard values.count == 3 * columns + 3 * input.features.count + 46,
      values.allSatisfy(\.isFinite)
    else { throw BenchmarkFailure("Invalid supervised classifier output layout") }
    return .values(values)
  }

  private static func supervisedClassificationScores(
    actual: [Int], predicted: [Int], probabilities: [Double]
  ) -> [Double] {
    var confusion = [Double](repeating: 0, count: 4)
    for (truth, prediction) in zip(actual, predicted) {
      confusion[2 * truth + prediction] += 1
    }
    return confusion + [
      Metrics.accuracy(yTrue: actual, yPred: predicted),
      Metrics.precision(yTrue: actual, yPred: predicted, label: 1),
      Metrics.recall(yTrue: actual, yPred: predicted, label: 1),
      Metrics.f1Score(yTrue: actual, yPred: predicted, label: 1),
      Metrics.logLoss(yTrue: actual, yScore: probabilities, eps: 1e-15),
      Metrics.rocAUC(yTrue: actual, yScore: probabilities),
      Metrics.brierScore(yTrue: actual, yScore: probabilities),
    ]
  }
}
