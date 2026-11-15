import Darwin
import Foundation
import MLX
import SwiftCluster
import SwiftExplain
import SwiftForecast
import SwiftLLM
import SwiftML
import SwiftNLP
import SwiftPreprocessing
import SwiftSciBenchmarkSupport
import SwiftStats

struct ScientificSuiteRequest: Decodable {
  let schema_version: Int
  let workload_id: String
  let operation: String
  let warmups: Int
  let samples: Int
  let expect_error: Bool?

  let series: [Double]?
  let horizon: Int?
  let p: Int?
  let d: Int?
  let q: Int?
  let P: Int?
  let D: Int?
  let Q: Int?
  let s: Int?
  let max_p: Int?
  let max_d: Int?
  let max_q: Int?
  let max_sp: Int?
  let max_sd: Int?
  let max_sq: Int?
  let max_candidates: Int?
  let seasonal: Bool?
  let alpha: Double?
  let beta: Double?
  let gamma: Double?
  let coefficients: [Double]?
  let period: Int?
  let max_anomalies_ratio: Double?
  let threshold_z: Double?

  let x_train: [[Double]]?
  let y_train: [Double]?
  let x_test: [[Double]]?
  let n_estimators: Int?
  let max_depth: Int?
  let max_bins: Int?
  let learning_rate: Double?
  let epochs: Int?
  let c_param: Double?
  let n_clusters: Int?
  let n_components: Int?
  let seed: Int?
  let instances: [[Double]]?
  let prompt_len: Int?
  let decode_steps: Int?
}

struct ScientificSuiteResponse: Encodable {
  let workload_id: String
  let operation: String
  var fit_samples_ns: [UInt64] = []
  var infer_samples_ns: [UInt64] = []
  var stage_samples_ns: [String: [UInt64]]? = nil
  var peak_rss_bytes: Int64 = 0
  var predictions: [Double]? = nil
  var probabilities: [[Double]]? = nil
  var fitted_values: [Double]? = nil
  var residuals: [Double]? = nil
  var is_fitted: Bool? = nil
  var error_occurred: Bool = false
  var error_type: String? = nil
  var error_message: String? = nil
  var best_order: [Int]? = nil
  var bool_flags: [Bool]? = nil
  var anomaly_indices: [Int]? = nil
  var anomaly_scores: [Double]? = nil
  var anomaly_threshold: Double? = nil
  var matrix_output: [[Double]]? = nil
  var shap_base_values: [Double]? = nil
  var shap_predictions: [Double]? = nil
  var flat_trees: [[[Double]]]? = nil
  var metrics: [String: Double]? = nil
}

@inline(__always)
private func nowUptimeNs() -> UInt64 {
  clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
}

private func currentPeakRSSBytes() -> Int64 {
  var usage = rusage()
  guard getrusage(RUSAGE_SELF, &usage) == 0 else { return -1 }
  return Int64(usage.ru_maxrss)
}

private func writeScientificResponse(
  _ response: ScientificSuiteResponse,
  to outputURL: URL
) throws {
  var finalized = response
  finalized.peak_rss_bytes = currentPeakRSSBytes()
  let data = try JSONEncoder().encode(finalized)
  try data.write(to: outputURL, options: .atomic)
}

private struct SuiteDummyTokenizer: Tokenizer {
  func tokenize(text: String) -> [String] { ["tok"] }
  func encode(text: String) -> [Int] { [1, 2, 3] }
  func decode(tokens: [Int]) -> String { "tok" }
  func tokenToString(token: Int) -> String? { "tok" }
  func stringToToken(string: String) -> Int? { 1 }
}

private func timedLoop(
  warmups: Int,
  samples: Int,
  _ body: () async throws -> Void
) async throws -> [UInt64] {
  var timings = [UInt64]()
  timings.reserveCapacity(samples)
  for index in 0..<(warmups + samples) {
    let start = nowUptimeNs()
    try await body()
    let elapsed = nowUptimeNs() &- start
    if index >= warmups {
      timings.append(elapsed)
    }
  }
  return timings
}

func runScientificSuite(requestURL: URL, responseURL: URL) async -> Int32 {
  var workloadId = "unknown"
  var operation = "unknown"
  var expectError = false
  do {
    let request = try JSONDecoder().decode(
      ScientificSuiteRequest.self,
      from: Data(contentsOf: requestURL))
    workloadId = request.workload_id
    operation = request.operation
    expectError = request.expect_error ?? false
    guard request.schema_version == 1, request.warmups >= 0, request.samples > 0 else {
      throw BenchmarkFailure("Invalid scientific suite request")
    }
    let response = try await executeScientificWorkload(request)
    try writeScientificResponse(response, to: responseURL)
    return 0
  } catch {
    if expectError {
      let errResponse = ScientificSuiteResponse(
        workload_id: workloadId,
        operation: operation,
        is_fitted: false,
        error_occurred: true,
        error_type: String(describing: type(of: error)),
        error_message: String(describing: error))
      do {
        try writeScientificResponse(errResponse, to: responseURL)
        return 0
      } catch {
        fputs("Failed writing expected-error response: \(error)\n", stderr)
        return 1
      }
    }
    let failResponse = ScientificSuiteResponse(
      workload_id: workloadId,
      operation: operation,
      error_occurred: true,
      error_type: String(describing: type(of: error)),
      error_message: String(describing: error))
    if let encoded = try? JSONEncoder().encode(failResponse) {
      try? encoded.write(to: responseURL, options: .atomic)
    }
    fputs("Scientific suite error (\(workloadId)): \(error)\n", stderr)
    return 1
  }
}

private func executeScientificWorkload(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  switch req.operation {
  case "holt_winters":
    return try await runHoltWinters(req)
  case "arima_fit_forecast":
    return try await runARIMA(req)
  case "sarima_fit_forecast":
    return try await runSARIMA(req)
  case "auto_arima_fit_forecast":
    return try await runAutoARIMA(req)
  case "auto_arima_polynomial_check":
    return try runPolynomialCheck(req)
  case "anomaly_detect":
    return try runAnomalyDetect(req)
  case "ols_regression":
    return try await runOLS(req)
  case "logistic_regression":
    return try await runLogisticRegression(req)
  case "linear_svc":
    return try await runLinearSVC(req)
  case "decision_tree_regressor":
    return try await runDecisionTreeRegressor(req)
  case "decision_tree_classifier":
    return try await runDecisionTreeClassifier(req)
  case "random_forest_regressor":
    return try await runRandomForestRegressor(req)
  case "random_forest_classifier":
    return try await runRandomForestClassifier(req)
  case "gbdt_regressor":
    return try await runGBDTRegressor(req)
  case "hist_gbdt_regressor":
    return try await runHistGBDTRegressor(req)
  case "hist_gbdt_classifier":
    return try await runHistGBDTClassifier(req)
  case "pca_fit_transform":
    return try await runPCA(req)
  case "kmeans_fit_predict":
    return try await runKMeans(req)
  case "treeshap_decision_tree_regressor":
    return try await runTreeSHAPDecisionTreeRegressor(req)
  case "treeshap_random_forest_regressor":
    return try await runTreeSHAPRandomForestRegressor(req)
  case "stats_skewness_kurtosis":
    return try runStatsSkewKurt(req)
  case "swiftllm_kvcache_ablation":
    return try runSwiftLLMKVCacheAblation(req)
  default:
    throw BenchmarkFailure("Unsupported scientific operation: \(req.operation)")
  }
}

private func runHoltWinters(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let series = req.series, let horizon = req.horizon else {
    throw BenchmarkFailure("Missing series or horizon")
  }
  let period = req.period ?? 12
  let alpha = req.alpha
  let beta = req.beta ?? 0.1
  let gamma = req.gamma ?? 0.1
  var preds = [Double]()
  var fitted = [Double]()
  var fitModel = ExponentialSmoothing(
    method: .holtWinters(beta: beta, gamma: gamma, period: period, seasonal: .additive),
    alpha: alpha)
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = ExponentialSmoothing(
      method: .holtWinters(beta: beta, gamma: gamma, period: period, seasonal: .additive),
      alpha: alpha)
    try await m.fit(series: series)
    fitModel = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let res = try await fitModel.forecast(horizon: horizon)
    preds = res.predictions
    fitted = try await fitModel.fittedValues()
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    fitted_values: fitted,
    is_fitted: true)
}

private func runARIMA(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let series = req.series, let horizon = req.horizon else {
    throw BenchmarkFailure("Missing series or horizon")
  }
  let p = req.p ?? 1
  let d = req.d ?? 1
  let q = req.q ?? 1
  var lastResult: ARIMAResult?
  var lastFittedModel = try ARIMAModel(p: p, d: d, q: q, method: .hannanRissanen)
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let model = try ARIMAModel(p: p, d: d, q: q, method: .hannanRissanen)
    try await model.fit(series: series)
    lastFittedModel = model
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    lastResult = try await lastFittedModel.forecast(horizon: horizon)
  }
  let isFitted = await lastFittedModel.isFitted
  let aic = try? await lastFittedModel.aic()
  let bic = try? await lastFittedModel.bic()
  var extra = [String: Double]()
  if let aic { extra["aic"] = aic }
  if let bic { extra["bic"] = bic }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: lastResult?.forecast.predictions,
    fitted_values: await lastFittedModel.fittedValues,
    residuals: await lastFittedModel.residuals,
    is_fitted: isFitted,
    metrics: extra.isEmpty ? nil : extra)
}

private func runSARIMA(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let series = req.series, let horizon = req.horizon else {
    throw BenchmarkFailure("Missing series or horizon")
  }
  let p = req.p ?? 1
  let d = req.d ?? 1
  let q = req.q ?? 1
  let P = req.P ?? 0
  let D = req.D ?? 0
  let Q = req.Q ?? 0
  let s = req.s ?? 12

  var lastModel = try SARIMAModel(p: p, d: d, q: q, P: P, D: D, Q: Q, s: s)
  if req.expect_error ?? false {
    do {
      try await lastModel.fit(series: series)
      throw BenchmarkFailure("Expected SARIMAModel.fit to fail closed but succeeded")
    } catch {
      let fittedAfterErr = await lastModel.isFitted
      return ScientificSuiteResponse(
        workload_id: req.workload_id,
        operation: req.operation,
        is_fitted: fittedAfterErr,
        error_occurred: true,
        error_type: String(describing: type(of: error)),
        error_message: String(describing: error))
    }
  }

  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try SARIMAModel(p: p, d: d, q: q, P: P, D: D, Q: Q, s: s)
    try await m.fit(series: series)
    lastModel = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await lastModel.forecast(steps: horizon)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    fitted_values: await lastModel.fittedValues,
    residuals: await lastModel.residuals,
    is_fitted: await lastModel.isFitted)
}

private func runAutoARIMA(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let series = req.series, let horizon = req.horizon else {
    throw BenchmarkFailure("Missing series or horizon")
  }
  let maxP = req.max_p ?? 2
  let maxD = req.max_d ?? 2
  let maxQ = req.max_q ?? 2
  let seasonal = req.seasonal ?? false
  let s = req.s ?? 12
  let maxSP = req.max_sp ?? 1
  let maxSD = req.max_sd ?? 1
  let maxSQ = req.max_sq ?? 1
  let maxCand = req.max_candidates ?? 25

  var lastAuto = try AutoARIMA(
    maxP: maxP, maxD: maxD, maxQ: maxQ,
    seasonal: seasonal, seasonalPeriod: s,
    maxSeasonalP: maxSP, maxSeasonalD: maxSD, maxSeasonalQ: maxSQ,
    maxCandidates: maxCand, criterion: .aic)
  var lastFitResult: ARIMAResult?
  var lastForecastResult: ARIMAResult?

  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let engine = try AutoARIMA(
      maxP: maxP, maxD: maxD, maxQ: maxQ,
      seasonal: seasonal, seasonalPeriod: s,
      maxSeasonalP: maxSP, maxSeasonalD: maxSD, maxSeasonalQ: maxSQ,
      maxCandidates: maxCand, criterion: .aic)
    lastFitResult = try await engine.fit(series: series)
    lastAuto = engine
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    lastForecastResult = try await lastAuto.forecast(horizon: horizon)
  }
  let order = await lastAuto.bestOrder
  let orderArr = order.map { [$0.p, $0.d, $0.q, $0.P, $0.D, $0.Q, $0.s] }
  let fitVals = lastFitResult?.forecast.fittedValues ?? lastForecastResult?.forecast.fittedValues
  let resVals = lastFitResult?.forecast.residuals ?? lastForecastResult?.forecast.residuals
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: lastForecastResult?.forecast.predictions,
    fitted_values: fitVals,
    residuals: resVals,
    is_fitted: order != nil,
    best_order: orderArr)
}

private func runPolynomialCheck(_ req: ScientificSuiteRequest) throws -> ScientificSuiteResponse {
  guard let coeffs = req.coefficients else {
    throw BenchmarkFailure("Missing coefficients")
  }
  let start = nowUptimeNs()
  let arStable = AutoARIMA.isPolynomialStable(coefficients: coeffs)
  let maInvertible = AutoARIMA.isPolynomialStable(coefficients: coeffs.map(-))
  let elapsed = nowUptimeNs() &- start
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: [elapsed],
    infer_samples_ns: [elapsed],
    bool_flags: [arStable, maInvertible])
}

private func runAnomalyDetect(_ req: ScientificSuiteRequest) throws -> ScientificSuiteResponse {
  guard let series = req.series else { throw BenchmarkFailure("Missing series") }
  let period = req.period
  let ratio = req.max_anomalies_ratio ?? 0.10
  let thresholdZ = req.threshold_z ?? 3.0
  var lastResult = AnomalyDetectionResult(anomalies: [], threshold: thresholdZ)
  var timings = [UInt64]()
  timings.reserveCapacity(req.samples)
  for index in 0..<(req.warmups + req.samples) {
    let start = nowUptimeNs()
    lastResult = TimeSeriesAnomalyDetector.detectAnomalies(
      series: series, period: period, maxAnomaliesRatio: ratio, thresholdZ: thresholdZ)
    let elapsed = nowUptimeNs() &- start
    if index >= req.warmups { timings.append(elapsed) }
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: timings,
    infer_samples_ns: timings,
    anomaly_indices: lastResult.anomalyIndices,
    anomaly_scores: lastResult.anomalies.map(\.score),
    anomaly_threshold: lastResult.threshold)
}

private func requireSupervised(_ req: ScientificSuiteRequest) throws -> ([[Double]], [Double], [[Double]]) {
  guard let xTrain = req.x_train, let yTrain = req.y_train, let xTest = req.x_test else {
    throw BenchmarkFailure("Missing supervised train/test inputs")
  }
  return (xTrain, yTrain, xTest)
}

private func runOLS(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  var model = LinearRegression(device: .cpu)
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = LinearRegression(device: .cpu)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await model.predict(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds)
}

private func runLogisticRegression(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let lr = Float(req.learning_rate ?? 0.25)
  let epochs = req.epochs ?? 1000
  var model = LogisticRegression(device: .cpu)
  var probs = [[Double]]()
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = LogisticRegression(device: .cpu)
    try await m.fit(features: xTrain, targets: yTrain, learningRate: lr, epochs: epochs)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    probs = try await model.predictProbability(features: xTest)
    preds = probs.map { row in (row.count >= 2 && row[1] >= 0.5) ? 1.0 : 0.0 }
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    probabilities: probs)
}

private func runLinearSVC(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let cParam = req.c_param ?? 1.0
  let lr = Float(req.learning_rate ?? 0.05)
  let epochs = req.epochs ?? 800
  var model = LinearSVC(C: cParam, device: .cpu)
  var preds = [Double]()
  var scores = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = LinearSVC(C: cParam, device: .cpu)
    try await m.fit(features: xTrain, targets: yTrain, learningRate: lr, epochs: epochs)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let labels = try await model.predict(features: xTest)
    preds = labels.map(Double.init)
    scores = try await model.decisionFunction(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    anomaly_scores: scores)
}

private func runDecisionTreeRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let maxDepth = req.max_depth ?? 5
  var model = DecisionTreeRegressor(maxDepth: maxDepth, minSamplesSplit: 2)
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = DecisionTreeRegressor(maxDepth: maxDepth, minSamplesSplit: 2)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await model.predict(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds)
}

private func runDecisionTreeClassifier(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let maxDepth = req.max_depth ?? 5
  var model = DecisionTreeClassifier(maxDepth: maxDepth, minSamplesSplit: 2, criterion: .gini)
  var preds = [Double]()
  var probs = [[Double]]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = DecisionTreeClassifier(maxDepth: maxDepth, minSamplesSplit: 2, criterion: .gini)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let labels = try await model.predict(features: xTest)
    preds = labels.map(Double.init)
    probs = try await model.predictProbability(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    probabilities: probs)
}

private func runRandomForestRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let nEst = req.n_estimators ?? 50
  let maxDepth = req.max_depth ?? 6
  let seed = req.seed ?? 42
  var model = RandomForestRegressor(nEstimators: nEst, maxDepth: maxDepth, randomState: seed)
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = RandomForestRegressor(nEstimators: nEst, maxDepth: maxDepth, randomState: seed)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await model.predict(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds)
}

private func runRandomForestClassifier(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let nEst = req.n_estimators ?? 50
  let maxDepth = req.max_depth ?? 6
  let seed = req.seed ?? 42
  var model = RandomForestClassifier(
    nEstimators: nEst, maxDepth: maxDepth, criterion: .gini, randomState: seed)
  var preds = [Double]()
  var probs = [[Double]]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = RandomForestClassifier(
      nEstimators: nEst, maxDepth: maxDepth, criterion: .gini, randomState: seed)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let labels = try await model.predict(features: xTest)
    preds = labels.map(Double.init)
    probs = try await model.predictProbability(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    probabilities: probs)
}

private func runGBDTRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let nEst = req.n_estimators ?? 50
  let maxDepth = req.max_depth ?? 3
  let lr = req.learning_rate ?? 0.1
  var model = try GradientBoostedTreesRegressor(
    nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, minSamplesSplit: 2, loss: .squaredError)
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try GradientBoostedTreesRegressor(
      nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, minSamplesSplit: 2, loss: .squaredError)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await model.predict(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds)
}

private func runHistGBDTRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let nEst = req.n_estimators ?? 50
  let maxDepth = req.max_depth ?? 5
  let maxBins = req.max_bins ?? 256
  let lr = req.learning_rate ?? 0.1
  var model = try HistGradientBoostingRegressor(
    nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, maxBins: maxBins)
  var preds = [Double]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try HistGradientBoostingRegressor(
      nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, maxBins: maxBins)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    preds = try await model.predict(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds)
}

private func runHistGBDTClassifier(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let nEst = req.n_estimators ?? 50
  let maxDepth = req.max_depth ?? 5
  let maxBins = req.max_bins ?? 256
  let lr = req.learning_rate ?? 0.1
  let seed = req.seed ?? 42
  var model = try HistGradientBoostingClassifier(
    nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, maxBins: maxBins, randomState: seed)
  var preds = [Double]()
  var probs = [[Double]]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try HistGradientBoostingClassifier(
      nEstimators: nEst, learningRate: lr, maxDepth: maxDepth, maxBins: maxBins, randomState: seed)
    try await m.fit(features: xTrain, targets: yTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let labels = try await model.predict(features: xTest)
    preds = labels.map(Double.init)
    probs = try await model.predictProbability(features: xTest)
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: preds,
    probabilities: probs)
}

private func runPCA(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let xTrain = req.x_train else { throw BenchmarkFailure("Missing x_train") }
  let xTest = req.x_test ?? xTrain
  let k = req.n_components ?? 10
  var model = try PCA(nComponents: k, svdSolver: .full, device: .cpu)
  var projected = [[Double]]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try PCA(nComponents: k, svdSolver: .full, device: .cpu)
    try await m.fit(xTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    projected = try await model.transform(xTest)
  }
  let params = try await model.parameters()
  let ratioSum = params.totalVariance > 0
    ? params.explainedVariance.reduce(0, +) / params.totalVariance
    : 0.0
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    matrix_output: projected,
    metrics: [
      "total_variance": params.totalVariance,
      "explained_variance_ratio_sum": ratioSum,
    ])
}

private func runKMeans(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  guard let xTrain = req.x_train else { throw BenchmarkFailure("Missing x_train") }
  let xTest = req.x_test ?? xTrain
  let k = req.n_clusters ?? 3
  let seed = req.seed ?? 42
  var model = try KMeans(nClusters: k, maxIterations: 50, tolerance: 1e-4, seed: seed, device: .cpu)
  var labels = [Int]()
  let fitNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    let m = try KMeans(nClusters: k, maxIterations: 50, tolerance: 1e-4, seed: seed, device: .cpu)
    try await m.fit(features: xTrain)
    model = m
  }
  let inferNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    labels = try await model.predict(features: xTest)
  }
  let centroids = await model.getCentroids() ?? []
  let trainLabels = try await model.predict(features: xTrain)
  var wcss = 0.0
  for (rowIdx, row) in xTrain.enumerated() {
    let c = centroids[trainLabels[rowIdx]]
    for j in 0..<row.count {
      let diff = row[j] - c[j]
      wcss += diff * diff
    }
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fitNs,
    infer_samples_ns: inferNs,
    predictions: labels.map(Double.init),
    matrix_output: centroids,
    metrics: ["wcss": wcss])
}

private func serializeTrees(_ trees: [[FlatTreeNode]]) -> [[[Double]]] {
  trees.map { tree in
    tree.map { node in
      [
        Double(node.featureIndex),
        node.threshold,
        Double(node.leftChild),
        Double(node.rightChild),
        node.value,
        node.isLeaf ? 1.0 : 0.0,
      ]
    }
  }
}

private func runTreeSHAPDecisionTreeRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let instances = req.instances ?? Array(xTest.prefix(25))
  let maxDepth = req.max_depth ?? 4
  let dt = DecisionTreeRegressor(maxDepth: maxDepth, minSamplesSplit: 2)
  try await dt.fit(features: xTrain, targets: yTrain)
  let nodes = await dt.getTreeNodes()
  let numFeatures = xTrain[0].count
  let explainer = TreeSHAP(permutationSeed: UInt64(req.seed ?? 42))
  let exactFlag = explainer.isExact(tree: nodes, numFeatures: numFeatures)
  var matrix = [[Double]]()
  var bases = [Double]()
  var preds = [Double]()
  let explainNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    var localMatrix = [[Double]]()
    var localBases = [Double]()
    var localPreds = [Double]()
    localMatrix.reserveCapacity(instances.count)
    for row in instances {
      let detailed = explainer.explainDetailed(tree: nodes, instance: row, numFeatures: numFeatures)
      localMatrix.append(detailed.values)
      localBases.append(detailed.baseValue)
      localPreds.append(detailed.prediction)
    }
    matrix = localMatrix
    bases = localBases
    preds = localPreds
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: explainNs,
    infer_samples_ns: explainNs,
    bool_flags: [exactFlag],
    matrix_output: matrix,
    shap_base_values: bases,
    shap_predictions: preds,
    flat_trees: serializeTrees([nodes]))
}

private func runTreeSHAPRandomForestRegressor(_ req: ScientificSuiteRequest) async throws -> ScientificSuiteResponse {
  let (xTrain, yTrain, xTest) = try requireSupervised(req)
  let instances = req.instances ?? Array(xTest.prefix(25))
  let nEst = req.n_estimators ?? 20
  let maxDepth = req.max_depth ?? 4
  let seed = req.seed ?? 42
  let rf = RandomForestRegressor(nEstimators: nEst, maxDepth: maxDepth, randomState: seed)
  try await rf.fit(features: xTrain, targets: yTrain)
  let trees = await rf.getForestTrees()
  let numFeatures = xTrain[0].count
  let explainer = TreeSHAP(permutationSeed: UInt64(seed))
  let allExact = trees.allSatisfy { explainer.isExact(tree: $0, numFeatures: numFeatures) }
  var matrix = [[Double]]()
  var bases = [Double]()
  var preds = [Double]()
  let explainNs = try await timedLoop(warmups: req.warmups, samples: req.samples) {
    var localMatrix = [[Double]]()
    var localBases = [Double]()
    var localPreds = [Double]()
    localMatrix.reserveCapacity(instances.count)
    for row in instances {
      let detailed = explainer.explainDetailed(trees: trees, instance: row, numFeatures: numFeatures)
      localMatrix.append(detailed.values)
      localBases.append(detailed.baseValue)
      localPreds.append(detailed.prediction)
    }
    matrix = localMatrix
    bases = localBases
    preds = localPreds
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: explainNs,
    infer_samples_ns: explainNs,
    bool_flags: [allExact],
    matrix_output: matrix,
    shap_base_values: bases,
    shap_predictions: preds,
    flat_trees: trees.count <= 20 && numFeatures <= 8 ? serializeTrees(trees) : nil)
}

private func runStatsSkewKurt(_ req: ScientificSuiteRequest) throws -> ScientificSuiteResponse {
  guard let values = req.series else { throw BenchmarkFailure("Missing series") }
  var skew = 0.0
  var kurt = 0.0
  var timings = [UInt64]()
  timings.reserveCapacity(req.samples)
  for index in 0..<(req.warmups + req.samples) {
    let start = nowUptimeNs()
    skew = try Stats.skewness(values)
    kurt = try Stats.kurtosis(values)
    let elapsed = nowUptimeNs() &- start
    if index >= req.warmups { timings.append(elapsed) }
  }
  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: timings,
    infer_samples_ns: timings,
    metrics: ["skewness": skew, "kurtosis": kurt])
}

private func runSwiftLLMKVCacheAblation(_ req: ScientificSuiteRequest) throws -> ScientificSuiteResponse {
  let promptLen = req.prompt_len ?? 24
  let decodeSteps = req.decode_steps ?? 16
  let vocabSize = 64
  let hiddenDim = 32
  let numHeads = 4
  let numLayers = 2
  let config = LLMConfig(
    vocabSize: vocabSize,
    numLayers: numLayers,
    hiddenDim: hiddenDim,
    numHeads: numHeads,
    numKVHeads: numHeads,
    maxSeqLen: max(64, promptLen + decodeSteps + 8),
    positionalEncoding: .rope(base: 10_000.0))
  let decoder = TransformerDecoder(config: config, tokenizer: SuiteDummyTokenizer())

  let promptTokens = (0..<promptLen).map { ($0 * 7 + 3) % vocabSize }
  let genTokens = (0..<decodeSteps).map { ($0 * 11 + 5) % vocabSize }

  var cachedNs = [UInt64]()
  var fullNs = [UInt64]()
  var maxAbsDiff = 0.0

  for index in 0..<(req.warmups + req.samples) {
    let caches = (0..<numLayers).map { _ in KVCache() }
    let prefill = decoder.forward(MLXArray(promptTokens.map(Int32.init)), caches: caches, offset: 0)
    eval(prefill)

    let cachedStart = nowUptimeNs()
    var cachedLogits = [Float]()
    for step in 0..<decodeSteps {
      let nextTok = MLXArray([Int32(genTokens[step])])
      let stepOut = decoder.forward(nextTok, caches: caches, offset: promptLen + step)
      eval(stepOut)
      if step == decodeSteps - 1 {
        cachedLogits = stepOut.reshaped([vocabSize]).asArray(Float.self)
      }
    }
    let cElapsed = nowUptimeNs() &- cachedStart

    let fullStart = nowUptimeNs()
    var fullLogits = [Float]()
    var running = promptTokens
    for step in 0..<decodeSteps {
      running.append(genTokens[step])
      let seqOut = decoder.forward(MLXArray(running.map(Int32.init)), caches: nil, offset: 0)
      let lastSlice = seqOut[0, running.count - 1]
      eval(lastSlice)
      if step == decodeSteps - 1 {
        fullLogits = lastSlice.asArray(Float.self)
      }
    }
    let fElapsed = nowUptimeNs() &- fullStart

    var stepDiff = 0.0
    for j in 0..<vocabSize {
      stepDiff = max(stepDiff, Double(abs(cachedLogits[j] - fullLogits[j])))
    }
    maxAbsDiff = max(maxAbsDiff, stepDiff)
    if index >= req.warmups {
      cachedNs.append(cElapsed)
      fullNs.append(fElapsed)
    }
  }

  return ScientificSuiteResponse(
    workload_id: req.workload_id,
    operation: req.operation,
    fit_samples_ns: fullNs,
    infer_samples_ns: cachedNs,
    stage_samples_ns: [
      "full_forward_decode": fullNs,
      "kvcache_decode": cachedNs,
    ],
    metrics: [
      "max_abs_logit_diff": maxAbsDiff,
      "prompt_len": Double(promptLen),
      "decode_steps": Double(decodeSteps),
    ])
}
