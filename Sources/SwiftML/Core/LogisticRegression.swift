#if os(macOS)
import Foundation
import MLX
import SwiftPreprocessing

private protocol LogisticFeatures: Sendable {
    var count: Int { get }
    var width: Int { get }
    subscript(_ row: Int, _ column: Int) -> Double { get }
    func withGPUInput<T>(_ operation: (MLXArray) throws -> T) throws -> T
}

private struct LogisticRows: LogisticFeatures {
    let values: [[Double]]
    var count: Int { values.count }
    var width: Int { values.first?.count ?? 0 }
    subscript(_ row: Int, _ column: Int) -> Double { values[row][column] }

    func withGPUInput<T>(_ operation: (MLXArray) throws -> T) throws -> T {
        if width == 0 {
            return try operation(MLXArray([Float]()).reshaped([count, 0]))
        }
        return try ScopedGPURead.withMatrix(rows: count, columns: width,
                                            value: { values[$0][$1] }, operation)
    }
}

private struct LogisticColumns: LogisticFeatures {
    let batch: PreparedNumericBatch
    var count: Int { batch.rowCount }
    var width: Int { batch.columnCount }
    subscript(_ row: Int, _ column: Int) -> Double { batch.columns[column].values[row] }

    func withGPUInput<T>(_ operation: (MLXArray) throws -> T) throws -> T {
        try ScopedGPURead.withMatrix(batch: batch, operation)
    }
}

private struct FittedLogisticState {
    let cpuWeights: [Double]
    let cpuBias: Double
    let weights: MLXArray
    let bias: MLXArray
    let device: ExecutionDevice
    let featureNames: [String]?
}

/// Binary logistic regression with Double CPU and Float MLX gradient descent.
public actor LogisticRegression: ClassifierEstimator {
    private var fitted: FittedLogisticState?

    /// Learned Float parameters for MLX callers. CPU prediction retains Double parameters.
    public var weights: MLXArray? { fitted?.weights }
    /// Learned Float bias for MLX callers.
    public var bias: MLXArray? { fitted?.bias }
    /// The requested device, preserved across fits.
    public let requestedDevice: ExecutionDevice
    /// The device used by the last successful fit or supplied-parameter initialization.
    public var resolvedDevice: ExecutionDevice? { fitted?.device }

    /// Creates an unfitted classifier with the requested device policy.
    public init(device: ExecutionDevice = .auto) {
        requestedDevice = device
    }

    /// Creates a positional model from supplied parameters, without a named feature schema.
    public init(weights: [Double], bias: Double, device: ExecutionDevice = .auto) {
        requestedDevice = device
        fitted = FittedLogisticState(
            cpuWeights: weights, cpuBias: bias,
            weights: MLXArray(weights.map { Float($0) }).reshaped([weights.count, 1]),
            bias: MLXArray([Float(bias)]),
            device: (device == .auto || device == .cpu) ? .cpu : .gpu,
            featureNames: nil)
    }

    /// Returns learned parameters, retaining Double precision for CPU fits.
    public func getWeightsAndBias() -> (weights: [Double]?, bias: Double?) {
        (fitted?.cpuWeights, fitted?.cpuBias)
    }

    /// Fits rows with learning rate 0.1 and 1,000 full-batch updates.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 0.1, epochs: 1000)
    }

    /// Fits finite rectangular rows and aligned finite targets, including soft targets.
    /// Zero epochs initializes zero parameters. Every fit starts from zero.
    /// Failure or cancellation preserves the previous complete fitted state.
    public func fit(features: [[Double]], targets: [Double],
                    learningRate: Float = 0.1, epochs: Int = 1000) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        let width = features[0].count
        guard features.allSatisfy({ $0.count == width }) else {
            throw SwiftMLError.invalidParameter("Invalid logistic feature shape")
        }
        guard features.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw SwiftMLError.invalidParameter("Logistic training features must be finite")
        }
        try await fitValidated(features: LogisticRows(values: features), targets: targets,
                               learningRate: learningRate, epochs: epochs, names: nil)
    }

    /// Fits compact named columns with the default training controls, without expanding rows.
    public func fit(features: PreparedNumericBatch, targets: [Double]) async throws {
        try await fit(features: features, targets: targets, learningRate: 0.1, epochs: 1000)
    }

    /// Fits compact columns with explicit full-batch controls. Targets follow the row API.
    /// Prepared features must be finite, nonmissing and uniquely named. Later prepared
    /// predictions require the same names in the same order. Input snapshots are preserved.
    public func fit(features: PreparedNumericBatch, targets: [Double],
                    learningRate: Float = 0.1, epochs: Int = 1000) async throws {
        guard features.rowCount > 0, features.columnCount > 0 else { throw SwiftMLError.emptyInput }
        guard Set(features.columnNames).count == features.columnCount else {
            throw SwiftMLError.invalidParameter("Prepared logistic features require unique names")
        }
        try features.requireFinite()
        try await fitValidated(features: LogisticColumns(batch: features), targets: targets,
                               learningRate: learningRate, epochs: epochs, names: features.columnNames)
    }

    private func fitValidated<F: LogisticFeatures>(features: F, targets: [Double],
        learningRate: Float, epochs: Int, names: [String]?) async throws {
        guard !targets.isEmpty else { throw SwiftMLError.emptyInput }
        guard targets.count == features.count, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Logistic targets must be finite and match feature rows")
        }
        guard learningRate.isFinite, learningRate > 0, epochs >= 0 else {
            throw SwiftMLError.invalidParameter("Learning rate must be finite and positive; epochs must be nonnegative")
        }
        guard features.count <= Int32.max, features.width <= Int32.max,
              !features.count.multipliedReportingOverflow(by: features.width).overflow else {
            throw SwiftMLError.invalidParameter("Logistic dimensions exceed the solver capacity")
        }
        try Task.checkCancellation()
        let device = await HardwareRouter.shared.resolveDevice(
            for: "LogisticRegression", sampleCount: features.count,
            featureCount: features.width, requestedDevice: requestedDevice)
        try Task.checkCancellation()
        let candidate: FittedLogisticState
        switch device {
        case .cpu:
            let result = try fitCPU(features: features, targets: targets,
                                    learningRate: Double(learningRate), epochs: epochs)
            candidate = FittedLogisticState(
                cpuWeights: result.weights, cpuBias: result.bias,
                weights: MLXArray(result.weights.map { Float($0) }).reshaped([features.width, 1]),
                bias: MLXArray([Float(result.bias)]), device: device, featureNames: names)
        case .gpu, .ane, .auto:
            candidate = try Device.withDefaultDevice(.gpu) {
                try features.withGPUInput { input in
                    let y = MLXArray(targets.map { Float($0) })
                    let result = try fitGPU(X: input, y: y, learningRate: learningRate, epochs: epochs)
                    return FittedLogisticState(
                        cpuWeights: result.weights.asArray(Float.self).map { Double($0) },
                        cpuBias: Double(result.bias.item(Float.self)),
                        weights: result.weights, bias: result.bias, device: device, featureNames: names)
                }
            }
        }
        try Task.checkCancellation()
        fitted = candidate
    }

    private func fitCPU<F: LogisticFeatures>(features: F, targets: [Double],
        learningRate: Double, epochs: Int) throws -> (weights: [Double], bias: Double) {
        let count = features.count, width = features.width
        var w = [Double](repeating: 0, count: width)
        var b = 0.0
        var predictions = [Double](repeating: 0, count: count)
        var gradW = [Double](repeating: 0, count: width)
        for _ in 0..<epochs {
            try Task.checkCancellation()
            for i in 0..<count {
                var z = 0.0
                for j in 0..<width { z += features[i, j] * w[j] }
                predictions[i] = sigmoid(z + b)
            }
            for j in 0..<width { gradW[j] = 0 }
            var gradB = 0.0
            for i in 0..<count {
                let diff = predictions[i] - targets[i]
                gradB += diff
                for j in 0..<width { gradW[j] += diff * features[i, j] }
            }
            let factor = 1.0 / Double(count)
            gradB *= factor
            for j in 0..<width { gradW[j] *= factor }
            guard gradB.isFinite, gradW.allSatisfy(\.isFinite) else {
                throw SwiftMLError.trainingFailed("Logistic gradient descent diverged; try a lower learning rate")
            }
            for j in 0..<width { w[j] -= learningRate * gradW[j] }
            b -= learningRate * gradB
        }
        guard b.isFinite, w.allSatisfy(\.isFinite) else {
            throw SwiftMLError.trainingFailed("Logistic gradient descent produced nonfinite parameters")
        }
        return (w, b)
    }

    private func fitGPU(X: MLXArray, y: MLXArray, learningRate: Float,
                        epochs: Int) throws -> (weights: MLXArray, bias: MLXArray) {
        let yReshaped = y.reshaped([X.shape[0], 1])
        var w = MLXArray.zeros([X.shape[1], 1])
        var b = MLXArray.zeros([1])
        func lossFn(params: [MLXArray]) -> [MLXArray] {
            let logits = matmul(X, params[0]) + params[1]
            return [Losses.binaryCrossEntropy(logits: logits, targets: yReshaped)]
        }
        let gradFn = valueAndGrad(lossFn, argumentNumbers: [0, 1])
        for _ in 0..<epochs {
            try Task.checkCancellation()
            let (_, grads) = gradFn([w, b])
            w = w - learningRate * grads[0]
            b = b - learningRate * grads[1]
        }
        eval(w, b)
        guard w.asArray(Float.self).allSatisfy(\.isFinite), b.item(Float.self).isFinite else {
            throw SwiftMLError.trainingFailed("Logistic gradient descent diverged; try a lower learning rate")
        }
        return (w, b)
    }

    /// Positive-class probabilities for existing internal row-array consumers.
    func binaryPositiveClassProbability(features: [[Double]]) throws -> [Double] {
        guard !features.isEmpty else { return [] }
        let width = features[0].count
        guard features.allSatisfy({ $0.count == width }) else {
            throw SwiftMLError.invalidParameter("Invalid logistic prediction shape")
        }
        return try positiveProbabilities(features: LogisticRows(values: features))
    }

    private func binaryPositiveClassProbability(features: PreparedNumericBatch) throws -> [Double] {
        if let names = fitted?.featureNames, names != features.columnNames {
            throw SwiftMLError.invalidParameter("Prediction feature names or order differ from training")
        }
        try features.requireFinite()
        guard features.rowCount > 0 else { return [] }
        return try positiveProbabilities(features: LogisticColumns(batch: features))
    }

    private func positiveProbabilities<F: LogisticFeatures>(features: F) throws -> [Double] {
        guard let fitted else { throw SwiftMLError.modelNotFitted }
        let w = fitted.cpuWeights, b = fitted.cpuBias
        guard features.width == w.count else {
            throw SwiftMLError.dimensionMismatch(expected: w.count, got: features.width)
        }
        if fitted.device == .cpu {
            return (0..<features.count).map { row in
                var z = 0.0
                for column in w.indices { z += features[row, column] * w[column] }
                return sigmoid(z + b)
            }
        }
        return try features.withGPUInput { input in
            try predictProbability(X: input).asArray(Float.self).map { Double($0) }
        }
    }

    /// Returns probability rows `[P(class 0), P(class 1)]` in input row order.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        try binaryPositiveClassProbability(features: features).map { [1 - $0, $0] }
    }

    /// Returns probabilities from finite prepared columns in the fitted named order.
    /// A row-fitted or supplied-parameter model interprets prepared columns by position.
    public func predictProbability(features: PreparedNumericBatch) async throws -> [[Double]] {
        try binaryPositiveClassProbability(features: features).map { [1 - $0, $0] }
    }

    /// Returns positive-class Float probabilities for a two-dimensional MLX input.
    public func predictProbability(X: MLXArray) throws -> MLXArray {
        guard let fitted else { throw SwiftMLError.modelNotFitted }
        guard X.size > 0 else { throw SwiftMLError.emptyInput }
        guard X.shape.count == 2 else {
            throw SwiftMLError.dimensionMismatch(expected: 2, got: X.shape.count)
        }
        let width = fitted.cpuWeights.count
        guard X.shape[1] == width else {
            throw SwiftMLError.dimensionMismatch(expected: width, got: X.shape[1])
        }
        return sigmoid(matmul(X, fitted.weights) + fitted.bias)
    }

    /// Predicts binary labels using the strict positive-class probability threshold 0.5.
    public func predict(features: [[Double]]) async throws -> [Int] {
        try predict(features: features, threshold: 0.5)
    }

    /// Predicts binary labels. A probability equal to the threshold is class 0.
    public func predict(features: [[Double]], threshold: Float = 0.5) throws -> [Int] {
        try binaryPositiveClassProbability(features: features).map { $0 > Double(threshold) ? 1 : 0 }
    }

    /// Predicts labels from finite prepared columns using threshold 0.5.
    public func predict(features: PreparedNumericBatch) async throws -> [Int] {
        try predict(features: features, threshold: 0.5)
    }

    /// Predicts labels from prepared columns with the same strict threshold as the row API.
    public func predict(features: PreparedNumericBatch, threshold: Float = 0.5) throws -> [Int] {
        try binaryPositiveClassProbability(features: features).map { $0 > Double(threshold) ? 1 : 0 }
    }

    /// Predicts binary labels for MLX input with a strict probability threshold.
    public func predict(X: MLXArray, threshold: Float = 0.5) throws -> MLXArray {
        greater(try predictProbability(X: X), threshold).asType(.int32)
    }

    /// Returns learned weights, or nil before the first successful fit.
    public func getWeights() -> [Double]? { fitted?.cpuWeights }
    /// Returns learned bias, or nil before the first successful fit.
    public func getBias() -> Double? { fitted?.cpuBias }
}
#endif
