import Foundation
import SwiftPreprocessing

// MARK: - Codable Model State Containers

public struct LinearRegressionModelState: Codable, Sendable {
    public let weights: [Double]
    public let bias: Double
}

public struct LogisticRegressionModelState: Codable, Sendable {
    public let weights: [Double]
    public let bias: Double
}

public struct DecisionTreeModelState: Codable, Sendable {
    public let maxDepth: Int
    public let minSamplesSplit: Int
    public let nodes: [FlatTreeNode]
    public let numFeatures: Int
}

public struct RandomForestModelState: Codable, Sendable {
    public let nEstimators: Int
    public let maxDepth: Int
    public let minSamplesSplit: Int
    public let trees: [[FlatTreeNode]]
    public let numFeatures: Int
}

// MARK: - Model Persistence Extensions

extension LinearRegression {
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw MLError.notFitted
        }
        let state = LinearRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LinearRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LinearRegressionModelState.self, from: data)
        return LinearRegression(weights: state.weights, bias: state.bias, device: device)
    }
}

extension LogisticRegression {
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw MLError.notFitted
        }
        let state = LogisticRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LogisticRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LogisticRegressionModelState.self, from: data)
        return LogisticRegression(weights: state.weights, bias: state.bias, device: device)
    }
}
