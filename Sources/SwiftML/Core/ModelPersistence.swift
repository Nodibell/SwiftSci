import Foundation
import SwiftPreprocessing

// MARK: - Codable Model State Containers

/// Represents linear regression model state.
public struct LinearRegressionModelState: Codable, Sendable {
    /// The weights.
    public let weights: [Double]
    /// The bias.
    public let bias: Double
}

/// Represents logistic regression model state.
public struct LogisticRegressionModelState: Codable, Sendable {
    /// The weights.
    public let weights: [Double]
    /// The bias.
    public let bias: Double
}

/// Represents decision tree model state.
public struct DecisionTreeModelState: Codable, Sendable {
    /// The max depth.
    public let maxDepth: Int
    /// The min samples split.
    public let minSamplesSplit: Int
    /// The nodes.
    public let nodes: [FlatTreeNode]
    /// The num features.
    public let numFeatures: Int
}

/// Represents random forest model state.
public struct RandomForestModelState: Codable, Sendable {
    /// The n estimators.
    public let nEstimators: Int
    /// The max depth.
    public let maxDepth: Int
    /// The min samples split.
    public let minSamplesSplit: Int
    /// The trees.
    public let trees: [[FlatTreeNode]]
    /// The num features.
    public let numFeatures: Int
}

// MARK: - Model Persistence Extensions

extension LinearRegression {
    /// Save.
    /// - Throws: An error if the operation fails.
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw MLError.notFitted
        }
        let state = LinearRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Load.
    /// - Parameters:
    ///   - device: The device.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `LinearRegression` result.
    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LinearRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LinearRegressionModelState.self, from: data)
        return LinearRegression(weights: state.weights, bias: state.bias, device: device)
    }
}

extension LogisticRegression {
    /// Save.
    /// - Throws: An error if the operation fails.
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw MLError.notFitted
        }
        let state = LogisticRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Load.
    /// - Parameters:
    ///   - device: The device.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `LogisticRegression` result.
    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LogisticRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LogisticRegressionModelState.self, from: data)
        return LogisticRegression(weights: state.weights, bias: state.bias, device: device)
    }
}
