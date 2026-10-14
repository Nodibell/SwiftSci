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
    /// The observed unique classes (classifiers only).
    public let classes: [Double]?
    /// The splitting impurity criterion ("gini", "entropy", "mse").
    public let criterion: String?

    public init(
        maxDepth: Int,
        minSamplesSplit: Int,
        nodes: [FlatTreeNode],
        numFeatures: Int,
        classes: [Double]? = nil,
        criterion: String? = nil
    ) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.nodes = nodes
        self.numFeatures = numFeatures
        self.classes = classes
        self.criterion = criterion
    }
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
    /// The number of unique classes (classifiers only).
    public let numClasses: Int?
    /// The max features per split.
    public let maxFeatures: Int?
    /// The max bootstrap samples per tree.
    public let maxSamples: Int?
    /// The splitting criterion ("gini", "entropy", "mse").
    public let criterion: String?
    /// The random seed.
    public let randomState: Int?

    public init(
        nEstimators: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        trees: [[FlatTreeNode]],
        numFeatures: Int,
        numClasses: Int? = nil,
        maxFeatures: Int? = nil,
        maxSamples: Int? = nil,
        criterion: String? = nil,
        randomState: Int? = nil
    ) {
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.trees = trees
        self.numFeatures = numFeatures
        self.numClasses = numClasses
        self.maxFeatures = maxFeatures
        self.maxSamples = maxSamples
        self.criterion = criterion
        self.randomState = randomState
    }
}

// MARK: - Model Persistence Extensions

extension LinearRegression {
    /// Saves trained Linear Regression model parameters to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    /// - Throws: `MLError.notFitted` if model is not fitted, or I/O encoding errors.
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw SwiftMLError.notFitted
        }
        let state = LinearRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Linear Regression model from a JSON file URL.
    /// - Parameters:
    ///   - url: Source file `URL`.
    ///   - device: Preferred execution device (`.auto`, `.cpu`, or `.metal`).
    /// - Throws: I/O decoding errors if file is invalid.
    /// - Returns: A restored `LinearRegression` model instance.
    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LinearRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LinearRegressionModelState.self, from: data)
        return LinearRegression(weights: state.weights, bias: state.bias, device: device)
    }
}

extension LogisticRegression {
    /// Saves trained Logistic Regression model parameters to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    /// - Throws: `MLError.notFitted` if model is not fitted, or I/O encoding errors.
    public func save(to url: URL) async throws {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            throw SwiftMLError.notFitted
        }
        let state = LogisticRegressionModelState(weights: weights, bias: bias)
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Logistic Regression model from a JSON file URL.
    /// - Parameters:
    ///   - url: Source file `URL`.
    ///   - device: Preferred execution device (`.auto`, `.cpu`, or `.metal`).
    /// - Throws: I/O decoding errors if file is invalid.
    /// - Returns: A restored `LogisticRegression` model instance.
    public static func load(from url: URL, device: ExecutionDevice = .auto) throws -> LogisticRegression {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(LogisticRegressionModelState.self, from: data)
        return LogisticRegression(weights: state.weights, bias: state.bias, device: device)
    }
}

extension DecisionTreeClassifier {
    /// Saves trained Decision Tree Classifier parameters and tree structure to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    public func save(to url: URL) async throws {
        let state = getModelState()
        guard !state.nodes.isEmpty else { throw SwiftMLError.notFitted }
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Decision Tree Classifier from a JSON file URL.
    /// - Parameter url: Source file `URL`.
    public static func load(from url: URL) throws -> DecisionTreeClassifier {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(DecisionTreeModelState.self, from: data)
        let crit: SplitCriterion = state.criterion == "entropy" ? .entropy : .gini
        return DecisionTreeClassifier(
            maxDepth: state.maxDepth,
            minSamplesSplit: state.minSamplesSplit,
            criterion: crit,
            nodes: state.nodes,
            numFeatures: state.numFeatures,
            classes: state.classes ?? []
        )
    }
}

extension DecisionTreeRegressor {
    /// Saves trained Decision Tree Regressor parameters and tree structure to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    public func save(to url: URL) async throws {
        let state = getModelState()
        guard !state.nodes.isEmpty else { throw SwiftMLError.notFitted }
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Decision Tree Regressor from a JSON file URL.
    /// - Parameter url: Source file `URL`.
    public static func load(from url: URL) throws -> DecisionTreeRegressor {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(DecisionTreeModelState.self, from: data)
        return DecisionTreeRegressor(
            maxDepth: state.maxDepth,
            minSamplesSplit: state.minSamplesSplit,
            nodes: state.nodes,
            numFeatures: state.numFeatures
        )
    }
}

extension RandomForestClassifier {
    /// Saves trained Random Forest Classifier parameters and tree ensemble to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    public func save(to url: URL) async throws {
        let state = getModelState()
        guard !state.trees.isEmpty else { throw SwiftMLError.notFitted }
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Random Forest Classifier from a JSON file URL.
    /// - Parameter url: Source file `URL`.
    public static func load(from url: URL) throws -> RandomForestClassifier {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(RandomForestModelState.self, from: data)
        let crit: SplitCriterion = state.criterion == "entropy" ? .entropy : .gini
        return RandomForestClassifier(
            nEstimators: state.nEstimators,
            maxDepth: state.maxDepth,
            minSamplesSplit: state.minSamplesSplit,
            maxFeatures: state.maxFeatures,
            maxSamples: state.maxSamples,
            criterion: crit,
            randomState: state.randomState,
            trees: state.trees,
            numFeatures: state.numFeatures,
            numClasses: state.numClasses ?? 2
        )
    }
}

extension RandomForestRegressor {
    /// Saves trained Random Forest Regressor parameters and tree ensemble to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    public func save(to url: URL) async throws {
        let state = getModelState()
        guard !state.trees.isEmpty else { throw SwiftMLError.notFitted }
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Random Forest Regressor from a JSON file URL.
    /// - Parameter url: Source file `URL`.
    public static func load(from url: URL) throws -> RandomForestRegressor {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(RandomForestModelState.self, from: data)
        return RandomForestRegressor(
            nEstimators: state.nEstimators,
            maxDepth: state.maxDepth,
            minSamplesSplit: state.minSamplesSplit,
            maxFeatures: state.maxFeatures,
            maxSamples: state.maxSamples,
            randomState: state.randomState,
            trees: state.trees,
            numFeatures: state.numFeatures
        )
    }
}

