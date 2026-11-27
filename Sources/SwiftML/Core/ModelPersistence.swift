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

    /// Creates a serialized decision tree model state.
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
    /// The observed unique class labels (classifiers only).
    public let classes: [Double]?
    /// The max features per split.
    public let maxFeatures: Int?
    /// The max bootstrap samples per tree.
    public let maxSamples: Int?
    /// The splitting criterion ("gini", "entropy", "mse").
    public let criterion: String?
    /// The random seed.
    public let randomState: Int?

    /// Creates a serialized random forest model state.
    public init(
        nEstimators: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        trees: [[FlatTreeNode]],
        numFeatures: Int,
        numClasses: Int? = nil,
        classes: [Double]? = nil,
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
        self.classes = classes
        self.maxFeatures = maxFeatures
        self.maxSamples = maxSamples
        self.criterion = criterion
        self.randomState = randomState
    }
}

/// Represents gradient boosted decision trees regressor model state.
public struct GradientBoostedTreesModelState: Codable, Sendable {
    /// The number of boosting estimators configured.
    public let nEstimators: Int
    /// The shrinkage learning rate applied to each boosting tree.
    public let learningRate: Double
    /// The maximum depth of individual regression trees.
    public let maxDepth: Int
    /// The minimum number of samples required to split an internal node.
    public let minSamplesSplit: Int
    /// The loss function utilized during boosting.
    public let loss: GBDTLoss
    /// The fitted decision trees in the boosting ensemble.
    public let trees: [[FlatTreeNode]]
    /// The initial base prediction offset before boosting updates.
    public let initialPrediction: Double
    /// The number of input features observed during training.
    public let numFeatures: Int

    /// Creates a serialized gradient boosted decision trees model state.
    /// - Parameters:
    ///   - nEstimators: The number of boosting estimators configured.
    ///   - learningRate: The shrinkage learning rate applied to each boosting tree.
    ///   - maxDepth: The maximum depth of individual regression trees.
    ///   - minSamplesSplit: The minimum number of samples required to split an internal node.
    ///   - loss: The loss function utilized during boosting.
    ///   - trees: The fitted decision trees in the boosting ensemble.
    ///   - initialPrediction: The initial base prediction offset before boosting updates.
    ///   - numFeatures: The number of input features observed during training.
    public init(
        nEstimators: Int,
        learningRate: Double,
        maxDepth: Int,
        minSamplesSplit: Int,
        loss: GBDTLoss,
        trees: [[FlatTreeNode]],
        initialPrediction: Double,
        numFeatures: Int
    ) {
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.loss = loss
        self.trees = trees
        self.initialPrediction = initialPrediction
        self.numFeatures = numFeatures
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
        let resolvedClasses = state.classes ?? (state.numClasses != nil ? (0..<state.numClasses!).map { Double($0) } : [])
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
            numClasses: state.numClasses ?? resolvedClasses.count,
            classes: resolvedClasses
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

extension GradientBoostedTreesRegressor {
    /// Saves trained Gradient Boosted Decision Trees parameters, base offset, and tree ensemble to a JSON file URL.
    /// - Parameter url: Target file `URL`.
    /// - Throws: `SwiftMLError.notFitted` if the model has not been fitted, or I/O encoding errors.
    public func save(to url: URL) async throws {
        let state = getModelState()
        guard !state.trees.isEmpty else { throw SwiftMLError.notFitted }
        let data = try JSONEncoder().encode(state)
        try data.write(to: url)
    }

    /// Loads a trained Gradient Boosted Decision Trees Regressor from a JSON file URL.
    /// - Parameter url: Source file `URL`.
    /// - Throws: I/O or decoding errors if the file is invalid, or `SwiftMLError` if parameters are invalid.
    /// - Returns: A restored ``GradientBoostedTreesRegressor`` model instance.
    public static func load(from url: URL) throws -> GradientBoostedTreesRegressor {
        let data = try Data(contentsOf: url)
        let state = try JSONDecoder().decode(GradientBoostedTreesModelState.self, from: data)
        return try GradientBoostedTreesRegressor(
            nEstimators: state.nEstimators,
            learningRate: state.learningRate,
            maxDepth: state.maxDepth,
            minSamplesSplit: state.minSamplesSplit,
            loss: state.loss,
            trees: state.trees,
            initialPrediction: state.initialPrediction,
            numFeatures: state.numFeatures
        )
    }
}

