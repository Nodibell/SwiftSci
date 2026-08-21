import Foundation

// MARK: - CoreMLExportable Protocol

/// A protocol for SwiftML estimators that support exporting a trained model
/// as a binary Apple Core ML `.mlmodel` artifact.
///
/// Conforming types produce a binary payload compatible with `MLModel(contentsOf:)`
/// on macOS 14+, iOS 17+, and visionOS 1+, following the Core ML specification v4
/// (`GLMRegressor`, `GLMClassifier`, `TreeEnsembleClassifier`, `TreeEnsembleRegressor`).
///
/// ## Supported conformances
///
/// | Type | Core ML message |
/// |------|----------------|
/// | ``LinearRegression`` | `GLMRegressor` |
/// | ``LogisticRegression`` | `GLMClassifier` |
/// | ``DecisionTreeClassifier`` | `TreeEnsembleClassifier` |
/// | ``DecisionTreeRegressor`` | `TreeEnsembleRegressor` |
/// | ``RandomForestClassifier`` | `TreeEnsembleClassifier` (multi-tree) |
/// | ``RandomForestRegressor`` | `TreeEnsembleRegressor` (multi-tree) |
///
/// ## Example
///
/// ```swift
/// import SwiftML
///
/// let clf = RandomForestClassifier(nEstimators: 50)
/// try await clf.fit(features: X_train, targets: y_train)
///
/// // Export as binary .mlmodel
/// let url = URL(fileURLWithPath: "/tmp/Forest.mlmodel")
/// try await clf.writeCoreML(to: url, featureNames: ["age", "income", "score"], outputName: "label")
/// ```
///
/// - Note: `MLPClassifier` / `MLPRegressor` export is deferred beyond v3.1 (requires
///   the `NeuralNetwork` Core ML proto subtree). Gradient boosted tree export is similarly deferred.
public protocol CoreMLExportable {
    /// Exports the fitted model as binary Apple Core ML `.mlmodel` `Data`.
    ///
    /// - Parameters:
    ///   - featureNames: Ordered list of input feature column names, matching the training data order.
    ///   - outputName: Name of the predicted output feature.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the model has not been fitted,
    ///   or ``SwiftMLError/exportFailed(_:)`` on encoding failure.
    /// - Returns: Binary `Data` representing a complete, loadable `.mlmodel` file.
    func exportCoreML(featureNames: [String], outputName: String) async throws -> Data

    /// Writes the fitted model as a binary Apple Core ML `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL (`.mlmodel` extension recommended).
    ///   - featureNames: Ordered list of input feature column names.
    ///   - outputName: Name of the predicted output feature.
    /// - Throws: ``SwiftMLError/modelNotFitted`` if not fitted,
    ///   or ``SwiftMLError/exportFailed(_:)`` on I/O failure.
    func writeCoreML(to url: URL, featureNames: [String], outputName: String) async throws
}

// MARK: - Default writeCoreML implementation

extension CoreMLExportable {
    /// Default implementation: encodes via ``exportCoreML(featureNames:outputName:)`` and writes to disk.
    public func writeCoreML(to url: URL, featureNames: [String], outputName: String) async throws {
        let data = try await exportCoreML(featureNames: featureNames, outputName: outputName)
        do {
            try data.write(to: url)
        } catch {
            throw SwiftMLError.exportFailed("Failed to write .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }
}

// MARK: - DecisionTreeClassifier Conformance

extension DecisionTreeClassifier: CoreMLExportable {
    /// Exports the fitted decision tree classifier as a binary `.mlmodel` (`TreeEnsembleClassifier`).
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching training order.
    ///   - outputName: Output predicted class label name (stored as `Int64`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the tree has not been fitted.
    public func exportCoreML(featureNames: [String], outputName: String = "label") async throws -> Data {
        let nodes = getTreeNodes()
        guard !nodes.isEmpty else { throw SwiftMLError.modelNotFitted }
        return CoreMLExporter.exportBinaryDecisionTreeClassifier(
            state: DecisionTreeModelState(maxDepth: 0, minSamplesSplit: 2, nodes: nodes, numFeatures: featureNames.count),
            featureNames: featureNames,
            outputName: outputName
        )
    }
}

// MARK: - DecisionTreeRegressor Conformance

extension DecisionTreeRegressor: CoreMLExportable {
    /// Exports the fitted decision tree regressor as a binary `.mlmodel` (`TreeEnsembleRegressor`).
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching training order.
    ///   - outputName: Output prediction column name (stored as `Double`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the tree has not been fitted.
    public func exportCoreML(featureNames: [String], outputName: String = "prediction") async throws -> Data {
        let nodes = getTreeNodes()
        guard !nodes.isEmpty else { throw SwiftMLError.modelNotFitted }
        return CoreMLExporter.exportBinaryDecisionTreeRegressor(
            state: DecisionTreeModelState(maxDepth: 0, minSamplesSplit: 2, nodes: nodes, numFeatures: featureNames.count),
            featureNames: featureNames,
            outputName: outputName
        )
    }
}

// MARK: - RandomForestClassifier Conformance

extension RandomForestClassifier: CoreMLExportable {
    /// Exports the fitted random forest classifier as a binary `.mlmodel` (`TreeEnsembleClassifier`).
    ///
    /// All trees are encoded in a single `TreeEnsembleClassifier` message with per-tree IDs.
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching training order.
    ///   - outputName: Output predicted class label name (stored as `Int64`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the forest has not been fitted.
    public func exportCoreML(featureNames: [String], outputName: String = "label") async throws -> Data {
        let treesNodes = getForestTrees()
        guard !treesNodes.isEmpty else { throw SwiftMLError.modelNotFitted }
        let nf = getNumFeatures()
        let state = RandomForestModelState(
            nEstimators: treesNodes.count,
            maxDepth: 0,
            minSamplesSplit: 2,
            trees: treesNodes,
            numFeatures: nf
        )
        return CoreMLExporter.exportBinaryRandomForestClassifier(
            state: state,
            featureNames: featureNames,
            outputName: outputName
        )
    }
}

// MARK: - RandomForestRegressor Conformance

extension RandomForestRegressor: CoreMLExportable {
    /// Exports the fitted random forest regressor as a binary `.mlmodel` (`TreeEnsembleRegressor`).
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching training order.
    ///   - outputName: Output prediction column name (stored as `Double`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the forest has not been fitted.
    public func exportCoreML(featureNames: [String], outputName: String = "prediction") async throws -> Data {
        let treesNodes = getForestTrees()
        guard !treesNodes.isEmpty else { throw SwiftMLError.modelNotFitted }
        let nf = getNumFeatures()
        let state = RandomForestModelState(
            nEstimators: treesNodes.count,
            maxDepth: 0,
            minSamplesSplit: 2,
            trees: treesNodes,
            numFeatures: nf
        )
        return CoreMLExporter.exportBinaryRandomForestRegressor(
            state: state,
            featureNames: featureNames,
            outputName: outputName
        )
    }
}

// MARK: - macOS-only conformances (LinearRegression, LogisticRegression use MLX)

#if os(macOS)

// MARK: - LinearRegression Conformance

extension LinearRegression: CoreMLExportable {
    /// Exports the fitted linear regression model as a binary `.mlmodel` (`GLMRegressor`).
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching the training data column order.
    ///   - outputName: Output predicted value column name (stored as `Double`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if weights have not been fitted yet.
    public func exportCoreML(featureNames: [String], outputName: String = "prediction") async throws -> Data {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else { throw SwiftMLError.modelNotFitted }
        return CoreMLExporter.exportBinaryLinearModel(
            name: "LinearRegressionModel",
            inputNames: featureNames,
            outputName: outputName,
            weights: weights,
            bias: bias
        )
    }
}

// MARK: - LogisticRegression Conformance

extension LogisticRegression: CoreMLExportable {
    /// Exports the fitted binary logistic regression model as a binary `.mlmodel` (`GLMClassifier`).
    ///
    /// Class labels default to `[0, 1]` for standard binary classification. Use
    /// ``CoreMLExporter/exportBinaryLogisticModel(name:inputNames:outputName:weights:bias:classLabels:)``
    /// directly if custom label integers are required.
    ///
    /// - Parameters:
    ///   - featureNames: Input feature column names matching training order.
    ///   - outputName: Output predicted class label name (stored as `Int64`).
    /// - Throws: ``SwiftMLError/modelNotFitted`` if the model has not been fitted.
    public func exportCoreML(featureNames: [String], outputName: String = "label") async throws -> Data {
        let (weightsOpt, biasOpt) = getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else { throw SwiftMLError.modelNotFitted }
        return CoreMLExporter.exportBinaryLogisticModel(
            name: "LogisticRegressionModel",
            inputNames: featureNames,
            outputName: outputName,
            weights: weights,
            bias: bias
        )
    }
}

// MARK: - MLPClassifier Conformance

extension MLPClassifier: CoreMLExportable {
    /// Exports the fitted Multi-Layer Perceptron classifier as a binary `.mlmodel` (`NeuralNetwork`).
    public func exportCoreML(featureNames: [String], outputName: String = "label") async throws -> Data {
        guard let layers = trainedLayers, !layers.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        let labels = trainedClasses?.map { String(Int($0)) }
        return CoreMLExporter.exportBinaryMLPClassifier(
            name: "SwiftSciMLPClassifier",
            inputNames: featureNames,
            outputName: outputName,
            layers: layers,
            activation: activation.rawValue,
            classLabels: labels
        )
    }
}

// MARK: - MLPRegressor Conformance

extension MLPRegressor: CoreMLExportable {
    /// Exports the fitted Multi-Layer Perceptron regressor as a binary `.mlmodel` (`NeuralNetwork`).
    public func exportCoreML(featureNames: [String], outputName: String = "target") async throws -> Data {
        guard let layers = trainedLayers, !layers.isEmpty else {
            throw SwiftMLError.modelNotFitted
        }
        return CoreMLExporter.exportBinaryMLPRegressor(
            name: "SwiftSciMLPRegressor",
            inputNames: featureNames,
            outputName: outputName,
            layers: layers,
            activation: activation.rawValue
        )
    }
}

#endif // os(macOS)
