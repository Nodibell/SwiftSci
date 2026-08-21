import Foundation

// MARK: - CoreMLExporter

/// Exports fitted SwiftML models to Apple Core ML artifact formats.
///
/// ## Supported export formats
///
/// | Method | Output | Status |
/// |--------|--------|--------|
/// | ``exportLinearModel(name:inputNames:outputName:weights:bias:)`` | JSON metadata | ⚠️ Deprecated |
/// | ``exportBinaryLinearModel(name:inputNames:outputName:weights:bias:)`` | `.mlmodel` binary | ✅ Recommended |
/// | ``exportBinaryLogisticModel(name:inputNames:outputName:weights:bias:classLabels:)`` | `.mlmodel` binary | ✅ |
/// | ``exportBinaryDecisionTreeClassifier(state:featureNames:outputName:classLabels:)`` | `.mlmodel` binary | ✅ |
/// | ``exportBinaryDecisionTreeRegressor(state:featureNames:outputName:)`` | `.mlmodel` binary | ✅ |
/// | ``exportBinaryRandomForestClassifier(state:featureNames:outputName:classLabels:)`` | `.mlmodel` binary | ✅ |
/// | ``exportBinaryRandomForestRegressor(state:featureNames:outputName:)`` | `.mlmodel` binary | ✅ |
///
/// ## Binary format compatibility
///
/// All `exportBinary*` methods produce `specificationVersion = 4` Core ML models,
/// compatible with Xcode 15+, macOS 14+, iOS 17+, and visionOS 1+.
/// Binary `.mlmodel` files can be loaded with `MLModel(contentsOf:)` and deployed to
/// CPU, GPU, or Apple Neural Engine via Core ML inference pipeline.
///
/// ## Out of scope
///
/// - `MLPClassifier` / `MLPRegressor` (`NeuralNetwork` proto subtree — deferred beyond v3.1)
/// - `.mlpackage` directory bundles with manifest (single `.mlmodel` satisfies G-001)
/// - GradientBoostedTrees ensemble export
///
/// ## Example
///
/// ```swift
/// import SwiftML
///
/// // Fit a linear regression model
/// let model = LinearRegression()
/// try await model.fit(features: X_train, targets: y_train)
///
/// // Export as binary .mlmodel
/// let url = URL(fileURLWithPath: "/tmp/MyModel.mlmodel")
/// try await model.writeCoreML(to: url, featureNames: ["x1", "x2"], outputName: "prediction")
/// ```
public enum CoreMLExporter {

    // MARK: - Deprecated JSON export (legacy)

    /// Serializes linear regression weights and bias into a CoreML JSON specification.
    ///
    /// - Important: This method produces a **JSON metadata blob**, not a loadable
    ///   `.mlmodel` binary. Use ``exportBinaryLinearModel(name:inputNames:outputName:weights:bias:)``
    ///   or ``CoreMLExportable/exportCoreML(featureNames:outputName:)`` for production deployments.
    ///
    /// - Parameters:
    ///   - name: Display name embedded in the JSON payload.
    ///   - inputNames: Input feature identifiers.
    ///   - outputName: Output target feature identifier.
    ///   - weights: Fitted regression coefficient array.
    ///   - bias: Fitted intercept term.
    /// - Throws: JSON encoding errors.
    /// - Returns: JSON `Data` representing model metadata.
    @available(*, deprecated, message: "Produces JSON metadata only — not a loadable .mlmodel. Use exportBinaryLinearModel or model.exportCoreML() instead.")
    public static func exportLinearModel(
        name: String = "SwiftSciLinearModel",
        inputNames: [String],
        outputName: String = "target",
        weights: [Double],
        bias: Double
    ) throws -> Data {
        let spec = CoreMLModelSpec(
            modelName: name,
            author: "SwiftSci Ecosystem",
            license: "MIT",
            inputFeatures: inputNames,
            outputFeature: outputName,
            weights: weights,
            bias: bias
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(spec)
    }

    // MARK: - Binary Linear Regression (GLMRegressor)

    /// Encodes a fitted linear regression model as a binary Apple Core ML `.mlmodel` artifact.
    ///
    /// The output uses the `GLMRegressor` Core ML message with `specificationVersion = 4`.
    /// Feature weights and bias are encoded as double-precision packed fields.
    ///
    /// - Parameters:
    ///   - name: Model display name (embedded in the `.mlmodel` spec description).
    ///   - inputNames: Ordered list of input feature names, matching the training feature order.
    ///   - outputName: Output prediction feature name.
    ///   - weights: Fitted regression coefficients (one per feature).
    ///   - bias: Fitted intercept / bias term.
    /// - Returns: Binary `Data` that can be written to a `.mlmodel` file and loaded via
    ///   `MLModel(contentsOf:)` on macOS 14+ or iOS 17+.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let data = CoreMLExporter.exportBinaryLinearModel(
    ///     name: "PricePredictor",
    ///     inputNames: ["sqft", "bedrooms"],
    ///     outputName: "price",
    ///     weights: [250.0, 15000.0],
    ///     bias: 50000.0
    /// )
    /// try data.write(to: URL(fileURLWithPath: "/tmp/PricePredictor.mlmodel"))
    /// ```
    public static func exportBinaryLinearModel(
        name: String = "SwiftSciLinearModel",
        inputNames: [String],
        outputName: String = "target",
        weights: [Double],
        bias: Double
    ) -> Data {
        buildGLMRegressorModel(name: name, inputNames: inputNames, outputName: outputName, weights: weights, bias: bias)
    }

    /// Writes a fitted linear regression model as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL (`.mlmodel` extension recommended).
    ///   - name: Model display name.
    ///   - inputNames: Ordered list of input feature names.
    ///   - outputName: Output prediction feature name.
    ///   - weights: Fitted regression coefficients.
    ///   - bias: Fitted intercept term.
    /// - Throws: `SwiftMLError.exportFailed` if the file cannot be written.
    public static func writeLinearModel(
        to url: URL,
        name: String = "SwiftSciLinearModel",
        inputNames: [String],
        outputName: String = "target",
        weights: [Double],
        bias: Double
    ) throws {
        let data = exportBinaryLinearModel(name: name, inputNames: inputNames, outputName: outputName, weights: weights, bias: bias)
        do {
            try data.write(to: url)
        } catch {
            throw SwiftMLError.exportFailed("Failed to write linear .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Logistic Regression (GLMClassifier)

    /// Encodes a fitted binary logistic regression model as a binary Apple Core ML `.mlmodel` artifact.
    ///
    /// The output uses the `GLMClassifier` Core ML message with `postEvaluationTransform = Logit`
    /// (sigmoid probability conversion). Class labels default to `[0, 1]` for binary classification.
    ///
    /// - Parameters:
    ///   - name: Model display name.
    ///   - inputNames: Ordered input feature names.
    ///   - outputName: Predicted class label output name (Int64).
    ///   - weights: Fitted logistic regression coefficients.
    ///   - bias: Fitted intercept term.
    ///   - classLabels: Integer class label values (default `[0, 1]`).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryLogisticModel(
        name: String = "SwiftSciLogisticModel",
        inputNames: [String],
        outputName: String = "label",
        weights: [Double],
        bias: Double,
        classLabels: [Int64] = [0, 1]
    ) -> Data {
        buildGLMClassifierModel(name: name, inputNames: inputNames, outputName: outputName, weights: weights, bias: bias, classLabels: classLabels)
    }

    /// Writes a fitted logistic regression model as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - name: Model display name.
    ///   - inputNames: Ordered input feature names.
    ///   - outputName: Predicted class label output name.
    ///   - weights: Fitted coefficients.
    ///   - bias: Fitted intercept.
    ///   - classLabels: Integer class labels.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeLogisticModel(
        to url: URL,
        name: String = "SwiftSciLogisticModel",
        inputNames: [String],
        outputName: String = "label",
        weights: [Double],
        bias: Double,
        classLabels: [Int64] = [0, 1]
    ) throws {
        let data = exportBinaryLogisticModel(name: name, inputNames: inputNames, outputName: outputName, weights: weights, bias: bias, classLabels: classLabels)
        do {
            try data.write(to: url)
        } catch {
            throw SwiftMLError.exportFailed("Failed to write logistic .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Decision Tree Classifier (TreeEnsembleClassifier)

    /// Encodes a fitted decision tree classifier model state as a binary Apple Core ML `.mlmodel`.
    ///
    /// - Parameters:
    ///   - state: The serialised ``DecisionTreeModelState`` containing fitted `FlatTreeNode` array.
    ///   - featureNames: Ordered list of input feature names matching training column order.
    ///   - outputName: Output predicted class label name (Int64).
    ///   - classLabels: Integer class labels (default `[0, 1]`).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryDecisionTreeClassifier(
        state: DecisionTreeModelState,
        featureNames: [String],
        outputName: String = "label",
        classLabels: [Int64] = [0, 1]
    ) -> Data {
        buildTreeEnsembleClassifierModel(
            name: "SwiftSciDecisionTreeClassifier",
            inputNames: featureNames,
            outputName: outputName,
            treesNodes: [state.nodes],
            classLabels: classLabels
        )
    }

    /// Writes a fitted decision tree classifier as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - state: Fitted model state.
    ///   - featureNames: Input feature names.
    ///   - outputName: Output label feature name.
    ///   - classLabels: Integer class labels.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeDecisionTreeClassifier(
        to url: URL,
        state: DecisionTreeModelState,
        featureNames: [String],
        outputName: String = "label",
        classLabels: [Int64] = [0, 1]
    ) throws {
        let data = exportBinaryDecisionTreeClassifier(state: state, featureNames: featureNames, outputName: outputName, classLabels: classLabels)
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write decision tree classifier .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Decision Tree Regressor (TreeEnsembleRegressor)

    /// Encodes a fitted decision tree regressor model state as a binary Apple Core ML `.mlmodel`.
    ///
    /// - Parameters:
    ///   - state: The serialised ``DecisionTreeModelState`` with fitted `FlatTreeNode` array.
    ///   - featureNames: Input feature names matching training column order.
    ///   - outputName: Output prediction column name (Double).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryDecisionTreeRegressor(
        state: DecisionTreeModelState,
        featureNames: [String],
        outputName: String = "prediction"
    ) -> Data {
        buildTreeEnsembleRegressorModel(
            name: "SwiftSciDecisionTreeRegressor",
            inputNames: featureNames,
            outputName: outputName,
            treesNodes: [state.nodes]
        )
    }

    /// Writes a fitted decision tree regressor as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - state: Fitted model state.
    ///   - featureNames: Input feature names.
    ///   - outputName: Output prediction feature name.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeDecisionTreeRegressor(
        to url: URL,
        state: DecisionTreeModelState,
        featureNames: [String],
        outputName: String = "prediction"
    ) throws {
        let data = exportBinaryDecisionTreeRegressor(state: state, featureNames: featureNames, outputName: outputName)
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write decision tree regressor .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Random Forest Classifier (TreeEnsembleClassifier, multi-tree)

    /// Encodes a fitted random forest classifier state as a binary Apple Core ML `.mlmodel`.
    ///
    /// All trees in the forest are encoded as separate tree IDs within a single
    /// `TreeEnsembleClassifier` message. Core ML applies majority-vote aggregation by default.
    ///
    /// - Parameters:
    ///   - state: The serialised ``RandomForestModelState`` containing all fitted trees.
    ///   - featureNames: Input feature names matching training column order.
    ///   - outputName: Output predicted class label name (Int64).
    ///   - classLabels: Integer class labels (default `[0, 1]`).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryRandomForestClassifier(
        state: RandomForestModelState,
        featureNames: [String],
        outputName: String = "label",
        classLabels: [Int64] = [0, 1]
    ) -> Data {
        buildTreeEnsembleClassifierModel(
            name: "SwiftSciRandomForestClassifier",
            inputNames: featureNames,
            outputName: outputName,
            treesNodes: state.trees,
            classLabels: classLabels
        )
    }

    /// Writes a fitted random forest classifier as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - state: Fitted model state.
    ///   - featureNames: Input feature names.
    ///   - outputName: Output label feature name.
    ///   - classLabels: Integer class labels.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeRandomForestClassifier(
        to url: URL,
        state: RandomForestModelState,
        featureNames: [String],
        outputName: String = "label",
        classLabels: [Int64] = [0, 1]
    ) throws {
        let data = exportBinaryRandomForestClassifier(state: state, featureNames: featureNames, outputName: outputName, classLabels: classLabels)
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write random forest classifier .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Random Forest Regressor (TreeEnsembleRegressor, multi-tree)

    /// Encodes a fitted random forest regressor state as a binary Apple Core ML `.mlmodel`.
    ///
    /// All trees are encoded in a single `TreeEnsembleRegressor` message. Core ML applies
    /// average aggregation across trees by default.
    ///
    /// - Parameters:
    ///   - state: The serialised ``RandomForestModelState`` containing fitted trees.
    ///   - featureNames: Input feature names.
    ///   - outputName: Output prediction column name (Double).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryRandomForestRegressor(
        state: RandomForestModelState,
        featureNames: [String],
        outputName: String = "prediction"
    ) -> Data {
        buildTreeEnsembleRegressorModel(
            name: "SwiftSciRandomForestRegressor",
            inputNames: featureNames,
            outputName: outputName,
            treesNodes: state.trees
        )
    }

    /// Writes a fitted random forest regressor as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - state: Fitted model state.
    ///   - featureNames: Input feature names.
    ///   - outputName: Output prediction feature name.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeRandomForestRegressor(
        to url: URL,
        state: RandomForestModelState,
        featureNames: [String],
        outputName: String = "prediction"
    ) throws {
        let data = exportBinaryRandomForestRegressor(state: state, featureNames: featureNames, outputName: outputName)
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write random forest regressor .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Scaler (StandardScaler)

    /// Encodes a feature standard scaler as a binary Apple Core ML `.mlmodel` artifact.
    ///
    /// The output uses the `Scaler` Core ML message with `specificationVersion = 4`.
    ///
    /// - Parameters:
    ///   - name: Model display name.
    ///   - inputNames: Input feature identifiers.
    ///   - outputNames: Output scaled feature identifiers.
    ///   - shiftValues: Shift offsets (typically `-mean`).
    ///   - scaleValues: Scale multipliers (typically `1 / std`).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryStandardScaler(
        name: String = "SwiftSciStandardScaler",
        inputNames: [String],
        outputNames: [String]? = nil,
        shiftValues: [Double],
        scaleValues: [Double]
    ) -> Data {
        buildScalerModel(
            name: name,
            inputNames: inputNames,
            outputNames: outputNames,
            shiftValues: shiftValues,
            scaleValues: scaleValues
        )
    }

    /// Writes a standard scaler as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - name: Model display name.
    ///   - inputNames: Input feature identifiers.
    ///   - outputNames: Output scaled feature identifiers.
    ///   - shiftValues: Shift offsets.
    ///   - scaleValues: Scale multipliers.
    /// - Throws: `SwiftMLError.exportFailed` on I/O failure.
    public static func writeStandardScaler(
        to url: URL,
        name: String = "SwiftSciStandardScaler",
        inputNames: [String],
        outputNames: [String]? = nil,
        shiftValues: [Double],
        scaleValues: [Double]
    ) throws {
        let data = exportBinaryStandardScaler(
            name: name,
            inputNames: inputNames,
            outputNames: outputNames,
            shiftValues: shiftValues,
            scaleValues: scaleValues
        )
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write standard scaler .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Multi-Layer Perceptron (NeuralNetwork)

    /// Encodes a fitted Multi-Layer Perceptron classifier as a binary Apple Core ML `.mlmodel` artifact.
    public static func exportBinaryMLPClassifier(
        name: String = "SwiftSciMLPClassifier",
        inputNames: [String],
        outputName: String = "target",
        layers: [LayerWeights],
        activation: String = "relu",
        classLabels: [String]? = nil
    ) -> Data {
        let weights = layers.map { layer -> [[Double]] in
            var w2D = [[Double]](repeating: [Double](repeating: 0.0, count: layer.inDim), count: layer.outDim)
            for i in 0..<layer.inDim {
                for j in 0..<layer.outDim {
                    let flatIdx = i * layer.outDim + j
                    if flatIdx < layer.W.count {
                        w2D[j][i] = layer.W[flatIdx]
                    }
                }
            }
            return w2D
        }
        let biases = layers.map { $0.b }

        return buildNeuralNetworkModel(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            layerWeights: weights,
            layerBiases: biases,
            hiddenActivation: activation,
            outputActivation: layers.last?.outDim == 1 ? "sigmoid" : "softmax"
        )
    }

    /// Encodes a fitted Multi-Layer Perceptron regressor as a binary Apple Core ML `.mlmodel` artifact.
    public static func exportBinaryMLPRegressor(
        name: String = "SwiftSciMLPRegressor",
        inputNames: [String],
        outputName: String = "target",
        layers: [LayerWeights],
        activation: String = "relu"
    ) -> Data {
        let weights = layers.map { layer -> [[Double]] in
            var w2D = [[Double]](repeating: [Double](repeating: 0.0, count: layer.inDim), count: layer.outDim)
            for i in 0..<layer.inDim {
                for j in 0..<layer.outDim {
                    let flatIdx = i * layer.outDim + j
                    if flatIdx < layer.W.count {
                        w2D[j][i] = layer.W[flatIdx]
                    }
                }
            }
            return w2D
        }
        let biases = layers.map { $0.b }

        return buildNeuralNetworkModel(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            layerWeights: weights,
            layerBiases: biases,
            hiddenActivation: activation,
            outputActivation: "linear"
        )
    }

    /// Writes a binary `.mlmodel` file for a fitted MLP classifier.
    public static func writeMLPClassifier(
        to url: URL,
        name: String = "SwiftSciMLPClassifier",
        inputNames: [String],
        outputName: String = "target",
        layers: [LayerWeights],
        activation: String = "relu",
        classLabels: [String]? = nil
    ) throws {
        let data = exportBinaryMLPClassifier(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            layers: layers,
            activation: activation,
            classLabels: classLabels
        )
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write MLP classifier .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    /// Writes a binary `.mlmodel` file for a fitted MLP regressor.
    public static func writeMLPRegressor(
        to url: URL,
        name: String = "SwiftSciMLPRegressor",
        inputNames: [String],
        outputName: String = "target",
        layers: [LayerWeights],
        activation: String = "relu"
    ) throws {
        let data = exportBinaryMLPRegressor(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            layers: layers,
            activation: activation
        )
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write MLP regressor .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Binary Pipeline Exporter (PipelineClassifier / PipelineRegressor)

    /// Encodes a composite sequence of submodels into a single binary Apple Core ML `PipelineClassifier` `.mlmodel`.
    ///
    /// - Parameters:
    ///   - name: Composite pipeline model display name.
    ///   - inputNames: Input feature identifiers for the initial pipeline stage.
    ///   - outputName: Final classification label output name.
    ///   - submodelsData: Ordered array of serialized submodel `.mlmodel` binary Data (e.g. `[scalerData, modelData]`).
    ///   - submodelNames: Optional display names for each submodel stage.
    ///   - classLabels: Class label values (default `[0, 1]`).
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryPipelineClassifier(
        name: String = "SwiftSciPipelineClassifier",
        inputNames: [String],
        outputName: String = "label",
        submodelsData: [Data],
        submodelNames: [String]? = nil,
        classLabels: [Int64] = [0, 1]
    ) -> Data {
        buildPipelineClassifierModel(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            submodelsData: submodelsData,
            submodelNames: submodelNames,
            classLabels: classLabels
        )
    }

    /// Writes a composite `PipelineClassifier` as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination `.mlmodel` file URL.
    ///   - name: Composite pipeline model display name.
    ///   - inputNames: Input feature identifiers.
    ///   - outputName: Final classification label output name.
    ///   - submodelsData: Ordered array of serialized submodel `.mlmodel` binaries.
    ///   - submodelNames: Optional display names for each stage.
    ///   - classLabels: Class label values.
    /// - Throws: `SwiftMLError.exportFailed` on I/O write failure.
    public static func writePipelineClassifier(
        to url: URL,
        name: String = "SwiftSciPipelineClassifier",
        inputNames: [String],
        outputName: String = "label",
        submodelsData: [Data],
        submodelNames: [String]? = nil,
        classLabels: [Int64] = [0, 1]
    ) throws {
        let data = exportBinaryPipelineClassifier(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            submodelsData: submodelsData,
            submodelNames: submodelNames,
            classLabels: classLabels
        )
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write pipeline classifier .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    /// Encodes a composite sequence of submodels into a single binary Apple Core ML `PipelineRegressor` `.mlmodel`.
    ///
    /// - Parameters:
    ///   - name: Composite pipeline model display name.
    ///   - inputNames: Input feature identifiers for the initial pipeline stage.
    ///   - outputName: Final continuous prediction output name.
    ///   - submodelsData: Ordered array of serialized submodel `.mlmodel` binary Data.
    ///   - submodelNames: Optional display names for each submodel stage.
    /// - Returns: Binary `.mlmodel` `Data`.
    public static func exportBinaryPipelineRegressor(
        name: String = "SwiftSciPipelineRegressor",
        inputNames: [String],
        outputName: String = "target",
        submodelsData: [Data],
        submodelNames: [String]? = nil
    ) -> Data {
        buildPipelineRegressorModel(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            submodelsData: submodelsData,
            submodelNames: submodelNames
        )
    }

    /// Writes a composite `PipelineRegressor` as a binary `.mlmodel` file to disk.
    ///
    /// - Parameters:
    ///   - url: Destination `.mlmodel` file URL.
    ///   - name: Composite pipeline model display name.
    ///   - inputNames: Input feature identifiers.
    ///   - outputName: Final continuous prediction output name.
    ///   - submodelsData: Ordered array of serialized submodel `.mlmodel` binaries.
    ///   - submodelNames: Optional display names for each stage.
    /// - Throws: `SwiftMLError.exportFailed` on I/O write failure.
    public static func writePipelineRegressor(
        to url: URL,
        name: String = "SwiftSciPipelineRegressor",
        inputNames: [String],
        outputName: String = "target",
        submodelsData: [Data],
        submodelNames: [String]? = nil
    ) throws {
        let data = exportBinaryPipelineRegressor(
            name: name,
            inputNames: inputNames,
            outputName: outputName,
            submodelsData: submodelsData,
            submodelNames: submodelNames
        )
        do { try data.write(to: url) } catch {
            throw SwiftMLError.exportFailed("Failed to write pipeline regressor .mlmodel to \(url.path): \(error.localizedDescription)")
        }
    }

    // MARK: - Modern .mlpackage Directory Bundle Exporter

    /// Writes a Core ML model payload into a modern `.mlpackage` directory bundle format.
    ///
    /// - Parameters:
    ///   - modelData: Serialized `.mlmodel` binary payload data.
    ///   - packageURL: Destination `.mlpackage` bundle directory URL.
    ///   - author: Model author metadata string.
    ///   - description: Model summary description string.
    /// - Throws: `SwiftMLError.exportFailed` if directory creation or file writing fails.
    public static func writeMLPackage(
        modelData: Data,
        to packageURL: URL,
        author: String = "SwiftSci",
        description: String = "CoreML Model exported by SwiftSci"
    ) throws {
        let fm = FileManager.default
        let dataDir = packageURL.appendingPathComponent("Data/com.apple.CoreML", isDirectory: true)

        do {
            try fm.createDirectory(at: dataDir, withIntermediateDirectories: true)
            let modelURL = dataDir.appendingPathComponent("model.mlmodel")
            try modelData.write(to: modelURL)

            let manifestJSON = """
            {
                "fileFormatVersion": "1.0.0",
                "itemInfoEntries": {
                    "com.apple.CoreML/model.mlmodel": {
                        "author": "\(author)",
                        "description": "\(description)",
                        "name": "model.mlmodel",
                        "path": "com.apple.CoreML/model.mlmodel"
                    }
                },
                "rootModelIdentifier": "com.apple.CoreML/model.mlmodel"
            }
            """
            let manifestURL = packageURL.appendingPathComponent("Manifest.json")
            try manifestJSON.write(to: manifestURL, atomically: true, encoding: .utf8)
        } catch {
            throw SwiftMLError.exportFailed("Failed to generate .mlpackage at \(packageURL.path): \(error.localizedDescription)")
        }
    }
}

// MARK: - Legacy JSON Spec (kept for backward compatibility)

extension CoreMLExporter {
    /// Represents a CoreML model specification metadata and parameter payload (JSON format).
    ///
    /// - Note: Retained for backward compatibility with ``exportLinearModel(name:inputNames:outputName:weights:bias:)``.
    ///   New code should use binary export methods and ``CoreMLExportable`` protocol conformances.
    @available(*, deprecated, message: "Use CoreMLExportable.exportCoreML() or CoreMLExporter.exportBinaryLinearModel() for loadable .mlmodel artifacts.")
    public struct CoreMLModelSpec: Codable, Sendable {
        /// Display name of the CoreML model.
        public let modelName: String
        /// Author metadata string.
        public let author: String
        /// License metadata string.
        public let license: String
        /// Array of input feature identifier names.
        public let inputFeatures: [String]
        /// Output target feature identifier name.
        public let outputFeature: String
        /// Feature weight coefficient array.
        public let weights: [Double]
        /// Intercept bias term.
        public let bias: Double
    }
}
