import Foundation

// MARK: - Apple Core ML Model.proto Wire-Format Builders
//
// Encodes binary `.mlmodel` artifacts following Apple's CoreML specification v4
// (compatible with Xcode 15+ / macOS 14+ / iOS 17+).
//
// Field numbers are derived directly from Apple's public CoreML proto repository:
//   https://github.com/apple/coremltools/tree/main/mlmodel/format
//
// ## Supported model types
//   - GLMRegressor           (field 300) → linear regression
//   - GLMClassifier          (field 400) → binary logistic regression
//   - TreeEnsembleClassifier (field 402) → decision tree & random forest classifiers
//   - TreeEnsembleRegressor  (field 302) → decision tree & random forest regressors

// MARK: - Model.proto field numbers

private enum ModelField {
    static let specificationVersion: Int   = 1   // int32
    static let description: Int            = 2   // ModelDescription
    static let glmRegressor: Int           = 300 // GLMRegressor
    static let treeEnsembleRegressor: Int  = 302 // TreeEnsembleRegressor
    static let glmClassifier: Int          = 400 // GLMClassifier
    static let treeEnsembleClassifier: Int = 402 // TreeEnsembleClassifier
    static let neuralNetwork: Int          = 500 // NeuralNetwork
    static let scaler: Int                 = 604 // Scaler
}

private enum ScalerField {
    static let shiftValue: Int = 1 // repeated double (packed)
    static let scaleValue: Int = 2 // repeated double (packed)
}

private enum NeuralNetworkField {
    static let layers: Int = 1                // repeated NeuralNetworkLayer
    static let arrayInputShapeMapping: Int = 5 // enum: 0 = RANK5_ARRAY_MAPPING, 1 = EXACT_ARRAY_MAPPING
}

private enum NeuralNetworkLayerField {
    static let name: Int = 1           // string
    static let input: Int = 2          // repeated string
    static let output: Int = 3         // repeated string
    static let activation: Int = 130   // ActivationParams
    static let innerProduct: Int = 140 // InnerProductLayerParams
    static let softmax: Int = 175      // SoftmaxLayerParams
    static let concat: Int = 320       // ConcatLayerParams
}

private enum InnerProductLayerParamsField {
    static let inputChannels: Int = 1  // uint64
    static let outputChannels: Int = 2 // uint64
    static let hasBias: Int = 10       // bool
    static let weights: Int = 20       // WeightParams
    static let bias: Int = 21          // WeightParams
}

private enum WeightParamsField {
    static let floatValue: Int = 1 // repeated float (packed)
}

private enum ActivationParamsField {
    static let linear: Int = 5
    static let reLU: Int = 10
    static let tanh: Int = 30
    static let sigmoid: Int = 40
}

private enum ModelDescriptionField {
    static let input: Int                = 1  // repeated FeatureDescription
    static let output: Int               = 10 // repeated FeatureDescription
    static let predictedFeatureName: Int = 11 // string
}

private enum FeatureDescriptionField {
    static let name: Int = 1  // string
    static let type: Int = 3  // FeatureType
}

private enum FeatureTypeField {
    static let int64Type: Int      = 1 // Int64FeatureType
    static let doubleType: Int     = 2 // DoubleFeatureType
    static let stringType: Int     = 3 // StringFeatureType
    static let imageType: Int      = 4 // ImageFeatureType
    static let multiArrayType: Int = 5 // ArrayFeatureType
}

private enum ArrayFeatureTypeField {
    static let shape: Int    = 1 // repeated int64
    static let dataType: Int = 2 // ArrayDataType enum (DOUBLE = 65600, FLOAT32 = 65568)
}

// MARK: - GLMRegressor field numbers (GLMRegressor.proto)

private enum GLMRegressorField {
    static let weights: Int        = 1 // repeated DoubleArray
    static let offset: Int         = 2 // repeated double (bias)
    static let postTransform: Int  = 3 // PostEvaluationTransform enum (0 = NoTransform)
}

private enum DoubleArrayField {
    static let values: Int = 1 // repeated double (packed)
}

// MARK: - GLMClassifier field numbers (GLMClassifier.proto)

private enum GLMClassifierField {
    static let weights: Int          = 1   // repeated DoubleArray
    static let offset: Int           = 2   // repeated double (bias)
    static let postTransform: Int    = 3   // PostEvaluationTransform (0 = Logit)
    static let classEncoding: Int    = 4   // ClassEncoding (0 = ReferenceClass)
    static let stringClassLabels: Int = 100 // StringVector
    static let int64ClassLabels: Int = 101 // Int64Vector
}

private enum Int64VectorField {
    static let vector: Int = 1 // repeated int64 (packed)
}

// MARK: - TreeEnsemble field numbers (TreeEnsemble.proto)

private enum TreeEnsembleClassifierField {
    static let treeEnsemble: Int     = 1   // TreeEnsembleParameters
    static let postTransform: Int    = 2   // TreeEnsemblePostEvaluationTransform (0 = NoTransform)
    static let int64ClassLabels: Int = 101 // Int64Vector
}

private enum TreeEnsembleRegressorField {
    static let treeEnsemble: Int  = 1 // TreeEnsembleParameters
    static let postTransform: Int = 2 // TreeEnsemblePostEvaluationTransform (0 = NoTransform)
}

private enum TreeEnsembleParametersField {
    static let nodes: Int                   = 1 // repeated TreeNode
    static let numPredictionDimensions: Int = 2 // uint64
    static let basePredictionValue: Int     = 3 // repeated double
}

private enum TreeNodeField {
    static let treeId: Int              = 1  // uint64
    static let nodeId: Int              = 2  // uint64
    static let nodeBehavior: Int        = 3  // TreeNodeBehavior enum
    static let branchFeatureIndex: Int  = 10 // uint64
    static let branchFeatureValue: Int  = 11 // double
    static let trueChildNodeId: Int     = 12 // uint64
    static let falseChildNodeId: Int    = 13 // uint64
    static let evaluationInfo: Int      = 20 // repeated EvaluationInfo
}

// TreeNodeBehavior enum values in TreeEnsemble.proto
private enum TreeNodeBehavior: UInt64 {
    case branchOnValueLessThanEqual = 0
    case leafNode                   = 6
}

private enum EvaluationInfoField {
    static let evaluationIndex: Int = 1 // uint64 — which output dimension
    static let evaluationValue: Int = 2 // double — prediction value
}

// MARK: - FeatureDescription builder helpers

private func encodeDoubleFeature(name: String) -> Data {
    var fw = ProtobufWriter()
    fw.writeStringField(fieldNumber: FeatureDescriptionField.name, value: name)
    // DoubleFeatureType (field 2 in FeatureType) is an empty message — presence signals Double
    var tw = ProtobufWriter()
    tw.writeBytesField(fieldNumber: FeatureTypeField.doubleType, bytes: Data())
    fw.writeBytesField(fieldNumber: FeatureDescriptionField.type, bytes: tw.data)
    return fw.data
}

private func encodeMultiArrayFeature(name: String, shape: [Int] = [1], isDouble: Bool = true) -> Data {
    var fw = ProtobufWriter()
    fw.writeStringField(fieldNumber: FeatureDescriptionField.name, value: name)
    var arrayType = ProtobufWriter()
    for s in shape {
        arrayType.writeVarintField(fieldNumber: ArrayFeatureTypeField.shape, value: UInt64(s))
    }
    arrayType.writeVarintField(fieldNumber: ArrayFeatureTypeField.dataType, value: isDouble ? 65600 : 65568)
    var tw = ProtobufWriter()
    tw.writeBytesField(fieldNumber: FeatureTypeField.multiArrayType, bytes: arrayType.data)
    fw.writeBytesField(fieldNumber: FeatureDescriptionField.type, bytes: tw.data)
    return fw.data
}

private func encodeInt64Feature(name: String) -> Data {
    var fw = ProtobufWriter()
    fw.writeStringField(fieldNumber: FeatureDescriptionField.name, value: name)
    var tw = ProtobufWriter()
    tw.writeBytesField(fieldNumber: FeatureTypeField.int64Type, bytes: Data())
    fw.writeBytesField(fieldNumber: FeatureDescriptionField.type, bytes: tw.data)
    return fw.data
}

private func buildModelDescription(inputNames: [String], outputName: String, outputIsInt64: Bool = false) -> Data {
    var desc = ProtobufWriter()
    for name in inputNames {
        desc.writeBytesField(fieldNumber: ModelDescriptionField.input, bytes: encodeDoubleFeature(name: name))
    }
    let outData = outputIsInt64 ? encodeInt64Feature(name: outputName) : encodeDoubleFeature(name: outputName)
    desc.writeBytesField(fieldNumber: ModelDescriptionField.output, bytes: outData)
    desc.writeStringField(fieldNumber: ModelDescriptionField.predictedFeatureName, value: outputName)
    return desc.data
}

// MARK: - GLMRegressor builder

/// Builds a binary `.mlmodel` payload encoding a linear regression model as a `GLMRegressor`.
///
/// - Parameters:
///   - name: Model display name embedded in the spec description.
///   - inputNames: Names of the input double features.
///   - outputName: Name of the predicted double output.
///   - weights: Fitted regression coefficients (one per input feature).
///   - bias: Fitted intercept term.
/// - Returns: Binary `Data` representing a complete, loadable Apple Core ML `.mlmodel` file.
internal func buildGLMRegressorModel(
    name: String,
    inputNames: [String],
    outputName: String,
    weights: [Double],
    bias: Double
) -> Data {
    var model = ProtobufWriter()

    // specificationVersion = 4 (CoreML v4, Xcode 15+ / macOS 14+)
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)

    // ModelDescription
    model.writeBytesField(
        fieldNumber: ModelField.description,
        bytes: buildModelDescription(inputNames: inputNames, outputName: outputName)
    )

    // GLMRegressor payload
    var glm = ProtobufWriter()
    // weights: repeated DoubleArray — one DoubleArray per output dimension (1 for regression)
    var doubleArray = ProtobufWriter()
    doubleArray.writePackedDoublesField(fieldNumber: DoubleArrayField.values, values: weights)
    glm.writeBytesField(fieldNumber: GLMRegressorField.weights, bytes: doubleArray.data)
    // offset (bias)
    glm.writePackedDoublesField(fieldNumber: GLMRegressorField.offset, values: [bias])
    // postEvaluationTransform = 0 (NoTransform — identity)
    glm.writeVarintField(fieldNumber: GLMRegressorField.postTransform, value: 0)

    model.writeBytesField(fieldNumber: ModelField.glmRegressor, bytes: glm.data)
    return model.data
}

// MARK: - GLMClassifier builder

/// Builds a binary `.mlmodel` payload encoding a binary logistic regression model as a `GLMClassifier`.
///
/// - Parameters:
///   - name: Model display name.
///   - inputNames: Names of the input double features.
///   - outputName: Name of the predicted label output (Int64: 0 or 1).
///   - weights: Fitted logistic regression coefficients.
///   - bias: Fitted intercept term.
///   - classLabels: Integer class labels (default: `[0, 1]`).
/// - Returns: Binary `Data` representing a complete, loadable Apple Core ML `.mlmodel` file.
internal func buildGLMClassifierModel(
    name: String,
    inputNames: [String],
    outputName: String,
    weights: [Double],
    bias: Double,
    classLabels: [Int64] = [0, 1]
) -> Data {
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)
    model.writeBytesField(
        fieldNumber: ModelField.description,
        bytes: buildModelDescription(inputNames: inputNames, outputName: outputName, outputIsInt64: true)
    )

    var glm = ProtobufWriter()
    var doubleArray = ProtobufWriter()
    doubleArray.writePackedDoublesField(fieldNumber: DoubleArrayField.values, values: weights)
    glm.writeBytesField(fieldNumber: GLMClassifierField.weights, bytes: doubleArray.data)
    glm.writePackedDoublesField(fieldNumber: GLMClassifierField.offset, values: [bias])
    // postEvaluationTransform = 0 (Logit in GLMClassifier.proto)
    glm.writeVarintField(fieldNumber: GLMClassifierField.postTransform, value: 0)
    // classEncoding = 0 (ReferenceClass)
    glm.writeVarintField(fieldNumber: GLMClassifierField.classEncoding, value: 0)
    // int64ClassLabels
    var labelVec = ProtobufWriter()
    labelVec.writePackedVarintsField(
        fieldNumber: Int64VectorField.vector,
        values: classLabels.map { UInt64(bitPattern: Int64($0)) }
    )
    glm.writeBytesField(fieldNumber: GLMClassifierField.int64ClassLabels, bytes: labelVec.data)

    model.writeBytesField(fieldNumber: ModelField.glmClassifier, bytes: glm.data)
    return model.data
}

// MARK: - TreeEnsembleClassifier builder

/// Builds a binary `.mlmodel` for a decision tree or random forest **classifier**.
///
/// - Parameters:
///   - name: Model display name.
///   - inputNames: Feature column names.
///   - outputName: Output label column name (Int64).
///   - treesNodes: Array of per-tree node arrays (one element for single tree, multiple for forest).
///   - classLabels: Integer class labels (default: `[0, 1]`).
/// - Returns: Binary `.mlmodel` `Data`.
internal func buildTreeEnsembleClassifierModel(
    name: String,
    inputNames: [String],
    outputName: String,
    treesNodes: [[FlatTreeNode]],
    classLabels: [Int64] = [0, 1]
) -> Data {
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)
    model.writeBytesField(
        fieldNumber: ModelField.description,
        bytes: buildModelDescription(inputNames: inputNames, outputName: outputName, outputIsInt64: true)
    )

    var clf = ProtobufWriter()
    var params = ProtobufWriter()
    let numClasses = classLabels.count
    params.writeVarintField(fieldNumber: TreeEnsembleParametersField.numPredictionDimensions, value: UInt64(numClasses))
    let basePred = [Double](repeating: 0.0, count: numClasses)
    params.writePackedDoublesField(fieldNumber: TreeEnsembleParametersField.basePredictionValue, values: basePred)

    for (treeIdx, nodes) in treesNodes.enumerated() {
        for (i, node) in nodes.enumerated() {
            var n = ProtobufWriter()
            n.writeVarintField(fieldNumber: TreeNodeField.treeId, value: UInt64(treeIdx))
            n.writeVarintField(fieldNumber: TreeNodeField.nodeId, value: UInt64(i))
            if node.isLeaf {
                n.writeVarintField(fieldNumber: TreeNodeField.nodeBehavior, value: TreeNodeBehavior.leafNode.rawValue)
                var leaf = ProtobufWriter()
                let evalIdx = min(max(0, Int(node.value)), numClasses - 1)
                leaf.writeVarintField(fieldNumber: EvaluationInfoField.evaluationIndex, value: UInt64(evalIdx))
                leaf.writeDoubleField(fieldNumber: EvaluationInfoField.evaluationValue, value: 1.0)
                n.writeBytesField(fieldNumber: TreeNodeField.evaluationInfo, bytes: leaf.data)
            } else {
                n.writeVarintField(fieldNumber: TreeNodeField.nodeBehavior, value: TreeNodeBehavior.branchOnValueLessThanEqual.rawValue)
                n.writeVarintField(fieldNumber: TreeNodeField.branchFeatureIndex, value: UInt64(node.featureIndex))
                n.writeDoubleField(fieldNumber: TreeNodeField.branchFeatureValue, value: node.threshold)
                n.writeVarintField(fieldNumber: TreeNodeField.trueChildNodeId, value: UInt64(node.leftChild))
                n.writeVarintField(fieldNumber: TreeNodeField.falseChildNodeId, value: UInt64(node.rightChild))
            }
            params.writeBytesField(fieldNumber: TreeEnsembleParametersField.nodes, bytes: n.data)
        }
    }

    clf.writeBytesField(fieldNumber: TreeEnsembleClassifierField.treeEnsemble, bytes: params.data)
    clf.writeVarintField(fieldNumber: TreeEnsembleClassifierField.postTransform, value: 0) // NoTransform
    var labelVec = ProtobufWriter()
    labelVec.writePackedVarintsField(
        fieldNumber: Int64VectorField.vector,
        values: classLabels.map { UInt64(bitPattern: Int64($0)) }
    )
    clf.writeBytesField(fieldNumber: TreeEnsembleClassifierField.int64ClassLabels, bytes: labelVec.data)

    model.writeBytesField(fieldNumber: ModelField.treeEnsembleClassifier, bytes: clf.data)
    return model.data
}

// MARK: - TreeEnsembleRegressor builder

/// Builds a binary `.mlmodel` for a decision tree or random forest **regressor**.
///
/// - Parameters:
///   - name: Model display name.
///   - inputNames: Feature column names.
///   - outputName: Output prediction column name (Double).
///   - treesNodes: Per-tree node arrays.
/// - Returns: Binary `.mlmodel` `Data`.
internal func buildTreeEnsembleRegressorModel(
    name: String,
    inputNames: [String],
    outputName: String,
    treesNodes: [[FlatTreeNode]]
) -> Data {
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)
    model.writeBytesField(
        fieldNumber: ModelField.description,
        bytes: buildModelDescription(inputNames: inputNames, outputName: outputName)
    )

    var reg = ProtobufWriter()
    var params = ProtobufWriter()
    params.writeVarintField(fieldNumber: TreeEnsembleParametersField.numPredictionDimensions, value: 1)
    params.writePackedDoublesField(fieldNumber: TreeEnsembleParametersField.basePredictionValue, values: [0.0])

    let treeWeight = 1.0 / Double(max(1, treesNodes.count))
    for (treeIdx, nodes) in treesNodes.enumerated() {
        for (i, node) in nodes.enumerated() {
            var n = ProtobufWriter()
            n.writeVarintField(fieldNumber: TreeNodeField.treeId, value: UInt64(treeIdx))
            n.writeVarintField(fieldNumber: TreeNodeField.nodeId, value: UInt64(i))
            if node.isLeaf {
                n.writeVarintField(fieldNumber: TreeNodeField.nodeBehavior, value: TreeNodeBehavior.leafNode.rawValue)
                var leaf = ProtobufWriter()
                leaf.writeVarintField(fieldNumber: EvaluationInfoField.evaluationIndex, value: 0)
                leaf.writeDoubleField(fieldNumber: EvaluationInfoField.evaluationValue, value: node.value * treeWeight)
                n.writeBytesField(fieldNumber: TreeNodeField.evaluationInfo, bytes: leaf.data)
            } else {
                n.writeVarintField(fieldNumber: TreeNodeField.nodeBehavior, value: TreeNodeBehavior.branchOnValueLessThanEqual.rawValue)
                n.writeVarintField(fieldNumber: TreeNodeField.branchFeatureIndex, value: UInt64(node.featureIndex))
                n.writeDoubleField(fieldNumber: TreeNodeField.branchFeatureValue, value: node.threshold)
                n.writeVarintField(fieldNumber: TreeNodeField.trueChildNodeId, value: UInt64(node.leftChild))
                n.writeVarintField(fieldNumber: TreeNodeField.falseChildNodeId, value: UInt64(node.rightChild))
            }
            params.writeBytesField(fieldNumber: TreeEnsembleParametersField.nodes, bytes: n.data)
        }
    }

    reg.writeBytesField(fieldNumber: TreeEnsembleRegressorField.treeEnsemble, bytes: params.data)
    reg.writeVarintField(fieldNumber: TreeEnsembleRegressorField.postTransform, value: 0)

    model.writeBytesField(fieldNumber: ModelField.treeEnsembleRegressor, bytes: reg.data)
    return model.data
}

// MARK: - Scaler builder

/// Builds a binary `.mlmodel` payload encoding a feature standard scaler as a `Scaler`.
///
/// - Parameters:
///   - name: Model display name.
///   - inputNames: Names of the input double features.
///   - outputName: Name of the output feature.
///   - shiftValues: Offset shifts for each feature (typically -mean).
///   - scaleValues: Scaling factors for each feature (typically 1 / std).
/// - Returns: Binary `Data` representing a loadable Apple Core ML `.mlmodel` file.
internal func buildScalerModel(
    name: String,
    inputNames: [String],
    outputNames: [String]? = nil,
    shiftValues: [Double],
    scaleValues: [Double]
) -> Data {
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)

    var desc = ProtobufWriter()
    for name in inputNames {
        desc.writeBytesField(fieldNumber: ModelDescriptionField.input, bytes: encodeDoubleFeature(name: name))
    }
    let outs = outputNames ?? inputNames.map { "scaled_\($0)" }
    for name in outs {
        desc.writeBytesField(fieldNumber: ModelDescriptionField.output, bytes: encodeDoubleFeature(name: name))
    }
    model.writeBytesField(fieldNumber: ModelField.description, bytes: desc.data)

    var scalerMsg = ProtobufWriter()
    if !shiftValues.isEmpty {
        scalerMsg.writePackedDoublesField(fieldNumber: ScalerField.shiftValue, values: shiftValues)
    }
    if !scaleValues.isEmpty {
        scalerMsg.writePackedDoublesField(fieldNumber: ScalerField.scaleValue, values: scaleValues)
    }

    model.writeBytesField(fieldNumber: ModelField.scaler, bytes: scalerMsg.data)
    return model.data
}

// MARK: - NeuralNetwork builder

/// Builds a binary `.mlmodel` payload encoding a Multi-Layer Perceptron (MLP) as a Core ML `NeuralNetwork`.
///
/// - Parameters:
///   - name: Model display name.
///   - inputNames: Names of the input double features.
///   - outputName: Name of the output feature.
///   - layerWeights: Array of 2D weight matrices for each layer.
///   - layerBiases: Array of 1D bias vectors for each layer.
///   - hiddenActivation: Activation function for hidden layers ("relu", "sigmoid", "tanh", "linear").
///   - outputActivation: Activation function for the output layer (or nil for linear).
/// - Returns: Binary `Data` representing a loadable Apple Core ML `.mlmodel` file.
internal func buildNeuralNetworkModel(
    name: String,
    inputNames: [String],
    outputName: String,
    layerWeights: [[[Double]]],
    layerBiases: [[Double]],
    hiddenActivation: String = "relu",
    outputActivation: String? = nil
) -> Data {
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: ModelField.specificationVersion, value: 4)

    let inDim = layerWeights.first?.first?.count ?? inputNames.count
    let inFeatureName = inputNames.count == 1 ? inputNames[0] : (inputNames.first ?? "features")

    var desc = ProtobufWriter()
    desc.writeBytesField(fieldNumber: ModelDescriptionField.input, bytes: encodeMultiArrayFeature(name: inFeatureName, shape: [inDim], isDouble: true))
    let lastOutDim = layerBiases.last?.count ?? 1
    desc.writeBytesField(fieldNumber: ModelDescriptionField.output, bytes: encodeMultiArrayFeature(name: outputName, shape: [lastOutDim], isDouble: true))
    model.writeBytesField(fieldNumber: ModelField.description, bytes: desc.data)

    var nnMsg = ProtobufWriter()
    var currentInputs = [inFeatureName]

    for layerIdx in 0..<layerWeights.count {
        let weights2D = layerWeights[layerIdx]
        let biases = layerBiases[layerIdx]
        let outDim = biases.count
        let inDim = weights2D.count > 0 ? weights2D[0].count : currentInputs.count

        // Flatten weights to [outDim * inDim] in row-major order
        var flatWeights: [Float] = []
        flatWeights.reserveCapacity(outDim * inDim)
        for r in 0..<outDim {
            for c in 0..<inDim {
                if r < weights2D.count && c < weights2D[r].count {
                    flatWeights.append(Float(weights2D[r][c]))
                } else if c < weights2D.count && r < weights2D[c].count {
                    flatWeights.append(Float(weights2D[c][r]))
                } else {
                    flatWeights.append(0.0)
                }
            }
        }

        let isLastLayer = (layerIdx == layerWeights.count - 1)
        let actType = isLastLayer ? (outputActivation ?? "linear") : hiddenActivation
        let hasActivation = actType.lowercased() != "linear"
        let ipOutName = (!hasActivation && isLastLayer) ? outputName : "dense_out_\(layerIdx)"

        var ipLayer = ProtobufWriter()
        ipLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.name, value: "dense_\(layerIdx)")
        for inp in currentInputs {
            ipLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.input, value: inp)
        }
        ipLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.output, value: ipOutName)

        var ipParams = ProtobufWriter()
        ipParams.writeVarintField(fieldNumber: InnerProductLayerParamsField.inputChannels, value: UInt64(inDim))
        ipParams.writeVarintField(fieldNumber: InnerProductLayerParamsField.outputChannels, value: UInt64(outDim))
        ipParams.writeVarintField(fieldNumber: InnerProductLayerParamsField.hasBias, value: 1)

        var wParams = ProtobufWriter()
        wParams.writePackedFloatsField(fieldNumber: WeightParamsField.floatValue, values: flatWeights)
        ipParams.writeBytesField(fieldNumber: InnerProductLayerParamsField.weights, bytes: wParams.data)

        var bParams = ProtobufWriter()
        bParams.writePackedFloatsField(fieldNumber: WeightParamsField.floatValue, values: biases.map { Float($0) })
        ipParams.writeBytesField(fieldNumber: InnerProductLayerParamsField.bias, bytes: bParams.data)

        ipLayer.writeBytesField(fieldNumber: NeuralNetworkLayerField.innerProduct, bytes: ipParams.data)
        nnMsg.writeBytesField(fieldNumber: NeuralNetworkField.layers, bytes: ipLayer.data)

        // Activation
        if hasActivation {
            let finalLayerOut = isLastLayer ? outputName : "act_out_\(layerIdx)"
            var actLayer = ProtobufWriter()
            actLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.name, value: "act_\(layerIdx)")
            actLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.input, value: ipOutName)
            actLayer.writeStringField(fieldNumber: NeuralNetworkLayerField.output, value: finalLayerOut)

            var actParams = ProtobufWriter()
            switch actType.lowercased() {
            case "relu":
                actParams.writeBytesField(fieldNumber: ActivationParamsField.reLU, bytes: Data())
            case "sigmoid":
                actParams.writeBytesField(fieldNumber: ActivationParamsField.sigmoid, bytes: Data())
            case "tanh":
                actParams.writeBytesField(fieldNumber: ActivationParamsField.tanh, bytes: Data())
            default:
                break
            }
            actLayer.writeBytesField(fieldNumber: NeuralNetworkLayerField.activation, bytes: actParams.data)
            nnMsg.writeBytesField(fieldNumber: NeuralNetworkField.layers, bytes: actLayer.data)
            currentInputs = [finalLayerOut]
        } else {
            currentInputs = [ipOutName]
        }
    }

    nnMsg.writeVarintField(fieldNumber: NeuralNetworkField.arrayInputShapeMapping, value: 1)
    model.writeBytesField(fieldNumber: ModelField.neuralNetwork, bytes: nnMsg.data)
    return model.data
}

