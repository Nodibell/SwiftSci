import Testing
import Foundation
@testable import SwiftML

@Suite("SwiftSci 3.13.3 G-042 to G-047 Regression Tests")
struct SwiftSci3133RegressionTests {

    // MARK: - G-042 & G-043: Numerical Gradient Checking (Central Finite Differences)

    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func numericalGradientCheckClassifierAndRegressor(_ activation: ActivationFunction) throws {
        let eps = 1e-6

        // 11 samples, 4 features -> tests both full batch (0..<8) and partial batch (8..<11 with batchSize = 4)
        let features: [[Double]] = (0..<11).map { (r: Int) -> [Double] in
            (0..<4).map { (c: Int) -> Double in
                let angle = Double(r * 5 + c * 3 + 1)
                let offset = Double(c - 1) * 0.23
                let raw = sin(angle) * 0.8 + offset
                return raw == 0.0 ? 0.17 : raw
            }
        }

        let scenarios: [(name: String, numClasses: Int, targets: [Double], lossKind: MLPLossKind)] = [
            ("binary", 1, (0..<11).map { Double($0 % 2) }, .binaryCrossEntropy(positiveClass: 1.0)),
            ("3-class", 3, (0..<11).map { Double($0 % 3) }, .softmaxCrossEntropy(classes: [0.0, 1.0, 2.0])),
            ("5-class", 5, (0..<11).map { Double($0 % 5) }, .softmaxCrossEntropy(classes: [0.0, 1.0, 2.0, 3.0, 4.0])),
            ("regressor", 1, (0..<11).map { Double($0) * 0.35 - 1.2 }, .mse),
        ]

        let batchSlices: [ArraySlice<Int>] = [
            Array(0..<8)[0..<8],    // full batch
            Array(0..<11)[8..<11],   // final partial batch (N=11, batchSize=4 -> 3 rows)
        ]

        for scenario in scenarios {
            let layerSizes = [4, 5, 3, scenario.numClasses]
            var rng = SeededRandom(seed: 12345 + scenario.numClasses)
            var baseLayers: [LayerWeights] = []
            for l in 0..<(layerSizes.count - 1) {
                let inD = layerSizes[l]
                let outD = layerSizes[l + 1]
                let w = (0..<(inD * outD)).map { _ in (rng.nextDouble() * 2.0 - 1.0) * 0.45 }
                let b = (0..<outD).map { idx in 0.13 + Double(idx) * 0.03 }
                baseLayers.append(LayerWeights(W: w, b: b, inDim: inD, outDim: outD))
            }

            for slice in batchSlices {
                let workspace = MLPMiniBatchWorkspace(layers: baseLayers, maxBatchSize: slice.count)
                _ = mlpComputeMiniBatchLossAndGradients(
                    layers: baseLayers,
                    features: features,
                    targets: scenario.targets,
                    indices: slice,
                    activation: activation,
                    lossKind: scenario.lossKind,
                    workspace: workspace,
                    computeGradients: true
                )

                var maxError = 0.0
                for l in 0..<baseLayers.count {
                    let gradW = workspace.gradW[l].data
                    let gradB = workspace.gradB[l].data

                    for p in 0..<baseLayers[l].W.count {
                        var layersPlus = baseLayers
                        layersPlus[l].W[p] += eps
                        let lossPlus = mlpComputeMiniBatchLossAndGradients(
                            layers: layersPlus,
                            features: features,
                            targets: scenario.targets,
                            indices: slice,
                            activation: activation,
                            lossKind: scenario.lossKind,
                            workspace: workspace,
                            computeGradients: false
                        )

                        var layersMinus = baseLayers
                        layersMinus[l].W[p] -= eps
                        let lossMinus = mlpComputeMiniBatchLossAndGradients(
                            layers: layersMinus,
                            features: features,
                            targets: scenario.targets,
                            indices: slice,
                            activation: activation,
                            lossKind: scenario.lossKind,
                            workspace: workspace,
                            computeGradients: false
                        )

                        let numGrad = (lossPlus - lossMinus) / (2.0 * eps)
                        let err = abs(numGrad - gradW[p])
                        maxError = max(maxError, err)
                        #expect(err < 1e-6, "Weight grad mismatch for \(scenario.name) \(activation) layer \(l) idx \(p): analytical=\(gradW[p]) numerical=\(numGrad)")
                    }

                    for j in 0..<baseLayers[l].b.count {
                        var layersPlus = baseLayers
                        layersPlus[l].b[j] += eps
                        let lossPlus = mlpComputeMiniBatchLossAndGradients(
                            layers: layersPlus,
                            features: features,
                            targets: scenario.targets,
                            indices: slice,
                            activation: activation,
                            lossKind: scenario.lossKind,
                            workspace: workspace,
                            computeGradients: false
                        )

                        var layersMinus = baseLayers
                        layersMinus[l].b[j] -= eps
                        let lossMinus = mlpComputeMiniBatchLossAndGradients(
                            layers: layersMinus,
                            features: features,
                            targets: scenario.targets,
                            indices: slice,
                            activation: activation,
                            lossKind: scenario.lossKind,
                            workspace: workspace,
                            computeGradients: false
                        )

                        let numGrad = (lossPlus - lossMinus) / (2.0 * eps)
                        let err = abs(numGrad - gradB[j])
                        maxError = max(maxError, err)
                        #expect(err < 1e-6, "Bias grad mismatch for \(scenario.name) \(activation) layer \(l) idx \(j): analytical=\(gradB[j]) numerical=\(numGrad)")
                    }
                }
                #expect(maxError < 1e-6)
            }
        }
    }

    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func mlpPartialBatchMulticlassAndSigmoidTanhConvergence(_ activation: ActivationFunction) async throws {
        // N = 11 with batchSize = 4 exercises partial final batches end-to-end across all activations
        var features: [[Double]] = []
        var targets: [Double] = []
        for i in 0..<11 {
            let c = i % 3
            let offset = Double(i / 3) * 0.1
            features.append([Double(c * 4) + offset, Double(c * -4) - offset])
            targets.append(Double(c))
        }

        let clf = MLPClassifier(
            hiddenLayerSizes: [12],
            activation: activation,
            solver: .adam,
            maxIter: 250,
            learningRate: 0.03,
            batchSize: 4,
            seed: 42
        )
        try await clf.fit(features: features, targets: targets)
        let preds = try await clf.predict(features: features)
        let correct = zip(preds, targets).filter { $0.0 == Int($0.1) }.count
        #expect(correct >= 10, "Expected high multiclass convergence with \(activation), got \(correct)/11")
    }

    // MARK: - G-042: Extreme Logit Stability

    @Test func extremeLogitStability() throws {
        let layer3 = [LayerWeights(W: [0.0, 0.0, 0.0], b: [1000.0, -1000.0, 999.0], inDim: 1, outDim: 3)]
        let workspace3 = MLPMiniBatchWorkspace(layers: layer3, maxBatchSize: 1)
        let loss3 = mlpComputeMiniBatchLossAndGradients(
            layers: layer3,
            features: [[1.0]],
            targets: [2.0],
            indices: [0][0..<1],
            activation: .relu,
            lossKind: .softmaxCrossEntropy(classes: [0.0, 1.0, 2.0]),
            workspace: workspace3,
            computeGradients: true
        )
        #expect(loss3.isFinite)
        #expect(abs(loss3 - (1.0 + log1p(exp(-1.0)))) < 1e-12)
        #expect(workspace3.gradW[0].data.allSatisfy { $0.isFinite })
        #expect(workspace3.gradB[0].data.allSatisfy { $0.isFinite })

        let layer2 = [LayerWeights(W: [0.0, 0.0], b: [-1000.0, -1000.0], inDim: 1, outDim: 2)]
        let workspace2 = MLPMiniBatchWorkspace(layers: layer2, maxBatchSize: 1)
        let loss2 = mlpComputeMiniBatchLossAndGradients(
            layers: layer2,
            features: [[1.0]],
            targets: [0.0],
            indices: [0][0..<1],
            activation: .relu,
            lossKind: .softmaxCrossEntropy(classes: [0.0, 1.0]),
            workspace: workspace2,
            computeGradients: true
        )
        #expect(loss2.isFinite)
        #expect(abs(loss2 - log(2.0)) < 1e-12)
        #expect(abs(workspace2.gradB[0].data[0] - (-0.5)) < 1e-12)
        #expect(abs(workspace2.gradB[0].data[1] - 0.5) < 1e-12)

        let layerBin = [LayerWeights(W: [0.0], b: [1000.0], inDim: 1, outDim: 1)]
        let workspaceBin = MLPMiniBatchWorkspace(layers: layerBin, maxBatchSize: 2)
        let lossBin = mlpComputeMiniBatchLossAndGradients(
            layers: layerBin,
            features: [[1.0], [1.0]],
            targets: [1.0, 0.0],
            indices: [0, 1][0..<2],
            activation: .relu,
            lossKind: .binaryCrossEntropy(positiveClass: 1.0),
            workspace: workspaceBin,
            computeGradients: true
        )
        #expect(lossBin.isFinite)
        #expect(abs(lossBin - 500.0) < 1e-9)
        #expect(workspaceBin.gradB[0].data.allSatisfy { $0.isFinite })
    }

    // MARK: - G-042: 768-D Convergence & Throughput

    @Test func mlp768DConvergenceAndBatchedThroughput() async throws {
        let dim = 768
        let nTrain = 96
        var trainX: [[Double]] = []
        var trainY: [Double] = []
        trainX.reserveCapacity(nTrain)
        trainY.reserveCapacity(nTrain)

        for i in 0..<nTrain {
            let c = i % 3
            var row = [Double](repeating: 0.0, count: dim)
            for j in 0..<dim {
                row[j] = ((j % 3 == c) ? 1.0 : -0.4) + Double((i + j) % 7 - 3) * 0.02
            }
            trainX.append(row)
            trainY.append(Double(c))
        }

        let clf = MLPClassifier(
            hiddenLayerSizes: [32],
            activation: .relu,
            solver: .adam,
            maxIter: 20,
            learningRate: 0.01,
            batchSize: 32,
            seed: 42,
            requestedDevice: .cpu
        )

        let clock = ContinuousClock()
        let elapsed = try await clock.measure {
            try await clf.fit(features: trainX, targets: trainY)
        }
        #expect(elapsed < .seconds(10))

        let preds = try await clf.predict(features: trainX)
        let accuracy = Double(zip(preds, trainY).filter { $0.0 == Int($0.1) }.count) / Double(nTrain)
        #expect(accuracy >= 0.95)

        let probs = try await clf.predictProbability(features: trainX)
        #expect(probs.count == nTrain)
        for p in probs {
            #expect(p.count == 3)
            #expect(p.allSatisfy { $0.isFinite && $0 >= 0.0 && $0 <= 1.0 })
            #expect(abs(p.reduce(0.0, +) - 1.0) < 1e-9)
        }
    }

    // MARK: - G-044: MLP Dimension, Rectangularity & Single-Class Regression Tests

    @Test func mlpValidationAndSingleClassConsistency() async throws {
        let clf = MLPClassifier(hiddenLayerSizes: [4], maxIter: 5, seed: 42)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            try await clf.fit(features: [[1.0, 2.0], [3.0, 4.0]], targets: [0.0])
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            try await clf.fit(features: [[1.0, 2.0], [3.0, 4.0, 5.0]], targets: [0.0, 1.0])
        }
        await #expect(throws: SwiftMLError.invalidInput("Validation features and targets must both be provided")) {
            try await clf.fit(
                features: [[1.0, 2.0], [3.0, 4.0]],
                targets: [0.0, 1.0],
                validationFeatures: [[1.0, 2.0]],
                validationTargets: nil,
                earlyStopping: nil
            )
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            try await clf.fit(
                features: [[1.0, 2.0], [3.0, 4.0]],
                targets: [0.0, 1.0],
                validationFeatures: [[1.0]],
                validationTargets: [0.0],
                earlyStopping: nil
            )
        }

        // Single-class fit & predict consistency
        try await clf.fit(features: [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], targets: [5.0, 5.0, 5.0])
        let singlePreds = try await clf.predict(features: [[1.0, 2.0]])
        #expect(singlePreds == [5])
        let singleProbs = try await clf.predictProbability(features: [[1.0, 2.0]])
        #expect(singleProbs.count == 1 && singleProbs[0].count == 2)
        #expect(abs(singleProbs[0].reduce(0.0, +) - 1.0) < 1e-12)

        // Prediction dimension mismatch (small row path & batched path)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            _ = try await clf.predict(features: [[1.0, 2.0, 3.0]])
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            _ = try await clf.predictProbability(features: Array(repeating: [1.0], count: 35))
        }

        let reg = MLPRegressor(hiddenLayerSizes: [4], maxIter: 5, seed: 42)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            try await reg.fit(features: [[1.0, 2.0], [3.0, 4.0, 5.0]], targets: [1.0, 2.0])
        }
        try await reg.fit(features: [[1.0, 2.0], [3.0, 4.0]], targets: [1.0, 2.0])
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            _ = try await reg.predict(features: [[1.0]])
        }
        let layers = try #require(await reg.trainedLayers)
        #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 4)) {
            _ = try mlpPredictRowValidated([1.0, 2.0, 3.0, 4.0], layers: layers, activate: { $0 })
        }
    }

    // MARK: - G-045: HistGradientBoosting Multiclass & Validation

    @Test func histGradientBoostingMulticlassAndValidation() async throws {
        let clf = try HistGradientBoostingClassifier(
            nEstimators: 30,
            learningRate: 0.15,
            maxDepth: 4,
            maxBins: 32,
            minSamplesLeaf: 2
        )

        // Ragged input validation in fit
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            try await clf.fit(features: [[1.0, 2.0], [3.0, 4.0, 5.0]], targets: [0.0, 1.0])
        }

        // 3-class separable data with non-zero-based labels [10, 20, 30]
        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<24 {
            let d = Double(i % 6) * 0.1
            X.append([-5.0 + d, -5.0 - d])
            y.append(10.0)
            X.append([0.0 + d, 5.0 - d])
            y.append(20.0)
            X.append([5.0 + d, -5.0 - d])
            y.append(30.0)
        }
        try await clf.fit(features: X, targets: y)

        let testX: [[Double]] = [[-4.8, -5.2], [0.2, 4.8], [5.2, -5.2]]
        let preds = try await clf.predict(features: testX)
        #expect(preds == [10, 20, 30])

        let probs = try await clf.predictProbability(features: testX)
        #expect(probs.count == 3)
        for (idx, row) in probs.enumerated() {
            #expect(row.count == 3)
            #expect(row.allSatisfy { $0.isFinite && $0 >= 0.0 && $0 <= 1.0 })
            #expect(abs(row.reduce(0.0, +) - 1.0) < 1e-6)
            #expect(row.argmax() == idx)
        }

        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            _ = try await clf.predict(features: [[1.0, 2.0, 3.0]])
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            _ = try await clf.predictProbability(features: [[1.0]])
        }

        let reg = try HistGradientBoostingRegressor(nEstimators: 10, minSamplesLeaf: 2)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            try await reg.fit(features: [[1.0, 2.0], [3.0]], targets: [1.0, 2.0])
        }
        try await reg.fit(features: [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], targets: [1.0, 2.0, 3.0, 4.0])
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            _ = try await reg.predict(features: [[1.0, 2.0, 3.0]])
        }
    }

    // MARK: - G-046: LinearSVC Zero/Negative Epochs & Rectangularity Validation

    @Test func linearSVCZeroEpochsAndValidation() async throws {
        let svcCPU = LinearSVC(C: 1.0, device: .cpu)
        // Zero and negative epochs must not crash
        try await svcCPU.fit(features: [[-1.0, -1.0], [1.0, 1.0]], targets: [0.0, 1.0], epochs: 0)
        try await svcCPU.fit(features: [[-1.0, -1.0], [1.0, 1.0]], targets: [0.0, 1.0], epochs: -5)

        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            try await svcCPU.fit(features: [[1.0, 2.0], [3.0, 4.0, 5.0]], targets: [0.0, 1.0])
        }
        try await svcCPU.fit(features: [[-2.0, -2.0], [2.0, 2.0]], targets: [0.0, 1.0], epochs: 50)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            _ = try await svcCPU.decisionFunction(features: [[1.0, 2.0, 3.0]])
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 1)) {
            _ = try await svcCPU.predict(features: [[1.0]])
        }

        let svcGPU = LinearSVC(C: 1.0, device: .gpu)
        try await svcGPU.fit(features: [[-2.0, -2.0], [2.0, 2.0]], targets: [0.0, 1.0], epochs: 10)
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 2, got: 3)) {
            _ = try await svcGPU.decisionFunction(features: [[1.0, 2.0, 3.0]])
        }
    }

    // MARK: - G-047: PlattScaling & CalibratedClassifier

    @Test func plattScalingAndCalibratedClassifierFixes() async throws {
        let platt = PlattScaling()
        #expect(platt.a == 1.0)
        #expect(platt.predictProbability(score: 1000.0).isFinite)
        #expect(platt.predictProbability(score: -1000.0).isFinite)

        platt.fit(scores: [-1000.0, -1.0, 1.0, 1000.0], targets: [0, 0, 1, 1], maxIterations: 50, lr: 0.05)
        #expect(platt.a > 0.0)
        #expect(platt.predictProbability(score: -1000.0).isFinite)
        #expect(platt.predictProbability(score: 1000.0).isFinite)
        #expect(platt.predictProbability(score: 1000.0) > platt.predictProbability(score: -1000.0))

        // Unfitted CalibratedClassifier throws modelNotFitted
        let unfitted = CalibratedClassifier(baseEstimator: DecisionTreeClassifier(maxDepth: 2))
        await #expect(throws: SwiftMLError.modelNotFitted) {
            _ = try await unfitted.predict(features: [[1.0, 2.0]])
        }

        // Multiclass rejection in CalibratedClassifier
        let multi = CalibratedClassifier(baseEstimator: DecisionTreeClassifier(maxDepth: 2))
        await #expect(throws: SwiftMLError.self) {
            try await multi.fit(features: [[0.0], [1.0], [2.0]], targets: [0.0, 1.0, 2.0])
        }

        // Arbitrary binary labels [-1, 1] preserved in predict()
        let cal = CalibratedClassifier(baseEstimator: DecisionTreeClassifier(maxDepth: 3))
        let X: [[Double]] = [[-2.0, -2.0], [-1.5, -1.5], [-1.0, -1.0], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]]
        let y: [Double] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        try await cal.fit(features: X, targets: y)
        let preds = try await cal.predict(features: [[-2.0, -2.0], [2.0, 2.0]])
        #expect(preds == [-1, 1])
        let probs = try await cal.predictProbability(features: [[-2.0, -2.0], [2.0, 2.0]])
        #expect(probs[0][0] > probs[0][1])
        #expect(probs[1][1] > probs[1][0])
    }
}
