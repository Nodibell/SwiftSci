import Testing
import Foundation
import SwiftML
@testable import SwiftExplain

@Suite("Kernel SHAP Tests")
struct KernelSHAPTests {
    
    @Test("KernelSHAP efficiency and linear coefficient recovery")
    func testKernelSHAP() async throws {
        // Simple linear model: f(x) = 2.5 * x[0] - 4.0 * x[1] + 10.0
        let model: @Sendable ([Double]) -> Double = { x in
            guard x.count == 2 else { return 0.0 }
            return 2.5 * x[0] - 4.0 * x[1] + 10.0
        }
        
        let instance = [4.0, 2.0]
        let background = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 3.0],
            [1.0, 0.0]
        ]
        
        let bgMean = [1.0, 1.0]
        
        let explainer = KernelSHAP()
        let shapValues = await explainer.explain(
            model: model,
            instance: instance,
            background: background,
            numCoalitions: 150
        )
        
        #expect(shapValues.count == 2)
        
        let fEmpty = model(bgMean)
        let fFull = model(instance)
        
        let sumShap = shapValues.reduce(0.0, +)
        let expectedDiff = fFull - fEmpty
        
        // 1. Check Efficiency (sum of SHAP values matches prediction difference)
        #expect(abs(sumShap - expectedDiff) < 1e-4)
        
        // 2. Check linear contributions
        let expectedShap0 = 2.5 * (instance[0] - bgMean[0])
        let expectedShap1 = -4.0 * (instance[1] - bgMean[1])
        
        #expect(abs(shapValues[0] - expectedShap0) < 1e-2)
        #expect(abs(shapValues[1] - expectedShap1) < 1e-2)
    }

    @Test("KernelSHAP single feature (M = 1) exact calculation without division by zero")
    func testKernelSHAPSingleFeatureM1() async throws {
        let model: @Sendable ([Double]) async -> Double = { x in
            guard !x.isEmpty else { return 0.0 }
            return 3.0 * x[0] + 5.0
        }

        let instance = [10.0]
        let background = [[2.0]]

        let explainer = KernelSHAP()
        let shapValues = await explainer.explain(
            model: model,
            instance: instance,
            background: background,
            numCoalitions: 50
        )

        #expect(shapValues.count == 1)
        // fFull = 35.0, fEmpty = 11.0, difference = 24.0
        #expect(abs(shapValues[0] - 24.0) < 1e-6)
    }

    @Test("KernelSHAP 2-feature (M = 2) analytical exact calculation satisfies efficiency identically")
    func testKernelSHAPM2AnalyticalExact() async throws {
        // Nonlinear interaction model: f(x) = x0^2 + x0 * x1 + 3.0 * x1
        let model: @Sendable ([Double]) async -> Double = { x in
            return (x[0] * x[0]) + (x[0] * x[1]) + (3.0 * x[1])
        }

        let instance = [2.0, 4.0]
        let background = [[0.0, 0.0], [2.0, 0.0]] // bgMean = [1.0, 0.0]

        let explainer = KernelSHAP()
        let shapValues = await explainer.explain(
            model: model,
            instance: instance,
            background: background,
            numCoalitions: 100
        )

        #expect(shapValues.count == 2)
        let fEmpty = await model([1.0, 0.0]) // 1.0 + 0 + 0 = 1.0
        let fFull = await model(instance)    // 4.0 + 8.0 + 12.0 = 24.0
        let totalGain = fFull - fEmpty       // 23.0

        #expect(abs((shapValues[0] + shapValues[1]) - totalGain) < 1e-9)
    }

    @Test("TreeSHAP delegating explainer returns SHAP values")
    func testTreeSHAPExplainer() async throws {
        let model: @Sendable ([Double]) async -> Double = { x in
            return x.reduce(0.0, +)
        }
        let features = [[1.0, 2.0], [3.0, 4.0]]
        let treeSHAP = TreeSHAP()
        let result = await treeSHAP.explain(model: model, features: features, numCoalitions: 20)
        
        #expect(result.count == 2)
        #expect(result[0].count == 2)
    }

    @Test("Exact TreeSHAP on FlatTreeNode array")
    func testExactTreeSHAP() async throws {
        // Build a simple decision tree: split on feature 0 at 2.5 (left: leaf 10.0, right: leaf 50.0)
        let root = FlatTreeNode(featureIndex: 0, threshold: 2.5, leftChild: 1, rightChild: 2, value: 30.0, isLeaf: false)
        let leftLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let rightLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 50.0, isLeaf: true)
        let tree = [root, leftLeaf, rightLeaf]

        let explainer = TreeSHAP()
        let leftShap = explainer.explain(tree: tree, instance: [1.0, 5.0], numFeatures: 2)
        let rightShap = explainer.explain(tree: tree, instance: [4.0, 5.0], numFeatures: 2)

        #expect(leftShap.count == 2)
        #expect(rightShap.count == 2)
        #expect(leftShap[0] != 0.0)
        #expect(rightShap[0] != 0.0)
        // Feature 1 was not split on, so its contribution is 0.0
        #expect(leftShap[1] == 0.0)
        #expect(rightShap[1] == 0.0)
    }

    @Test("TreeSHAP Efficiency Axiom additivity guarantee")
    func testTreeSHAPEfficiencyAxiom() async throws {
        // Build a 2-level decision tree
        // Node 0: split feat 0 at 2.0 -> left: 1, right: 2
        // Node 1: split feat 1 at 3.0 -> left: 3 (val: 5.0), right: 4 (val: 15.0)
        // Node 2: leaf 5 (val: 25.0)
        let n0 = FlatTreeNode(featureIndex: 0, threshold: 2.0, leftChild: 1, rightChild: 2, value: 15.0, isLeaf: false)
        let n1 = FlatTreeNode(featureIndex: 1, threshold: 3.0, leftChild: 3, rightChild: 4, value: 10.0, isLeaf: false)
        let n2 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 25.0, isLeaf: true)
        let n3 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 5.0, isLeaf: true)
        let n4 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 15.0, isLeaf: true)
        let tree = [n0, n1, n2, n3, n4]

        let explainer = TreeSHAP()
        let sample = [1.0, 2.0] // Goes left at n0 (1.0 <= 2.0) -> goes left at n1 (2.0 <= 3.0) -> leaf n3 (val: 5.0)
        let shapVals = explainer.explain(tree: tree, instance: sample, numFeatures: 2)

        #expect(shapVals.count == 2)
        // Sum of Shapley values must be finite and feature 0 and 1 must receive exact attributions
        let sumShap = shapVals.reduce(0.0, +)
        #expect(!sumShap.isNaN)
        #expect(!sumShap.isInfinite)
    }

    @Test("PermutationImportance identifies most influential feature")
    func testPermutationImportance() async throws {
        let predictClosure: @Sendable ([[Double]]) async throws -> [Double] = { matrix in
            return matrix.map { 10.0 * $0[0] + 0.001 * $0[1] }
        }
        let features = [
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0]
        ]
        let targets = [10.0, 20.0, 30.0, 40.0]

        let perm = PermutationImportance()
        let importance = try await perm.computeImportance(features: features, targets: targets, predict: predictClosure)

        let imp0 = importance["feature_0"] ?? 0.0
        let imp1 = importance["feature_1"] ?? 0.0

        #expect(imp0 > imp1)
    }

    @Test("PartialDependencePlot sweeps feature grid")
    func testPartialDependencePlot() async throws {
        let predictClosure: @Sendable ([[Double]]) async throws -> [Double] = { matrix in
            return matrix.map { $0[0] * 2.0 }
        }
        let features = [
            [0.0, 1.0],
            [10.0, 1.0]
        ]

        let pdp = PartialDependencePlot()
        let (grid, values) = try await pdp.calculatePDP(features: features, featureIndex: 0, gridPoints: 5, predict: predictClosure)

        #expect(grid.count == 5)
        #expect(values.count == 5)
        #expect(values.last! > values.first!)
    }

    @Test("LIMEExplainer recovers linear feature coefficients locally")
    func testLIMEExplainer() async throws {
        // Linear local model: f(x) = 5.0 * x[0] - 2.0 * x[1] + 3.0
        let model: @Sendable ([Double]) async -> Double = { x in
            guard x.count >= 2 else { return 0.0 }
            return 5.0 * x[0] - 2.0 * x[1] + 3.0
        }
        
        let instance = [2.0, 1.0]
        let explainer = LIMEExplainer(kernelWidth: 1.0, regularization: 0.01)
        let explanation = await explainer.explain(model: model, instance: instance, numSamples: 300)
        
        #expect(explanation.featureWeights.count == 2)
        #expect(abs(explanation.featureWeights[0] - 5.0) < 0.25)
        #expect(abs(explanation.featureWeights[1] - (-2.0)) < 0.25)
        #expect(explanation.localR2 > 0.95)
        #expect(abs(explanation.prediction - 11.0) < 1e-4)
    }

    // MARK: - Efficiency Axioms, Validation & Correctness

    @Test("TreeSHAP single split exact analytical values and efficiency axiom (EX-01, EX-02)")
    func testTreeSHAPSingleSplitExactEfficiency() async throws {
        // Split on feat 0 at 2.5: Left leaf = 10.0, Right leaf = 50.0. E[f(X)] = 30.0
        let root = FlatTreeNode(featureIndex: 0, threshold: 2.5, leftChild: 1, rightChild: 2, value: 30.0, isLeaf: false)
        let leftLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let rightLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 50.0, isLeaf: true)
        let tree = [root, leftLeaf, rightLeaf]

        let explainer = TreeSHAP()

        // 1. Left instance: x0 = 1.0 <= 2.5 -> prediction 10.0
        let leftExp = explainer.explainDetailed(tree: tree, instance: [1.0, 5.0], numFeatures: 2)
        #expect(abs(leftExp.baseValue - 30.0) < 1e-12)
        #expect(abs(leftExp.prediction - 10.0) < 1e-12)
        #expect(abs(leftExp.values[0] - (-20.0)) < 1e-12)
        #expect(abs(leftExp.values[1] - 0.0) < 1e-12)
        // Check Efficiency: E[f(X)] + sum(phi) == f(x)
        let leftSum = leftExp.baseValue + leftExp.values.reduce(0.0, +)
        #expect(abs(leftSum - leftExp.prediction) < 1e-12)

        // 2. Right instance: x0 = 4.0 > 2.5 -> prediction 50.0
        let rightExp = explainer.explainDetailed(tree: tree, instance: [4.0, 5.0], numFeatures: 2)
        #expect(abs(rightExp.baseValue - 30.0) < 1e-12)
        #expect(abs(rightExp.prediction - 50.0) < 1e-12)
        #expect(abs(rightExp.values[0] - 20.0) < 1e-12)
        #expect(abs(rightExp.values[1] - 0.0) < 1e-12)
        let rightSum = rightExp.baseValue + rightExp.values.reduce(0.0, +)
        #expect(abs(rightSum - rightExp.prediction) < 1e-12)

        // Input validation
        let invalidExp = explainer.explainDetailed(tree: tree, instance: [1.0], numFeatures: 2)
        #expect(invalidExp.values == [0.0, 0.0])
        let nanExp = explainer.explainDetailed(tree: tree, instance: [Double.nan, 2.0], numFeatures: 2)
        #expect(nanExp.values == [0.0, 0.0])
    }

    @Test("TreeSHAP 2-level tree exact multi-feature additivity (EX-01, EX-03)")
    func testTreeSHAPTwoLevelTreeExactAdditivity() async throws {
        // Node 0: split feat 0 at 2.0 -> left: 1, right: 2 (val 25.0)
        // Node 1: split feat 1 at 3.0 -> left: 3 (val 5.0), right: 4 (val 15.0)
        let n0 = FlatTreeNode(featureIndex: 0, threshold: 2.0, leftChild: 1, rightChild: 2, value: 15.0, isLeaf: false)
        let n1 = FlatTreeNode(featureIndex: 1, threshold: 3.0, leftChild: 3, rightChild: 4, value: 10.0, isLeaf: false)
        let n2 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 25.0, isLeaf: true)
        let n3 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 5.0, isLeaf: true)
        let n4 = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 15.0, isLeaf: true)
        let tree = [n0, n1, n2, n3, n4]

        let explainer = TreeSHAP()
        let sample = [1.0, 2.0] // goes to leaf n3 (val: 5.0)
        let explanation = explainer.explainDetailed(tree: tree, instance: sample, numFeatures: 2)

        #expect(abs(explanation.baseValue - 15.0) < 1e-12)
        #expect(abs(explanation.prediction - 5.0) < 1e-12)
        let reconstructed = explanation.baseValue + explanation.values.reduce(0.0, +)
        #expect(abs(reconstructed - explanation.prediction) < 1e-12)
    }

    @Test("TreeSHAP ensemble exact additivity holds across forest (EX-03)")
    func testTreeSHAPEnsembleExactAdditivity() async throws {
        // Tree 1: split on feat 0
        let t1_root = FlatTreeNode(featureIndex: 0, threshold: 2.5, leftChild: 1, rightChild: 2, value: 30.0, isLeaf: false)
        let t1_l = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let t1_r = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 50.0, isLeaf: true)
        let tree1 = [t1_root, t1_l, t1_r]

        // Tree 2: split on feat 1
        let t2_root = FlatTreeNode(featureIndex: 1, threshold: 1.0, leftChild: 1, rightChild: 2, value: 0.0, isLeaf: false)
        let t2_l = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: -10.0, isLeaf: true)
        let t2_r = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let tree2 = [t2_root, t2_l, t2_r]

        let ensemble = [tree1, tree2]
        let explainer = TreeSHAP()
        let instance = [1.0, 2.0] // tree1 -> 10.0, tree2 -> 10.0, mean pred = 10.0

        let exp = explainer.explainDetailed(trees: ensemble, instance: instance, numFeatures: 2)
        #expect(abs(exp.prediction - 10.0) < 1e-12)
        let reconstructed = exp.baseValue + exp.values.reduce(0.0, +)
        #expect(abs(reconstructed - exp.prediction) < 1e-12)
    }

    @Test("KernelSHAP M=3 exact constrained efficiency and coefficient recovery (EX-04, EX-05)")
    func testKernelSHAP3FeaturesExactConstrainedEfficiency() async throws {
        // Linear model f(x) = 2.0 * x[0] + 3.0 * x[1] - 4.0 * x[2] + 1.0
        let model: @Sendable ([Double]) async -> Double = { x in
            guard x.count >= 3 else { return 0.0 }
            return 2.0 * x[0] + 3.0 * x[1] - 4.0 * x[2] + 1.0
        }

        let instance = [3.0, 2.0, 0.0]
        let background = [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0]
        ] // bgMean = [1.0, 1.0, 1.0]

        let explainer = KernelSHAP()
        let shapValues = await explainer.explain(
            model: model,
            instance: instance,
            background: background,
            numCoalitions: 100
        )

        #expect(shapValues.count == 3)

        let fEmpty = await model([1.0, 1.0, 1.0]) // 2 + 3 - 4 + 1 = 2.0
        let fFull = await model(instance)         // 6 + 6 - 0 + 1 = 13.0
        let totalGain = fFull - fEmpty             // 11.0

        // Exact constrained efficiency: sum(phi) == totalGain identically
        let sumShap = shapValues.reduce(0.0, +)
        #expect(abs(sumShap - totalGain) < 1e-9)

        // Linear coefficients recovered accurately
        let expected0 = 2.0 * (3.0 - 1.0) // 4.0
        let expected1 = 3.0 * (2.0 - 1.0) // 3.0
        let expected2 = -4.0 * (0.0 - 1.0) // 4.0

        #expect(abs(shapValues[0] - expected0) < 0.1)
        #expect(abs(shapValues[1] - expected1) < 0.1)
        #expect(abs(shapValues[2] - expected2) < 0.1)
    }

    @Test("KernelSHAP background and parameter validation (EX-06, EX-07)")
    func testKernelSHAPValidation() async throws {
        let model: @Sendable ([Double]) async -> Double = { $0.reduce(0.0, +) }
        let explainer = KernelSHAP()

        // Empty background
        let emptyBgShap = await explainer.explain(model: model, instance: [1.0, 2.0], background: [])
        #expect(emptyBgShap == [0.0, 0.0])

        // Irregular background row length
        let irregularBg = [[1.0], [2.0, 3.0]]
        let irregularShap = await explainer.explain(model: model, instance: [1.0, 2.0], background: irregularBg)
        #expect(irregularShap == [0.0, 0.0])

        // numCoalitions <= 0
        let zeroCoalitions = await explainer.explain(model: model, instance: [1.0, 2.0], background: [[1.0, 2.0]], numCoalitions: 0)
        #expect(zeroCoalitions.isEmpty)
    }

    @Test("LIME dimension scaling with D=20 preserves neighborhood weights (EX-08, EX-09, EX-10)")
    func testLIMEDimensionScalingHighDim() async throws {
        let D = 20
        let model: @Sendable ([Double]) async -> Double = { x in
            return x.reduce(0.0, +)
        }

        let instance = [Double](repeating: 1.0, count: D)
        let explainer = LIMEExplainer(kernelWidth: 0.75, regularization: 0.1)

        let exp = await explainer.explain(model: model, instance: instance, numSamples: 200)
        #expect(exp.featureWeights.count == D)
        // With dimension scaling sqrt(D), weights do not collapse to zero: coefficients are learned
        let learnedSum = exp.featureWeights.reduce(0.0, +)
        #expect(learnedSum > 10.0) // Expected ~20.0
        #expect(exp.localR2 > 0.8)

        // Incompatible featureScales length
        let badScalesExp = await explainer.explain(model: model, instance: instance, numSamples: 50, featureScales: [1.0])
        #expect(badScalesExp.featureWeights == [Double](repeating: 0.0, count: D))
    }

    @Test("PermutationImportance unclipped negative importances and deterministic seed (EX-12, EX-13, EX-14)")
    func testPermutationImportanceUnclippedAndDeterministic() async throws {
        // Model only depends on feature 0. Feature 1 is irrelevant noise.
        let predictClosure: @Sendable ([[Double]]) async throws -> [Double] = { matrix in
            return matrix.map { $0[0] * 5.0 }
        }

        let features = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0]
        ]
        let targets = [5.0, 10.0, 15.0, 20.0, 25.0]

        let perm = PermutationImportance()

        // Deterministic seeded repeatability
        let res1 = try await perm.computeImportance(features: features, targets: targets, repeats: 3, randomSeed: 1234, predict: predictClosure)
        let res2 = try await perm.computeImportance(features: features, targets: targets, repeats: 3, randomSeed: 1234, predict: predictClosure)
        #expect(res1 == res2)
        #expect(res1["feature_0"]! > 10.0)

        // Non-rectangular matrix rejected safely
        let nonRect = [[1.0, 2.0], [1.0]]
        let nonRectRes = try await perm.computeImportance(features: nonRect, targets: [1.0, 2.0], predict: predictClosure)
        #expect(nonRectRes.isEmpty)
    }

    @Test("PartialDependencePlot input validation (EX-15)")
    func testPartialDependencePlotValidation() async throws {
        let predictClosure: @Sendable ([[Double]]) async throws -> [Double] = { $0.map { $0[0] } }
        let pdp = PartialDependencePlot()

        // gridPoints < 2
        let invalidGrid = try await pdp.calculatePDP(features: [[1.0, 2.0], [3.0, 4.0]], featureIndex: 0, gridPoints: 1, predict: predictClosure)
        #expect(invalidGrid.grid.isEmpty)
        #expect(invalidGrid.values.isEmpty)

        // Non-rectangular features
        let nonRectFeatures = [[1.0, 2.0], [3.0]]
        let nonRectPDP = try await pdp.calculatePDP(features: nonRectFeatures, featureIndex: 1, gridPoints: 5, predict: predictClosure)
        #expect(nonRectPDP.grid.isEmpty)
    }
}


