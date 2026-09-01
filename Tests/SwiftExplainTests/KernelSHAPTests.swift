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
}
