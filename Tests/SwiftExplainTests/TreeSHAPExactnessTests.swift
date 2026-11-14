import Testing
import Foundation
@testable import SwiftExplain
import SwiftML

/// Independent brute-force Shapley reference for the path-dependent value function
/// (leaf-count weighted marginalisation), used to validate `TreeSHAP`.
private enum ShapReference {
    static func leafCount(_ tree: [FlatTreeNode], _ i: Int) -> Double {
        let n = tree[i]
        return n.isLeaf ? 1 : leafCount(tree, n.leftChild) + leafCount(tree, n.rightChild)
    }

    static func value(_ tree: [FlatTreeNode], _ x: [Double], known: Set<Int>, node i: Int = 0) -> Double {
        let n = tree[i]
        if n.isLeaf { return n.value }
        if known.contains(n.featureIndex) {
            return value(tree, x, known: known, node: x[n.featureIndex] <= n.threshold ? n.leftChild : n.rightChild)
        }
        let wl = leafCount(tree, n.leftChild), wr = leafCount(tree, n.rightChild)
        return (wl * value(tree, x, known: known, node: n.leftChild)
              + wr * value(tree, x, known: known, node: n.rightChild)) / (wl + wr)
    }

    static func factorial(_ n: Int) -> Double { n <= 1 ? 1 : Double(n) * factorial(n - 1) }

    static func shapley(_ tree: [FlatTreeNode], _ x: [Double], features: [Int]) -> [Int: Double] {
        let k = features.count
        var out: [Int: Double] = [:]
        for (bit, f) in features.enumerated() {
            var phi = 0.0
            for mask in 0..<(1 << k) where mask & (1 << bit) == 0 {
                var s = Set<Int>()
                for (b, g) in features.enumerated() where mask & (1 << b) != 0 { s.insert(g) }
                let size = s.count
                let w = factorial(size) * factorial(k - size - 1) / factorial(k)
                phi += w * (value(tree, x, known: s.union([f])) - value(tree, x, known: s))
            }
            out[f] = phi
        }
        return out
    }

    /// Chain tree: node i splits feature `features[i]`; left child is a leaf, right child continues.
    static func chain(features: [Int], leaves: [Double]) -> [FlatTreeNode] {
        var nodes: [FlatTreeNode] = []
        let k = features.count
        for i in 0..<k {
            nodes.append(FlatTreeNode(featureIndex: features[i], threshold: 0.5, leftChild: 2 * i + 1, rightChild: 2 * i + 2, value: 0, isLeaf: false))
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: leaves[i], isLeaf: true))
        }
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: leaves[k], isLeaf: true))
        return nodes
    }
}

@Suite("TreeSHAP exact vs approximate contract")
struct TreeSHAPExactnessTests {
    private func leaves(_ n: Int) -> [Double] { (0...n).map { sin(Double($0) * 1.7) * 10 + Double($0) } }

    @Test(arguments: [1, 2, 5, 8, 12])
    func exactBranchMatchesBruteForce(k: Int) {
        let features = Array(0..<k)
        let tree = ShapReference.chain(features: features, leaves: leaves(k))
        let x = (0..<k).map { $0 % 3 == 0 ? 0.0 : 1.0 }
        let shap = TreeSHAP()
        #expect(shap.isExact(tree: tree, numFeatures: k))
        let got = shap.explainDetailed(tree: tree, instance: x, numFeatures: k)
        let ref = ShapReference.shapley(tree, x, features: features)
        for f in features { #expect(abs(got.values[f] - ref[f]!) < 1e-9) }
        #expect(abs(got.values.reduce(0, +) - (got.prediction - got.baseValue)) < 1e-9)
    }

    @Test func repeatedSplitFeatureMatchesBruteForce() {
        // Feature 0 is used twice (root and in the right subtree), feature 1 once.
        let leafNode = { (v: Double) in FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: v, isLeaf: true) }
        let tree = [
            FlatTreeNode(featureIndex: 0, threshold: 0.5, leftChild: 1, rightChild: 2, value: 0, isLeaf: false),
            leafNode(3),
            FlatTreeNode(featureIndex: 1, threshold: 0.5, leftChild: 3, rightChild: 4, value: 0, isLeaf: false),
            FlatTreeNode(featureIndex: 0, threshold: 0.8, leftChild: 5, rightChild: 6, value: 0, isLeaf: false),
            leafNode(-4),
            leafNode(0),   // zero leaf weight
            leafNode(9)
        ]
        for x in [[0.9, 0.1], [0.2, 0.9], [0.6, 0.3]] {
            let got = TreeSHAP().explainDetailed(tree: tree, instance: x, numFeatures: 2)
            let ref = ShapReference.shapley(tree, x, features: [0, 1])
            #expect(abs(got.values[0] - ref[0]!) < 1e-9)
            #expect(abs(got.values[1] - ref[1]!) < 1e-9)
        }
    }

    @Test func dummyFeatureGetsZero() {
        let tree = ShapReference.chain(features: [0, 2], leaves: [1, 5, 9])
        let phi = TreeSHAP().explain(tree: tree, instance: [0, 1, 0], numFeatures: 3)
        #expect(phi[1] == 0)
    }

    @Test func approximateBranchIsSeededAndReproducible() {
        let k = 14
        let tree = ShapReference.chain(features: Array(0..<k), leaves: leaves(k))
        let x = (0..<k).map { $0 % 2 == 0 ? 0.0 : 1.0 }
        #expect(!TreeSHAP().isExact(tree: tree, numFeatures: k))
        let a = TreeSHAP(permutationSeed: 7).explain(tree: tree, instance: x, numFeatures: k)
        let b = TreeSHAP(permutationSeed: 7).explain(tree: tree, instance: x, numFeatures: k)
        let c = TreeSHAP(permutationSeed: 8).explain(tree: tree, instance: x, numFeatures: k)
        #expect(a == b)
        #expect(a != c)
        #expect(TreeSHAP().explain(tree: tree, instance: x, numFeatures: k) == TreeSHAP().explain(tree: tree, instance: x, numFeatures: k))
    }

    @Test(arguments: [13, 14])
    func approximateBranchIsCloseToBruteForceAndEfficient(k: Int) {
        let features = Array(0..<k)
        let tree = ShapReference.chain(features: features, leaves: leaves(k))
        let x = (0..<k).map { $0 % 3 == 1 ? 0.0 : 1.0 }
        let ref = ShapReference.shapley(tree, x, features: features)
        let range = (leaves(k).max() ?? 1) - (leaves(k).min() ?? 0)
        for seed: UInt64 in [1, 2, 3] {
            let got = TreeSHAP(permutationSeed: seed).explainDetailed(tree: tree, instance: x, numFeatures: k)
            #expect(abs(got.values.reduce(0, +) - (got.prediction - got.baseValue)) < 1e-9)
            for f in features { #expect(abs(got.values[f] - ref[f]!) < 0.06 * range) }
        }
    }

    @Test func boundaryTwelveIsExactThirteenIsNot() {
        let t12 = ShapReference.chain(features: Array(0..<12), leaves: leaves(12))
        let t13 = ShapReference.chain(features: Array(0..<13), leaves: leaves(13))
        #expect(TreeSHAP().isExact(tree: t12, numFeatures: 12))
        #expect(!TreeSHAP().isExact(tree: t13, numFeatures: 13))
    }
}
