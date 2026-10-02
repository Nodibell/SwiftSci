import Testing
import Foundation
@testable import SwiftExplain
import SwiftML

@Suite("Batch TreeSHAP Tests (G-023)")
struct TreeSHAPBatchTests {

    @Test("TreeSHAP explainBatch matches sequential explain on decision trees")
    func testTreeSHAPBatchMatchesSequential() async {
        // Build a simple 3-node decision tree:
        // Root: feature 0 <= 2.0 -> left child (leaf 1, val 10.0), right child (leaf 2, val 50.0)
        let root = FlatTreeNode(featureIndex: 0, threshold: 2.0, leftChild: 1, rightChild: 2, value: 0.0, isLeaf: false)
        let leftLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let rightLeaf = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 50.0, isLeaf: true)
        let tree = [root, leftLeaf, rightLeaf]

        let explainer = TreeSHAP()
        let instances: [[Double]] = [
            [1.0, 5.0],
            [3.0, 2.0],
            [0.5, 9.0],
            [4.0, 1.0],
            [1.5, 3.0],
            [2.5, 7.0],
            [0.1, 0.0],
            [10.0, 10.0]
        ]

        let sequentialPhis = instances.map { explainer.explain(tree: tree, instance: $0, numFeatures: 2) }
        let batchPhis = await explainer.explainBatch(trees: [tree], instances: instances, numFeatures: 2)

        #expect(batchPhis.count == instances.count)
        for i in 0..<instances.count {
            #expect(abs(batchPhis[i][0] - sequentialPhis[i][0]) < 1e-9)
            #expect(abs(batchPhis[i][1] - sequentialPhis[i][1]) < 1e-9)
        }
    }

    @Test("TreeSHAP explainBatch handles empty instance input safely")
    func testTreeSHAPBatchEmptyInput() async {
        let explainer = TreeSHAP()
        let phis = await explainer.explainBatch(trees: [], instances: [], numFeatures: 3)
        #expect(phis.isEmpty)
    }

    @Test("TreeSHAP explainBatch concurrency and performance stress test with 100 instances and multi-tree ensemble")
    func testTreeSHAPBatchConcurrencyAndPerformance() async {
        // Multi-tree ensemble (3 trees with 5 features)
        // Tree 1: split on feature 0, leaves at 10.0 and 50.0
        let t1_root = FlatTreeNode(featureIndex: 0, threshold: 3.0, leftChild: 1, rightChild: 2, value: 0.0, isLeaf: false)
        let t1_l = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true)
        let t1_r = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 50.0, isLeaf: true)
        let tree1 = [t1_root, t1_l, t1_r]

        // Tree 2: split on feature 2, leaves at -5.0 and 25.0
        let t2_root = FlatTreeNode(featureIndex: 2, threshold: 1.5, leftChild: 1, rightChild: 2, value: 0.0, isLeaf: false)
        let t2_l = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: -5.0, isLeaf: true)
        let t2_r = FlatTreeNode(featureIndex: -1, threshold: 0.0, leftChild: -1, rightChild: -1, value: 25.0, isLeaf: true)
        let tree2 = [t2_root, t2_l, t2_r]

        let ensemble = [tree1, tree2]
        let numFeatures = 5

        // Generate 100 distinct synthetic instances
        let count = 100
        var instances: [[Double]] = []
        for i in 0..<count {
            let row = (0..<numFeatures).map { f in Double(i * 13 + f * 7 % 29) * 0.1 }
            instances.append(row)
        }

        let explainer = TreeSHAP()

        let start = Date()
        let batchResults = await explainer.explainBatch(trees: ensemble, instances: instances, numFeatures: numFeatures)
        let duration = Date().timeIntervalSince(start)

        #expect(batchResults.count == count)
        #expect(duration < 2.0) // Must compute 100 explanations in well under 2 seconds

        // Verify concurrency thread-safety and exact numerical match against sequential computation
        for i in 0..<count {
            let seqPhi = explainer.explain(trees: ensemble, instance: instances[i], numFeatures: numFeatures)
            #expect(batchResults[i].count == numFeatures)
            for f in 0..<numFeatures {
                #expect(abs(batchResults[i][f] - seqPhi[f]) < 1e-10)
            }
        }
    }
}
