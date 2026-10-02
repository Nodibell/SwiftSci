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
}
