import Testing
import Foundation
@testable import SwiftExplain
import SwiftML

/// Independent brute-force 2^K Shapley coalition oracle supporting both `.uniformLeaf` and `.empiricalCover`
/// leaf weights and arbitrary leaf value projections.
private enum ExactCoalitionOracle3140 {
    static func leafWeight(
        tree: [FlatTreeNode],
        nodeIdx: Int,
        weighting: TreeNodeWeighting,
        useCover: Bool
    ) -> Double {
        let node = tree[nodeIdx]
        if node.isLeaf {
            return useCover ? (node.cover ?? 1.0) : 1.0
        }
        let wl = leafWeight(tree: tree, nodeIdx: node.leftChild, weighting: weighting, useCover: useCover)
        let wr = leafWeight(tree: tree, nodeIdx: node.rightChild, weighting: weighting, useCover: useCover)
        return wl + wr
    }

    static func allReachableLeavesHaveCover(tree: [FlatTreeNode]) -> Bool {
        guard !tree.isEmpty else { return false }
        var stack: [Int] = [0]
        var visited = Set<Int>()
        var found = false
        while let idx = stack.popLast() {
            guard idx >= 0 && idx < tree.count && visited.insert(idx).inserted else { continue }
            let n = tree[idx]
            if n.isLeaf {
                found = true
                guard let c = n.cover, c.isFinite, c > 0.0 else { return false }
            } else {
                stack.append(n.leftChild)
                stack.append(n.rightChild)
            }
        }
        return found
    }

    static func activeFeatures(tree: [FlatTreeNode], numFeatures: Int) -> [Int] {
        var set = Set<Int>()
        for node in tree where !node.isLeaf && node.featureIndex >= 0 && node.featureIndex < numFeatures {
            set.insert(node.featureIndex)
        }
        return Array(set).sorted()
    }

    static func evalCoalition(
        tree: [FlatTreeNode],
        instance: [Double],
        known: Set<Int>,
        nodeIdx: Int,
        weighting: TreeNodeWeighting,
        useCover: Bool,
        leafValue: (FlatTreeNode) -> Double
    ) -> Double {
        let node = tree[nodeIdx]
        if node.isLeaf {
            return leafValue(node)
        }
        if known.contains(node.featureIndex) {
            let nextIdx = instance[node.featureIndex] <= node.threshold ? node.leftChild : node.rightChild
            return evalCoalition(
                tree: tree,
                instance: instance,
                known: known,
                nodeIdx: nextIdx,
                weighting: weighting,
                useCover: useCover,
                leafValue: leafValue
            )
        }
        let wl = leafWeight(tree: tree, nodeIdx: node.leftChild, weighting: weighting, useCover: useCover)
        let wr = leafWeight(tree: tree, nodeIdx: node.rightChild, weighting: weighting, useCover: useCover)
        let total = wl + wr
        let vl = evalCoalition(
            tree: tree,
            instance: instance,
            known: known,
            nodeIdx: node.leftChild,
            weighting: weighting,
            useCover: useCover,
            leafValue: leafValue
        )
        let vr = evalCoalition(
            tree: tree,
            instance: instance,
            known: known,
            nodeIdx: node.rightChild,
            weighting: weighting,
            useCover: useCover,
            leafValue: leafValue
        )
        return (wl * vl + wr * vr) / total
    }

    static func factorial(_ n: Int) -> Double {
        if n <= 1 { return 1.0 }
        var acc = 1.0
        for i in 2...n { acc *= Double(i) }
        return acc
    }

    static func shapleyTree(
        tree: [FlatTreeNode],
        instance: [Double],
        numFeatures: Int,
        weighting: TreeNodeWeighting,
        leafValue: (FlatTreeNode) -> Double
    ) -> TreeSHAPExplanation {
        let useCover = (weighting == .empiricalCover) && allReachableLeavesHaveCover(tree: tree)
        let active = activeFeatures(tree: tree, numFeatures: numFeatures)
        let k = active.count
        let baseValue = evalCoalition(
            tree: tree,
            instance: instance,
            known: [],
            nodeIdx: 0,
            weighting: weighting,
            useCover: useCover,
            leafValue: leafValue
        )
        let prediction = evalCoalition(
            tree: tree,
            instance: instance,
            known: Set(active),
            nodeIdx: 0,
            weighting: weighting,
            useCover: useCover,
            leafValue: leafValue
        )
        var phi = [Double](repeating: 0.0, count: numFeatures)
        if k == 0 {
            return TreeSHAPExplanation(values: phi, baseValue: baseValue, prediction: prediction)
        }
        let totalSubsets = 1 << k
        for (bit, feat) in active.enumerated() {
            var sum = 0.0
            for mask in 0..<totalSubsets where (mask & (1 << bit)) == 0 {
                var subset = Set<Int>()
                for (b, f) in active.enumerated() where (mask & (1 << b)) != 0 {
                    subset.insert(f)
                }
                let s = subset.count
                let weight = factorial(s) * factorial(k - s - 1) / factorial(k)
                let withFeat = evalCoalition(
                    tree: tree,
                    instance: instance,
                    known: subset.union([feat]),
                    nodeIdx: 0,
                    weighting: weighting,
                    useCover: useCover,
                    leafValue: leafValue
                )
                let withoutFeat = evalCoalition(
                    tree: tree,
                    instance: instance,
                    known: subset,
                    nodeIdx: 0,
                    weighting: weighting,
                    useCover: useCover,
                    leafValue: leafValue
                )
                sum += weight * (withFeat - withoutFeat)
            }
            phi[feat] = sum
        }
        return TreeSHAPExplanation(values: phi, baseValue: baseValue, prediction: prediction)
    }
}

@Suite("SwiftSci 3.14.0 TreeSHAP GBDT, Per-Class Probability & Cover Weighting Tests")
struct TreeSHAP3140Tests {

    private func make4DClassificationDataset(numClasses: Int) -> (features: [[Double]], targets: [Double]) {
        var features: [[Double]] = []
        var targets: [Double] = []
        // Create 4-feature samples with varying counts per region so empiricalCover != uniformLeaf
        for c in 0..<numClasses {
            let count = 3 + c * 2
            for j in 0..<count {
                let base = Double(c) * 3.0
                let offset = Double(j) * 0.15
                features.append([
                    base + offset,
                    Double((c + j) % numClasses) * 1.2 + 0.1,
                    Double(c % 2) * 2.0 + offset * 0.5,
                    Double(j % 3) * 0.7 - base * 0.2
                ])
                // Inject a couple of borderline/noisy labels to create non-trivial class distributions
                let label: Double = (j == 1 && numClasses >= 2) ? Double((c + 1) % numClasses) : Double(c)
                targets.append(label)
            }
        }
        return (features, targets)
    }

    private func make4DRegressionDataset() -> (features: [[Double]], targets: [Double]) {
        var features: [[Double]] = []
        var targets: [Double] = []
        for i in 0..<24 {
            let x0 = Double(i % 4) * 1.1 + (i < 12 ? 0.0 : 0.3)
            let x1 = Double((i / 2) % 3) * 1.5
            let x2 = Double((i * 3) % 5) * 0.8
            let x3 = Double((i + 1) % 4) * 0.9 - 1.0
            features.append([x0, x1, x2, x3])
            targets.append(2.5 * x0 - 1.4 * x1 + 0.8 * x2 + 0.4 * x3 + (i < 6 ? 1.0 : -0.5))
        }
        return (features, targets)
    }

    private func verifyCrossClassBalance(
        explanations: [TreeSHAPExplanation],
        numFeatures: Int,
        tolerance: Double = 1e-11
    ) {
        let baseSum = explanations.reduce(0.0) { (acc: Double, exp: TreeSHAPExplanation) -> Double in
            acc + exp.baseValue
        }
        let predSum = explanations.reduce(0.0) { (acc: Double, exp: TreeSHAPExplanation) -> Double in
            acc + exp.prediction
        }
        #expect(abs(baseSum - 1.0) < tolerance)
        #expect(abs(predSum - 1.0) < tolerance)
        for f in 0..<numFeatures {
            var phiSum = 0.0
            for exp in explanations {
                phiSum += exp.values[f]
            }
            #expect(abs(phiSum) < tolerance)
        }
    }

    // MARK: 1, 2, 3, 4: GBDT & Multiclass DT/RF Local Accuracy, Cross-Class Balance, Consistency & Class Specificity

    @Test("G-035: GBDT TreeSHAP satisfies local accuracy, prediction parity, batch parity, and 2^K oracle across losses")
    func testGBDTTreeSHAPExactnessAndParity() async throws {
        let (features, targets) = make4DRegressionDataset()
        let testInstances: [[Double]] = [
            [0.5, 1.0, 1.2, 0.0],
            [2.4, 3.0, 2.5, -0.4],
            [1.1, 0.2, 0.5, 1.5],
            [3.2, 2.2, 1.6, -0.8],
            [0.1, 1.4, 2.9, 0.2],
            [2.9, 0.1, 0.2, 1.1]
        ]

        let losses: [GBDTLoss] = [.squaredError, .absoluteError, .quantile(alpha: 0.7)]
        for loss in losses {
            let gbdt = try GradientBoostedTreesRegressor(
                nEstimators: 6,
                learningRate: 0.2,
                maxDepth: 3,
                minSamplesSplit: 2,
                loss: loss
            )
            try await gbdt.fit(features: features, targets: targets)
            let trees = await gbdt.getEnsembleTrees()
            let initPred = await gbdt.getInitialPrediction()
            let lr = await gbdt.learningRate
            let modelPreds = try await gbdt.predict(features: testInstances)

            for weighting in [TreeNodeWeighting.uniformLeaf, TreeNodeWeighting.empiricalCover] {
                let explainer = TreeSHAP(permutationSeed: 42, weighting: weighting)

                for (rowIdx, instance) in testInstances.enumerated() {
                    let exp = await explainer.explainDetailed(gbdt: gbdt, instance: instance)
                    let vec = await explainer.explain(gbdt: gbdt, instance: instance)

                    #expect(exp.values.count == 4)
                    #expect(vec == exp.values)

                    // Point 1: Local accuracy (< 1e-11)
                    let reconstructed = exp.baseValue + exp.values.reduce(0.0, +)
                    #expect(abs(reconstructed - exp.prediction) < 1e-11)

                    // Point 3: Model output consistency (< 1e-11)
                    #expect(abs(exp.prediction - modelPreds[rowIdx]) < 1e-11)

                    // Point 5: Independent 2^K coalition oracle parity (< 1e-10)
                    var oraclePhi = [Double](repeating: 0.0, count: 4)
                    var oracleBase = initPred
                    var oraclePred = initPred
                    for tree in trees {
                        let treeOracle = ExactCoalitionOracle3140.shapleyTree(
                            tree: tree,
                            instance: instance,
                            numFeatures: 4,
                            weighting: weighting,
                            leafValue: { (node: FlatTreeNode) -> Double in node.value }
                        )
                        for f in 0..<4 {
                            oraclePhi[f] += lr * treeOracle.values[f]
                        }
                        oracleBase += lr * treeOracle.baseValue
                        oraclePred += lr * treeOracle.prediction
                    }
                    #expect(abs(exp.baseValue - oracleBase) < 1e-10)
                    #expect(abs(exp.prediction - oraclePred) < 1e-10)
                    for f in 0..<4 {
                        #expect(abs(exp.values[f] - oraclePhi[f]) < 1e-10)
                    }
                }

                // Batch parity (triggers TaskGroup for count > 4)
                let batch = await explainer.explainBatch(gbdt: gbdt, instances: testInstances)
                #expect(batch.count == testInstances.count)
                for (rowIdx, instance) in testInstances.enumerated() {
                    let seq = await explainer.explain(gbdt: gbdt, instance: instance)
                    for f in 0..<4 {
                        #expect(abs(batch[rowIdx][f] - seq[f]) < 1e-12)
                    }
                }
            }
        }
    }

    @Test("G-036: DecisionTreeClassifier per-class TreeSHAP satisfies local accuracy, balance, oracle, and class specificity across C = 2, 3, 5")
    func testDecisionTreeClassifierPerClassTreeSHAP() async throws {
        for numClasses in [2, 3, 5] {
            let (features, targets) = make4DClassificationDataset(numClasses: numClasses)
            let dt = DecisionTreeClassifier(maxDepth: 4, minSamplesSplit: 2)
            try await dt.fit(features: features, targets: targets)
            let nodes = await dt.flatNodes
            let classes = await dt.classes
            #expect(classes.count == numClasses)

            let probes: [[Double]] = [
                features[0],
                features[features.count / 2],
                features[features.count - 1],
                [1.5, 1.0, 0.5, 0.2],
                [4.2, 2.0, 1.1, -0.3]
            ]
            let dtProbs = try await dt.predictProbability(features: probes)

            // Default TreeSHAP preserves scalar explain(decisionTree:)
            let defaultExplainer = TreeSHAP()
            let explicitUniform = TreeSHAP(permutationSeed: 42, weighting: .uniformLeaf)
            let scalarA = await defaultExplainer.explain(decisionTree: dt, instance: probes[0])
            let scalarB = await explicitUniform.explain(decisionTree: dt, instance: probes[0])
            #expect(scalarA == scalarB)

            for weighting in [TreeNodeWeighting.uniformLeaf, TreeNodeWeighting.empiricalCover] {
                let explainer = TreeSHAP(permutationSeed: 42, weighting: weighting)

                for (rowIdx, instance) in probes.enumerated() {
                    let allExps = await explainer.explainAllClassesDetailed(decisionTree: dt, instance: instance)
                    #expect(allExps.count == numClasses)

                    // Point 2: Cross-class balance
                    verifyCrossClassBalance(explanations: allExps, numFeatures: 4)

                    for c in 0..<numClasses {
                        let singleExp = try await explainer.explainClassDetailed(decisionTree: dt, instance: instance, classIndex: c)
                        #expect(singleExp == allExps[c])

                        // Point 1: Local accuracy (< 1e-11)
                        let reconstructed = singleExp.baseValue + singleExp.values.reduce(0.0, +)
                        #expect(abs(reconstructed - singleExp.prediction) < 1e-11)

                        // Point 3: Model output consistency (< 1e-11)
                        #expect(abs(singleExp.prediction - dtProbs[rowIdx][c]) < 1e-11)

                        // Point 5: Independent 2^K oracle on first probe
                        if rowIdx == 0 {
                            let targetClass = classes[c]
                            let oracle = ExactCoalitionOracle3140.shapleyTree(
                                tree: nodes,
                                instance: instance,
                                numFeatures: 4,
                                weighting: weighting,
                                leafValue: { (node: FlatTreeNode) -> Double in
                                    if let dist = node.classDistribution, c < dist.count {
                                        return dist[c]
                                    }
                                    return node.value == targetClass ? 1.0 : 0.0
                                }
                            )
                            #expect(abs(singleExp.baseValue - oracle.baseValue) < 1e-10)
                            #expect(abs(singleExp.prediction - oracle.prediction) < 1e-10)
                            for f in 0..<4 {
                                #expect(abs(singleExp.values[f] - oracle.values[f]) < 1e-10)
                            }
                        }
                    }

                    // Point 4: Class-specific explanations differ across classes
                    if numClasses >= 2 {
                        var maxPairDiff = 0.0
                        for c1 in 0..<numClasses {
                            for c2 in (c1 + 1)..<numClasses {
                                for f in 0..<4 {
                                    let d = abs(allExps[c1].values[f] - allExps[c2].values[f])
                                    if d > maxPairDiff { maxPairDiff = d }
                                }
                            }
                        }
                        #expect(maxPairDiff > 1e-6)
                    }
                }

                // Batch class parity
                let batchClass0 = try await explainer.explainBatchClass(decisionTree: dt, instances: probes, classIndex: 0)
                #expect(batchClass0.count == probes.count)
                for (rowIdx, instance) in probes.enumerated() {
                    let single = try await explainer.explainClassDetailed(decisionTree: dt, instance: instance, classIndex: 0)
                    for f in 0..<4 {
                        #expect(abs(batchClass0[rowIdx][f] - single.values[f]) < 1e-12)
                    }
                }

                // Bounds checking on classIndex
                await #expect(throws: SwiftMLError.self) {
                    _ = try await explainer.explainClassDetailed(decisionTree: dt, instance: probes[0], classIndex: -1)
                }
                await #expect(throws: SwiftMLError.self) {
                    _ = try await explainer.explainClassDetailed(decisionTree: dt, instance: probes[0], classIndex: numClasses)
                }
                await #expect(throws: SwiftMLError.self) {
                    _ = try await explainer.explainBatchClass(decisionTree: dt, instances: probes, classIndex: numClasses)
                }
            }
        }
    }

    @Test("G-036: RandomForestClassifier per-class TreeSHAP matches hardVote & softLeafDistribution and 2^K oracle")
    func testRandomForestPerClassTreeSHAP() async throws {
        let (features, targets) = make4DClassificationDataset(numClasses: 3)
        let rf = RandomForestClassifier(nEstimators: 5, maxDepth: 3, minSamplesSplit: 2, randomState: 17)
        try await rf.fit(features: features, targets: targets)
        let trees = await rf.flatTrees
        let classes = await rf.classes
        #expect(classes.count == 3)

        let probes: [[Double]] = [
            features[0],
            features[features.count / 2],
            features[features.count - 1],
            [2.1, 1.3, 0.6, -0.1],
            [5.0, 0.5, 1.4, 0.3]
        ]

        // Preserve scalar explain(randomForest:)
        let scalarDefault = await TreeSHAP().explain(randomForest: rf, instance: probes[0])
        let scalarUniform = await TreeSHAP(permutationSeed: 42, weighting: .uniformLeaf).explain(randomForest: rf, instance: probes[0])
        #expect(scalarDefault == scalarUniform)

        for mode in [ForestProbabilityMode.hardVote, ForestProbabilityMode.softLeafDistribution] {
            let rfProbs = try await rf.predictProbability(features: probes, mode: mode)

            for weighting in [TreeNodeWeighting.uniformLeaf, TreeNodeWeighting.empiricalCover] {
                let explainer = TreeSHAP(permutationSeed: 42, weighting: weighting)

                for (rowIdx, instance) in probes.enumerated() {
                    let allExps = await explainer.explainAllClassesDetailed(
                        randomForest: rf,
                        instance: instance,
                        probabilityMode: mode
                    )
                    #expect(allExps.count == 3)

                    // Point 2: Cross-class balance
                    verifyCrossClassBalance(explanations: allExps, numFeatures: 4)

                    for c in 0..<3 {
                        let singleExp = try await explainer.explainClassDetailed(
                            randomForest: rf,
                            instance: instance,
                            classIndex: c,
                            probabilityMode: mode
                        )
                        #expect(singleExp == allExps[c])

                        // Point 1: Local accuracy (< 1e-11)
                        let reconstructed = singleExp.baseValue + singleExp.values.reduce(0.0, +)
                        #expect(abs(reconstructed - singleExp.prediction) < 1e-11)

                        // Point 3: Model output consistency (< 1e-11)
                        #expect(abs(singleExp.prediction - rfProbs[rowIdx][c]) < 1e-11)

                        // Point 5: Independent 2^K oracle parity (< 1e-10)
                        if rowIdx == 0 {
                            var oraclePhi = [Double](repeating: 0.0, count: 4)
                            var oracleBase = 0.0
                            var oraclePred = 0.0
                            let targetClass = classes[c]
                            for tree in trees {
                                let treeOracle = ExactCoalitionOracle3140.shapleyTree(
                                    tree: tree,
                                    instance: instance,
                                    numFeatures: 4,
                                    weighting: weighting,
                                    leafValue: { (node: FlatTreeNode) -> Double in
                                        switch mode {
                                        case .hardVote:
                                            return node.value == targetClass ? 1.0 : 0.0
                                        case .softLeafDistribution:
                                            if let dist = node.classDistribution, c < dist.count {
                                                return dist[c]
                                            }
                                            return node.value == targetClass ? 1.0 : 0.0
                                        }
                                    }
                                )
                                for f in 0..<4 { oraclePhi[f] += treeOracle.values[f] }
                                oracleBase += treeOracle.baseValue
                                oraclePred += treeOracle.prediction
                            }
                            let tCount = Double(trees.count)
                            oracleBase /= tCount
                            oraclePred /= tCount
                            #expect(abs(singleExp.baseValue - oracleBase) < 1e-10)
                            #expect(abs(singleExp.prediction - oraclePred) < 1e-10)
                            for f in 0..<4 {
                                #expect(abs(singleExp.values[f] - (oraclePhi[f] / tCount)) < 1e-10)
                            }
                        }
                    }
                }

                // Batch parity
                let batchC1 = try await explainer.explainBatchClass(
                    randomForest: rf,
                    instances: probes,
                    classIndex: 1,
                    probabilityMode: mode
                )
                #expect(batchC1.count == probes.count)
                for (rowIdx, instance) in probes.enumerated() {
                    let single = try await explainer.explainClassDetailed(
                        randomForest: rf,
                        instance: instance,
                        classIndex: 1,
                        probabilityMode: mode
                    )
                    for f in 0..<4 {
                        #expect(abs(batchC1[rowIdx][f] - single.values[f]) < 1e-12)
                    }
                }

                // Invalid classIndex validation
                await #expect(throws: SwiftMLError.self) {
                    _ = try await explainer.explainClassDetailed(randomForest: rf, instance: probes[0], classIndex: -1, probabilityMode: mode)
                }
                await #expect(throws: SwiftMLError.self) {
                    _ = try await explainer.explainBatchClass(randomForest: rf, instances: probes, classIndex: 3, probabilityMode: mode)
                }
            }
        }
    }

    // MARK: 6: Post-prune(alpha:) Invariance

    @Test("Post-prune(alpha:) invariance holds for DecisionTreeClassifier under TreeSHAP empiricalCover")
    func testPostPruneInvariance() async throws {
        let (features, targets) = make4DClassificationDataset(numClasses: 3)
        let dt = DecisionTreeClassifier(maxDepth: 5, minSamplesSplit: 2)
        try await dt.fit(features: features, targets: targets)

        let unprunedNodes = await dt.flatNodes
        let beforeLeafCount = unprunedNodes.filter { (n: FlatTreeNode) -> Bool in n.isLeaf }.count
        let twigGains = unprunedNodes.compactMap { (node: FlatTreeNode) -> Double? in
            guard !node.isLeaf else { return nil }
            let leftIsLeaf = node.leftChild >= 0 && node.leftChild < unprunedNodes.count && unprunedNodes[node.leftChild].isLeaf
            let rightIsLeaf = node.rightChild >= 0 && node.rightChild < unprunedNodes.count && unprunedNodes[node.rightChild].isLeaf
            return (leftIsLeaf && rightIsLeaf) ? node.impurityGain : nil
        }.sorted()
        #expect(!twigGains.isEmpty)
        // Choose alpha that collapses at least one twig into a leaf
        let alpha = twigGains[0] + 1e-9

        await dt.prune(alpha: alpha)
        let prunedNodes = await dt.flatNodes
        let classes = await dt.classes

        // Verify reachable leaves in pruned tree have valid classDistribution and cover
        var stack: [Int] = [0]
        var visited = Set<Int>()
        var afterLeafCount = 0
        while let idx = stack.popLast() {
            guard idx >= 0 && idx < prunedNodes.count && visited.insert(idx).inserted else { continue }
            let node = prunedNodes[idx]
            let cover = try #require(node.cover)
            #expect(cover > 0.0)
            let dist = try #require(node.classDistribution)
            #expect(dist.count == 3)
            #expect(abs(dist.reduce(0.0, +) - 1.0) < 1e-12)
            if node.isLeaf {
                afterLeafCount += 1
            } else {
                stack.append(node.leftChild)
                stack.append(node.rightChild)
            }
        }
        #expect(afterLeafCount < beforeLeafCount)

        let probe: [Double] = [1.8, 1.1, 0.4, 0.2]
        let probs = try await dt.predictProbability(features: [probe])[0]
        let explainer = TreeSHAP(permutationSeed: 42, weighting: .empiricalCover)
        let allExps = await explainer.explainAllClassesDetailed(decisionTree: dt, instance: probe)
        #expect(allExps.count == 3)
        verifyCrossClassBalance(explanations: allExps, numFeatures: 4)

        for c in 0..<3 {
            let exp = allExps[c]
            let reconstructed = exp.baseValue + exp.values.reduce(0.0, +)
            #expect(abs(reconstructed - exp.prediction) < 1e-11)
            #expect(abs(exp.prediction - probs[c]) < 1e-11)

            let targetClass = classes[c]
            let oracle = ExactCoalitionOracle3140.shapleyTree(
                tree: prunedNodes,
                instance: probe,
                numFeatures: 4,
                weighting: .empiricalCover,
                leafValue: { (node: FlatTreeNode) -> Double in
                    if let dist = node.classDistribution, c < dist.count {
                        return dist[c]
                    }
                    return node.value == targetClass ? 1.0 : 0.0
                }
            )
            #expect(abs(exp.baseValue - oracle.baseValue) < 1e-10)
            for f in 0..<4 {
                #expect(abs(exp.values[f] - oracle.values[f]) < 1e-10)
            }
        }
    }

    // MARK: 7: Backward-Compatible 3.13.2 JSON Persistence + TreeSHAP Fallback + GBDT Persistence

    @Test("Legacy 3.13.2 JSON models work with TreeSHAP empiricalCover fallback, and GBDT round-trip preserves TreeSHAP")
    func testLegacy3132TreeSHAPFallbackAndGBDTPersistence() async throws {
        // Legacy DecisionTreeClassifier (no "cover")
        let dtClfJSON = """
        {
          "maxDepth": 2,
          "minSamplesSplit": 2,
          "numFeatures": 2,
          "classes": [0.0, 1.0],
          "criterion": "gini",
          "nodes": [
            {"featureIndex": 0, "threshold": 1.0, "leftChild": 1, "rightChild": 2, "value": 0.0, "isLeaf": false, "impurityGain": 0.4},
            {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 0.0, "isLeaf": true, "impurityGain": 0.0, "classDistribution": [0.8, 0.2]},
            {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 1.0, "isLeaf": true, "impurityGain": 0.0, "classDistribution": [0.1, 0.9]}
          ]
        }
        """
        let dtClfURL = FileManager.default.temporaryDirectory.appendingPathComponent("shap_legacy_dt_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: dtClfURL) }
        try Data(dtClfJSON.utf8).write(to: dtClfURL)
        let loadedDTClf = try DecisionTreeClassifier.load(from: dtClfURL)

        let coverExplainer = TreeSHAP(permutationSeed: 42, weighting: .empiricalCover)
        let uniformExplainer = TreeSHAP(permutationSeed: 42, weighting: .uniformLeaf)
        let probe2D: [Double] = [0.5, 2.0]

        let dtCoverExps = await coverExplainer.explainAllClassesDetailed(decisionTree: loadedDTClf, instance: probe2D)
        let dtUniformExps = await uniformExplainer.explainAllClassesDetailed(decisionTree: loadedDTClf, instance: probe2D)
        #expect(dtCoverExps == dtUniformExps)
        verifyCrossClassBalance(explanations: dtCoverExps, numFeatures: 2)
        let dtProbs = try await loadedDTClf.predictProbability(features: [probe2D])[0]
        for c in 0..<2 {
            let reconstructed = dtCoverExps[c].baseValue + dtCoverExps[c].values.reduce(0.0, +)
            #expect(abs(reconstructed - dtCoverExps[c].prediction) < 1e-11)
            #expect(abs(dtCoverExps[c].prediction - dtProbs[c]) < 1e-11)
        }

        // Legacy DecisionTreeRegressor (no "cover")
        let dtRegJSON = """
        {
          "maxDepth": 1,
          "minSamplesSplit": 2,
          "numFeatures": 2,
          "criterion": "mse",
          "nodes": [
            {"featureIndex": 1, "threshold": 0.0, "leftChild": 1, "rightChild": 2, "value": 5.0, "isLeaf": false, "impurityGain": 9.0},
            {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 2.0, "isLeaf": true, "impurityGain": 0.0},
            {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 8.0, "isLeaf": true, "impurityGain": 0.0}
          ]
        }
        """
        let dtRegURL = FileManager.default.temporaryDirectory.appendingPathComponent("shap_legacy_dt_reg_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: dtRegURL) }
        try Data(dtRegJSON.utf8).write(to: dtRegURL)
        let loadedDTReg = try DecisionTreeRegressor.load(from: dtRegURL)
        let regCoverPhi = await coverExplainer.explain(decisionTree: loadedDTReg, instance: probe2D)
        let regUniformPhi = await uniformExplainer.explain(decisionTree: loadedDTReg, instance: probe2D)
        #expect(regCoverPhi == regUniformPhi)
        #expect(abs(regCoverPhi[0]) < 1e-12)
        #expect(abs(regCoverPhi[1] - 3.0) < 1e-11)

        // Legacy RandomForestClassifier (no "cover", no "classDistribution")
        let rfClfJSON = """
        {
          "nEstimators": 2,
          "maxDepth": 2,
          "minSamplesSplit": 2,
          "numFeatures": 2,
          "numClasses": 2,
          "classes": [0.0, 1.0],
          "criterion": "gini",
          "trees": [
            [
              {"featureIndex": 0, "threshold": 1.0, "leftChild": 1, "rightChild": 2, "value": 0.0, "isLeaf": false, "impurityGain": 0.5},
              {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 0.0, "isLeaf": true, "impurityGain": 0.0},
              {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 1.0, "isLeaf": true, "impurityGain": 0.0}
            ],
            [
              {"featureIndex": 1, "threshold": 1.0, "leftChild": 1, "rightChild": 2, "value": 0.0, "isLeaf": false, "impurityGain": 0.5},
              {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 0.0, "isLeaf": true, "impurityGain": 0.0},
              {"featureIndex": -1, "threshold": 0.0, "leftChild": -1, "rightChild": -1, "value": 1.0, "isLeaf": true, "impurityGain": 0.0}
            ]
          ]
        }
        """
        let rfClfURL = FileManager.default.temporaryDirectory.appendingPathComponent("shap_legacy_rf_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: rfClfURL) }
        try Data(rfClfJSON.utf8).write(to: rfClfURL)
        let loadedRFClf = try RandomForestClassifier.load(from: rfClfURL)

        for mode in [ForestProbabilityMode.hardVote, ForestProbabilityMode.softLeafDistribution] {
            let rfCoverExps = await coverExplainer.explainAllClassesDetailed(randomForest: loadedRFClf, instance: probe2D, probabilityMode: mode)
            let rfUniformExps = await uniformExplainer.explainAllClassesDetailed(randomForest: loadedRFClf, instance: probe2D, probabilityMode: mode)
            #expect(rfCoverExps == rfUniformExps)
            verifyCrossClassBalance(explanations: rfCoverExps, numFeatures: 2)
            let rfProbs = try await loadedRFClf.predictProbability(features: [probe2D], mode: mode)[0]
            for c in 0..<2 {
                let reconstructed = rfCoverExps[c].baseValue + rfCoverExps[c].values.reduce(0.0, +)
                #expect(abs(reconstructed - rfCoverExps[c].prediction) < 1e-11)
                #expect(abs(rfCoverExps[c].prediction - rfProbs[c]) < 1e-11)
            }
        }

        // GBDT save/load round-trip preserves TreeSHAP empiricalCover attributions
        let (regX, regY) = make4DRegressionDataset()
        let gbdt = try GradientBoostedTreesRegressor(nEstimators: 5, learningRate: 0.2, maxDepth: 3, loss: .squaredError)
        try await gbdt.fit(features: regX, targets: regY)

        let gbdtURL = FileManager.default.temporaryDirectory.appendingPathComponent("shap_gbdt_rt_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: gbdtURL) }
        try await gbdt.save(to: gbdtURL)
        let loadedGBDT = try GradientBoostedTreesRegressor.load(from: gbdtURL)

        let probe4D: [Double] = [1.5, 1.5, 1.0, 0.0]
        let origExp = await coverExplainer.explainDetailed(gbdt: gbdt, instance: probe4D)
        let loadedExp = await coverExplainer.explainDetailed(gbdt: loadedGBDT, instance: probe4D)
        #expect(abs(origExp.baseValue - loadedExp.baseValue) < 1e-12)
        #expect(abs(origExp.prediction - loadedExp.prediction) < 1e-12)
        for f in 0..<4 {
            #expect(abs(origExp.values[f] - loadedExp.values[f]) < 1e-12)
        }
    }
}
