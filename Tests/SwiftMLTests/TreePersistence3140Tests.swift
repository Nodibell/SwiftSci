import Testing
import Foundation
@testable import SwiftML

@Suite("SwiftSci 3.14.0 Tree Metadata, Pruning, Probability Mode & Persistence Tests")
struct TreePersistence3140Tests {

    private func reachableIndices(tree: [FlatTreeNode]) -> [Int] {
        guard !tree.isEmpty else { return [] }
        var result: [Int] = []
        var stack: [Int] = [0]
        var visited = Set<Int>()
        while let idx = stack.popLast() {
            guard idx >= 0 && idx < tree.count && visited.insert(idx).inserted else { continue }
            result.append(idx)
            let node = tree[idx]
            if !node.isLeaf {
                stack.append(node.leftChild)
                stack.append(node.rightChild)
            }
        }
        return result
    }

    @Test("DecisionTreeRegressor and DecisionTreeClassifier populate cover and classDistribution on all nodes")
    func testDecisionTreeCoverAndClassDistribution() async throws {
        let features: [[Double]] = [
            [0.1, 1.0], [0.2, 1.2], [0.3, 0.8], [0.4, 1.1],
            [2.0, 0.1], [2.2, 0.3], [2.4, 0.2], [2.5, 0.4],
            [5.0, 5.0], [5.2, 4.8], [5.4, 5.1], [5.6, 4.9]
        ]
        let regTargets: [Double] = [1.0, 1.2, 0.9, 1.1, 4.0, 4.2, 3.9, 4.1, 9.0, 9.2, 8.9, 9.1]
        let clfTargets: [Double] = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0]

        let dtReg = DecisionTreeRegressor(maxDepth: 3, minSamplesSplit: 2)
        try await dtReg.fit(features: features, targets: regTargets)
        let regNodes = await dtReg.flatNodes
        #expect(!regNodes.isEmpty)
        #expect(regNodes[0].cover == Double(features.count))
        for node in regNodes {
            let cover = try #require(node.cover)
            #expect(cover > 0.0)
            if !node.isLeaf {
                let leftCover = try #require(regNodes[node.leftChild].cover)
                let rightCover = try #require(regNodes[node.rightChild].cover)
                #expect(abs(cover - (leftCover + rightCover)) < 1e-12)
            }
        }

        let dtClf = DecisionTreeClassifier(maxDepth: 3, minSamplesSplit: 2)
        try await dtClf.fit(features: features, targets: clfTargets)
        let clfNodes = await dtClf.flatNodes
        #expect(!clfNodes.isEmpty)
        #expect(clfNodes[0].cover == Double(features.count))
        for node in clfNodes {
            let cover = try #require(node.cover)
            #expect(cover > 0.0)
            let dist = try #require(node.classDistribution)
            #expect(dist.count == 3)
            #expect(abs(dist.reduce(0.0, +) - 1.0) < 1e-12)
            if !node.isLeaf {
                let leftCover = try #require(clfNodes[node.leftChild].cover)
                let rightCover = try #require(clfNodes[node.rightChild].cover)
                #expect(abs(cover - (leftCover + rightCover)) < 1e-12)
            }
        }
    }

    @Test("RandomForestClassifier and RandomForestRegressor populate cover and classDistribution on all nodes")
    func testRandomForestMetadataAndProbabilityModes() async throws {
        let features: [[Double]] = [
            [0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [0.3, 0.3],
            [0.4, 0.4], [1.0, 1.0], [1.1, 0.9], [1.2, 1.1],
            [2.0, 2.0], [2.1, 1.9], [2.2, 2.1], [2.3, 2.2]
        ]
        // Intentional overlap so shallow depth-1/2 trees produce impure leaves
        let clfTargets: [Double] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0]
        let regTargets: [Double] = [1.0, 1.2, 2.0, 1.5, 2.2, 3.0, 4.0, 3.2, 5.0, 5.2, 4.8, 5.5]

        let rfClf = RandomForestClassifier(nEstimators: 6, maxDepth: 2, minSamplesSplit: 2, randomState: 7)
        try await rfClf.fit(features: features, targets: clfTargets)
        let clfTrees = await rfClf.flatTrees
        #expect(clfTrees.count == 6)
        for tree in clfTrees {
            for node in tree {
                let cover = try #require(node.cover)
                #expect(cover > 0.0)
                let dist = try #require(node.classDistribution)
                #expect(dist.count == 3)
                #expect(abs(dist.reduce(0.0, +) - 1.0) < 1e-12)
            }
        }

        let defaultProbs = try await rfClf.predictProbability(features: features)
        let hardProbs = try await rfClf.predictProbability(features: features, mode: .hardVote)
        let softProbs = try await rfClf.predictProbability(features: features, mode: .softLeafDistribution)
        #expect(defaultProbs.count == features.count)
        #expect(hardProbs.count == features.count)
        #expect(softProbs.count == features.count)

        var anyModeDifference = false
        for i in 0..<features.count {
            #expect(abs(softProbs[i].reduce(0.0, +) - 1.0) < 1e-12)
            for c in 0..<3 {
                #expect(abs(defaultProbs[i][c] - hardProbs[i][c]) < 1e-12)
                if abs(hardProbs[i][c] - softProbs[i][c]) > 1e-6 {
                    anyModeDifference = true
                }
            }
        }
        #expect(anyModeDifference)

        let rfReg = RandomForestRegressor(nEstimators: 5, maxDepth: 3, randomState: 7)
        try await rfReg.fit(features: features, targets: regTargets)
        let regTrees = await rfReg.flatTrees
        for tree in regTrees {
            for node in tree {
                let cover = try #require(node.cover)
                #expect(cover > 0.0)
            }
        }
    }

    @Test("DecisionTreeClassifier.prune preserves classDistribution and cover on collapsed leaves")
    func testDecisionTreePrunePreservesMetadata() async throws {
        let features: [[Double]] = [
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
            [1.0, 0.0], [1.1, 0.1], [1.2, 0.2], [1.3, 0.3],
            [2.0, 1.0], [2.1, 1.1], [2.2, 1.2], [2.3, 1.3]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 2.0]

        let dt = DecisionTreeClassifier(maxDepth: 4, minSamplesSplit: 2)
        try await dt.fit(features: features, targets: targets)
        let beforeNodes = await dt.flatNodes
        let beforeReachable = reachableIndices(tree: beforeNodes)
        let beforeLeaves = beforeReachable.filter { (idx: Int) -> Bool in beforeNodes[idx].isLeaf }.count

        // Find an alpha that collapses at least one twig (internal node whose both children are leaves).
        let twigGains = beforeReachable.compactMap { (idx: Int) -> Double? in
            let node = beforeNodes[idx]
            guard !node.isLeaf else { return nil }
            let leftIsLeaf = node.leftChild >= 0 && node.leftChild < beforeNodes.count && beforeNodes[node.leftChild].isLeaf
            let rightIsLeaf = node.rightChild >= 0 && node.rightChild < beforeNodes.count && beforeNodes[node.rightChild].isLeaf
            return (leftIsLeaf && rightIsLeaf) ? node.impurityGain : nil
        }.sorted()
        #expect(!twigGains.isEmpty)
        let pruneAlpha = twigGains[0] + 1e-9

        await dt.prune(alpha: pruneAlpha)
        let afterNodes = await dt.flatNodes
        let afterReachable = reachableIndices(tree: afterNodes)
        let afterLeaves = afterReachable.filter { (idx: Int) -> Bool in afterNodes[idx].isLeaf }.count
        #expect(afterLeaves < beforeLeaves)

        for idx in afterReachable {
            let node = afterNodes[idx]
            let cover = try #require(node.cover)
            #expect(cover > 0.0)
            let dist = try #require(node.classDistribution)
            #expect(dist.count == 3)
            #expect(abs(dist.reduce(0.0, +) - 1.0) < 1e-12)
        }

        let probs = try await dt.predictProbability(features: features)
        for row in probs {
            #expect(row.count == 3)
            #expect(abs(row.reduce(0.0, +) - 1.0) < 1e-12)
        }
    }

    @Test("GradientBoostedTreesRegressor populates cover across losses and round-trips via JSON persistence")
    func testGBDTPersistenceAndCoverRoundtrip() async throws {
        let features: [[Double]] = [
            [0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 2.5],
            [2.0, 1.0], [2.5, 0.5], [3.0, 1.5], [3.5, 2.0]
        ]
        let targets: [Double] = [1.2, 2.1, 3.4, 4.2, 3.0, 2.5, 4.8, 5.5]

        let unfitted = try GradientBoostedTreesRegressor(nEstimators: 3, learningRate: 0.1, maxDepth: 2)
        let unfittedURL = FileManager.default.temporaryDirectory.appendingPathComponent("gbdt_unfitted_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: unfittedURL) }
        await #expect(throws: SwiftMLError.self) {
            try await unfitted.save(to: unfittedURL)
        }

        let losses: [GBDTLoss] = [.squaredError, .absoluteError, .quantile(alpha: 0.75)]
        for (idx, loss) in losses.enumerated() {
            let gbdt = try GradientBoostedTreesRegressor(
                nEstimators: 6,
                learningRate: 0.15,
                maxDepth: 2,
                minSamplesSplit: 2,
                loss: loss
            )
            try await gbdt.fit(features: features, targets: targets)

            let trees = await gbdt.getEnsembleTrees()
            #expect(trees.count == 6)
            for tree in trees {
                for node in tree {
                    let cover = try #require(node.cover)
                    #expect(cover > 0.0)
                }
            }

            let origPreds = try await gbdt.predict(features: features)
            let origInit = await gbdt.getInitialPrediction()
            let origFeatures = await gbdt.getNumFeatures()

            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("gbdt_\(idx)_\(UUID().uuidString).json")
            defer { try? FileManager.default.removeItem(at: tempURL) }

            try await gbdt.save(to: tempURL)
            let loaded = try GradientBoostedTreesRegressor.load(from: tempURL)

            #expect(await loaded.nEstimators == 6)
            #expect(await loaded.learningRate == 0.15)
            #expect(await loaded.maxDepth == 2)
            #expect(await loaded.minSamplesSplit == 2)
            #expect(await loaded.loss == loss)
            #expect(await loaded.getInitialPrediction() == origInit)
            #expect(await loaded.getNumFeatures() == origFeatures)

            let loadedPreds = try await loaded.predict(features: features)
            #expect(loadedPreds.count == origPreds.count)
            for i in 0..<origPreds.count {
                #expect(abs(origPreds[i] - loadedPreds[i]) < 1e-12)
            }
        }
    }

    @Test("Legacy 3.13.2 JSON models without cover or classDistribution decode and predict cleanly")
    func testLegacy3132JSONCompatibility() async throws {
        // 1. Legacy DecisionTreeClassifier JSON (no "cover" key anywhere, no "classDistribution" on internal node)
        let dtClfJSON = """
        {
          "maxDepth": 2,
          "minSamplesSplit": 2,
          "numFeatures": 2,
          "classes": [10.0, 20.0],
          "criterion": "gini",
          "nodes": [
            {
              "featureIndex": 0,
              "threshold": 1.5,
              "leftChild": 1,
              "rightChild": 2,
              "value": 10.0,
              "isLeaf": false,
              "impurityGain": 1.0
            },
            {
              "featureIndex": -1,
              "threshold": 0.0,
              "leftChild": -1,
              "rightChild": -1,
              "value": 10.0,
              "isLeaf": true,
              "impurityGain": 0.0,
              "classDistribution": [0.75, 0.25]
            },
            {
              "featureIndex": -1,
              "threshold": 0.0,
              "leftChild": -1,
              "rightChild": -1,
              "value": 20.0,
              "isLeaf": true,
              "impurityGain": 0.0,
              "classDistribution": [0.0, 1.0]
            }
          ]
        }
        """

        let dtClfURL = FileManager.default.temporaryDirectory.appendingPathComponent("legacy_dt_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: dtClfURL) }
        try Data(dtClfJSON.utf8).write(to: dtClfURL)

        let dtClf = try DecisionTreeClassifier.load(from: dtClfURL)
        let dtClfNodes = await dtClf.flatNodes
        #expect(dtClfNodes.count == 3)
        for node in dtClfNodes {
            #expect(node.cover == nil)
        }
        let dtProbs = try await dtClf.predictProbability(features: [[1.0, 0.0], [2.0, 0.0]])
        #expect(abs(dtProbs[0][0] - 0.75) < 1e-12)
        #expect(abs(dtProbs[0][1] - 0.25) < 1e-12)
        #expect(abs(dtProbs[1][0] - 0.0) < 1e-12)
        #expect(abs(dtProbs[1][1] - 1.0) < 1e-12)

        // 2. Legacy DecisionTreeRegressor JSON (no "cover" key)
        let dtRegJSON = """
        {
          "maxDepth": 1,
          "minSamplesSplit": 2,
          "numFeatures": 1,
          "criterion": "mse",
          "nodes": [
            {
              "featureIndex": 0,
              "threshold": 2.0,
              "leftChild": 1,
              "rightChild": 2,
              "value": 5.0,
              "isLeaf": false,
              "impurityGain": 4.0
            },
            {
              "featureIndex": -1,
              "threshold": 0.0,
              "leftChild": -1,
              "rightChild": -1,
              "value": 2.0,
              "isLeaf": true,
              "impurityGain": 0.0
            },
            {
              "featureIndex": -1,
              "threshold": 0.0,
              "leftChild": -1,
              "rightChild": -1,
              "value": 8.0,
              "isLeaf": true,
              "impurityGain": 0.0
            }
          ]
        }
        """

        let dtRegURL = FileManager.default.temporaryDirectory.appendingPathComponent("legacy_dt_reg_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: dtRegURL) }
        try Data(dtRegJSON.utf8).write(to: dtRegURL)

        let dtReg = try DecisionTreeRegressor.load(from: dtRegURL)
        let dtRegNodes = await dtReg.flatNodes
        for node in dtRegNodes {
            #expect(node.cover == nil)
        }
        let dtRegPreds = try await dtReg.predict(features: [[1.0], [3.0]])
        #expect(abs(dtRegPreds[0] - 2.0) < 1e-12)
        #expect(abs(dtRegPreds[1] - 8.0) < 1e-12)

        // 3. Legacy RandomForestClassifier JSON (no "cover" and no "classDistribution" on leaves)
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

        let rfClfURL = FileManager.default.temporaryDirectory.appendingPathComponent("legacy_rf_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: rfClfURL) }
        try Data(rfClfJSON.utf8).write(to: rfClfURL)

        let rfClf = try RandomForestClassifier.load(from: rfClfURL)
        let rfTrees = await rfClf.flatTrees
        for tree in rfTrees {
            for node in tree {
                #expect(node.cover == nil)
                #expect(node.classDistribution == nil)
            }
        }
        let hardProbs = try await rfClf.predictProbability(features: [[0.5, 1.5]], mode: .hardVote)
        let softFallbackProbs = try await rfClf.predictProbability(features: [[0.5, 1.5]], mode: .softLeafDistribution)
        #expect(abs(hardProbs[0][0] - 0.5) < 1e-12)
        #expect(abs(hardProbs[0][1] - 0.5) < 1e-12)
        #expect(abs(softFallbackProbs[0][0] - 0.5) < 1e-12)
        #expect(abs(softFallbackProbs[0][1] - 0.5) < 1e-12)
    }
}
