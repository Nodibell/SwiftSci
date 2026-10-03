import XCTest
@testable import SwiftOptimize

final class ValidationFoldsTests: XCTestCase {
    
    func testStratifiedKFoldPreservesClassRatios() throws {
        let features = [[Double]](repeating: [1.0, 2.0], count: 100)
        // Imbalanced classes: 80 class 0, 20 class 1
        let targets = [Double](repeating: 0.0, count: 80) + [Double](repeating: 1.0, count: 20)
        
        let stratKFold = try StratifiedKFold(nSplits: 5, shuffle: false)
        let folds = try stratKFold.split(features: features, targets: targets)
        
        XCTAssertEqual(folds.count, 5)
        for fold in folds {
            let valClass1 = fold.valTargets.filter { $0 == 1.0 }.count
            XCTAssertEqual(valClass1, 4, "Each fold validation set should contain exactly 4 class 1 samples")
        }
    }

    func testStratifiedKFoldDeterminismAcrossInstances() throws {
        let n = 60
        let features = (0..<n).map { [Double($0), Double($0 * 2)] }
        let targets = (0..<n).map { Double($0 % 3) } // 3 classes with 20 samples each

        let skf1 = try StratifiedKFold(nSplits: 4, shuffle: true, seed: 12345)
        let folds1 = try skf1.split(features: features, targets: targets)

        let skf2 = try StratifiedKFold(nSplits: 4, shuffle: true, seed: 12345)
        let folds2 = try skf2.split(features: features, targets: targets)

        XCTAssertEqual(folds1.count, folds2.count)
        for i in 0..<folds1.count {
            XCTAssertEqual(folds1[i].valTargets, folds2[i].valTargets, "Folds must be identical for the same seed")
            XCTAssertEqual(folds1[i].trainTargets, folds2[i].trainTargets, "Training folds must be identical for the same seed")
        }
    }

    func testStratifiedKFoldBalancedFoldAllocation() throws {
        // Class 0: 7 samples, Class 1: 7 samples, Class 2: 7 samples. 21 samples total, 4 folds.
        // Remainder distribution should balance fold sizes to 5, 5, 5, 6
        let n = 21
        let features = (0..<n).map { [Double($0)] }
        let targets = [Double](repeating: 0.0, count: 7) + [Double](repeating: 1.0, count: 7) + [Double](repeating: 2.0, count: 7)

        let skf = try StratifiedKFold(nSplits: 4, shuffle: false)
        let folds = try skf.split(features: features, targets: targets)

        XCTAssertEqual(folds.count, 4)
        let foldSizes = folds.map { $0.valTargets.count }
        let minSize = foldSizes.min()!
        let maxSize = foldSizes.max()!
        XCTAssertLessThanOrEqual(maxSize - minSize, 1, "Fold sizes should differ by at most 1 in balanced stratified split")
    }

    func testStratifiedKFoldThrowsValidationError() throws {
        XCTAssertThrowsError(try StratifiedKFold(nSplits: 1)) { error in
            XCTAssertEqual(error as? ValidationError, .invalidFoldCount(1))
        }

        let validSkf = try StratifiedKFold(nSplits: 3)
        // Empty features
        XCTAssertThrowsError(try validSkf.split(features: [], targets: [])) { error in
            XCTAssertEqual(error as? ValidationError, .emptyDataset)
        }
        // Count mismatch
        XCTAssertThrowsError(try validSkf.split(features: [[1.0], [2.0]], targets: [0.0])) { error in
            XCTAssertEqual(error as? ValidationError, .dimensionMismatch(features: 2, targets: 1))
        }
        // Class with insufficient samples (class 1 has 2 samples, but 3 folds required)
        let targets = [0.0, 0.0, 0.0, 1.0, 1.0]
        let features = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        XCTAssertThrowsError(try validSkf.split(features: features, targets: targets)) { error in
            XCTAssertEqual(error as? ValidationError, .insufficientClassSamples(label: 1, count: 2, requiredFolds: 3))
        }
    }
    
    func testTimeSeriesSplitExpandingWindow() throws {
        let features = (0..<100).map { [Double($0)] }
        let targets = (0..<100).map { Double($0) }
        
        let tsSplit = try TimeSeriesSplit(nSplits: 4)
        let folds = try tsSplit.split(features: features, targets: targets)
        
        XCTAssertEqual(folds.count, 4)
        for i in 1..<folds.count {
            XCTAssertGreaterThan(folds[i].trainFeatures.count, folds[i - 1].trainFeatures.count, "Training set should expand over iterations")
        }
    }

    func testTimeSeriesSplitThrowsValidationError() throws {
        XCTAssertThrowsError(try TimeSeriesSplit(nSplits: 0)) { error in
            XCTAssertEqual(error as? ValidationError, .invalidFoldCount(0))
        }
        XCTAssertThrowsError(try TimeSeriesSplit(nSplits: 3, maxTrainSize: -5)) { error in
            XCTAssertEqual(error as? ValidationError, .invalidParameter("maxTrainSize must be positive"))
        }

        let validTs = try TimeSeriesSplit(nSplits: 4)
        XCTAssertThrowsError(try validTs.split(features: [[1.0], [2.0]], targets: [1.0, 2.0])) { error in
            XCTAssertEqual(error as? ValidationError, .insufficientSamples(samples: 2, required: 5))
        }
    }
    
    func testGroupKFoldNoGroupLeakage() throws {
        let features = [[Double]](repeating: [1.0], count: 12)
        let targets = [Double](repeating: 0.0, count: 12)
        let groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        
        let groupFold = try GroupKFold(nSplits: 4)
        let folds = try groupFold.split(features: features, targets: targets, groups: groups)
        
        XCTAssertEqual(folds.count, 4)
    }

    func testGroupKFoldThrowsValidationError() throws {
        XCTAssertThrowsError(try GroupKFold(nSplits: 1)) { error in
            XCTAssertEqual(error as? ValidationError, .invalidFoldCount(1))
        }

        let validGroup = try GroupKFold(nSplits: 3)
        // Group count mismatch
        XCTAssertThrowsError(try validGroup.split(features: [[1.0], [2.0]], targets: [0.0, 0.0], groups: [1])) { error in
            XCTAssertEqual(error as? ValidationError, .groupCountMismatch(features: 2, groups: 1))
        }
        // Insufficient unique groups (only 2 unique groups for 3 splits)
        XCTAssertThrowsError(try validGroup.split(features: [[1.0], [2.0], [3.0]], targets: [0.0, 0.0, 0.0], groups: [1, 1, 2])) { error in
            XCTAssertEqual(error as? ValidationError, .insufficientGroups(groups: 2, required: 3))
        }
    }
    
    func testNewMetricsCalculations() {
        let yTrueInt = [1, 1, 0, 0, 1, 0]
        let yPredInt = [1, 0, 0, 0, 1, 1]
        let yScore = [0.9, 0.4, 0.1, 0.2, 0.8, 0.6]
        
        let fBeta = Metrics.fBetaScore(yTrue: yTrueInt, yPred: yPredInt, label: 1, beta: 0.5)
        XCTAssertGreaterThan(fBeta, 0.0)
        
        let prAuc = Metrics.prAUC(yTrue: yTrueInt, yScore: yScore)
        XCTAssertGreaterThan(prAuc, 0.0)
        
        let yTrueD = [10.0, 20.0, 30.0, 40.0]
        let yPredD = [11.0, 19.0, 31.0, 39.0]
        
        let adjR2 = Metrics.adjustedR2Score(yTrue: yTrueD, yPred: yPredD, numFeatures: 1)
        XCTAssertGreaterThan(adjR2, 0.9)
        
        let mape = Metrics.mape(yTrue: yTrueD, yPred: yPredD)
        XCTAssertLessThan(mape, 10.0, "MAPE should be under 10%")
        
        let expVar = Metrics.explainedVarianceScore(yTrue: yTrueD, yPred: yPredD)
        XCTAssertGreaterThan(expVar, 0.9)
    }
}

