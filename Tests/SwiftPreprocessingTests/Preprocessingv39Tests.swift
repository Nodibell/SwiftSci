import Testing
import Foundation
@testable import SwiftPreprocessing
import SwiftDataFrame

@Suite("Preprocessing v3.9 Tests - Stratification & Advanced Resampling")
struct Preprocessingv39Tests {
    
    @Test("trainTestSplit with stratify preserves class proportions")
    func testTrainTestSplitStratify() throws {
        // Class 0: 80 samples, Class 1: 20 samples (80/20 ratio)
        var X = [[Double]]()
        var y = [Double]()
        for i in 0..<80 {
            X.append([Double(i), 1.0])
            y.append(0.0)
        }
        for i in 0..<20 {
            X.append([Double(i + 100), 2.0])
            y.append(1.0)
        }
        
        let split = try trainTestSplit(X, y, testSize: 0.25, stratify: y, shuffle: true, seed: 42)
        
        // Train: 75 samples (60 of 0.0, 15 of 1.0)
        // Test: 25 samples (20 of 0.0, 5 of 1.0)
        let train0 = split.trainTargets.filter { $0 == 0.0 }.count
        let train1 = split.trainTargets.filter { $0 == 1.0 }.count
        let test0 = split.testTargets.filter { $0 == 0.0 }.count
        let test1 = split.testTargets.filter { $0 == 1.0 }.count
        
        #expect(train0 == 60)
        #expect(train1 == 15)
        #expect(test0 == 20)
        #expect(test1 == 5)
    }
    
    @Test("DataFrame.trainTestSplit with stratifyColumn preserves proportions")
    func testDataFrameTrainTestSplitStratify() throws {
        var labels = [String]()
        var values = [Double]()
        for i in 0..<60 {
            labels.append("A")
            values.append(Double(i))
        }
        for i in 0..<30 {
            labels.append("B")
            values.append(Double(100 + i))
        }
        for i in 0..<10 {
            labels.append("C")
            values.append(Double(200 + i))
        }
        
        let df = try DataFrame(columns: [
            TypedColumn(name: "val", values: values),
            TypedColumn(name: "label", values: labels)
        ])
        
        let (trainDf, testDf) = try df.trainTestSplit(testSize: 0.2, stratifyColumn: "label", shuffle: true, seed: 99)
        
        #expect(trainDf.shape.rows + testDf.shape.rows == 100)
        
        let testLabels = testDf["label"]?.toStrings() ?? []
        let countA = testLabels.filter { $0 == "A" }.count
        let countB = testLabels.filter { $0 == "B" }.count
        let countC = testLabels.filter { $0 == "C" }.count
        
        #expect(countA == 12)
        #expect(countB == 6)
        #expect(countC == 2)
    }
    
    @Test("StratifiedKFold generates balanced folds")
    func testStratifiedKFold() throws {
        // 30 samples of 0.0, 15 samples of 1.0
        var y = [Double]()
        for _ in 0..<30 { y.append(0.0) }
        for _ in 0..<15 { y.append(1.0) }
        
        let skf = StratifiedKFold(nSplits: 3, shuffle: true, seed: 42)
        let folds = try skf.split(targets: y)
        
        #expect(folds.count == 3)
        for fold in folds {
            #expect(fold.trainIndices.count == 30)
            #expect(fold.testIndices.count == 15)
            
            let testY = fold.testIndices.map { y[$0] }
            let count0 = testY.filter { $0 == 0.0 }.count
            let count1 = testY.filter { $0 == 1.0 }.count
            #expect(count0 == 10)
            #expect(count1 == 5)
        }
    }
    
    @Test("BorderlineSMOTE balances minority class")
    func testBorderlineSMOTE() throws {
        // Class 0 (majority): 15 samples
        // Class 1 (minority): 5 samples
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<15 {
            features.append([Double(i), 0.0])
            targets.append(0.0)
        }
        for i in 0..<5 {
            features.append([Double(i + 10), 1.0])
            targets.append(1.0)
        }
        
        let bSmote = BorderlineSMOTE(kNeighbors: 3, mNeighbors: 5, seed: 42)
        let resampled = try bSmote.fitResample(features: features, targets: targets)
        
        let c0 = resampled.targets.filter { $0 == 0.0 }.count
        let c1 = resampled.targets.filter { $0 == 1.0 }.count
        
        #expect(c0 == 15)
        #expect(c1 == 15)
        #expect(resampled.features.count == 30)
    }
    
    @Test("ADASYN balances minority class adaptively")
    func testADASYN() throws {
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<20 {
            features.append([Double(i), Double(i * 2)])
            targets.append(0.0)
        }
        for i in 0..<5 {
            features.append([Double(15 + i), Double(30 + i)])
            targets.append(1.0)
        }
        
        let adasyn = ADASYN(kNeighbors: 3, seed: 123)
        let resampled = try adasyn.fitResample(features: features, targets: targets)
        
        let c0 = resampled.targets.filter { $0 == 0.0 }.count
        let c1 = resampled.targets.filter { $0 == 1.0 }.count
        
        #expect(c0 == 20)
        #expect(c1 >= 19 && c1 <= 21) // ADASYN generates roughly balanced amounts based on density
    }
    
    @Test("DataFrame.resample with SMOTE and string labels")
    func testDataFrameResample() throws {
        var col1 = [Double]()
        var col2 = [Double]()
        var labels = [String]()
        
        // 12 samples of "cat", 4 samples of "dog"
        for i in 0..<12 {
            col1.append(Double(i))
            col2.append(Double(i * 2))
            labels.append("cat")
        }
        for i in 0..<4 {
            col1.append(Double(50 + i))
            col2.append(Double(100 + i))
            labels.append("dog")
        }
        
        let df = try DataFrame(columns: [
            TypedColumn(name: "feat1", values: col1),
            TypedColumn(name: "feat2", values: col2),
            TypedColumn(name: "species", values: labels)
        ])
        
        let balancedDf = try df.resample(using: SMOTE(kNeighbors: 3, seed: 42), targetColumn: "species")
        
        #expect(balancedDf.shape.rows == 24)
        let targetStrings = balancedDf["species"]?.toStrings() ?? []
        let catCount = targetStrings.filter { $0 == "cat" }.count
        let dogCount = targetStrings.filter { $0 == "dog" }.count
        
        #expect(catCount == 12)
        #expect(dogCount == 12)
    }
}
