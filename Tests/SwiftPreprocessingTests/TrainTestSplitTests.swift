import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("TrainTestSplit Tests")
struct TrainTestSplitTests {
    
    @Test("trainTestSplit basic array splitting")
    func testArraySplit() throws {
        let features = [
            [1.0], [2.0], [3.0], [4.0]
        ]
        let targets = [
            10.0, 20.0, 30.0, 40.0
        ]
        
        let split = try trainTestSplit(features, targets, testSize: 0.25, shuffle: false)
        
        #expect(split.trainFeatures == [[1.0], [2.0], [3.0]])
        #expect(split.testFeatures == [[4.0]])
        #expect(split.trainTargets == [10.0, 20.0, 30.0])
        #expect(split.testTargets == [40.0])
    }
    
    @Test("trainTestSplit shuffling reproducibility with seed")
    func testArraySplitSeed() throws {
        let features = [
            [1.0], [2.0], [3.0], [4.0]
        ]
        let targets = [
            10.0, 20.0, 30.0, 40.0
        ]
        
        let split1 = try trainTestSplit(features, targets, testSize: 0.5, shuffle: true, seed: 42)
        let split2 = try trainTestSplit(features, targets, testSize: 0.5, shuffle: true, seed: 42)
        
        #expect(split1.trainFeatures == split2.trainFeatures)
        #expect(split1.testFeatures == split2.testFeatures)
        #expect(split1.trainTargets == split2.trainTargets)
        #expect(split1.testTargets == split2.testTargets)
    }
    
    @Test("DataFrame trainTestSplit extension")
    func testDataFrameSplit() throws {
        let colA = TypedColumn<Double>(name: "A", values: [1.0, 2.0, 3.0, 4.0])
        let colB = TypedColumn<Double>(name: "B", values: [10.0, 20.0, 30.0, 40.0])
        let df = try DataFrame(columns: [colA, colB])
        
        let (train, test) = try df.trainTestSplit(testSize: 0.25, shuffle: false)
        
        #expect(train.shape.rows == 3)
        #expect(test.shape.rows == 1)
        
        let testA = test[column: "A", as: Double.self]
        #expect(testA?[0] == 4.0)
    }

    @Test("trainTestSplit handles tiny dataset (N < 8, N = 3) without throwing (G-022)")
    func testTinyDatasetSplit() throws {
        let features = [[1.0], [2.0], [3.0]]
        let targets = [10.0, 20.0, 30.0]

        let split = try trainTestSplit(features, targets, testSize: 0.25, shuffle: false)
        #expect(split.trainFeatures.count == 2)
        #expect(split.testFeatures.count == 1)
        #expect(split.trainTargets.count == 2)
        #expect(split.testTargets.count == 1)
    }

    @Test("trainTestSplit stratify handles singleton classes gracefully (G-022)")
    func testStratifyWithSingletons() throws {
        // Class 0 has 3 items, Class 1 has 1 item, Class 2 has 1 item
        let features = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let targets = [0.0, 0.0, 0.0, 1.0, 2.0]

        let split = try trainTestSplit(features, targets, testSize: 0.25, stratify: targets, shuffle: false)
        #expect(!split.trainFeatures.isEmpty)
        #expect(!split.testFeatures.isEmpty)
        #expect(split.trainFeatures.count + split.testFeatures.count == 5)
    }

    @Test("DataFrame trainTestSplit handles singletons with stratifyColumn (G-022)")
    func testDataFrameStratifyWithSingletons() throws {
        let colX = TypedColumn<Double>(name: "X", values: [1.0, 2.0, 3.0, 4.0, 5.0])
        let colY = TypedColumn<String>(name: "label", values: ["A", "A", "A", "B", "C"])
        let df = try DataFrame(columns: [colX, colY])

        let (train, test) = try df.trainTestSplit(testSize: 0.25, stratifyColumn: "label", shuffle: false)
        #expect(train.shape.rows > 0)
        #expect(test.shape.rows > 0)
        #expect(train.shape.rows + test.shape.rows == 5)
    }
}
