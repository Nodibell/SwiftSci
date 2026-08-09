import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("SMOTE & Undersampling Tests")
struct SMOTETests {
    
    @Test("SMOTE balances binary classification dataset")
    func testSMOTEBinaryBalancing() throws {
        // 10 samples of class 0.0, 4 samples of class 1.0
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<10 {
            features.append([Double(i), Double(i * 2)])
            targets.append(0.0)
        }
        for i in 0..<4 {
            features.append([Double(100 + i), Double(200 + i)])
            targets.append(1.0)
        }
        
        let smote = SMOTE(kNeighbors: 3, seed: 42)
        let resampled = try smote.fitResample(features: features, targets: targets)
        
        let count0 = resampled.targets.filter { $0 == 0.0 }.count
        let count1 = resampled.targets.filter { $0 == 1.0 }.count
        
        #expect(count0 == 10)
        #expect(count1 == 10)
        #expect(resampled.features.count == 20)
    }
    
    @Test("SMOTE balances multi-class dataset")
    func testSMOTEMultiClassBalancing() throws {
        // Class 0: 6 samples, Class 1: 3 samples, Class 2: 2 samples
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<6 {
            features.append([Double(i), 1.0])
            targets.append(0.0)
        }
        for i in 0..<3 {
            features.append([Double(10 + i), 5.0])
            targets.append(1.0)
        }
        for i in 0..<2 {
            features.append([Double(20 + i), 10.0])
            targets.append(2.0)
        }
        
        let smote = SMOTE(kNeighbors: 2, seed: 123)
        let resampled = try smote.fitResample(features: features, targets: targets)
        
        let count0 = resampled.targets.filter { $0 == 0.0 }.count
        let count1 = resampled.targets.filter { $0 == 1.0 }.count
        let count2 = resampled.targets.filter { $0 == 2.0 }.count
        
        #expect(count0 == 6)
        #expect(count1 == 6)
        #expect(count2 == 6)
        #expect(resampled.features.count == 18)
    }
    
    @Test("SMOTE input validation errors")
    func testSMOTEInputValidation() throws {
        let smote = SMOTE()
        
        // Empty features
        #expect(throws: PreprocessingError.self) {
            try smote.fitResample(features: [], targets: [])
        }
        
        // Mismatched lengths
        #expect(throws: PreprocessingError.self) {
            try smote.fitResample(features: [[1.0, 2.0]], targets: [1.0, 2.0])
        }
    }
    
    @Test("RandomUndersampler balances majority classes")
    func testRandomUndersampler() throws {
        // Class 0: 10 samples, Class 1: 3 samples
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<10 {
            features.append([Double(i)])
            targets.append(0.0)
        }
        for i in 0..<3 {
            features.append([Double(100 + i)])
            targets.append(1.0)
        }
        
        let sampler = RandomUndersampler(seed: 42)
        let resampled = try sampler.fitResample(features: features, targets: targets)
        
        let count0 = resampled.targets.filter { $0 == 0.0 }.count
        let count1 = resampled.targets.filter { $0 == 1.0 }.count
        
        #expect(count0 == 3)
        #expect(count1 == 3)
        #expect(resampled.features.count == 6)
    }
}
