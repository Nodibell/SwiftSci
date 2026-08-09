import XCTest
@testable import SwiftStats

final class SurvivalAnalysisTests: XCTestCase {
    
    func testKaplanMeierEstimation() {
        let durations = [5.0, 10.0, 15.0, 20.0, 25.0]
        let events = [1, 0, 1, 1, 0]
        
        let km = KaplanMeier(durations: durations, events: events)
        XCTAssertEqual(km.timeline.count, 5)
        
        let initialProb = km.predictSurvivalProbability(at: 0.0)
        XCTAssertEqual(initialProb, 1.0, accuracy: 1e-5)
        
        let probAt5 = km.predictSurvivalProbability(at: 5.0)
        XCTAssertEqual(probAt5, 0.8, accuracy: 1e-3)
        
        let probAt30 = km.predictSurvivalProbability(at: 30.0)
        XCTAssertGreaterThan(probAt30, 0.0)
        XCTAssertLessThanOrEqual(probAt30, 1.0)
    }
    
    func testCoxProportionalHazardsFitting() {
        let features = [
            [1.0, 0.5],
            [2.0, 1.0],
            [0.5, 0.2],
            [3.0, 1.5]
        ]
        let durations = [10.0, 20.0, 5.0, 30.0]
        let events = [1, 1, 1, 1]
        
        let cox = CoxProportionalHazards()
        XCTAssertFalse(cox.isFitted)
        
        cox.fit(features: features, durations: durations, events: events, maxIterations: 20, lr: 0.05)
        XCTAssertTrue(cox.isFitted)
        XCTAssertEqual(cox.coefficients.count, 2)
        
        let hazard = cox.predictPartialHazard(features: [1.0, 0.5])
        XCTAssertGreaterThan(hazard, 0.0)
    }
}
