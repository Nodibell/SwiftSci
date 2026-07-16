import Testing
import Foundation
@testable import SwiftForecast

@Suite("Kalman Filter Tests")
struct KalmanFilterTests {
    
    @Test("1D constant signal filtering noise reduction")
    func test1DFilterNoiseReduction() async throws {
        // True value = 10.0 constant
        // Noisy measurements centered around 10.0
        let measurements: [[Double]] = [
            [10.2], [9.7], [10.5], [9.6], [10.1],
            [10.3], [9.8], [10.2], [9.9], [10.1]
        ]
        
        let kf = try await KalmanFilter.oneDimensional(processNoise: 0.01, measurementNoise: 0.1)
        try await kf.setInitialState(mean: [10.2, 0.0], covariance: [[1.0, 0.0], [0.0, 1.0]])
        let filteredStates = try await kf.filter(observations: measurements)
        #expect(filteredStates.count == measurements.count)
        
        // Calculate raw RMSE from 10.0
        var rawSSE = 0.0
        for m in measurements {
            let diff = m[0] - 10.0
            rawSSE += diff * diff
        }
        let rawRMSE = (rawSSE / Double(measurements.count)).squareRoot()
        
        // Calculate filtered RMSE from 10.0 (using state position estimate mean[0])
        var filteredSSE = 0.0
        for state in filteredStates {
            let diff = state.mean[0] - 10.0
            filteredSSE += diff * diff
        }
        let filteredRMSE = (filteredSSE / Double(filteredStates.count)).squareRoot()
        
        // Filtered signal must have lower RMSE than raw measurements due to smoothing
        #expect(filteredRMSE < rawRMSE)
    }
    
    @Test("Kalman Filter RTS smoother reduces variance")
    func testRTSSmoother() async throws {
        // Noisy sine wave
        var observations: [[Double]] = []
        for t in 0..<20 {
            let clean = sin(Double(t) * 0.5) * 10.0
            let noise = cos(Double(t) * 2.3) * 1.5 // pseudo-random noise
            observations.append([clean + noise])
        }
        
        let kf = try await KalmanFilter.oneDimensional(processNoise: 0.1, measurementNoise: 2.0)
        
        // Run forward filter
        let filteredStates = try await kf.filter(observations: observations)
        
        // Reset state to initial to run smoother from the same starting conditions
        try await kf.setInitialState(
            mean: [0.0, 0.0],
            covariance: [
                [10.0, 0.0],
                [0.0, 10.0]
            ]
        )
        
        // Run smoother
        let smoothedStates = try await kf.smooth(observations: observations)
        
        #expect(smoothedStates.count == observations.count)
        
        // The variance (uncertainty) of the state covariance P[0][0] in the middle of the run
        // should be smaller in the RTS smoother than in the forward filter
        // because the smoother uses both past and future data.
        let midIdx = 10
        let filtVar = filteredStates[midIdx].covariance[0][0]
        let smoothVar = smoothedStates[midIdx].covariance[0][0]
        
        #expect(smoothVar < filtVar)
    }
    
    @Test("Kalman Filter dimension checks and errors")
    func testKFDimensionChecks() async throws {
        let kf = try KalmanFilter(stateSize: 3, observationSize: 2)
        
        // Correct matrices setup
        try await kf.setTransitionMatrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        try await kf.setObservationMatrix([
            [1, 0, 0],
            [0, 1, 0]
        ])
        try await kf.setProcessNoise([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1]
        ])
        try await kf.setMeasurementNoise([
            [1.0, 0],
            [0, 1.0]
        ])
        // Wrong state size in mean
        await #expect(throws: ForecastError.self) {
            try await kf.setInitialState(mean: [1.0, 2.0], covariance: [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        }
        
        // Wrong matrix dimension for F
        await #expect(throws: ForecastError.self) {
            try await kf.setTransitionMatrix([
                [1, 0],
                [0, 1]
            ])
        }
    }
}
