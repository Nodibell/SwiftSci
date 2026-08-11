import Testing
import Foundation
@testable import SwiftForecast

@Suite("Koopman Operator Tests (EDMD)")
struct KoopmanOperatorTests {

    @Test("Linear 1D system exact forecasting")
    func testLinearSystemForecasting() async throws {
        // x_{t+1} = 0.95 * x_t
        var series: [Double] = [1.0]
        for _ in 0..<20 {
            let next = series.last! * 0.95
            series.append(next)
        }

        let koopman = KoopmanOperator(dictionary: .identity, regularization: 1e-6)
        try await koopman.fit(series: series)

        let predictions = try await koopman.predict1D(horizon: 5)
        #expect(predictions.count == 5)

        // Compare against ground truth 0.95^t
        var expectedLast = series.last!
        for p in predictions {
            expectedLast *= 0.95
            #expect(abs(p - expectedLast) < 1e-3)
        }
    }

    @Test("Nonlinear 1D system polynomial EDMD forecasting")
    func testNonlinearPolynomialSystem() async throws {
        // Quadratic nonlinear map x_{t+1} = 0.8 * x_t + 0.15 * x_t^2
        var series: [Double] = [0.5]
        for _ in 0..<30 {
            let x = series.last!
            let next = 0.8 * x + 0.15 * x * x
            series.append(next)
        }

        let koopman = KoopmanOperator(dictionary: .polynomial(degree: 2), regularization: 1e-6)
        try await koopman.fit(series: series)

        let predictions = try await koopman.predict1D(horizon: 3)
        #expect(predictions.count == 3)

        var xCurr = series.last!
        for p in predictions {
            xCurr = 0.8 * xCurr + 0.15 * xCurr * xCurr
            #expect(abs(p - xCurr) < 1e-2)
        }
    }

    @Test("Multi-dimensional state trajectory forecasting with RBF dictionary")
    func testMultiDimStateRBF() async throws {
        // 2D oscillatory trajectory [cos(t), sin(t)]
        var trajectory = [[Double]]()
        for t in 0..<50 {
            let angle = Double(t) * 0.1
            trajectory.append([cos(angle), sin(angle)])
        }

        let centers = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        let dict = ObservableDictionary.combined([
            .identity,
            .rbf(centers: centers, gamma: 0.5)
        ])

        let koopman = KoopmanOperator(dictionary: dict, regularization: 1e-5)
        try await koopman.fit(trajectory: trajectory)

        let predictedTrajectory = try await koopman.predict(horizon: 5)
        #expect(predictedTrajectory.count == 5)

        for (idx, state) in predictedTrajectory.enumerated() {
            let t = Double(50 + idx) * 0.1
            let expectedX = cos(t)
            let expectedY = sin(t)
            #expect(abs(state[0] - expectedX) < 0.1)
            #expect(abs(state[1] - expectedY) < 0.1)
        }
    }

    @Test("Fourier dictionary forecasting")
    func testFourierDictionary() async throws {
        var series = [Double]()
        for t in 0..<40 {
            series.append(sin(Double(t) * 0.2))
        }

        let koopman = KoopmanOperator(
            dictionary: .fourier(frequencies: [0.2, 0.4]),
            regularization: 1e-5
        )
        try await koopman.fit(series: series)

        let preds = try await koopman.predict1D(horizon: 5)
        #expect(preds.count == 5)
        #expect(abs(preds[0] - sin(40.0 * 0.2)) < 0.15)
    }

    @Test("Hankel time-delay embedding forecasting")
    func testHankelEmbedding() async throws {
        // Sine wave with lag embedding = 3
        var series = [Double]()
        for t in 0..<30 {
            series.append(sin(Double(t) * 0.3))
        }

        let koopman = KoopmanOperator(
            dictionary: .polynomial(degree: 2),
            regularization: 1e-5,
            embeddingLags: 3
        )
        try await koopman.fit(series: series)

        let preds = try await koopman.predict1D(horizon: 3)
        #expect(preds.count == 3)
    }

    @Test("Koopman eigenvalues spectrum calculation")
    func testEigenvalues() async throws {
        // Linear damped oscillator -> eigenvalues inside unit circle |lambda| <= 1
        var trajectory = [[Double]]()
        var x = 1.0
        var v = 0.0
        let dt = 0.05
        for _ in 0..<50 {
            trajectory.append([x, v])
            let a = -1.0 * x - 0.2 * v
            x += v * dt
            v += a * dt
        }

        let koopman = KoopmanOperator(dictionary: .identity, regularization: 1e-6)
        try await koopman.fit(trajectory: trajectory)

        let evs = try await koopman.eigenvalues()
        #expect(evs.count == 2)
        for ev in evs {
            let magnitude = sqrt(ev.real * ev.real + ev.imag * ev.imag)
            #expect(magnitude <= 1.05) // Damped or marginal stability
        }
    }

    @Test("Koopman error handling")
    func testErrorHandling() async throws {
        let koopman = KoopmanOperator()

        // Unfitted predict
        await #expect(throws: ForecastError.self) {
            _ = try await koopman.predict(horizon: 5)
        }

        // Empty series
        await #expect(throws: ForecastError.self) {
            try await koopman.fit(series: [])
        }

        // Series with NaN
        await #expect(throws: ForecastError.self) {
            try await koopman.fit(series: [1.0, Double.nan, 3.0])
        }

        // Series with Infinity
        await #expect(throws: ForecastError.self) {
            try await koopman.fit(series: [1.0, Double.infinity, 3.0])
        }

        // Insufficient length
        await #expect(throws: ForecastError.self) {
            try await koopman.fit(series: [1.0])
        }

        // Fit valid series then test invalid horizon
        try await koopman.fit(series: [1.0, 2.0, 3.0, 4.0, 5.0])
        await #expect(throws: ForecastError.self) {
            _ = try await koopman.predict(horizon: 0)
        }
    }
}
