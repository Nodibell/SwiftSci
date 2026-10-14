import Testing
import Foundation
import SwiftPreprocessing
@testable import SwiftML

@Suite("LinearRegression Tests")
struct LinearRegressionTests {
    
    @Test("OLS recovers coefficients under large orthogonal residuals")
    func largeOrthogonalResiduals() async throws {
        let features = (0..<5).map { i -> [Double] in
            let x = Double(i)
            return [x, x * x, x * x * x]
        }
        let noise = [1.0, -4, 6, -4, 1]
        let targets = features.enumerated().map { i, row in
            7 - 2 * row[0] + 3 * row[1] + 0.5 * row[2] + 1e8 * noise[i]
        }
        let model = LinearRegression(device: .cpu)
        try await model.fit(features: features, targets: targets)
        let weights = try #require(await model.getWeights())
        let bias = try #require(await model.getBias())
        for (actual, expected) in zip(weights, [-2.0, 3, 0.5]) {
            #expect(abs(actual - expected) < 1e-10)
        }
        #expect(abs(bias - 7) < 1e-10)
    }

    @Test("Supplied linear parameters predict without fitting", arguments: [ExecutionDevice.cpu, .auto, .gpu, .ane])
    func suppliedParameters(device: ExecutionDevice) async throws {
        let model = LinearRegression(weights: [2, -3], bias: 5, device: device)
        let predictions = try await model.predict(features: [[1, 2], [-2, 1]])
        #expect(predictions == [1, -2])
        #expect(await model.resolvedDevice == ((device == .cpu || device == .auto) ? .cpu : .gpu))
    }

    @Test("Supplied CPU linear parameters retain Double precision")
    func suppliedDoublePrecision() async throws {
        let coefficient = 1 + Double(sign: .plus, exponent: -40, significand: 1)
        let model = LinearRegression(weights: [coefficient], bias: 0, device: .cpu)
        #expect(try await model.predict(features: [[1]]) == [coefficient])
    }

    @Test("LinearRegression convergence on clean data")
    func testConvergence() async throws {
        // Equation: y = 2 * x1 + 3 * x2 + 5
        let features: [[Double]] = [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ]
        let targets: [Double] = [10.0, 12.0, 13.0, 15.0, 20.0]
        
        let model = LinearRegression()
        try await model.fit(features: features, targets: targets, learningRate: 0.05, epochs: 1200)
        
        let w = await model.getWeights()
        let b = await model.getBias()
        
        #expect(w != nil)
        #expect(b != nil)
        
        // Check coefficients (approximate tolerance because of SGD)
        #expect(abs(w![0] - 2.0) < 0.05)
        #expect(abs(w![1] - 3.0) < 0.05)
        #expect(abs(b! - 5.0) < 0.15)
        
        // Predict
        let preds = try await model.predict(features: [[1.0, 3.0], [3.0, 1.0]])
        #expect(abs(preds[0] - 16.0) < 0.1)
        #expect(abs(preds[1] - 14.0) < 0.1)
    }
    
    @Test("LinearRegression errors")
    func testLinearRegressionErrors() async throws {
        let model = LinearRegression()
        
        // Empty input throws
        await #expect(throws: MLError.self) {
            try await model.fit(features: [], targets: [])
        }
        
        // Transform before fit throws
        await #expect(throws: MLError.self) {
            _ = try await model.predict(features: [[1.0, 2.0]])
        }
    }

    // MARK: - Input & Solver Validation

    @Test("LinearRegression rejects non-finite features and targets across row array API")
    func testFiniteValidationInRowAPI() async throws {
        let nanFeatures = [[1.0, Double.nan], [2.0, 3.0]]
        let cleanFeatures = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        let nanTargets = [1.0, Double.nan, 3.0]
        let infFeatures = [[1.0, Double.infinity], [2.0, 3.0], [3.0, 4.0]]
        let cleanTargets = [1.0, 2.0, 3.0]

        let model = LinearRegression()
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: nanFeatures, targets: [1.0, 2.0])
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: infFeatures, targets: cleanTargets)
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: cleanFeatures, targets: nanTargets)
        }

        try await model.fit(features: cleanFeatures, targets: cleanTargets)
        let preds = try await model.predict(features: [[1.0, Double.nan]])
        #expect(preds[0].isNaN)
    }

    @Test("LinearRegression exposes solverUsed and fallbackReason")
    func testSolverUsedAndFallback() async throws {
        // Overdetermined clean system on CPU -> analytical OLS
        let X = [[1.0], [2.0], [3.0], [4.0]]
        let y = [2.0, 4.0, 6.0, 8.0]
        let olsModel = LinearRegression(device: .cpu)
        try await olsModel.fit(features: X, targets: y)
        let solver1 = await olsModel.solverUsed
        let fallback1 = await olsModel.fallbackReason
        #expect(solver1 == .analyticalOLS)
        #expect(fallback1 == nil)

        // Underdetermined system (numSamples < numFeatures + 1): 2 samples, 2 features (requires 3 samples for OLS)
        let underX = [[1.0, 2.0], [3.0, 4.0]]
        let underY = [5.0, 11.0]
        let gdModel = LinearRegression(device: .cpu)
        try await gdModel.fit(features: underX, targets: underY, learningRate: 0.01, epochs: 100)
        let solver2 = await gdModel.solverUsed
        let fallback2 = await gdModel.fallbackReason
        #expect(solver2 == .gradientDescent)
        #expect(fallback2 != nil)
    }

    @Test("LinearRegression validates optimization parameters")
    func testParameterValidation() async throws {
        let X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        let y = [3.0, 5.0, 7.0]

        let model = LinearRegression()
        // Invalid learning rate
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: X, targets: y, learningRate: 0.0, epochs: 100)
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: X, targets: y, learningRate: -0.01, epochs: 100)
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: X, targets: y, learningRate: Float.nan, epochs: 100)
        }

        // Invalid epochs
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: X, targets: y, learningRate: 0.01, epochs: 0)
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(features: X, targets: y, learningRate: 0.01, epochs: -10)
        }
    }
}
