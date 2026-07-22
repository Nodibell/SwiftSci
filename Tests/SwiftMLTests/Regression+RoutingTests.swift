import Testing
import Foundation
@testable import SwiftML

@Suite("Regression Routing Tests")
struct RegressionRoutingTests {
    
    init() {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
           let bundle = Bundle(url: url) {
            _ = bundle.resourceURL
        }
        for bundle in Bundle.allBundles {
            if let url = bundle.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
               let b = Bundle(url: url) {
                _ = b.resourceURL
            }
        }
    }
    
    @Test("LinearRegression CPU vs GPU")
    func testLinearRegressionRouting() async throws {
        let features = [
            [1.0], [2.0], [3.0], [4.0], [5.0]
        ]
        let targets = [
            2.0, 4.0, 6.0, 8.0, 10.0
        ]
        
        let lrCPU = LinearRegression(device: .cpu)
        try await lrCPU.fit(features: features, targets: targets, learningRate: 0.05, epochs: 100)
        
        let lrGPU = LinearRegression(device: .gpu)
        try await lrGPU.fit(features: features, targets: targets, learningRate: 0.05, epochs: 100)
        
        #expect(await lrCPU.resolvedDevice == .cpu)
        #expect(await lrGPU.resolvedDevice == .gpu)
        
        let weightsCPU = await lrCPU.getWeights()
        let weightsGPU = await lrGPU.getWeights()
        #expect(weightsCPU != nil && weightsGPU != nil)
        #expect(abs(weightsCPU![0] - weightsGPU![0]) < 1e-1)
        
        let biasCPU = await lrCPU.getBias()
        let biasGPU = await lrGPU.getBias()
        #expect(biasCPU != nil && biasGPU != nil)
        #expect(abs(biasCPU! - biasGPU!) < 1e-1)
        
        let predsCPU = try await lrCPU.predict(features: features)
        let predsGPU = try await lrGPU.predict(features: features)
        #expect(abs(predsCPU[0] - predsGPU[0]) < 0.2)
    }
    
    @Test("LogisticRegression CPU vs GPU")
    func testLogisticRegressionRouting() async throws {
        let features = [
            [-2.0], [-1.0], [1.0], [2.0]
        ]
        let targets = [
            0.0, 0.0, 1.0, 1.0
        ]
        
        let lrCPU = LogisticRegression(device: .cpu)
        try await lrCPU.fit(features: features, targets: targets, learningRate: 0.2, epochs: 100)
        
        let lrGPU = LogisticRegression(device: .gpu)
        try await lrGPU.fit(features: features, targets: targets, learningRate: 0.2, epochs: 100)
        
        #expect(await lrCPU.resolvedDevice == .cpu)
        #expect(await lrGPU.resolvedDevice == .gpu)
        
        let probsCPU = try await lrCPU.predictProbability(features: features)
        let probsGPU = try await lrGPU.predictProbability(features: features)
        #expect(abs(probsCPU[0][1] - probsGPU[0][1]) < 0.1)
    }
}
