import Testing
import Foundation
import MLX
@testable import SwiftLLM

@Suite("Sampling Pipeline Tests")
struct SamplerTests {

    @Test("Greedy search returns exact argmax")
    func testGreedySearch() {
        let logits = MLXArray([1.0, 5.0, 2.0, 0.5] as [Float])
        let config = SamplingConfiguration.greedy
        let token = Sampler.sample(logits: logits, config: config)
        #expect(token == 1)
    }

    @Test("Top-K = 1 behaves deterministically like argmax")
    func testTopKOne() {
        let logits = MLXArray([0.1, 0.2, 8.5, 3.2, 1.0] as [Float])
        let config = SamplingConfiguration(temperature: 1.0, topK: 1, topP: 1.0)
        let token = Sampler.sample(logits: logits, config: config)
        #expect(token == 2)
    }

    @Test("Top-P nucleus masks out lower-probability tokens")
    func testTopPNucleusMasking() {
        // Logits where index 3 dominates overwhelmingly (approx 90%+ prob)
        let logits = MLXArray([0.0, 0.0, 0.0, 10.0, 0.0] as [Float])
        let config = SamplingConfiguration(temperature: 1.0, topK: 0, topP: 0.5)
        let token = Sampler.sample(logits: logits, config: config)
        #expect(token == 3)
    }

    @Test("Repetition penalty reduces probability of previous tokens")
    func testRepetitionPenalty() {
        // Initially index 0 is slightly higher than index 1
        let logits = MLXArray([3.0, 2.8, 0.1, 0.1] as [Float])
        
        // Without penalty, greedy picks index 0
        let greedyBefore = Sampler.sample(logits: logits, config: .greedy)
        #expect(greedyBefore == 0)

        // With repetition penalty on token 0:
        // logit 0 (3.0) becomes 3.0 / 1.5 = 2.0, so index 1 (2.8) becomes the new maximum
        let configWithPenalty = SamplingConfiguration(temperature: 0.0, topK: 0, topP: 1.0, repetitionPenalty: 1.5)
        let greedyAfter = Sampler.sample(logits: logits, config: configWithPenalty, pastTokens: [0])
        #expect(greedyAfter == 1)
    }

    @Test("LLMOptions backward compatibility and single source of truth")
    func testLLMOptionsCompatibility() {
        var options = LLMOptions(temperature: 0.8, topP: 0.85, topK: 50, repetitionPenalty: 1.2, maxTokens: 200)
        
        #expect(options.sampling.temperature == 0.8)
        #expect(options.sampling.topP == 0.85)
        #expect(options.sampling.topK == 50)
        #expect(options.sampling.repetitionPenalty == 1.2)
        #expect(options.maxTokens == 200)

        // Mutate via sampling struct
        options.sampling.temperature = 0.5
        #expect(options.temperature == 0.5)

        // Mutate via deprecated property
        options.topP = 0.95
        #expect(options.sampling.topP == 0.95)
    }
}
