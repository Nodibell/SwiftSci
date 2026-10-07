import Foundation
import Testing
import MLX
import MLXNN
import SwiftNLP
@testable import SwiftLLM

@Suite("Llama rotary scaling", .serialized)
struct LlamaRotaryScalingTests {
    @Test("Scaled frequency constants do not depend on the initialization device", arguments: [64, 128])
    func initializationDeviceParity(dimensions: Int) {
        for base: Float in [10_000, 500_000] {
            for scaling in [Llama3RoPEScaling(factor: 1), Llama3RoPEScaling(factor: 8),
                            Llama3RoPEScaling(factor: 32),
                            Llama3RoPEScaling(factor: 4, lowFrequencyFactor: 2,
                                highFrequencyFactor: 8, originalContextLength: 4096)] {
                let cpu = Device.withDefaultDevice(.cpu) {
                    scaling.frequencyDenominators(dimensions: dimensions, base: base).asArray(Float.self)
                }
                let gpu = Device.withDefaultDevice(.gpu) {
                    scaling.frequencyDenominators(dimensions: dimensions, base: base).asArray(Float.self)
                }
                #expect(cpu.count == dimensions / 2)
                #expect(cpu.allSatisfy { $0.isFinite && $0 > 0 })
                #expect(cpu.map(\.bitPattern) == gpu.map(\.bitPattern),
                    "dimensions=\(dimensions), base=\(base), scaling=\(scaling)")
            }
        }
    }

    @Test("Transition-band frequencies match the checkpoint reference", arguments: [false, true])
    func transitionFrequencies(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            let scaling = Llama3RoPEScaling(factor: 32)
            let actual = scaling.frequencyDenominators(dimensions: 64, base: 500_000)
                .asArray(Float.self)
            // mlx-lm 0.31.2 Llama 3 scaling, MLX 0.31.1, Float32, head dimension 64.
            // Allow one ULP for backend power rounding in the transition band.
            let reference: [(Int, Float)] = [
                (15, 774.86474609375),
                (16, 2327.9814453125),
                (17, 10300.4794921875),
            ]
            for (index, expected) in reference {
                #expect(abs(actual[index] - expected) <= expected.ulp,
                    "GPU=\(gpu), frequency=\(index), actual=\(actual[index]), expected=\(expected)")
            }
        }
    }

    @Test("Llama 3.2 preset applies wavelength-dependent scaling")
    func checkpointScaling() throws {
        try Device.withDefaultDevice(.cpu) {
            var config = LLMConfig.llama1B
            config.hiddenDim = 64
            config.numHeads = 1
            config.numKVHeads = nil
            config.intermediateSize = 8
            let rope = try #require(TransformerBlock(config: config).rope)
            let input = MLXArray.ones([1, 1, 2, 64])
            let actual = rope(input, offset: 8192).asArray(Float.self)
            var expected = [Double](repeating: 0, count: 128)
            for token in 0..<2 {
                for pair in 0..<32 {
                    let original = pow(500_000.0, -Double(2 * pair) / 64)
                    let wavelength = 2 * Double.pi / original
                    let inverseFrequency: Double
                    if wavelength > 8192 {
                        inverseFrequency = original / 32
                    } else if wavelength < 2048 {
                        inverseFrequency = original
                    } else {
                        let blend = (8192 / wavelength - 1) / 3
                        inverseFrequency = original * (blend + (1 - blend) / 32)
                    }
                    let angle = Double(8192 + token) * inverseFrequency
                    expected[token * 64 + pair] = cos(angle) - sin(angle)
                    expected[token * 64 + pair + 32] = sin(angle) + cos(angle)
                }
            }
            let error = zip(actual, expected).map { abs(Double($0) - $1) }.max()!
            #expect(error < 0.002, "Llama 3.2 rotary maximum absolute error: \(error)")
        }
    }

    private func inverseFrequency(pair: Int, dimensions: Int, base: Double,
                                  scaling: Llama3RoPEScaling) -> Double {
        let frequency = pow(base, -Double(2 * pair) / Double(dimensions))
        let wavelength = 2 * Double.pi / frequency
        let originalContext = Double(scaling.originalContextLength)
        let low = Double(scaling.lowFrequencyFactor)
        let high = Double(scaling.highFrequencyFactor)
        if wavelength >= originalContext / low { return frequency / Double(scaling.factor) }
        if wavelength <= originalContext / high { return frequency }
        let fraction = (originalContext / wavelength - low) / (high - low)
        return frequency * (fraction + (1 - fraction) / Double(scaling.factor))
    }

    @Test("Scaled rotations match a Double oracle across layouts and long positions",
          arguments: [64, 128], [false, true])
    func rotationOracle(dimensions: Int, gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            for scaling in [Llama3RoPEScaling(factor: 1), Llama3RoPEScaling(factor: 8),
                            Llama3RoPEScaling(factor: 32),
                            Llama3RoPEScaling(factor: 4, lowFrequencyFactor: 2,
                                highFrequencyFactor: 8, originalContextLength: 4096)] {
                let factor = scaling.factor
                for traditional in [false, true] {
                    let rope = RoPEEmbedding(dimensions: dimensions, base: 500_000,
                        traditional: traditional, scaling: scaling)
                    #expect(rope.parameters().flattened().isEmpty)
                    for length in [1, 3] {
                        let width = dimensions + 4
                        let shape = [2, 3, length, width]
                        let input = (0..<(2 * 3 * length * width)).map { Float(($0 * 7) % 19 - 9) / 16 }
                        for offset in [0, 1, 8192, 131069] {
                            let output = rope(MLXArray(input, shape), offset: offset).asArray(Float.self)
                            var worstRatio = 0.0
                            for row in 0..<(2 * 3 * length) {
                                let position = offset + row % length
                                for pair in 0..<(dimensions / 2) {
                                    let first = traditional ? 2 * pair : pair
                                    let second = traditional ? first + 1 : pair + dimensions / 2
                                    let a = Double(input[row * width + first])
                                    let b = Double(input[row * width + second])
                                    let angle = Double(position) * inverseFrequency(pair: pair,
                                        dimensions: dimensions, base: 500_000, scaling: scaling)
                                    // Bound Float32 phase rounding separately from the Double trig oracle.
                                    let budget = 4 * Double(Float(angle).ulp) * (abs(a) + abs(b)) + 2e-6
                                    let expectedA = a * cos(angle) - b * sin(angle)
                                    let expectedB = a * sin(angle) + b * cos(angle)
                                    worstRatio = max(worstRatio,
                                        abs(Double(output[row * width + first]) - expectedA) / budget,
                                        abs(Double(output[row * width + second]) - expectedB) / budget)
                                }
                                #expect(Array(output[(row * width + dimensions)..<((row + 1) * width)]) ==
                                    Array(input[(row * width + dimensions)..<((row + 1) * width)]))
                            }
                            #expect(worstRatio <= 1,
                                "dim=\(dimensions), GPU=\(gpu), factor=\(factor), traditional=\(traditional), length=\(length), offset=\(offset), error/budget=\(worstRatio)")
                        }
                    }
                }
            }
        }
    }

    @Test("Unscaled RoPE preserves its MLX implementation and public defaults")
    func standardCompatibility() {
        Device.withDefaultDevice(.cpu) {
            for traditional in [false, true] {
                let module = RoPEEmbedding(dimensions: 8, base: 10_000, scale: 0.5,
                    traditional: traditional)
                let input = MLXArray((0..<32).map { Float($0) / 16 }, [1, 4, 8])
                let expected = MLXNN.RoPE(dimensions: 8, traditional: traditional,
                    base: 10_000, scale: 0.5)(input, offset: 17)
                #expect(module(input, offset: 17).asArray(Float.self) == expected.asArray(Float.self))
                #expect(module.scaling == nil)
            }
            #expect(LLMConfig.llama8B.positionalEncoding == .rope(base: 500_000))
        }
    }

    @Test("Scaled RoPE preserves dtype and zero-position identity")
    func dtypePreservation() {
        Device.withDefaultDevice(.gpu) {
            let rope = RoPEEmbedding(dimensions: 64, base: 500_000,
                scaling: Llama3RoPEScaling(factor: 32))
            for dtype in [DType.float32, .float16, .bfloat16] {
                let input = MLXArray((0..<384).map { Float($0 % 17 - 8) / 16 }, [2, 3, 1, 64]).asType(dtype)
                let output = rope(input)
                #expect(output.dtype == dtype)
                #expect(output.asArray(Float.self) == input.asArray(Float.self))
            }
        }
    }

    @Test("Scaled rotary cached decoding matches a full pass at a long offset", arguments: [false, true])
    func cachedDecode(gpu: Bool) {
        Device.withDefaultDevice(gpu ? .gpu : .cpu) {
            let config = LLMConfig(vocabSize: 12, numLayers: 2, hiddenDim: 64, numHeads: 1,
                intermediateSize: 16, maxSeqLen: 8,
                positionalEncoding: .llama3RoPE(scaling: Llama3RoPEScaling(factor: 32)))
            let model = TransformerDecoder(config: config,
                tokenizer: BPETokenizer(vocab: ["<unk>": 0], merges: []))
            let parameters = model.parameters().flattened().map { name, tensor in
                (name, MLXArray((0..<tensor.size).map { Float(($0 * 3) % 17 - 8) / 32 }, tensor.shape))
            }
            model.update(parameters: NestedDictionary.unflattened(parameters))
            let tokens = (0..<18).map { $0 % 12 }
            let full = model.forward(MLXArray(tokens, [3, 6]), offset: 8192)
            eval(full)
            let caches = model.layers.map { _ in KVCache() }
            var pieces: [MLXArray] = []
            for range in [0..<2, 2..<3, 3..<4, 4..<5, 5..<6] {
                let values = (0..<3).flatMap { row in range.map { tokens[row * 6 + $0] } }
                let output = model.forward(MLXArray(values, [3, range.count]), caches: caches,
                    offset: 8192 + range.lowerBound)
                eval(output)
                pieces.append(output)
                #expect(caches.allSatisfy { $0.count == range.upperBound })
            }
            let error = MLX.abs(concatenated(pieces, axis: 1) - full).max().item(Float.self)
            #expect(error < 1e-4)
        }
    }
}
