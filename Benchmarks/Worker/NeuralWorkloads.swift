import CoreFoundation
import Foundation
import MLX
import MLXNN
import SwiftLLM
import SwiftNLP
import SwiftSciBenchmarkSupport

struct FixedDecoderInput {
  struct Weight {
    let shape: [Int]
    let values: [Float]
  }

  let device: String
  let position: String
  let execution: String
  let loading: String
  let tokens: [[Int32]]
  let weights: [String: Weight]

  static let shapes: [String: [Int]] = [
    "embedding.weight": [7, 8],
    "posEmbedding.weight": [8, 8],
    "finalNorm.weight": [8],
    "lmHead.weight": [7, 8],
    "layers.0.norm1.weight": [8],
    "layers.0.norm2.weight": [8],
    "layers.0.attention.query_proj.weight": [8, 8],
    "layers.0.attention.key_proj.weight": [8, 8],
    "layers.0.attention.value_proj.weight": [8, 8],
    "layers.0.attention.out_proj.weight": [8, 8],
    "layers.0.ffn.gate.weight": [6, 8],
    "layers.0.ffn.up.weight": [6, 8],
    "layers.0.ffn.down.weight": [8, 6],
  ]

  private init(
    device: String, position: String, execution: String, loading: String,
    tokens: [[Int32]], weights: [String: Weight]
  ) {
    self.device = device
    self.position = position
    self.execution = execution
    self.loading = loading
    self.tokens = tokens
    self.weights = weights
  }

  static func decode(_ data: Data, rows: Int) throws -> FixedDecoderInput {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == ["operation", "device", "position", "execution", "loading", "tokens", "weights"],
      object["operation"] as? String == "decoder-fixed-f32",
      let device = object["device"] as? String, ["cpu", "gpu"].contains(device),
      let position = object["position"] as? String, ["learned", "rope"].contains(position),
      let execution = object["execution"] as? String, ["full", "cached"].contains(execution),
      let loading = object["loading"] as? String, ["direct", "public-loader"].contains(loading),
      let rawTokens = object["tokens"] as? [[Any]], (1...2).contains(rawTokens.count),
      let sequence = rawTokens.first?.count, (1...4).contains(sequence),
      rawTokens.allSatisfy({ $0.count == sequence }),
      rows == rawTokens.count * sequence,
      execution != "cached" || sequence == 4,
      let rawWeights = object["weights"] as? [String: Any],
      Set(rawWeights.keys) == Set(shapes.keys)
    else { throw BenchmarkFailure("Invalid fixed decoder input contract") }

    let tokens = try rawTokens.map { row in
      try row.map { value -> Int32 in
        let integer = try boundedInteger(value, range: 0...6)
        return Int32(integer)
      }
    }
    var weights = [String: Weight]()
    for key in shapes.keys.sorted() {
      guard let expectedShape = shapes[key],
        let entry = rawWeights[key] as? [String: Any], Set(entry.keys) == ["shape", "values"],
        let dimensions = entry["shape"] as? [Any], dimensions.count == expectedShape.count,
        let values = entry["values"] as? [Any], values.count == expectedShape.reduce(1, *)
      else { throw BenchmarkFailure("Invalid decoder weight structure: \(key)") }
      let shape = try dimensions.map { try boundedInteger($0, range: 1...8) }
      guard shape == expectedShape else { throw BenchmarkFailure("Invalid decoder weight shape: \(key)") }
      let floats = try values.map { value -> Float in
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
          throw BenchmarkFailure("Decoder weights must be numbers: \(key)")
        }
        let double = number.doubleValue
        let float = Float(double)
        guard double.isFinite, abs(double) <= 2, float.isFinite, Double(float) == double else {
          throw BenchmarkFailure("Decoder weight is not bounded exact Float32: \(key)")
        }
        return float
      }
      weights[key] = Weight(shape: shape, values: floats)
    }
    return FixedDecoderInput(
      device: device, position: position, execution: execution, loading: loading,
      tokens: tokens, weights: weights)
  }

  private static func boundedInteger(_ value: Any, range: ClosedRange<Int>) throws -> Int {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
      !["d", "f"].contains(String(cString: number.objCType)),
      number.doubleValue >= Double(range.lowerBound), number.doubleValue <= Double(range.upperBound),
      number.doubleValue.rounded(.towardZero) == number.doubleValue
    else { throw BenchmarkFailure("Decoder dimensions and token IDs must be bounded integers") }
    return number.intValue
  }
}

extension Worker {
  @inline(never) static func executeFixedDecoder(_ input: FixedDecoderInput) throws -> Output {
    let device: Device = input.device == "cpu" ? .cpu : .gpu
    return try Device.withDefaultDevice(device) {
      try Stream.withNewDefaultStream(device: device) {
        let config = LLMConfig(
          vocabSize: 7, numLayers: 1, hiddenDim: 8, numHeads: 2, intermediateSize: 6,
          maxSeqLen: 8, rmsNormEps: 0.0009765625,
          positionalEncoding: input.position == "learned" ? .learned : .rope(base: 10000))
        let tokenizer = BPETokenizer(vocab: ["<unk>": 0], merges: [])
        let model = TransformerDecoder(config: config, tokenizer: tokenizer)
        let tensors = input.weights.mapValues { MLXArray($0.values, $0.shape) }
        if input.loading == "direct" {
          try model.update(
            parameters: NestedDictionary.unflattened(tensors),
            verify: [.noUnusedKeys, .allModelKeysSet, .shapeMismatch])
        } else {
          let mapping = [
            "embedding.weight": "model.embed_tokens.weight",
            "posEmbedding.weight": "model.pos_embed.weight",
            "finalNorm.weight": "model.norm.weight",
            "lmHead.weight": "lm_head.weight",
            "layers.0.norm1.weight": "model.layers.0.input_layernorm.weight",
            "layers.0.norm2.weight": "model.layers.0.post_attention_layernorm.weight",
            "layers.0.attention.query_proj.weight": "model.layers.0.self_attn.q_proj.weight",
            "layers.0.attention.key_proj.weight": "model.layers.0.self_attn.k_proj.weight",
            "layers.0.attention.value_proj.weight": "model.layers.0.self_attn.v_proj.weight",
            "layers.0.attention.out_proj.weight": "model.layers.0.self_attn.o_proj.weight",
            "layers.0.ffn.gate.weight": "model.layers.0.mlp.gate_proj.weight",
            "layers.0.ffn.up.weight": "model.layers.0.mlp.up_proj.weight",
            "layers.0.ffn.down.weight": "model.layers.0.mlp.down_proj.weight",
          ]
          let external = Dictionary(uniqueKeysWithValues: mapping.map { ($0.value, tensors[$0.key]!) })
          let missing = model.loadWeights(external)
          guard missing.isEmpty else { throw BenchmarkFailure("Decoder loader reported missing weights: \(missing)") }
        }
        try verifyDecoderWeights(model, expected: input.weights)
        model.train(false)
        let batch = input.tokens.count
        let sequence = input.tokens[0].count
        let logits: MLXArray
        var counts = [Double]()
        if input.execution == "full" {
          let tokens = MLXArray(input.tokens.flatMap { $0 }, [batch, sequence])
          logits = model.forward(tokens)
        } else {
          let cache = KVCache()
          var chunks = [MLXArray]()
          for range in [0..<2, 2..<3, 3..<4] {
            let tokens = MLXArray(input.tokens.flatMap { Array($0[range]) }, [batch, range.count])
            let chunk = model.forward(tokens, caches: [cache], offset: range.lowerBound)
            guard chunk.dtype == .float32, chunk.shape == [batch, range.count, 7] else {
              throw BenchmarkFailure("Invalid cached decoder output shape or dtype")
            }
            eval(chunk)
            chunks.append(chunk)
            counts.append(Double(cache.count))
          }
          guard counts == [2, 3, 4] else { throw BenchmarkFailure("Invalid decoder cache lengths") }
          logits = concatenated(chunks, axis: 1)
        }
        guard logits.dtype == .float32, logits.shape == [batch, sequence, 7] else {
          throw BenchmarkFailure("Invalid decoder output shape or dtype")
        }
        eval(logits)
        StreamOrDevice.default.stream.synchronize()
        let values = logits.asArray(Float.self)
        guard values.count == batch * sequence * 7, values.allSatisfy(\.isFinite) else {
          throw BenchmarkFailure("Invalid decoder output values")
        }
        var result = [Double(batch), Double(sequence), 7] + values.map(Double.init)
        if input.execution == "cached" { result += [Double(counts.count)] + counts }
        return .values(result)
      }
    }
  }

  private static func verifyDecoderWeights(
    _ model: TransformerDecoder, expected: [String: FixedDecoderInput.Weight]
  ) throws {
    let parameters = model.parameters().flattened()
    guard parameters.count == expected.count, Set(parameters.map(\.0)) == Set(expected.keys) else {
      throw BenchmarkFailure("Decoder parameter keys differ from the complete supplied weights")
    }
    for (key, tensor) in parameters.sorted(by: { $0.0 < $1.0 }) {
      guard let weight = expected[key], tensor.shape == weight.shape, tensor.dtype == .float32 else {
        throw BenchmarkFailure("Decoder parameter shape or dtype differs: \(key)")
      }
      eval(tensor)
      StreamOrDevice.default.stream.synchronize()
      let actual = tensor.asArray(Float.self)
      guard actual.count == weight.values.count,
        zip(actual, weight.values).allSatisfy({ $0.bitPattern == $1.bitPattern })
      else { throw BenchmarkFailure("Decoder parameter was not replaced exactly: \(key)") }
    }
  }
}
