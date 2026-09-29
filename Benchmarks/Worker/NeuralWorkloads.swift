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

  struct Config {
    let vocab: Int
    let hidden: Int
    let heads: Int
    let intermediate: Int
    let maxLength: Int
    static let tiny = Config(vocab: 7, hidden: 8, heads: 2, intermediate: 6, maxLength: 8)

    var shapes: [String: [Int]] {
      [
        "embedding.weight": [vocab, hidden], "posEmbedding.weight": [maxLength, hidden],
        "finalNorm.weight": [hidden], "lmHead.weight": [vocab, hidden],
        "layers.0.norm1.weight": [hidden], "layers.0.norm2.weight": [hidden],
        "layers.0.attention.query_proj.weight": [hidden, hidden],
        "layers.0.attention.key_proj.weight": [hidden, hidden],
        "layers.0.attention.value_proj.weight": [hidden, hidden],
        "layers.0.attention.out_proj.weight": [hidden, hidden],
        "layers.0.ffn.gate.weight": [intermediate, hidden],
        "layers.0.ffn.up.weight": [intermediate, hidden],
        "layers.0.ffn.down.weight": [hidden, intermediate],
      ]
    }
  }

  let config: Config
  let chunkSizes: [Int]
  let device: String
  let position: String
  let execution: String
  let loading: String
  let tokens: [[Int32]]
  let weights: [String: Weight]

  static let shapes = Config.tiny.shapes

  private init(
    device: String, position: String, execution: String, loading: String,
    tokens: [[Int32]], weights: [String: Weight], config: Config, chunkSizes: [Int]
  ) {
    self.config = config
    self.chunkSizes = chunkSizes
    self.device = device
    self.position = position
    self.execution = execution
    self.loading = loading
    self.tokens = tokens
    self.weights = weights
  }

  static func decode(_ data: Data, rows: Int) throws -> FixedDecoderInput {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let operation = object["operation"] as? String,
      ["decoder-fixed-f32", "decoder-shaped-f32"].contains(operation)
    else { throw BenchmarkFailure("Invalid decoder operation") }
    let shaped = operation == "decoder-shaped-f32"
    var config = Config.tiny
    if shaped {
      guard let raw = object["config"] as? [String: Any],
        Set(raw.keys) == ["vocab_size", "hidden_dim", "num_heads", "intermediate_size", "max_seq_len"]
      else { throw BenchmarkFailure("Invalid decoder config") }
      config = try Config(
        vocab: boundedInteger(raw["vocab_size"]!, range: 1...32),
        hidden: boundedInteger(raw["hidden_dim"]!, range: 1...32),
        heads: boundedInteger(raw["num_heads"]!, range: 1...4),
        intermediate: boundedInteger(raw["intermediate_size"]!, range: 1...64),
        maxLength: boundedInteger(raw["max_seq_len"]!, range: 1...128))
      guard config.hidden % config.heads == 0, (config.hidden / config.heads) % 2 == 0 else {
        throw BenchmarkFailure("Decoder head width must be even")
      }
    }
    let shapes = config.shapes
    let keys: Set<String> = ["operation", "device", "position", "execution", "loading", "tokens", "weights"]
    guard Set(object.keys) == (shaped ? keys.union(["config", "chunk_sizes"]) : keys),
      let device = object["device"] as? String, ["cpu", "gpu"].contains(device),
      let position = object["position"] as? String, ["learned", "rope"].contains(position),
      let execution = object["execution"] as? String, ["full", "cached"].contains(execution),
      let loading = object["loading"] as? String, ["direct", "public-loader"].contains(loading),
      !shaped || loading == "direct",
      let rawTokens = object["tokens"] as? [[Any]], (1...2).contains(rawTokens.count),
      let sequence = rawTokens.first?.count, (1...(shaped ? config.maxLength : 4)).contains(sequence),
      rawTokens.allSatisfy({ $0.count == sequence }),
      rows == rawTokens.count * sequence,
      shaped || execution != "cached" || sequence == 4,
      let rawWeights = object["weights"] as? [String: Any],
      Set(rawWeights.keys) == Set(shapes.keys)
    else { throw BenchmarkFailure("Invalid fixed decoder input contract") }

    var chunkSizes = execution == "cached" ? [2, 1, 1] : []
    if shaped {
      guard let chunks = object["chunk_sizes"] as? [Any] else {
        throw BenchmarkFailure("Invalid decoder chunks")
      }
      chunkSizes = try chunks.map { try boundedInteger($0, range: 1...128) }
      guard (execution == "full" && chunkSizes.isEmpty)
        || (execution == "cached" && chunkSizes.count >= 2 && chunkSizes.reduce(0, +) == sequence
          && chunkSizes.dropFirst().allSatisfy({ $0 == 1 }))
      else { throw BenchmarkFailure("Cache requires a prefix followed by single tokens") }
    }
    let tokens = try rawTokens.map { row in
      try row.map { value -> Int32 in
        let integer = try boundedInteger(value, range: 0...(config.vocab - 1))
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
      let shape = try dimensions.map { try boundedInteger($0, range: 1...128) }
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
      tokens: tokens, weights: weights, config: config, chunkSizes: chunkSizes)
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
          vocabSize: input.config.vocab, numLayers: 1, hiddenDim: input.config.hidden,
          numHeads: input.config.heads, intermediateSize: input.config.intermediate,
          maxSeqLen: input.config.maxLength, rmsNormEps: 0.0009765625,
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
          var start = 0
          for size in input.chunkSizes {
            let range = start..<(start + size)
            let tokens = MLXArray(input.tokens.flatMap { Array($0[range]) }, [batch, range.count])
            let chunk = model.forward(tokens, caches: [cache], offset: range.lowerBound)
            guard chunk.dtype == .float32, chunk.shape == [batch, range.count, input.config.vocab] else {
              throw BenchmarkFailure("Invalid cached decoder output shape or dtype")
            }
            eval(chunk)
            chunks.append(chunk)
            start += size
            guard cache.count == start else { throw BenchmarkFailure("Invalid decoder cache length") }
            counts.append(Double(cache.count))
          }
          logits = concatenated(chunks, axis: 1)
        }
        guard logits.dtype == .float32, logits.shape == [batch, sequence, input.config.vocab] else {
          throw BenchmarkFailure("Invalid decoder output shape or dtype")
        }
        eval(logits)
        StreamOrDevice.default.stream.synchronize()
        let values = logits.asArray(Float.self)
        guard values.count == batch * sequence * input.config.vocab, values.allSatisfy(\.isFinite) else {
          throw BenchmarkFailure("Invalid decoder output values")
        }
        var result = [Double(batch), Double(sequence), Double(input.config.vocab)] + values.map(Double.init)
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
