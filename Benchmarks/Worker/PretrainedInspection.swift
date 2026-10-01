import Foundation
import MLX
import MLXNN
import SwiftSciBenchmarkSupport
import SwiftLLM
import SwiftNLP

extension Worker {
  /// Inspect a verified external checkpoint through the current public model and tokenizer APIs.
  static func inspectLlama(requestURL: URL, responseURL: URL) -> Int32 {
    return Device.withDefaultDevice(.cpu) {
      do {
        guard let request = try JSONSerialization.jsonObject(with: Data(contentsOf: requestURL)) as? [String: Any],
          let path = request["model_path"] as? String,
          let cases = request["cases"] as? [[String: Any]]
        else { throw BenchmarkFailure("Invalid checkpoint inspection request") }
        let directory = URL(fileURLWithPath: path)
        guard let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any],
          raw["model_type"] as? String == "llama"
        else { throw BenchmarkFailure("Expected Llama configuration") }
        func integer(_ key: String) throws -> Int {
          guard let value = raw[key] as? Int, value > 0 else { throw BenchmarkFailure("Missing model config: \(key)") }
          return value
        }
        let hidden = try integer("hidden_size")
        let heads = try integer("num_attention_heads")
        let kvHeads = try integer("num_key_value_heads")
        guard hidden % heads == 0, heads % kvHeads == 0, (hidden / heads) % 2 == 0,
          let epsilon = raw["rms_norm_eps"] as? NSNumber,
          epsilon.floatValue.isFinite, epsilon.floatValue > 0,
          let theta = raw["rope_theta"] as? NSNumber,
          theta.floatValue.isFinite, theta.floatValue > 0,
          let tied = raw["tie_word_embeddings"] as? Bool
        else { throw BenchmarkFailure("Invalid attention, normalization, rotary base, or output-head configuration") }
        guard raw["hidden_act"] as? String == "silu",
          raw["attention_bias"] as? Bool == false,
          raw["mlp_bias"] as? Bool == false,
          raw["head_dim"] as? Int == hidden / heads
        else { throw BenchmarkFailure("Expected SiLU, bias-free attention and MLP, and matching head dimensions") }
        let tensors = try SafeTensorsParser.parse(url: directory.appendingPathComponent("model.safetensors"))
        guard let rawScaling = raw["rope_scaling"] as? [String: Any],
          rawScaling["rope_type"] as? String == "llama3",
          let factor = rawScaling["factor"] as? NSNumber,
          let low = rawScaling["low_freq_factor"] as? NSNumber,
          let high = rawScaling["high_freq_factor"] as? NSNumber,
          let original = rawScaling["original_max_position_embeddings"] as? Int,
          factor.floatValue.isFinite, factor.floatValue >= 1,
          low.floatValue.isFinite, low.floatValue > 0,
          high.floatValue.isFinite, high.floatValue > low.floatValue, original > 0
        else { throw BenchmarkFailure("Expected valid Llama 3 rotary scaling") }
        let scaling = Llama3RoPEScaling(factor: factor.floatValue,
          lowFrequencyFactor: low.floatValue, highFrequencyFactor: high.floatValue,
          originalContextLength: original)
        let config = try LLMConfig(
          vocabSize: integer("vocab_size"), numLayers: integer("num_hidden_layers"),
          hiddenDim: hidden, numHeads: heads, numKVHeads: kvHeads,
          intermediateSize: integer("intermediate_size"),
          maxSeqLen: 8, rmsNormEps: epsilon.floatValue,
          positionalEncoding: .llama3RoPE(base: theta.floatValue,
                                       scaling: scaling),
          tieWordEmbeddings: tied)
        let tokenizer = try BPETokenizer(llama3TokenizerJSON:
          Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
        let model = TransformerDecoder(config: config, tokenizer: tokenizer)
        let parameters: [String: MLXArray] = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var findings = [[String: Any]]()
        let suffixes = [
          "attention.query_proj.weight": "self_attn.q_proj.weight",
          "attention.key_proj.weight": "self_attn.k_proj.weight",
          "attention.value_proj.weight": "self_attn.v_proj.weight",
          "attention.out_proj.weight": "self_attn.o_proj.weight",
          "ffn.gate.weight": "mlp.gate_proj.weight",
          "ffn.up.weight": "mlp.up_proj.weight",
          "ffn.down.weight": "mlp.down_proj.weight",
          "norm1.weight": "input_layernorm.weight",
          "norm2.weight": "post_attention_layernorm.weight"
        ]
        var expected: [String: [Int]] = [:]
        for (name, parameter) in parameters {
          let key: String
          switch name {
          case "posEmbedding.weight": continue
          case "embedding.weight": key = "model.embed_tokens.weight"
          case "finalNorm.weight": key = "model.norm.weight"
          case "lmHead.weight": key = "lm_head.weight"
          default:
            let parts = name.split(separator: ".", maxSplits: 2).map(String.init)
            guard parts.count == 3, parts[0] == "layers", let suffix = suffixes[parts[2]]
            else { throw BenchmarkFailure("Unmapped public model parameter: \(name)") }
            key = "model.layers.\(parts[1]).\(suffix)"
          }
          expected[key] = parameter.shape
        }
        for key in Set(expected.keys).subtracting(tensors.keys).sorted() {
          findings.append(["id": "missing-tensor", "key": key])
        }
        for key in Set(tensors.keys).subtracting(expected.keys).sorted() {
          findings.append(["id": "unexpected-tensor", "key": key])
        }
        for (key, shape) in expected.sorted(by: { $0.key < $1.key }) {
          if let tensor = tensors[key], tensor.shape != shape {
            findings.append(["id": "tensor-shape", "key": key,
                             "checkpoint_shape": tensor.shape, "public_model_shape": shape])
          }
        }
        var tokenResults = [[String: Any]]()
        for item in cases {
          guard let id = item["id"] as? String, let text = item["text"] as? String,
            let expected = item["raw_tokens"] as? [Int]
          else { throw BenchmarkFailure("Invalid reference token case") }
          let actual = tokenizer.encode(text: text)
          let decoded = tokenizer.decode(tokens: actual)
          tokenResults.append(["id": id, "expected": expected, "actual": actual, "equal": actual == expected,
                               "decoded_actual": decoded, "decoded_equal": Array(decoded.utf8) == Array(text.utf8)])
        }
        if tokenResults.contains(where: { $0["equal"] as? Bool != true || $0["decoded_equal"] as? Bool != true }) {
          findings.append(["id": "tokenizer-parity", "detail": "Public BPETokenizer disagrees with the pinned checkpoint tokenizer"])
        }
        var weightLoadingVerified = false
        if findings.isEmpty {
          let missing = model.loadWeights(tensors)
          if missing.isEmpty { weightLoadingVerified = true }
          else { findings.append(["id": "missing-weights", "keys": missing]) }
        }
        let response: [String: Any] = [
          "schema_version": 1, "purpose": "pretrained-integration-preflight",
          "status": findings.isEmpty ? "preflight-passed" : "blocked",
          "safetensors_parsed": true, "tensor_count": tensors.count,
          "expected_tensor_count": expected.count,
          "inspection_model_max_sequence": 8,
          "generation_executed": false, "inspection_device": "cpu",
          "weight_loading_verified": weightLoadingVerified,
          "rotary_scaling": rawScaling, "findings": findings, "tokenizer_cases": tokenResults,
          "limitations": ["Inspection model uses eight positions to bound unused learned-position allocation.",
                          "Weights are installed only after the configuration and tokenizer checks pass.",
                          "Passing preflight is not successful generation or numerical conformance."]
        ]
        let data = try JSONSerialization.data(withJSONObject: response, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: responseURL, options: .atomic)
        return findings.isEmpty ? 0 : 1
      } catch {
        let response = ["status": "failed", "error": String(describing: error)]
        if let data = try? JSONSerialization.data(withJSONObject: response) { try? data.write(to: responseURL) }
        return 2
      }
    }
  }
}
