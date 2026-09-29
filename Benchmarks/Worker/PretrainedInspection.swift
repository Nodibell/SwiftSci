import Foundation
import MLX
import MLXNN
import SwiftSciBenchmarkSupport
import SwiftLLM
import SwiftNLP

extension Worker {
  /// Inspect a verified external checkpoint without bypassing incompatible public APIs.
  static func inspectLlama(requestURL: URL, responseURL: URL) -> Int32 {
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
      let tensors = try SafeTensorsParser.parse(url: directory.appendingPathComponent("model.safetensors"))
      let config = try LLMConfig(
        vocabSize: integer("vocab_size"), numLayers: integer("num_hidden_layers"),
        hiddenDim: hidden, numHeads: heads, intermediateSize: integer("intermediate_size"),
        maxSeqLen: 8, rmsNormEps: (raw["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5,
        positionalEncoding: .rope(base: (raw["rope_theta"] as? NSNumber)?.floatValue ?? 10000))
      guard let tokenJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [String: Any],
        let tokenModel = tokenJSON["model"] as? [String: Any],
        var vocab = tokenModel["vocab"] as? [String: Int], let rawMerges = tokenModel["merges"] as? [Any]
      else { throw BenchmarkFailure("Invalid BPE vocabulary") }
      for entry in tokenJSON["added_tokens"] as? [[String: Any]] ?? [] {
        if let text = entry["content"] as? String, let id = entry["id"] as? Int { vocab[text] = id }
      }
      let merges = try rawMerges.map { value -> String in
        if let text = value as? String { return text }
        if let pair = value as? [String], pair.count == 2 { return pair.joined(separator: " ") }
        throw BenchmarkFailure("Invalid BPE merge")
      }
      let tokenizer = BPETokenizer(vocab: vocab, merges: merges)
      let model = TransformerDecoder(config: config, tokenizer: tokenizer)
      let parameters: [String: MLXArray] = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
      var findings = [[String: Any]]()
      guard let actualKey = tensors["model.layers.0.self_attn.k_proj.weight"],
        let expectedKey = parameters["layers.0.attention.key_proj.weight"]
      else { throw BenchmarkFailure("Missing key projection for inspection") }
      if actualKey.shape != expectedKey.shape {
        findings.append(["id": "grouped-query-attention", "checkpoint_key_shape": actualKey.shape,
                         "public_model_key_shape": expectedKey.shape, "attention_heads": heads, "kv_heads": kvHeads])
      }
      if let scaling = raw["rope_scaling"] as? [String: Any] {
        findings.append(["id": "rotary-scaling", "checkpoint": scaling,
                         "public_config": "PositionalEncodingScheme.rope exposes base only"])
      }
      if raw["tie_word_embeddings"] as? Bool == true && tensors["lm_head.weight"] == nil {
        findings.append(["id": "tied-output-head", "checkpoint_has_lm_head": false,
                         "public_model_has_separate_head": parameters["lmHead.weight"] != nil])
      }
      var tokenResults = [[String: Any]]()
      for item in cases {
        guard let id = item["id"] as? String, let text = item["text"] as? String,
          let expected = item["raw_tokens"] as? [Int]
        else { throw BenchmarkFailure("Invalid reference token case") }
        let actual = tokenizer.encode(text: text)
        tokenResults.append(["id": id, "expected": expected, "actual": actual, "equal": actual == expected,
                             "decoded_actual": tokenizer.decode(tokens: actual)])
      }
      if tokenResults.contains(where: { $0["equal"] as? Bool == false }) {
        findings.append(["id": "tokenizer-parity", "detail": "Public BPETokenizer disagrees with the pinned checkpoint tokenizer"])
      }
      let response: [String: Any] = [
        "schema_version": 1, "purpose": "pretrained-integration-preflight",
        "status": findings.isEmpty ? "preflight-passed" : "blocked",
        "safetensors_parsed": true, "tensor_count": tensors.count,
        "key_dtype": String(describing: actualKey.dtype),
        "inspection_model_max_sequence": 8,
        "generation_executed": false, "findings": findings, "tokenizer_cases": tokenResults,
        "limitations": ["Inspection model uses eight positions to bound unused learned-position allocation.",
                        "No checkpoint weights are installed into an incompatible model.",
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
