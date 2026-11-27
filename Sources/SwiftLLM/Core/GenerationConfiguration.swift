import Foundation

/// Generation parameters and stop configuration parsed from a HuggingFace `generation_config.json`.
public struct GenerationConfiguration: Sendable, Equatable {
    /// End-of-sequence token IDs declared by the checkpoint.
    public var eosTokenIDs: Set<Int>
    /// Beginning-of-sequence token ID, if specified.
    public var bosTokenID: Int?
    /// Padding token ID, if specified.
    public var padTokenID: Int?
    /// Sampling temperature override, if specified.
    public var temperature: Float?
    /// Top-K truncation limit, if specified.
    public var topK: Int?
    /// Nucleus sampling cumulative probability threshold, if specified.
    public var topP: Float?
    /// Multiplicative repetition penalty factor, if specified.
    public var repetitionPenalty: Float?
    /// Default maximum number of new tokens to generate, if specified.
    public var maxNewTokens: Int?
    /// Stop strings configured by the checkpoint.
    public var stopStrings: [String]

    /// Creates a generation configuration with optional sampling and stop defaults.
    /// - Parameters:
    ///   - eosTokenIDs: End-of-sequence token IDs (default: empty).
    ///   - bosTokenID: Optional beginning-of-sequence token ID.
    ///   - padTokenID: Optional padding token ID.
    ///   - temperature: Optional sampling temperature.
    ///   - topK: Optional top-K truncation limit.
    ///   - topP: Optional nucleus sampling threshold.
    ///   - repetitionPenalty: Optional repetition penalty factor.
    ///   - maxNewTokens: Optional maximum new tokens budget.
    ///   - stopStrings: Stop strings configured for text generation (default: empty).
    public init(
        eosTokenIDs: Set<Int> = [],
        bosTokenID: Int? = nil,
        padTokenID: Int? = nil,
        temperature: Float? = nil,
        topK: Int? = nil,
        topP: Float? = nil,
        repetitionPenalty: Float? = nil,
        maxNewTokens: Int? = nil,
        stopStrings: [String] = []
    ) {
        self.eosTokenIDs = eosTokenIDs
        self.bosTokenID = bosTokenID
        self.padTokenID = padTokenID
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxNewTokens = maxNewTokens
        self.stopStrings = stopStrings
    }

    /// Parses a HuggingFace `generation_config.json` payload.
    /// - Parameter data: Raw JSON bytes.
    /// - Throws: ``LLMModelLoadingError/invalidConfig(_:)`` when the payload is not a JSON object.
    public init(huggingFaceGenerationConfigJSON data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            #if os(macOS)
            throw LLMModelLoadingError.invalidConfig("generation_config.json is not a JSON object")
            #else
            throw CocoaError(.coderReadCorrupt)
            #endif
        }

        func number(_ key: String) -> Double? {
            guard let n = root[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
            let d = n.doubleValue
            return d.isFinite ? d : nil
        }

        func integer(_ key: String) -> Int? {
            guard let d = number(key), d.rounded() == d, abs(d) < 1e12 else { return nil }
            return Int(d)
        }

        func float(_ key: String) -> Float? {
            guard let d = number(key) else { return nil }
            let value = Float(d)
            return value.isFinite ? value : nil
        }

        let parsedEOS = Self.parseTokenIDSet(from: root["eos_token_id"])
        let parsedStopStrings: [String]
        if let list = root["stop_strings"] as? [String] {
            parsedStopStrings = list
        } else if let single = root["stop_strings"] as? String {
            parsedStopStrings = [single]
        } else {
            parsedStopStrings = []
        }

        self.init(
            eosTokenIDs: parsedEOS,
            bosTokenID: integer("bos_token_id"),
            padTokenID: integer("pad_token_id"),
            temperature: float("temperature"),
            topK: integer("top_k"),
            topP: float("top_p"),
            repetitionPenalty: float("repetition_penalty"),
            maxNewTokens: integer("max_new_tokens"),
            stopStrings: parsedStopStrings
        )
    }

    /// Loads `generation_config.json` from a checkpoint directory if the file exists.
    /// - Parameter directory: Local checkpoint directory.
    /// - Throws: An error if the file exists but cannot be read or parsed.
    /// - Returns: The parsed `GenerationConfiguration`, or `nil` when `generation_config.json` is absent.
    public static func loadIfPresent(in directory: URL) throws -> GenerationConfiguration? {
        let url = directory.appendingPathComponent("generation_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        return try GenerationConfiguration(huggingFaceGenerationConfigJSON: data)
    }

    /// Builds an ``LLMOptions`` value applying this configuration's non-nil overrides on top of standard defaults.
    /// - Parameter maxTokens: Explicit token limit override; falls back to `maxNewTokens`, then `100`.
    /// - Returns: Configured ``LLMOptions`` instance.
    public func makeLLMOptions(maxTokens: Int? = nil) -> LLMOptions {
        let defaults = SamplingConfiguration()
        let sampling = SamplingConfiguration(
            temperature: temperature ?? defaults.temperature,
            topK: topK ?? defaults.topK,
            topP: topP ?? defaults.topP,
            repetitionPenalty: repetitionPenalty ?? defaults.repetitionPenalty
        )
        return LLMOptions(
            sampling: sampling,
            maxTokens: maxTokens ?? maxNewTokens ?? 100,
            stopSequences: stopStrings,
            stopTokenIDs: eosTokenIDs
        )
    }

    static func parseTokenIDSet(from container: Any?) -> Set<Int> {
        var ids = Set<Int>()
        if let num = container as? NSNumber, CFGetTypeID(num) != CFBooleanGetTypeID() {
            let d = num.doubleValue
            if d.rounded() == d, abs(d) < 1e12 {
                ids.insert(Int(d))
            }
        } else if let many = container as? [Any] {
            for case let value as NSNumber in many where CFGetTypeID(value) != CFBooleanGetTypeID() {
                let d = value.doubleValue
                if d.rounded() == d, abs(d) < 1e12 {
                    ids.insert(Int(d))
                }
            }
        }
        return ids
    }
}
