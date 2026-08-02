#if os(macOS)
import Foundation
import MLX

/// Key-Value cache storing previous Attention Keys and Values per layer for O(N) generation.
public final class KVCache: @unchecked Sendable {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    
    /// Creates an empty key-value attention cache instance for an autoregressive decoder layer.
    public init() {}
    
    /// Updates the cache with new Keys and Values for the current step.
    /// - Parameters:
    ///   - newKeys: New Key tensor, shape `[batch, 1, num_heads, head_dim]`.
    ///   - newValues: New Value tensor, shape `[batch, 1, num_heads, head_dim]`.
    /// - Returns: Accumulated (Key, Value) tensors over all past steps.
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        if let existingK = keys, let existingV = values {
            let updatedK = concatenated([existingK, newKeys], axis: 1)
            let updatedV = concatenated([existingV, newValues], axis: 1)
            self.keys = updatedK
            self.values = updatedV
            return (updatedK, updatedV)
        } else {
            self.keys = newKeys
            self.values = newValues
            return (newKeys, newValues)
        }
    }
    
    /// Resets the cache state for new prompt generation.
    public func reset() {
        self.keys = nil
        self.values = nil
    }
}
#endif
