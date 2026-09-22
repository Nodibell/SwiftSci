#if os(macOS)
import Foundation
import MLX

/// Key-Value cache storing previous Attention Keys and Values per layer for O(N) generation.
public final class KVCache: @unchecked Sendable {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    
    /// Number of cached token steps along the sequence dimension.
    public var count: Int {
        guard let k = keys else { return 0 }
        let seqAxis = k.ndim == 4 ? 2 : (k.ndim > 1 ? 1 : 0)
        return k.dim(seqAxis)
    }

    /// Creates an empty key-value attention cache instance for an autoregressive decoder layer.
    public init() {}
    
    /// Updates the cache with new Keys and Values for the current step.
    /// - Parameters:
    ///   - newKeys: New Key tensor.
    ///   - newValues: New Value tensor.
    ///   - axis: Optional explicit concatenation axis (defaults to 2 for 4D attention tensors, 1 for 2D).
    /// - Returns: Accumulated (Key, Value) tensors over all past steps.
    public func update(
        keys newKeys: MLXArray,
        values newValues: MLXArray,
        axis: Int? = nil
    ) -> (keys: MLXArray, values: MLXArray) {
        let concatAxis = axis ?? (newKeys.ndim == 4 ? 2 : 1)
        if let existingK = keys, let existingV = values {
            let updatedK = concatenated([existingK, newKeys], axis: concatAxis)
            let updatedV = concatenated([existingV, newValues], axis: concatAxis)
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
