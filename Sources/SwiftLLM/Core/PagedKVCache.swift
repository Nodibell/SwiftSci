#if os(macOS)
import Foundation
import MLX

/// A dynamic paged Key-Value attention cache allocator for high-throughput batch and stream inference.
///
/// `PagedKVCache` organizes Key-Value tensor storage into fixed-size physical memory pages (blocks),
/// eliminating frequent buffer re-allocations and memory fragmentation in unified memory.
public final class PagedKVCache: @unchecked Sendable {
    /// Number of token slots per page block.
    public let pageSize: Int
    /// Dimensionality of each attention head.
    public let headDim: Int
    /// Number of attention heads.
    public let numHeads: Int
    /// Number of transformer layers.
    public let numLayers: Int

    private var physicalKeyPages: [[[MLXArray]]] // [layer][pageIdx][slotIdx]
    private var physicalValuePages: [[[MLXArray]]] // [layer][pageIdx][slotIdx]
    private var sequenceBlockTables: [Int: [Int]] // sequenceId -> [pageIdx]
    private var sequenceLengths: [Int: Int] // sequenceId -> length

    /// Initializes a paged KV-cache allocator.
    ///
    /// - Parameters:
    ///   - pageSize: Number of tokens per page block (default: 16).
    ///   - numHeads: Number of attention heads.
    ///   - headDim: Dimension of each attention head.
    ///   - numLayers: Number of transformer layers.
    public init(
        pageSize: Int = 16,
        numHeads: Int = 4,
        headDim: Int = 32,
        numLayers: Int = 1
    ) {
        self.pageSize = pageSize
        self.numHeads = numHeads
        self.headDim = headDim
        self.numLayers = numLayers

        self.physicalKeyPages = Array(repeating: [], count: numLayers)
        self.physicalValuePages = Array(repeating: [], count: numLayers)
        self.sequenceBlockTables = [:]
        self.sequenceLengths = [:]
    }

    /// Appends single-step Key and Value tokens for a sequence and layer into the paged memory pool.
    ///
    /// - Parameters:
    ///   - sequenceId: Identifier of the sequence.
    ///   - layer: The layer index.
    ///   - key: New Key tensor of shape `[1, 1, numHeads, headDim]` or `[numHeads, headDim]`.
    ///   - value: New Value tensor of shape `[1, 1, numHeads, headDim]` or `[numHeads, headDim]`.
    public func append(sequenceId: Int, layer: Int, key: MLXArray, value: MLXArray) {
        let currentLen = sequenceLengths[sequenceId, default: 0]
        var blockTable = sequenceBlockTables[sequenceId, default: []]

        let pageIndexInSeq = currentLen / pageSize
        let offsetInPage = currentLen % pageSize

        if pageIndexInSeq >= blockTable.count {
            // Allocate new physical page for all layers
            let newPageIdx = physicalKeyPages[layer].count
            let emptyPageK = (0..<pageSize).map { _ in MLX.zeros([numHeads, headDim]) }
            let emptyPageV = (0..<pageSize).map { _ in MLX.zeros([numHeads, headDim]) }
            physicalKeyPages[layer].append(emptyPageK)
            physicalValuePages[layer].append(emptyPageV)
            blockTable.append(newPageIdx)
            sequenceBlockTables[sequenceId] = blockTable
        }

        let physicalPageIdx = blockTable[pageIndexInSeq]
        let reshapedK = key.reshaped([numHeads, headDim])
        let reshapedV = value.reshaped([numHeads, headDim])

        physicalKeyPages[layer][physicalPageIdx][offsetInPage] = reshapedK
        physicalValuePages[layer][physicalPageIdx][offsetInPage] = reshapedV

        if layer == numLayers - 1 {
            sequenceLengths[sequenceId] = currentLen + 1
        }
    }

    /// Retrieves concatenated Keys and Values for all past tokens of a sequence at a specified layer.
    ///
    /// - Parameters:
    ///   - sequenceId: Identifier of the sequence.
    ///   - layer: The layer index.
    /// - Returns: Accumulated (Keys, Values) of shape `[1, seqLen, numHeads, headDim]`.
    public func getKeysAndValues(sequenceId: Int, layer: Int) -> (keys: MLXArray, values: MLXArray) {
        let totalLen = sequenceLengths[sequenceId, default: 0]
        guard totalLen > 0 else {
            let empty = MLX.zeros([1, 0, numHeads, headDim])
            return (empty, empty)
        }

        guard let blockTable = sequenceBlockTables[sequenceId], !blockTable.isEmpty else {
            let empty = MLX.zeros([1, 0, numHeads, headDim])
            return (empty, empty)
        }

        var keySlots: [MLXArray] = []
        var valSlots: [MLXArray] = []

        var remaining = totalLen
        for physIdx in blockTable {
            let countInThisPage = min(remaining, pageSize)
            for slot in 0..<countInThisPage {
                keySlots.append(physicalKeyPages[layer][physIdx][slot])
                valSlots.append(physicalValuePages[layer][physIdx][slot])
            }
            remaining -= countInThisPage
            if remaining <= 0 { break }
        }

        let keys = MLX.stacked(keySlots, axis: 0).expandedDimensions(axis: 0)
        let vals = MLX.stacked(valSlots, axis: 0).expandedDimensions(axis: 0)
        return (keys, vals)
    }

    /// Resets the memory allocation table for a sequence.
    ///
    /// - Parameter sequenceId: Identifier of the sequence to free.
    public func reset(sequenceId: Int) {
        sequenceBlockTables.removeValue(forKey: sequenceId)
        sequenceLengths.removeValue(forKey: sequenceId)
    }

    /// Total number of physical page blocks allocated across all layers.
    public var totalAllocatedPages: Int {
        physicalKeyPages.reduce(0) { $0 + $1.count }
    }
}
#endif
