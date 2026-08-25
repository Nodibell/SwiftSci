#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftLLM

@Suite("Paged KV-Cache Allocator Tests")
struct PagedKVCacheTests {

    @Test("PagedKVCache allocates pages dynamically across token steps")
    func testPagedAllocation() throws {
        let pageSize = 4
        let numHeads = 2
        let headDim = 8
        let cache = PagedKVCache(pageSize: pageSize, numHeads: numHeads, headDim: headDim, numLayers: 1)

        let seqId = 1
        // Append 6 tokens (exceeds pageSize = 4 -> requires 2 pages)
        for i in 1...6 {
            let key = MLXArray([Float](repeating: Float(i), count: numHeads * headDim), [1, 1, numHeads, headDim])
            let val = MLXArray([Float](repeating: Float(i * 10), count: numHeads * headDim), [1, 1, numHeads, headDim])
            cache.append(sequenceId: seqId, layer: 0, key: key, value: val)
        }

        #expect(cache.totalAllocatedPages == 2)

        let (keys, values) = cache.getKeysAndValues(sequenceId: seqId, layer: 0)
        #expect(keys.shape == [1, 6, numHeads, headDim])
        #expect(values.shape == [1, 6, numHeads, headDim])

        cache.reset(sequenceId: seqId)
        let (resetKeys, _) = cache.getKeysAndValues(sequenceId: seqId, layer: 0)
        #expect(resetKeys.shape == [1, 0, numHeads, headDim])
    }
}
#endif
