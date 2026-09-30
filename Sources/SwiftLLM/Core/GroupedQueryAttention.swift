#if os(macOS)
import MLX
import MLXNN

/// Keeps the public attention type and parameter paths while narrowing K/V projections.
final class GroupedQueryAttention: MultiHeadAttention {
    private let numKVHeads: Int

    init(dimensions: Int, numHeads: Int, numKVHeads: Int) {
        self.numKVHeads = numKVHeads
        super.init(dimensions: dimensions, numHeads: numHeads)
        let kvDimensions = dimensions / numHeads * numKVHeads
        let projections: [String: Module] = [
            "key_proj": Linear(dimensions, kvDimensions, bias: false),
            "value_proj": Linear(dimensions, kvDimensions, bias: false)
        ]
        // MLX caches module metadata; replace registered children through its update API.
        update(modules: NestedDictionary.unflattened(projections))
    }

    override func callAsFunction(
        _ queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let q = unflatten(queryProjection(queries), axis: -1, shape: [numHeads, -1])
            .transposed(0, 2, 1, 3)
        let k = unflatten(keyProjection(keys), axis: -1, shape: [numKVHeads, -1])
            .transposed(0, 2, 1, 3)
        let v = unflatten(valueProjection(values), axis: -1, shape: [numKVHeads, -1])
            .transposed(0, 2, 1, 3)
        let mode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .none
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(q.dim(-1)).squareRoot(), mask: mode)
        return outProjection(output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1))
    }
}
#endif
