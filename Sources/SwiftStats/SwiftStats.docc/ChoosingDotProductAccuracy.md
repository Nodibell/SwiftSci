# Choosing dot-product accuracy

Select the reduction policy for each operation. Both policies accept `[Double]` inputs and return a `Double`.

```swift
import SwiftStats

let a: [Double] = [1e16, 1, -1e16]
let b: [Double] = [1, 1, 1]

let fast = try Stats.dotProduct(a, b)
let explicitFast = try Stats.dotProduct(a, b, accuracy: .performance)
let compensated = try Stats.dotProduct(a, b, accuracy: .compensated) // 1
```

## Performance policy

The existing two-argument function selects `.performance`. SwiftSci uses Accelerate's BLAS or vDSP implementation according to vector size and supported counts. This mode prioritizes throughput. It does not reduce the input precision to `Float`.

The reduction order is implementation-dependent. Results may differ in their last bits when the input size, processor, platform version or selected implementation changes. Switching from vDSP to BLAS can change rounding even though the function signature is unchanged.

## Compensated policy

Use `.compensated` when cancellation or product rounding can discard meaningful contributions. The reduction retains a leading sum and two correction components. It processes ordinary inputs with SIMD and fused multiply-add operations.

Unsafe product ranges, nonfinite intermediate values and severe cancellation restart the calculation using exact integer products. This fallback can recover representable answers when individual floating-point products overflow or underflow. It rounds the exact total once to the nearest `Double`, with ties to even. Its accumulator storage is bounded independently of vector length.

Compensation costs additional arithmetic. The exact fallback costs more, especially for large, nearly cancelling vectors. This policy does not promise universally correct rounding or identical bits across platforms because ordinary inputs still use finite-depth floating-point compensation. The documented arithmetic assumes the default round-to-nearest floating-point environment.

Neither policy can recover information lost before the function receives its inputs. Preserving original decimal observations requires an appropriate ingestion representation.

## Validation and exceptional values

Both policies reject empty inputs and unequal vector lengths. NaN propagates. An infinite operand contributes its signed infinity unless another contribution has the opposite infinity. Zero multiplied by infinity and opposing infinite contributions produce NaN. A finite exact result outside the representable range rounds to infinity in the exact fallback.

Exact cancellation returns positive zero in the fallback. A negative exact result that rounds below the minimum subnormal can return negative zero. Do not depend on the sign of an exact zero across reduction implementations.

This option currently controls `Stats.dotProduct`. Other reductions, matrix operations, model fitting and vector search retain their existing implementations. There is no mutable global accuracy setting.
