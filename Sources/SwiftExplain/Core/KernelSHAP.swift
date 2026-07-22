import Foundation

/// Model interpretation using the KernelSHAP algorithm.
public actor KernelSHAP {
    
    public init() {}
    
    /// Computes SHAP values explaining the model prediction for the given instance.
    /// - Parameters:
    ///   - model: A model prediction closure mapping feature vectors `[Double]` to a scalar prediction.
    ///   - instance: The target instance feature vector to explain.
    ///   - background: A background dataset of representative feature vectors.
    ///   - numCoalitions: Number of coalition masks to sample.
    /// - Returns: An array of SHAP values (one for each feature) indicating feature contributions.
    public func explain(
        model: @escaping @Sendable ([Double]) async -> Double,
        instance: [Double],
        background: [[Double]],
        numCoalitions: Int = 200
    ) async -> [Double] {
        let M = instance.count
        guard M > 0 else { return [] }
        
        // 1. Calculate the background mean vector
        var tempBgMean = [Double](repeating: 0.0, count: M)
        for col in 0..<M {
            let sum = background.map { $0[col] }.reduce(0.0, +)
            tempBgMean[col] = background.isEmpty ? 0.0 : sum / Double(background.count)
        }
        let bgMean = tempBgMean
        
        let fEmpty = await model(bgMean)
        let fFull = await model(instance)
        
        // 2. Generate coalition masks and their Shapley weights
        var masks = [[Double]]()
        var weights = [Double]()
        var targets = [Double]()
        
        masks.reserveCapacity(numCoalitions)
        weights.reserveCapacity(numCoalitions)
        targets.reserveCapacity(numCoalitions)
        
        // Always include all-zeros and all-ones coalitions with very high weight to enforce efficiency constraints
        let largeWeight = 1e6
        
        masks.append([Double](repeating: 0.0, count: M))
        weights.append(largeWeight)
        targets.append(0.0) // fEmpty - fEmpty
        
        masks.append([Double](repeating: 1.0, count: M))
        weights.append(largeWeight)
        targets.append(fFull - fEmpty)
        
        // Generate intermediate coalitions
        let coalitionSizes = M > 2 ? Array(1..<(M - 1)) : []
        let sizeWeights = coalitionSizes.map { k -> Double in
            return 1.0 / (Double(k) * Double(M - k))
        }
        let totalSizeWeight = sizeWeights.reduce(0.0, +)
        
        // Concurrently evaluate coalitions in batches using a TaskGroup
        let remaining = max(0, numCoalitions - 2)
        
        await withTaskGroup(of: (mask: [Double], weight: Double, target: Double).self) { group in
            for _ in 0..<remaining {
                group.addTask {
                    // Sample subset size k using helper function
                    let k = sampleSubsetSize(sizes: coalitionSizes, weights: sizeWeights, totalWeight: totalSizeWeight, fallback: max(1, M / 2))
                    
                    // Generate mask with k features present
                    var mask = [Double](repeating: 0.0, count: M)
                    var indices = Array(0..<M)
                    indices.shuffle()
                    for idx in indices.prefix(k) {
                        mask[idx] = 1.0
                    }
                    
                    // Compute Shapley kernel weight
                    let nCr = choose(M, k)
                    let weight = nCr > 0 ? (Double(M) - 1.0) / (nCr * Double(k) * Double(M - k)) : 1.0
                    
                    // Map coalition mask to feature space
                    var xMapped = [Double](repeating: 0.0, count: M)
                    for i in 0..<M {
                        xMapped[i] = (mask[i] == 1.0) ? instance[i] : bgMean[i]
                    }
                    
                    let y = await model(xMapped)
                    let target = y - fEmpty
                    
                    return (mask, weight, target)
                }
            }
            
            for await result in group {
                masks.append(result.mask)
                weights.append(result.weight)
                targets.append(result.target)
            }
        }
        
        // 3. Solve OLS system: (Z^T * W * Z) * beta = Z^T * W * Y
        let N = masks.count
        var ZTWZ = [Double](repeating: 0.0, count: M * M)
        var ZTWY = [Double](repeating: 0.0, count: M)
        
        for r in 0..<M {
            for c in 0..<M {
                var sum = 0.0
                for i in 0..<N {
                    sum += masks[i][r] * weights[i] * masks[i][c]
                }
                ZTWZ[r * M + c] = sum
            }
            
            // Add a tiny ridge regularization to ensure positive definiteness
            ZTWZ[r * M + r] += 1e-6
            
            var sum = 0.0
            for i in 0..<N {
                sum += masks[i][r] * weights[i] * targets[i]
            }
            ZTWY[r] = sum
        }
        
        // Solve system using Gauss-Jordan solver
        if let shapValues = solveLinearSystem(A: ZTWZ, B: ZTWY, M: M) {
            return shapValues
        }
        
        return [Double](repeating: 0.0, count: M)
    }
}

// MARK: - Top-Level Helpers (Free from Actor Isolation)

private func choose(_ n: Int, _ k: Int) -> Double {
    if k < 0 || k > n { return 0 }
    var val = 1.0
    for i in 1...min(k, n - k) {
        val *= Double(n - i + 1) / Double(i)
    }
    return val
}

private func sampleSubsetSize(sizes: [Int], weights: [Double], totalWeight: Double, fallback: Int) -> Int {
    guard !sizes.isEmpty, totalWeight > 0 else { return fallback }
    let r = Double.random(in: 0..<totalWeight)
    var cumulative = 0.0
    for i in 0..<sizes.count {
        cumulative += weights[i]
        if r <= cumulative {
            return sizes[i]
        }
    }
    return sizes.last ?? fallback
}

private func solveLinearSystem(A: [Double], B: [Double], M: Int) -> [Double]? {
    var a = A
    var b = B
    
    for i in 0..<M {
        var pivotRow = i
        var maxVal = abs(a[i * M + i])
        for r in (i + 1)..<M {
            let val = abs(a[r * M + i])
            if val > maxVal {
                maxVal = val
                pivotRow = r
            }
        }
        
        if pivotRow != i {
            for c in 0..<M {
                a.swapAt(i * M + c, pivotRow * M + c)
            }
            b.swapAt(i, pivotRow)
        }
        
        let pivot = a[i * M + i]
        if abs(pivot) < 1e-9 {
            return nil
        }
        
        for c in i..<M {
            a[i * M + c] /= pivot
        }
        b[i] /= pivot
        
        for r in 0..<M {
            if r == i { continue }
            let factor = a[r * M + i]
            for c in i..<M {
                a[r * M + c] -= factor * a[i * M + c]
            }
            b[r] -= factor * b[i]
        }
    }
    
    return b
}
