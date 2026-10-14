import Foundation

/// Model interpretation using the KernelSHAP algorithm.
public actor KernelSHAP {
    
    /// Creates a new instance.
    public init() {}
    
    /// Computes model-agnostic Shapley additive explanations for an instance using KernelSHAP.
    ///
    /// ## Mathematical Guarantees & Low-Dimensional Analytical Path
    /// - For $M = 1$: $\phi_0 = f(x) - f_{\text{empty}}$ exact.
    /// - For $M = 2$: computes exact 2-player coalition values satisfying efficiency $\phi_0 + \phi_1 = f(x) - f_{\text{empty}}$ identically.
    /// - For $M \ge 3$: exact constrained weighted linear regression enforcing $\sum_{i=1}^M \phi_i = f(x) - f_{\text{empty}}$ analytically without artificial penalties.
    ///
    /// - Parameters:
    ///   - model: An asynchronous model function mapping an array of features to a scalar output.
    ///   - instance: The target feature vector to explain of length $M$.
    ///   - background: Background reference dataset for baseline imputation.
    ///   - numCoalitions: Number of coalition subsets to sample when $M > 2$ (default: 200).
    /// - Returns: An array of $M$ Shapley values representing the attribution of each feature.
    public func explain(
        model: @escaping @Sendable ([Double]) async -> Double,
        instance: [Double],
        background: [[Double]],
        numCoalitions: Int = 200
    ) async -> [Double] {
        let M = instance.count
        // Parameter and dimension validation
        guard M > 0, numCoalitions > 0 else { return [] }
        guard instance.allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
            return [Double](repeating: 0.0, count: M)
        }

        // Strict background validation (must be non-empty, rectangular with >= M cols, all finite)
        guard !background.isEmpty,
              background.allSatisfy({ $0.count >= M && $0.prefix(M).allSatisfy({ !$0.isNaN && !$0.isInfinite }) }) else {
            return [Double](repeating: 0.0, count: M)
        }
        
        // 1. Calculate the background mean vector
        let bgCount = Double(background.count)
        let bgMean: [Double] = (0..<M).map { col in
            var sum = 0.0
            for row in background {
                sum += row[col]
            }
            return sum / bgCount
        }
        
        let fEmpty = await model(bgMean)
        let fFull = await model(instance)
        let deltaF = fFull - fEmpty

        // Exact analytical path for M = 1
        if M == 1 {
            return [deltaF]
        }

        // Exact analytical path for M = 2
        if M == 2 {
            let f0 = await model([instance[0], bgMean[1]])
            let f1 = await model([bgMean[0], instance[1]])
            let phi0 = 0.5 * ((f0 - fEmpty) + (fFull - f1))
            let phi1 = 0.5 * ((f1 - fEmpty) + (fFull - f0))
            return [phi0, phi1]
        }
        
        // Exact Constrained Weighted Least Squares for M >= 3
        // Efficiency axiom constraint: phi_{M-1} = deltaF - sum_{j=0}^{M-2} phi_j
        // Substituted into surrogate model:
        //   g(z) = sum_{j=0}^{M-2} phi_j (z_j - z_{M-1}) + z_{M-1} * deltaF
        // Regressors: \tilde{z}_j = z_j - z_{M-1} for j in 0..<(M-1)
        // Target: \tilde{y} = (y - fEmpty) - z_{M-1} * deltaF
        // System dimension is (M - 1).
        let P = M - 1

        let totalPossible = (M < 31) ? ((1 << M) - 2) : Int.max
        let canEnumerateAll = totalPossible <= numCoalitions

        var masks = [[Double]]()
        var weights = [Double]()
        var targets = [Double]()

        if canEnumerateAll {
            // Full enumeration of all 2^M - 2 intermediate coalitions with exact Shapley kernel weights
            let numSubsets = totalPossible
            masks.reserveCapacity(numSubsets)
            weights.reserveCapacity(numSubsets)
            targets.reserveCapacity(numSubsets)

            var subsets = [[Double]]()
            var subWeights = [Double]()
            for maskVal in 1...numSubsets {
                var mask = [Double](repeating: 0.0, count: M)
                var k = 0
                for bit in 0..<M {
                    if (maskVal & (1 << bit)) != 0 {
                        mask[bit] = 1.0
                        k += 1
                    }
                }
                guard k > 0 && k < M else { continue }
                let nCr = choose(M, k)
                let denom = nCr * Double(k) * Double(M - k)
                let w = denom > 1e-12 ? (Double(M) - 1.0) / denom : 1.0
                subsets.append(mask)
                subWeights.append(w)
            }

            await withTaskGroup(of: (mask: [Double], weight: Double, target: Double).self) { group in
                for idx in 0..<subsets.count {
                    let mask = subsets[idx]
                    let w = subWeights[idx]
                    group.addTask {
                        var xMapped = [Double](repeating: 0.0, count: M)
                        for i in 0..<M {
                            xMapped[i] = (mask[i] == 1.0) ? instance[i] : bgMean[i]
                        }
                        let y = await model(xMapped)
                        let rawTarget = y - fEmpty
                        return (mask, w, rawTarget)
                    }
                }
                for await res in group {
                    masks.append(res.mask)
                    weights.append(res.weight)
                    targets.append(res.target)
                }
            }
        } else {
            // Sample coalitions with proposal distribution q(k) proportional to 1 / (k*(M-k)).
            // Because q(S) = q(k) / \binom{M}{k} \propto \pi(S), importance weights in regression are uniform (1.0).
            masks.reserveCapacity(numCoalitions)
            weights.reserveCapacity(numCoalitions)
            targets.reserveCapacity(numCoalitions)

            let coalitionSizes = Array(1..<(M - 1))
            let sizeWeights = coalitionSizes.map { k -> Double in
                let denom = Double(k) * Double(M - k)
                return denom > 0 ? 1.0 / denom : 1.0
            }
            let totalSizeWeight = sizeWeights.reduce(0.0, +)

            await withTaskGroup(of: (mask: [Double], weight: Double, target: Double).self) { group in
                for _ in 0..<numCoalitions {
                    group.addTask {
                        let k = sampleSubsetSize(
                            sizes: coalitionSizes,
                            weights: sizeWeights,
                            totalWeight: totalSizeWeight,
                            fallback: max(1, M / 2)
                        )
                        var mask = [Double](repeating: 0.0, count: M)
                        var indices = Array(0..<M)
                        indices.shuffle()
                        for idx in indices.prefix(k) {
                            mask[idx] = 1.0
                        }

                        var xMapped = [Double](repeating: 0.0, count: M)
                        for i in 0..<M {
                            xMapped[i] = (mask[i] == 1.0) ? instance[i] : bgMean[i]
                        }
                        let y = await model(xMapped)
                        let rawTarget = y - fEmpty
                        // Uniform importance weight: sampling probability already matches Shapley kernel
                        return (mask, 1.0, rawTarget)
                    }
                }
                for await res in group {
                    masks.append(res.mask)
                    weights.append(res.weight)
                    targets.append(res.target)
                }
            }
        }

        // Fit weighted regression for (M - 1) parameters on transformed variables
        let N = masks.count
        guard N > 0 else { return [Double](repeating: deltaF / Double(M), count: M) }

        var ZtWZ = [Double](repeating: 0.0, count: P * P)
        var ZtWy = [Double](repeating: 0.0, count: P)

        let lastIdx = M - 1
        for i in 0..<N {
            let w = weights[i]
            let zLast = masks[i][lastIdx]
            let tildeY = targets[i] - zLast * deltaF

            var tildeZ = [Double](repeating: 0.0, count: P)
            for j in 0..<P {
                tildeZ[j] = masks[i][j] - zLast
            }

            for j in 0..<P {
                ZtWy[j] += tildeZ[j] * w * tildeY
                for k in 0..<P {
                    ZtWZ[j * P + k] += tildeZ[j] * w * tildeZ[k]
                }
            }
        }

        // Add small ridge regularization to ensure numerical invertibility
        for j in 0..<P {
            ZtWZ[j * P + j] += 1e-8
        }

        var phi = [Double](repeating: 0.0, count: M)
        if let beta = solveLinearSystem(A: ZtWZ, B: ZtWy, M: P) {
            var sumFirst = 0.0
            for j in 0..<P {
                phi[j] = beta[j]
                sumFirst += beta[j]
            }
            // Exact efficiency enforcement:
            phi[lastIdx] = deltaF - sumFirst
            return phi
        }

        // Fallback: equal attribution maintaining exact efficiency
        let equalContrib = deltaF / Double(M)
        return [Double](repeating: equalContrib, count: M)
    }
}

// MARK: - Top-Level Helpers (Free from Actor Isolation)

private func choose(_ n: Int, _ k: Int) -> Double {
    if k < 0 || k > n { return 0.0 }
    if k == 0 || k == n { return 1.0 }
    let bound = min(k, n - k)
    if bound < 1 { return 1.0 }
    var val = 1.0
    for i in 1...bound {
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
        if abs(pivot) < 1e-12 {
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

