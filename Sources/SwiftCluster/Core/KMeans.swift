import Foundation
import Accelerate
import MLX
import SwiftPreprocessing

private struct SendablePointer<T>: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<T>
}

/// K-Means clustering with CPU (vDSP) and GPU (MLX) backends via hardware routing.
public actor KMeans {
    public let nClusters: Int
    public let maxIterations: Int
    public let tolerance: Float
    public let requestedDevice: ExecutionDevice

    /// Device actually used for the last `fit` (after `.auto` resolution).
    public private(set) var resolvedDevice: ExecutionDevice?

    /// Centroids as an MLX array `[nClusters, nFeatures]` (populated by both backends).
    public private(set) var centroids: MLXArray?

    /// CPU-side centroid matrix (source of truth after a CPU fit).
    private var cpuCentroids: [[Double]]?

    public let seed: Int

    public init(
        nClusters: Int,
        maxIterations: Int = 300,
        tolerance: Double = 1e-4,
        seed: Int = 42,
        device: ExecutionDevice = .auto
    ) throws {
        guard nClusters > 0 else {
            throw ClusterError.invalidParameter("nClusters must be greater than 0.")
        }
        guard maxIterations > 0 else {
            throw ClusterError.invalidParameter("maxIterations must be greater than 0.")
        }
        self.nClusters = nClusters
        self.maxIterations = maxIterations
        self.tolerance = Float(tolerance)
        self.seed = seed
        self.requestedDevice = device
    }

    /// Fits K-Means on the input dataset (Sendable interface).
    public func fit(features: [[Double]]) async throws {
        guard !features.isEmpty else {
            throw ClusterError.emptyInput
        }

        let numSamples = features.count
        let numFeatures = features[0].count
        guard nClusters <= numSamples else {
            throw ClusterError.invalidParameter(
                "nClusters (\(nClusters)) cannot be greater than the number of samples (\(numSamples))."
            )
        }

        let device = await HardwareRouter.shared.resolveDevice(
            for: "KMeans",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )
        resolvedDevice = device

        switch device {
        case .cpu:
            try fitCPU(features: features)
        case .gpu, .ane, .auto:
            let X = MLXArray(features.flatMap { $0.map { Float($0) } })
                .reshaped([numSamples, numFeatures])
            let initCents = kmeansPlusPlusInitCPU(features: features, nClusters: nClusters, seed: seed)
            let initCentsArray = MLXArray(initCents.flatMap { $0.map { Float($0) } })
                .reshaped([nClusters, numFeatures])
            let maxIters = self.maxIterations
            let tol = self.tolerance
            let cents = Device.withDefaultDevice(.gpu) {
                Self.runFitGPU(X: X, initialCentroids: initCentsArray, maxIterations: maxIters, tolerance: Float(tol))
            }
            self.centroids = cents
            cpuCentroids = getCentroids()
        }
    }

    /// Fits K-Means on an MLX tensor (forces GPU path after setting MLX device).
    public func fit(X: MLXArray) async throws {
        guard X.size > 0 else { throw ClusterError.emptyInput }
        let shape = X.shape
        guard shape.count == 2 else {
            throw ClusterError.dimensionMismatch(expected: 2, got: shape.count)
        }
        guard nClusters <= shape[0] else {
            throw ClusterError.invalidParameter(
                "nClusters (\(nClusters)) cannot be greater than the number of samples (\(shape[0]))."
            )
        }

        let device = await HardwareRouter.shared.resolveDevice(
            for: "KMeans",
            sampleCount: shape[0],
            featureCount: shape[1],
            requestedDevice: requestedDevice == .auto ? .gpu : requestedDevice
        )
        resolvedDevice = device == .cpu ? .gpu : device  // MLXArray input → GPU path
        let flat = X.asArray(Float.self)
        let numSamples = shape[0]
        let numFeatures = shape[1]
        var features = [[Double]]()
        features.reserveCapacity(numSamples)
        for i in 0..<numSamples {
            var row = [Double]()
            row.reserveCapacity(numFeatures)
            for j in 0..<numFeatures {
                row.append(Double(flat[i * numFeatures + j]))
            }
            features.append(row)
        }
        let initCents = kmeansPlusPlusInitCPU(features: features, nClusters: nClusters, seed: seed)
        let initCentsArray = MLXArray(initCents.flatMap { $0.map { Float($0) } })
            .reshaped([nClusters, numFeatures])
        let maxIters = self.maxIterations
        let tol = self.tolerance
        let cents = Device.withDefaultDevice(.gpu) {
            Self.runFitGPU(X: X, initialCentroids: initCentsArray, maxIterations: maxIters, tolerance: Float(tol))
        }
        self.centroids = cents
        cpuCentroids = getCentroids()
    }

    public func predict(features: [[Double]]) throws -> [Int] {
        guard !features.isEmpty else { return [] }
        if let cpuCentroids, resolvedDevice == .cpu {
            return predictCPU(features: features, centroids: cpuCentroids)
        }
        let numSamples = features.count
        let numFeatures = features[0].count
        let X = MLXArray(features.flatMap { $0.map { Float($0) } })
            .reshaped([numSamples, numFeatures])
        let labels = try predict(X: X)
        return labels.asArray(Int32.self).map { Int($0) }
    }

    public func predict(X: MLXArray) throws -> MLXArray {
        guard let centroids = self.centroids else {
            throw ClusterError.fittingRequired
        }
        guard X.size > 0 else { throw ClusterError.emptyInput }
        let shape = X.shape
        guard shape.count == 2 else {
            throw ClusterError.dimensionMismatch(expected: 2, got: shape.count)
        }
        let numFeatures = centroids.shape[1]
        guard shape[1] == numFeatures else {
            throw ClusterError.dimensionMismatch(expected: numFeatures, got: shape[1])
        }

        let diff = X.expandedDimensions(axes: [1]) - centroids.expandedDimensions(axes: [0])
        let dists = sqrt((diff * diff).sum(axis: -1))
        return argMin(dists, axis: -1)
    }

    public func getCentroids() -> [[Double]]? {
        if let cpuCentroids { return cpuCentroids }
        guard let centroids = centroids else { return nil }
        let flatArray = centroids.asArray(Float.self)
        let numClusters = centroids.shape[0]
        let numFeatures = centroids.shape[1]
        var result = [[Double]]()
        result.reserveCapacity(numClusters)
        for i in 0..<numClusters {
            var row = [Double]()
            row.reserveCapacity(numFeatures)
            for j in 0..<numFeatures {
                row.append(Double(flatArray[i * numFeatures + j]))
            }
            result.append(row)
        }
        return result
    }

    // MARK: – GPU (MLX)

    private static func runFitGPU(X: MLXArray, initialCentroids: MLXArray, maxIterations: Int, tolerance: Float) -> MLXArray {
        let nClusters = initialCentroids.shape[0]
        var currentCentroids = initialCentroids

        for _ in 0..<maxIterations {
            let diff = X.expandedDimensions(axes: [1]) - currentCentroids.expandedDimensions(axes: [0])
            let dists = sqrt((diff * diff).sum(axis: -1))
            let labels = argMin(dists, axis: -1)

            var updatedCentroids = [MLXArray]()
            updatedCentroids.reserveCapacity(nClusters)
            for k in 0..<nClusters {
                let mask = equal(labels, MLXArray(k))
                let count = mask.sum().item(Int.self)
                if count > 0 {
                    let sumPoints = (X * mask.expandedDimensions(axes: [1])).sum(axis: 0)
                    updatedCentroids.append(sumPoints / Float(count))
                } else {
                    updatedCentroids.append(currentCentroids[k])
                }
            }

            let newCentroids = stacked(updatedCentroids)
            let diffCentroids = newCentroids - currentCentroids
            let distChange = sqrt((diffCentroids * diffCentroids).sum()).item(Float.self)
            currentCentroids = newCentroids
            eval(currentCentroids)
            if distChange < tolerance { break }
        }

        return currentCentroids
    }

    // MARK: – CPU (vDSP)

    private func fitCPU(features: [[Double]]) throws {
        let n = features.count
        let m = features[0].count
        var centroidsLocal = kmeansPlusPlusInitCPU(features: features, nClusters: nClusters, seed: seed)

        var labels = [Int](repeating: 0, count: n)
        let tol = Double(tolerance)

        for _ in 0..<maxIterations {
            // Assign labels (chunked parallel over points to avoid GCD scheduling overhead)
            let cents = centroidsLocal
            let chunkSize = 512
            let numChunks = (n + chunkSize - 1) / chunkSize
            labels.withUnsafeMutableBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                let sendableBase = SendablePointer(pointer: base)
                DispatchQueue.concurrentPerform(iterations: numChunks) { chunkIdx in
                    let start = chunkIdx * chunkSize
                    let end = min(start + chunkSize, n)
                    for i in start..<end {
                        sendableBase.pointer[i] = Self.nearestCentroid(point: features[i], centroids: cents)
                    }
                }
            }

            // Update centroids using vDSP
            var sums = Array(repeating: [Double](repeating: 0, count: m), count: nClusters)
            var counts = [Int](repeating: 0, count: nClusters)
            for i in 0..<n {
                let k = labels[i]
                counts[k] += 1
                vDSP_vaddD(sums[k], 1, features[i], 1, &sums[k], 1, vDSP_Length(m))
            }

            var newCentroids = centroidsLocal
            var maxShift = 0.0
            for k in 0..<nClusters {
                guard counts[k] > 0 else { continue }
                var updated = [Double](repeating: 0, count: m)
                var inv = 1.0 / Double(counts[k])
                vDSP_vsmulD(sums[k], 1, &inv, &updated, 1, vDSP_Length(m))
                maxShift = max(maxShift, Self.distanceSquared(centroidsLocal[k], updated).squareRoot())
                newCentroids[k] = updated
            }

            centroidsLocal = newCentroids
            if maxShift < tol { break }
        }

        cpuCentroids = centroidsLocal
    }

    private func predictCPU(features: [[Double]], centroids: [[Double]]) -> [Int] {
        let n = features.count
        var labels = [Int](repeating: 0, count: n)
        let chunkSize = 512
        let numChunks = (n + chunkSize - 1) / chunkSize
        labels.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            let sendableBase = SendablePointer(pointer: base)
            DispatchQueue.concurrentPerform(iterations: numChunks) { chunkIdx in
                let start = chunkIdx * chunkSize
                let end = min(start + chunkSize, n)
                for i in start..<end {
                    sendableBase.pointer[i] = Self.nearestCentroid(point: features[i], centroids: centroids)
                }
            }
        }
        return labels
    }

    private static func nearestCentroid(point: [Double], centroids: [[Double]]) -> Int {
        var best = 0
        var bestDist = Double.greatestFiniteMagnitude
        for (k, c) in centroids.enumerated() {
            let d = distanceSquared(point, c)
            if d < bestDist {
                bestDist = d
                best = k
            }
        }
        return best
    }

    private static func distanceSquared(_ a: [Double], _ b: [Double]) -> Double {
        let count = a.count
        if count == 4 {
            let d0 = a[0] - b[0]
            let d1 = a[1] - b[1]
            let d2 = a[2] - b[2]
            let d3 = a[3] - b[3]
            return d0*d0 + d1*d1 + d2*d2 + d3*d3
        }
        if count <= 16 {
            var sum = 0.0
            for i in 0..<count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sum
        }
        var dist = 0.0
        vDSP_distancesqD(a, 1, b, 1, &dist, vDSP_Length(count))
        return dist
    }

    /// Samples initial centroids using the KMeans++ algorithm.
    private func kmeansPlusPlusInitCPU(features: [[Double]], nClusters: Int, seed: Int) -> [[Double]] {
        let n = features.count
        var rng = SeededRandom(seed: seed)
        var sampledCentroids = [[Double]]()
        sampledCentroids.reserveCapacity(nClusters)

        // 1. Uniform random first centroid
        let firstIdx = rng.nextInt(upperBound: n)
        sampledCentroids.append(features[firstIdx])

        // 2. Sample remaining centroids with probability proportional to D(x)^2
        var dSquared = [Double](repeating: .infinity, count: n)

        for _ in 1..<nClusters {
            var totalDist = 0.0
            let lastCentroid = sampledCentroids.last!

            for i in 0..<n {
                let dist = Self.distanceSquared(features[i], lastCentroid)
                dSquared[i] = min(dSquared[i], dist)
                totalDist += dSquared[i]
            }

            if totalDist <= 0 {
                let fallbackIdx = rng.nextInt(upperBound: n)
                sampledCentroids.append(features[fallbackIdx])
                continue
            }

            let target = rng.nextDouble() * totalDist
            var cumulative = 0.0
            var chosenIdx = n - 1

            for i in 0..<n {
                cumulative += dSquared[i]
                if cumulative >= target {
                    chosenIdx = i
                    break
                }
            }
            sampledCentroids.append(features[chosenIdx])
        }

        return sampledCentroids
    }
}
