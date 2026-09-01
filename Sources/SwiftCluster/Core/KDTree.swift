import Foundation
import Accelerate

/// A k-dimensional tree (k-d tree) spatial index for efficient nearest-neighbor
/// and radius (range) queries in multi-dimensional Euclidean space.
///
/// Reduces range query complexity from O(N) to O(log N) on average,
/// enabling algorithms like DBSCAN and LOF to achieve O(N log N) overall complexity.
public final class KDTree: @unchecked Sendable {
    
    private final class Node {
        let point: [Double]
        let index: Int
        let axis: Int
        var left: Node?
        var right: Node?
        
        init(point: [Double], index: Int, axis: Int) {
            self.point = point
            self.index = index
            self.axis = axis
        }
    }
    
    private var root: Node?
    
    /// The dimensionality of points in the tree.
    public let dimensions: Int
    
    /// The total number of points indexed in the tree.
    public let count: Int
    
    /// Builds a KD-Tree from a collection of points.
    /// - Parameter points: Array of multi-dimensional points (each point is `[Double]`).
    public init(points: [[Double]]) {
        guard !points.isEmpty else {
            self.dimensions = 0
            self.count = 0
            self.root = nil
            return
        }
        
        let dim = points[0].count
        self.dimensions = dim
        self.count = points.count
        
        var indexedPoints: [(point: [Double], index: Int)] = points.enumerated().map { ($0.element, $0.offset) }
        self.root = Self.buildTree(points: &indexedPoints, start: 0, end: points.count, depth: 0, dim: dim)
    }
    
    private static func buildTree(
        points: inout [(point: [Double], index: Int)],
        start: Int,
        end: Int,
        depth: Int,
        dim: Int
    ) -> Node? {
        guard start < end else { return nil }
        
        let axis = depth % dim
        let mid = start + (end - start) / 2
        
        // Quickselect / median split along current axis
        points[start..<end].sort { $0.point[axis] < $1.point[axis] }
        
        let node = Node(point: points[mid].point, index: points[mid].index, axis: axis)
        node.left = buildTree(points: &points, start: start, end: mid, depth: depth + 1, dim: dim)
        node.right = buildTree(points: &points, start: mid + 1, end: end, depth: depth + 1, dim: dim)
        
        return node
    }
    
    /// Finds all point indices within the given Euclidean distance `radius` from the target point.
    /// - Parameters:
    ///   - point: The target query point.
    ///   - radius: The search radius in Euclidean distance.
    /// - Returns: An array of original sample indices within the radius.
    public func queryRadius(point: [Double], radius: Double) -> [Int] {
        guard let root = root, radius >= 0 else { return [] }
        var result = [Int]()
        let radiusSq = radius * radius
        searchRadius(node: root, target: point, radiusSq: radiusSq, radius: radius, result: &result)
        return result
    }
    
    private func searchRadius(
        node: Node?,
        target: [Double],
        radiusSq: Double,
        radius: Double,
        result: inout [Int]
    ) {
        guard let node = node else { return }
        
        let axis = node.axis
        let diff = target[axis] - node.point[axis]
        
        // Calculate squared Euclidean distance between target and node.point
        var distSq = 0.0
        for i in 0..<dimensions {
            let d = target[i] - node.point[i]
            distSq += d * d
            if distSq > radiusSq { break } // Early exit if distance exceeds radius
        }
        
        if distSq <= radiusSq {
            result.append(node.index)
        }
        
        // Check near and far subtrees
        let checkLeft = diff <= 0
        let nearChild = checkLeft ? node.left : node.right
        let farChild = checkLeft ? node.right : node.left
        
        searchRadius(node: nearChild, target: target, radiusSq: radiusSq, radius: radius, result: &result)
        
        // Check if hyperplane intersects hypersphere
        if diff * diff <= radiusSq {
            searchRadius(node: farChild, target: target, radiusSq: radiusSq, radius: radius, result: &result)
        }
    }
    
    /// Finds the `k` nearest neighbors to the target point.
    /// - Parameters:
    ///   - point: The target point.
    ///   - k: The number of nearest neighbors to retrieve.
    /// - Returns: Array of `(index: Int, distance: Double)` sorted by ascending distance.
    public func queryKNN(point: [Double], k: Int) -> [(index: Int, distance: Double)] {
        guard let root = root, k > 0 else { return [] }
        var heap = [Neighbor]()
        searchKNN(node: root, target: point, k: k, heap: &heap)
        
        return heap.sorted { $0.distSq < $1.distSq }.map { ($0.index, sqrt($0.distSq)) }
    }
    
    private struct Neighbor {
        let index: Int
        let distSq: Double
    }
    
    private func searchKNN(
        node: Node?,
        target: [Double],
        k: Int,
        heap: inout [Neighbor]
    ) {
        guard let node = node else { return }
        
        let axis = node.axis
        let diff = target[axis] - node.point[axis]
        
        var distSq = 0.0
        for i in 0..<dimensions {
            let d = target[i] - node.point[i]
            distSq += d * d
        }
        
        if heap.count < k {
            heap.append(Neighbor(index: node.index, distSq: distSq))
            heap.sort { $0.distSq > $1.distSq } // Max-heap behavior (largest at index 0)
        } else if let maxDistSq = heap.first?.distSq, distSq < maxDistSq {
            heap[0] = Neighbor(index: node.index, distSq: distSq)
            heap.sort { $0.distSq > $1.distSq }
        }
        
        let checkLeft = diff <= 0
        let nearChild = checkLeft ? node.left : node.right
        let farChild = checkLeft ? node.right : node.left
        
        searchKNN(node: nearChild, target: target, k: k, heap: &heap)
        
        let maxDistSq = heap.count < k ? Double.infinity : (heap.first?.distSq ?? Double.infinity)
        if diff * diff < maxDistSq {
            searchKNN(node: farChild, target: target, k: k, heap: &heap)
        }
    }
}
