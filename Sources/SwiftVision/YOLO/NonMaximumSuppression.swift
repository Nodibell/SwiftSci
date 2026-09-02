import Foundation

extension BoundingBox {
    /// Computes the area of the bounding box.
    @inlinable
    public var area: Double {
        max(0.0, xMax - xMin) * max(0.0, yMax - yMin)
    }

    /// Computes Intersection-over-Union (IoU) with another bounding box.
    @inlinable
    public func intersectionOverUnion(with other: BoundingBox) -> Double {
        let interXMin = max(self.xMin, other.xMin)
        let interYMin = max(self.yMin, other.yMin)
        let interXMax = min(self.xMax, other.xMax)
        let interYMax = min(self.yMax, other.yMax)

        let interWidth = max(0.0, interXMax - interXMin)
        let interHeight = max(0.0, interYMax - interYMin)
        let interArea = interWidth * interHeight

        if interArea == 0.0 { return 0.0 }

        let unionArea = self.area + other.area - interArea
        return unionArea > 0 ? interArea / unionArea : 0.0
    }
}

/// Pure Swift high-performance Non-Maximum Suppression (NMS) for filtering overlapping bounding boxes.
public enum NonMaximumSuppression: Sendable {

    /// Filters overlapping bounding boxes using Non-Maximum Suppression.
    ///
    /// - Parameters:
    ///   - boxes: List of detected candidate bounding boxes.
    ///   - iouThreshold: Intersection-over-Union threshold above which overlapping boxes are suppressed (typically 0.45).
    ///   - scoreThreshold: Minimum confidence score to retain (typically 0.25).
    /// - Returns: Filtered list of non-overlapping bounding boxes sorted by confidence in descending order.
    public static func filter(
        boxes: [BoundingBox],
        iouThreshold: Double = 0.45,
        scoreThreshold: Double = 0.25
    ) -> [BoundingBox] {
        // 1. Filter by minimum confidence
        let candidates = boxes.filter { $0.confidence >= scoreThreshold }
        if candidates.isEmpty { return [] }

        // 2. Sort by confidence descending
        let sorted = candidates.sorted { $0.confidence > $1.confidence }

        var results: [BoundingBox] = []
        var suppressed = [Bool](repeating: false, count: sorted.count)

        for i in 0..<sorted.count {
            if suppressed[i] { continue }
            let current = sorted[i]
            results.append(current)

            for j in (i + 1)..<sorted.count {
                if suppressed[j] { continue }
                if sorted[j].classLabel == current.classLabel {
                    let iou = current.intersectionOverUnion(with: sorted[j])
                    if iou > iouThreshold {
                        suppressed[j] = true
                    }
                }
            }
        }

        return results
    }
}
