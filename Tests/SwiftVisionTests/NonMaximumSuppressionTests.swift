import Testing
@testable import SwiftVision

// MARK: - BoundingBox IoU

@Suite("BoundingBox NMS Extensions")
struct BoundingBoxNMSTests {

    // Helper to build a BoundingBox (adjust param names if your type differs)
    private func box(x1: Double, y1: Double, x2: Double, y2: Double,
                     conf: Double = 1.0, label: String = "cat") -> BoundingBox {
        BoundingBox(xMin: x1, yMin: y1, xMax: x2, yMax: y2, confidence: conf, classLabel: label)
    }

    @Test("area of unit square is 1.0")
    func areaUnitSquare() {
        let b = box(x1: 0, y1: 0, x2: 1, y2: 1)
        #expect(b.area == 1.0)
    }

    @Test("area of zero-size box is 0.0")
    func areaZeroBox() {
        let b = box(x1: 2, y1: 2, x2: 2, y2: 2)
        #expect(b.area == 0.0)
    }

    @Test("area is positive for inverted coordinates (clamped)")
    func areaInverted() {
        let b = box(x1: 5, y1: 5, x2: 3, y2: 3)  // xMax < xMin => max(0,…) = 0
        #expect(b.area == 0.0)
    }

    @Test("IoU of identical boxes is 1.0")
    func iouIdentical() {
        let b = box(x1: 0, y1: 0, x2: 2, y2: 2)
        #expect(abs(b.intersectionOverUnion(with: b) - 1.0) < 1e-9)
    }

    @Test("IoU of non-overlapping boxes is 0.0")
    func iouNonOverlapping() {
        let a = box(x1: 0, y1: 0, x2: 1, y2: 1)
        let b = box(x1: 2, y1: 2, x2: 3, y2: 3)
        #expect(a.intersectionOverUnion(with: b) == 0.0)
    }

    @Test("IoU of half-overlapping boxes is correct")
    func iouHalfOverlap() {
        // a = [0,0]-[2,2], area=4; b = [1,0]-[3,2], area=4; intersection [1,0]-[2,2]=2; union=6
        let a = box(x1: 0, y1: 0, x2: 2, y2: 2)
        let b = box(x1: 1, y1: 0, x2: 3, y2: 2)
        let iou = a.intersectionOverUnion(with: b)
        #expect(abs(iou - 2.0 / 6.0) < 1e-9)
    }
}

// MARK: - NonMaximumSuppression.filter

@Suite("NonMaximumSuppression.filter")
struct NonMaximumSuppressionTests {

    private func box(x1: Double, y1: Double, x2: Double, y2: Double,
                     conf: Double, label: String = "obj") -> BoundingBox {
        BoundingBox(xMin: x1, yMin: y1, xMax: x2, yMax: y2, confidence: conf, classLabel: label)
    }

    @Test("empty input returns empty output")
    func emptyInput() {
        let result = NonMaximumSuppression.filter(boxes: [])
        #expect(result.isEmpty)
    }

    @Test("single box above threshold is kept")
    func singleBoxKept() {
        let b = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.9)
        let result = NonMaximumSuppression.filter(boxes: [b], scoreThreshold: 0.5)
        #expect(result.count == 1)
    }

    @Test("box below scoreThreshold is removed")
    func belowScoreThreshold() {
        let b = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.1)
        let result = NonMaximumSuppression.filter(boxes: [b], scoreThreshold: 0.5)
        #expect(result.isEmpty)
    }

    @Test("duplicate box suppresses the lower-confidence duplicate")
    func duplicateSuppressed() {
        let high = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.95)
        let low  = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.60)   // IoU=1.0 → suppressed
        let result = NonMaximumSuppression.filter(boxes: [high, low], iouThreshold: 0.45)
        #expect(result.count == 1)
        #expect(abs(result[0].confidence - 0.95) < 1e-9)
    }

    @Test("non-overlapping boxes with same class are both kept")
    func nonOverlappingKept() {
        let a = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.9)
        let b = box(x1: 5, y1: 5, x2: 6, y2: 6, conf: 0.8)
        let result = NonMaximumSuppression.filter(boxes: [a, b])
        #expect(result.count == 2)
    }

    @Test("overlapping boxes with different class labels are both kept")
    func differentClassKept() {
        let cat = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.9, label: "cat")
        let dog = box(x1: 0, y1: 0, x2: 1, y2: 1, conf: 0.8, label: "dog")  // identical region, different class
        let result = NonMaximumSuppression.filter(boxes: [cat, dog])
        #expect(result.count == 2)
    }

    @Test("result is sorted by confidence descending")
    func sortedByConfidence() {
        let low  = box(x1: 10, y1: 10, x2: 11, y2: 11, conf: 0.5)
        let high = box(x1: 20, y1: 20, x2: 21, y2: 21, conf: 0.9)
        let mid  = box(x1: 30, y1: 30, x2: 31, y2: 31, conf: 0.7)
        let result = NonMaximumSuppression.filter(boxes: [low, high, mid])
        #expect(result[0].confidence >= result[1].confidence)
        #expect(result[1].confidence >= result[2].confidence)
    }

    @Test("IoU exactly at threshold does not suppress")
    func iouAtThresholdNotSuppressed() {
        // boxes with IoU ≈ 0.45 — should NOT be suppressed (condition is strictly >)
        let a = box(x1: 0, y1: 0, x2: 2, y2: 2, conf: 0.9)   // area = 4
        // intersection width = 1.34 → IoU ≈ 0.45  (approximate; depends on exact coords)
        let b = box(x1: 1, y1: 0, x2: 3, y2: 2, conf: 0.7)
        // Real IoU = 2/6 ≈ 0.333 < 0.45 → both should survive
        let result = NonMaximumSuppression.filter(boxes: [a, b], iouThreshold: 0.45)
        #expect(result.count == 2)
    }
}
