import CoreML
import Foundation

struct ArrayFeature: Sendable {
    let name: String
    let shape: [Int]
    let dataType: MLMultiArrayDataType
    var width: Int { shape[shape.count - 1] }
    var elementBytes: Int {
        if dataType == .float16 { return MemoryLayout<Float16>.stride }
        return dataType == .double ? MemoryLayout<Double>.stride : MemoryLayout<Float>.stride
    }

    init(_ feature: MLFeatureDescription, rank: Int) throws {
        guard let constraint = feature.multiArrayConstraint, constraint.shape.count == rank,
              constraint.shape.allSatisfy({ $0.intValue > 0 }),
              [.double, .float32, .float16].contains(constraint.dataType) else {
            throw SwiftMLError.invalidParameter("Core ML arrays must have the requested rank, positive dimensions, and Double, Float32 or Float16 elements")
        }
        if constraint.dataType == .float16 {
            guard #available(macOS 15, *) else {
                throw SwiftMLError.invalidParameter("Core ML Float16 arrays require macOS 15 or later")
            }
        }
        name = feature.name
        shape = constraint.shape.map(\.intValue)
        dataType = constraint.dataType
        if rank == 2 {
            let flexibility = constraint.shapeConstraint
            switch flexibility.type {
            case .unspecified: break
            case .enumerated:
                guard flexibility.enumeratedShapes.allSatisfy({ $0 == constraint.shape }) else {
                    throw SwiftMLError.invalidParameter("Core ML matrix shapes must be fixed")
                }
            case .range:
                guard flexibility.sizeRangeForDimension.count == rank,
                      zip(flexibility.sizeRangeForDimension, shape).allSatisfy({ range, size in
                          range.rangeValue == NSRange(location: size, length: 1)
                      }) else {
                    throw SwiftMLError.invalidParameter("Core ML matrix shapes must be fixed")
                }
            @unknown default:
                throw SwiftMLError.invalidParameter("Unsupported Core ML matrix shape constraint")
            }
        }
    }
}
