import Foundation
import SwiftDataFrame
import Metal
import MLX
import Darwin

// Internal read-only imports. Callers finish evaluation in the closure; no lazy result may escape.
package enum ScopedGPURead {
    package static func withBuffer<R>(_ buffer: any MTLBuffer, range: Range<Int>, shape: [Int],
                                      _ operation: (MLXArray, Bool) throws -> R) throws -> R {
        guard buffer.storageMode == .shared, !range.isEmpty, range.lowerBound >= 0,
              range.upperBound <= buffer.length / MemoryLayout<Float>.stride,
              !shape.isEmpty, shape.allSatisfy({ $0 > 0 && $0 <= Int32.max }) else {
            throw SwiftMLError.invalidParameter("Invalid shared GPU input extent")
        }
        var count = 1
        for dimension in shape {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw SwiftMLError.invalidParameter("GPU input shape overflows Int") }
            count = next
        }
        guard count == range.count else { throw SwiftMLError.invalidParameter("GPU input shape differs from extent") }
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self).advanced(by: range.lowerBound)
        let page = Int(getpagesize()), bytes = count * MemoryLayout<Float>.stride
        let shared = Int(bitPattern:pointer) % page == 0 && bytes % page == 0
        let input: MLXArray
        if shared {
            input = MLXArray(rawPointer: UnsafeMutableRawPointer(pointer), shape, dtype: .float32) {
                withExtendedLifetime(buffer) {}
            }
        } else {
            input = MLXArray(UnsafeBufferPointer(start:pointer,count:count), shape)
        }
        defer {
            Stream.gpu.synchronize()
            withExtendedLifetime((input,buffer)) {}
        }
        return try operation(input,shared)
    }

    package static func withMatrix<R>(batch: PreparedNumericBatch,
                                      _ operation: (MLXArray) throws -> R) throws -> R {
        try withMatrix(rows: batch.rowCount, columns: batch.columnCount, initialize: { pointer in
            PreparedFloatPacking.initialize(batch, into: pointer)
        }, operation)
    }

    package static func withMatrix<R>(rows: Int, columns: Int,
                                      value: (Int, Int) -> Double,
                                      _ operation: (MLXArray) throws -> R) throws -> R {
        try withMatrix(rows: rows, columns: columns, initialize: { pointer in
            for row in 0..<rows {
                for column in 0..<columns {
                    pointer.baseAddress!.advanced(by: row * columns + column).initialize(to: Float(value(row, column)))
                }
            }
        }, operation)
    }

    private static func withMatrix<R>(rows: Int, columns: Int,
                                      initialize: (UnsafeMutableBufferPointer<Float>) -> Void,
                                      _ operation: (MLXArray) throws -> R) throws -> R {
        // Metal can retain autoreleased owners after Swift locals leave scope.
        // withBuffer drains GPU work before this synchronous pool closes.
        return try autoreleasepool {
            let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
            let (bytes, byteOverflow) = count.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
            guard rows > 0, columns > 0, rows <= Int32.max, columns <= Int32.max, !overflow, !byteOverflow,
                  let device = MetalDeviceOwner.defaultDevice, bytes <= device.maxBufferLength,
                  let buffer = device.makeBuffer(length:bytes,options:.storageModeShared) else {
                throw SwiftMLError.invalidParameter("Cannot allocate GPU feature matrix")
            }
            let pointer = buffer.contents().bindMemory(to:Float.self,capacity:count)
            initialize(UnsafeMutableBufferPointer(start: pointer, count: count))
            return try withBuffer(buffer,range:0..<count,shape:[rows,columns]) { input,_ in try operation(input) }
        }
    }
}
