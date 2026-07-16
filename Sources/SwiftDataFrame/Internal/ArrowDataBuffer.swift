import Foundation
import Arrow

/// A DataBuffer implementation backed by an Apache Arrow buffer's raw memory pointer.
/// Exposes read-only zero-copy access to Arrow memory.
internal struct ArrowDataBuffer<Element: Sendable>: DataBuffer, @unchecked Sendable {
    
    private let rawPointer: UnsafeRawPointer
    public let byteCount: Int
    public let elementCount: Int
    
    init(rawPointer: UnsafeRawPointer, byteCount: Int, elementCount: Int) {
        self.rawPointer = rawPointer
        self.byteCount = byteCount
        self.elementCount = elementCount
    }
    
    func withUnsafeBytes<R: Sendable>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        let bufferPointer = UnsafeRawBufferPointer(start: rawPointer, count: byteCount)
        return try body(bufferPointer)
    }
    
    func slice(from: Int, count: Int) -> ArrowDataBuffer<Element> {
        let elementSize = MemoryLayout<Element>.stride
        let byteOffset = from * elementSize
        let byteCountToCopy = count * elementSize
        
        return ArrowDataBuffer(
            rawPointer: rawPointer.advanced(by: byteOffset),
            byteCount: byteCountToCopy,
            elementCount: count
        )
    }
}
