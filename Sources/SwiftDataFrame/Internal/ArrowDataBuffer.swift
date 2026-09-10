import Foundation
import Arrow

/// A DataBuffer implementation backed by an Apache Arrow buffer's raw memory pointer.
/// Exposes read-only zero-copy access to Arrow memory.
///
/// ## Memory Safety & Lifetime Guarantees
/// This buffer holds a strong reference (`owner`) to the backing Arrow structure (`ArrowTable` or `ArrowArray`),
/// ensuring that the underlying memory is retained across asynchronous task boundaries.
///
/// ## Thread Safety
/// Conforms to `Sendable`. Immutable buffer slices can safely cross concurrency domains without use-after-free.
internal struct ArrowDataBuffer<Element: Sendable>: DataBuffer, @unchecked Sendable {
    
    private let rawPointer: UnsafeRawPointer
    /// The byte count.
    public let byteCount: Int
    /// The element count.
    public let elementCount: Int
    /// Retained owner object to prevent premature deallocation of the backing memory.
    private let owner: AnyObject?
    
    init(rawPointer: UnsafeRawPointer, byteCount: Int, elementCount: Int, owner: AnyObject? = nil) {
        self.rawPointer = rawPointer
        self.byteCount = byteCount
        self.elementCount = elementCount
        self.owner = owner
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
            elementCount: count,
            owner: self.owner
        )
    }
}
