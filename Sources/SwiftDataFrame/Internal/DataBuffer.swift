/// Internal abstraction over the memory backing a column.
/// In v0.1 this is a simple Swift Array wrapper.
/// v0.2 will introduce `ArrowDataBuffer` with true zero-copy via Apache Arrow C Data Interface.
internal protocol DataBuffer<Element>: Sendable {
    associatedtype Element: Sendable
    /// Total byte size of the buffer.
    var byteCount: Int { get }
    /// Number of elements stored.
    var elementCount: Int { get }
    /// Read-only access to underlying bytes (may copy in v0.1).
    func withUnsafeBytes<R: Sendable>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
    /// Returns a sub-buffer view. May copy in v0.1.
    func slice(from: Int, count: Int) -> Self
}

// MARK: – Array-backed implementation (v0.1)

/// Simple array-backed DataBuffer for v0.1.
/// All operations are O(n) copy-based. Will be replaced by ArrowDataBuffer in v0.2.
internal struct ArrayDataBuffer<Element: Sendable>: DataBuffer {

    let storage: [Element]

    init(_ storage: [Element]) { self.storage = storage }

    var byteCount: Int    { storage.count * MemoryLayout<Element>.stride }
    var elementCount: Int { storage.count }

    func withUnsafeBytes<R: Sendable>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }

    func slice(from: Int, count: Int) -> ArrayDataBuffer<Element> {
        let upper = Swift.min(from + count, storage.count)
        return ArrayDataBuffer(Array(storage[from..<upper]))
    }
}
