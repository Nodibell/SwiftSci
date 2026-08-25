import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// A memory-mapped file reader providing low-overhead, out-of-core file access.
///
/// `MemoryMappedReader` maps a file directly into the virtual address space using POSIX `mmap`,
/// allowing high-throughput tabular reading and chunk partitioning without loading entire files into physical RAM.
public final class MemoryMappedReader: @unchecked Sendable {
    /// The URL of the mapped file.
    public let fileURL: URL
    /// The size of the mapped region in bytes.
    public let size: Int

    private let fileDescriptor: Int32
    private let mappedPointer: UnsafeMutableRawPointer?

    /// Initializes a memory-mapped file reader for the file at the specified URL.
    ///
    /// - Parameter url: The file URL of the target file.
    /// - Throws: `SwiftMLError.fileNotFound` if the file cannot be opened, or `SwiftMLError.ioError` if `mmap` fails.
    public init(url: URL) throws {
        self.fileURL = url
        let path = url.path

        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw SwiftMLError.fileNotFound(url)
        }

        var statBuffer = stat()
        guard fstat(fd, &statBuffer) == 0 else {
            close(fd)
            throw SwiftMLError.parseError(line: 0, description: "Failed to stat file at \(path)")
        }

        let fileSize = Int(statBuffer.st_size)
        self.fileDescriptor = fd
        self.size = fileSize

        if fileSize == 0 {
            self.mappedPointer = nil
        } else {
            let ptr = mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0)
            if ptr == MAP_FAILED {
                close(fd)
                throw SwiftMLError.parseError(line: 0, description: "Failed to mmap file of size \(fileSize) bytes at \(path)")
            }
            self.mappedPointer = ptr
        }
    }

    deinit {
        if let ptr = mappedPointer, size > 0 {
            munmap(ptr, size)
        }
        if fileDescriptor >= 0 {
            close(fileDescriptor)
        }
    }

    /// Provides access to the raw memory buffer of the mapped file.
    ///
    /// - Parameter body: A closure taking an `UnsafeRawBufferPointer` to the mapped memory.
    /// - Returns: The value returned by `body`.
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        guard let ptr = mappedPointer, size > 0 else {
            return try body(UnsafeRawBufferPointer(start: nil, count: 0))
        }
        let buffer = UnsafeRawBufferPointer(start: ptr, count: size)
        return try body(buffer)
    }

    /// Partitions the memory-mapped file into byte ranges aligned to line endings (`\n`).
    ///
    /// - Parameter targetChunkByteSize: The approximate target byte size for each partition.
    /// - Returns: An array of byte ranges suitable for parallel or streaming parsing.
    public func partitionByLines(targetChunkByteSize: Int = 16 * 1024 * 1024) -> [Range<Int>] {
        guard size > 0, let ptr = mappedPointer else { return [] }
        let buffer = UnsafeRawBufferPointer(start: ptr, count: size)
        var ranges: [Range<Int>] = []
        var currentStart = 0

        while currentStart < size {
            var targetEnd = min(currentStart + targetChunkByteSize, size)
            if targetEnd < size {
                // Find next newline character '\n' (0x0A) to ensure boundary falls on a complete line
                while targetEnd < size && buffer[targetEnd] != 0x0A {
                    targetEnd += 1
                }
                if targetEnd < size {
                    targetEnd += 1 // Include newline character
                }
            }
            ranges.append(currentStart..<targetEnd)
            currentStart = targetEnd
        }

        return ranges
    }

    /// Extracts a `Data` copy of a specific byte subrange from the memory-mapped region.
    ///
    /// - Parameter range: The byte range to copy.
    /// - Returns: `Data` containing the requested bytes.
    public func data(in range: Range<Int>) -> Data {
        guard let ptr = mappedPointer else { return Data() }
        let clamped = max(0, range.lowerBound)..<min(size, range.upperBound)
        guard clamped.count > 0 else { return Data() }
        return Data(bytes: ptr.advanced(by: clamped.lowerBound), count: clamped.count)
    }
}
