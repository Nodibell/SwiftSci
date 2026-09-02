import Foundation

/// A high-performance, pure-Swift parser for NumPy binary tensor files (`.npy`).
///
/// Supports NumPy NPY format version 1.0 and 2.0 with multidimensional arrays,
/// arbitrary shapes, little-endian primitive numerical types (`<f8`, `<f4`, `<i8`, `<i4`, `|u1`, `|b1`),
/// and zero-copy conversion into `DataFrame` columnar tables.
public struct NPYArray: Sendable {
    /// The dimension extents of the tensor (e.g. `[100, 8, 8]` or `[500, 10]`).
    public let shape: [Int]
    /// The NumPy type descriptor string (e.g. `'<f8'`, `'<i8'`, `'<f4'`).
    public let descr: String
    /// Whether the array was written in Fortran column-major order.
    public let fortranOrder: Bool
    /// The raw binary payload following the NPY header.
    public let data: Data

    /// Total number of scalar elements in the multidimensional array.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    /// Converts the binary payload into an array of `Double` values.
    public func toDoubles() -> [Double] {
        let total = elementCount
        guard total > 0, !data.isEmpty else { return [] }

        return data.withUnsafeBytes { rawBuf in
            guard let basePtr = rawBuf.baseAddress else { return [] }

            if descr.contains("f8") {
                let count = min(total, rawBuf.count / 8)
                var result = [Double](repeating: 0.0, count: count)
                _ = result.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 8)
                }
                return result
            } else if descr.contains("f4") {
                let count = min(total, rawBuf.count / 4)
                var floats = [Float](repeating: 0.0, count: count)
                _ = floats.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 4)
                }
                return floats.map { Double($0) }
            } else if descr.contains("i8") {
                let count = min(total, rawBuf.count / 8)
                var ints = [Int64](repeating: 0, count: count)
                _ = ints.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 8)
                }
                return ints.map { Double($0) }
            } else if descr.contains("i4") {
                let count = min(total, rawBuf.count / 4)
                var ints = [Int32](repeating: 0, count: count)
                _ = ints.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 4)
                }
                return ints.map { Double($0) }
            } else {
                let count = min(total, rawBuf.count)
                let bytes = rawBuf.bindMemory(to: UInt8.self)
                return bytes.prefix(count).map { Double($0) }
            }
        }
    }

    /// Converts the binary payload into an array of `Int64` values.
    public func toInt64s() -> [Int64] {
        let total = elementCount
        guard total > 0, !data.isEmpty else { return [] }

        return data.withUnsafeBytes { rawBuf in
            guard let basePtr = rawBuf.baseAddress else { return [] }

            if descr.contains("i8") {
                let count = min(total, rawBuf.count / 8)
                var result = [Int64](repeating: 0, count: count)
                _ = result.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 8)
                }
                return result
            } else if descr.contains("i4") {
                let count = min(total, rawBuf.count / 4)
                var ints = [Int32](repeating: 0, count: count)
                _ = ints.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 4)
                }
                return ints.map { Int64($0) }
            } else if descr.contains("f8") {
                let count = min(total, rawBuf.count / 8)
                var doubles = [Double](repeating: 0.0, count: count)
                _ = doubles.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 8)
                }
                return doubles.map { Int64($0) }
            } else if descr.contains("f4") {
                let count = min(total, rawBuf.count / 4)
                var floats = [Float](repeating: 0.0, count: count)
                _ = floats.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, basePtr, count * 4)
                }
                return floats.map { Int64($0) }
            } else {
                let count = min(total, rawBuf.count)
                let bytes = rawBuf.bindMemory(to: UInt8.self)
                return bytes.prefix(count).map { Int64($0) }
            }
        }
    }

    /// Converts this tensor array into a `DataFrame`.
    ///
    /// - 1D array `(N,)`: yields a single column `"value"` of length `N`.
    /// - 2D array `(N, M)`: yields `M` columns (`"col_0" ... "col_M-1"`) with `N` rows each.
    /// - 3D+ array `(N, D1, D2, ...)`: flattens inner dimensions to `(N, D1*D2)` columns (`"pixel_0" ...`).
    public func toDataFrame(columnPrefix: String = "col") throws -> DataFrame {
        guard !shape.isEmpty else { return DataFrame.empty }

        if shape.count == 1 {
            let values = toDoubles()
            let col = TypedColumn(name: "value", values: values)
            return try DataFrame(columns: [col])
        }

        let numRows = shape[0]
        let numCols = shape.dropFirst().reduce(1, *)
        let doubles = toDoubles()

        var cols: [any AnyColumn] = []
        cols.reserveCapacity(numCols)

        for colIdx in 0..<numCols {
            var colVals: [Double] = []
            colVals.reserveCapacity(numRows)
            for rowIdx in 0..<numRows {
                let idx = rowIdx * numCols + colIdx
                colVals.append(idx < doubles.count ? doubles[idx] : 0.0)
            }
            let name = shape.count > 2 ? "pixel_\(colIdx)" : "\(columnPrefix)_\(colIdx)"
            cols.append(TypedColumn(name: name, values: colVals))
        }

        return try DataFrame(columns: cols)
    }
}

/// Pure-Swift reader for `.npy` files.
public enum NPYReader: Sendable {

    /// Reads a `.npy` file from a local URL.
    public static func read(url: URL) throws -> NPYArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SwiftMLError.fileNotFound(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try read(data: data)
    }

    /// Reads an NPY tensor array from raw data bytes.
    public static func read(data: Data) throws -> NPYArray {
        guard data.count >= 10 else {
            throw SwiftMLError.parseError(line: 0, description: "NPY file too small (< 10 bytes)")
        }

        // 1. Verify magic: \x93NUMPY
        let magic = data.subdata(in: 0..<6)
        let expectedMagic = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        guard magic == expectedMagic else {
            throw SwiftMLError.parseError(line: 0, description: "Invalid NPY magic prefix; expected '\\x93NUMPY'")
        }

        let major = data[6]
        let headerLen: Int
        let headerStart: Int

        if major == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else if major == 2 {
            guard data.count >= 14 else {
                throw SwiftMLError.parseError(line: 0, description: "Corrupted NPY v2 header length")
            }
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        } else {
            throw SwiftMLError.parseError(line: 0, description: "Unsupported NPY major version \(major)")
        }

        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else {
            throw SwiftMLError.parseError(line: 0, description: "NPY header exceeds data bounds")
        }

        let headerData = data.subdata(in: headerStart..<headerEnd)
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw SwiftMLError.parseError(line: 0, description: "Failed to decode ASCII NPY header dictionary")
        }

        // 2. Parse dictionary attributes: 'descr', 'fortran_order', 'shape'
        let descr = parseDescr(from: headerStr)
        let fortranOrder = parseFortranOrder(from: headerStr)
        let shape = parseShape(from: headerStr)

        // 3. Payload
        let payload = data.subdata(in: headerEnd..<data.count)

        return NPYArray(shape: shape, descr: descr, fortranOrder: fortranOrder, data: payload)
    }

    private static func parseDescr(from str: String) -> String {
        guard let range = str.range(of: "'descr':") ?? str.range(of: "\"descr\":") else { return "<f8" }
        let after = str[range.upperBound...]
        if let quote1 = after.firstIndex(where: { $0 == "'" || $0 == "\"" }) {
            let start = after.index(after: quote1)
            if let quote2 = after[start...].firstIndex(where: { $0 == "'" || $0 == "\"" }) {
                return String(after[start..<quote2])
            }
        }
        return "<f8"
    }

    private static func parseFortranOrder(from str: String) -> Bool {
        guard let range = str.range(of: "fortran_order") else { return false }
        let after = str[range.upperBound...]
        if let colon = after.firstIndex(of: ":") {
            let valStr = after[after.index(after: colon)...].prefix(10).lowercased()
            return valStr.contains("true")
        }
        return false
    }

    private static func parseShape(from str: String) -> [Int] {
        guard let range = str.range(of: "shape") else { return [] }
        let after = str[range.upperBound...]
        guard let openParen = after.firstIndex(of: "("),
              let closeParen = after[openParen...].firstIndex(of: ")") else {
            return []
        }
        let tupleContent = after[after.index(after: openParen)..<closeParen]
        let tokens = tupleContent.components(separatedBy: ",")
        return tokens.compactMap {
            let trimmed = $0.trimmingCharacters(in: .whitespacesAndNewlines)
            return Int(trimmed)
        }
    }
}
