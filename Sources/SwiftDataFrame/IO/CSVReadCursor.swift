import Foundation

/// Owns the mapping between requests; pointers and record metadata stay inside one request.
internal final class CSVReadCursor: @unchecked Sendable {
    private struct Input {
        let data: Data
        let headers: [String]
        var offset: Int
        var remainingRows: Int?
    }

    private enum State {
        case unopened
        case reading(Input)
        case finished
    }

    private let lock = NSLock()
    private let url: URL
    private let options: CSVReadOptions
    private let chunkSize: Int
    private var state: State = .unopened

    init(url: URL, options: CSVReadOptions, chunkSize: Int) {
        self.url = url
        self.options = options
        self.chunkSize = chunkSize
    }

    func next() throws -> DataFrame? {
        try lock.withLock {
            do {
                try Task.checkCancellation()
                let input: Input
                switch state {
                case .finished: return nil
                case .unopened:
                    guard chunkSize > 0 else {
                        throw SwiftMLError.invalidParameter("CSV chunk size must be positive")
                    }
                    guard let opened = try open() else {
                        state = .finished
                        return nil
                    }
                    input = opened
                case .reading(let current): input = current
                }
                return try read(input)
            } catch {
                state = .finished
                throw error
            }
        }
    }

    private func open() throws -> Input? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SwiftMLError.fileNotFound(url)
        }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        return try data.withUnsafeBytes { rawBuffer in
            let buffer = rawBuffer.bindMemory(to: UInt8.self)
            let first = try parser.scanBatch(buffer: buffer, maximumRows: 1)
            guard !first.records.isEmpty else { return nil }
            let headers: [String]
            let offset: Int
            if options.hasHeader {
                headers = CSVReader.deduplicateHeaders(first.records.row(at: 0).map {
                    VectorizedByteParsers.parseString(buffer: buffer, offset: $0)
                })
                offset = first.nextOffset
            } else {
                headers = (0..<first.records.row(at: 0).count).map { "col\($0)" }
                offset = 0
            }
            guard !headers.isEmpty, offset < data.count, options.maxRows != 0 else { return nil }
            return Input(data: data, headers: headers, offset: offset, remainingRows: options.maxRows)
        }
    }

    private var parser: SystemsCSVParser {
        SystemsCSVParser(delimiterByte: UInt8(options.delimiter.utf8.first ?? 44))
    }

    private func read(_ current: Input) throws -> DataFrame? {
        var input = current
        let rows = min(chunkSize, input.remainingRows ?? chunkSize)
        guard rows > 0 else {
            state = .finished
            return nil
        }
        let data = input.data
        let frame = try data.withUnsafeBytes { rawBuffer -> DataFrame? in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            let buffer = UnsafeBufferPointer(rebasing: bytes[input.offset...])
            let batch = try parser.scanBatch(buffer: buffer, maximumRows: rows)
            input.offset += batch.nextOffset
            guard !batch.records.isEmpty else { return nil }
            let result = try CSVReader.buildFrame(buffer: buffer, records: batch.records,
                startRow: 0, rowCount: batch.records.count, headers: input.headers, options: options)
            if let remaining = input.remainingRows {
                input.remainingRows = remaining - batch.records.count
            }
            return result
        }
        try Task.checkCancellation()
        if frame == nil || input.offset == input.data.count || input.remainingRows == 0 {
            state = .finished
        } else {
            state = .reading(input)
        }
        return frame
    }
}
