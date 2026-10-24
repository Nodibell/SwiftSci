import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV demand and ownership")
struct CSVReadCursorTests {
    @Test("Opening waits for the first request and one mapping supplies later chunks")
    func demandAndMapping() async throws {
        let url = temporaryURL()
        let original = url.appendingPathExtension("original")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: original)
        }
        let stream = DataFrame.readCSVStream(contentsOf: url, chunkSize: 1)
        try Data("id,text\n1,alpha\n2,beta\n3,gamma\n".utf8).write(to: url)
        var iterator = stream.makeAsyncIterator()
        let first = try #require(try await iterator.next())
        try FileManager.default.moveItem(at: url, to: original)
        try Data("id,text\n9,replacement\n".utf8).write(to: url)
        let second = try #require(try await iterator.next())
        let third = try #require(try await iterator.next())
        #expect(try await iterator.next() == nil)
        try FileManager.default.removeItem(at: original)
        #expect(first[column: "text", as: String.self]?.values == ["alpha"])
        #expect(second[column: "text", as: String.self]?.values == ["beta"])
        #expect(third[column: "id", as: Int64.self]?.values == [3])
    }

    @Test("Reader failures finish the cursor rather than reopening the source")
    func failedCursor() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let cursor = CSVReadCursor(url: url, options: .default, chunkSize: 1)
        #expect(throws: SwiftMLError.self) { try cursor.next() }
        try Data("id\n1\n".utf8).write(to: url)
        #expect(try cursor.next() == nil)
    }

    @Test("Invalid chunk sizes fail without starting a read", arguments: [0, -1])
    func invalidSize(_ size: Int) throws {
        let cursor = CSVReadCursor(url: temporaryURL(), options: .default, chunkSize: size)
        #expect(throws: SwiftMLError.invalidParameter("CSV chunk size must be positive")) {
            try cursor.next()
        }
        #expect(try cursor.next() == nil)
    }

    @Test("Direct cancellation finishes a retained cursor")
    func cancellation() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("id\n1\n2\n".utf8).write(to: url)
        let cursor = CSVReadCursor(url: url, options: .default, chunkSize: 1)
        #expect(try cursor.next()?.rowCount == 1)
        let gate = Gate()
        let task = Task {
            await gate.wait()
            return try cursor.next()
        }
        task.cancel()
        await gate.open()
        do {
            _ = try await task.value
            Issue.record("A cancelled cursor request must throw")
        } catch is CancellationError { }
        #expect(try cursor.next() == nil)
    }

    @Test("Cancelled unfolding releases its source even when the stream is retained")
    func cancelledStreamOwnership() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("id\n1\n2\n".utf8).write(to: url)
        var cursor: CSVReadCursor? = CSVReadCursor(url: url, options: .default, chunkSize: 1)
        weak var observed = cursor
        let stream = unfolding(try #require(cursor))
        cursor = nil
        var iterator = stream.makeAsyncIterator()
        #expect(try await iterator.next()?.rowCount == 1)
        #expect(observed != nil)
        let gate = Gate()
        let task = Task {
            await gate.wait()
            var next = stream.makeAsyncIterator()
            return try await next.next()
        }
        task.cancel()
        await gate.open()
        do {
            #expect(try await task.value == nil)
        } catch is CancellationError { }
        #expect(observed == nil)
        withExtendedLifetime(stream) {}
        withExtendedLifetime(iterator) {}
    }

    private func unfolding(_ cursor: CSVReadCursor) -> AsyncThrowingStream<DataFrame, any Error> {
        AsyncThrowingStream(unfolding: { try cursor.next() })
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("csv-demand-\(UUID().uuidString).csv")
    }

    private actor Gate {
        private var opened = false
        private var continuation: CheckedContinuation<Void, Never>?
        func wait() async {
            if opened { return }
            await withCheckedContinuation { continuation = $0 }
        }
        func open() {
            opened = true
            continuation?.resume()
            continuation = nil
        }
    }
}
