import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Chunked consumer demand")
struct ChunkedDemandTests {
    @Test("Composed transforms request only the input needed for one output")
    func composedDemand() async throws {
        let frames = try [1, 2, 3].map { value -> DataFrame in
            let values: [Int64?] = [Int64(value)]
            return try DataFrame(columns: [TypedColumn<Int64>(name: "id", values: values)])
        }
        let source = PullSource(frames)
        let base = ChunkedDataFrame {
            AsyncThrowingStream(unfolding: { await source.next() })
        }
        let keys: [Int64?] = [2, 3]
        let labels: [String?] = ["two", "three"]
        let right = try DataFrame(columns: [TypedColumn<Int64>(name: "id", values: keys),
            TypedColumn<String>(name: "label", values: labels)])
        let pipeline = base.filter { row in (row["id"] as? Int64 ?? 0) >= 2 }
            .select("id").join(right, on: "id").mapChunk { $0 }
        var iterator = pipeline.makeAsyncIterator()
        #expect(await source.requests == 0)
        let first = try #require(try await iterator.next())
        #expect(first[column: "label", as: String.self]?.values == ["two"])
        #expect(await source.requests == 2)
        let second = try #require(try await iterator.next())
        #expect(second[column: "label", as: String.self]?.values == ["three"])
        #expect(await source.requests == 3)
        #expect(try await iterator.next() == nil)
        #expect(await source.requests == 4)
    }

    @Test("Mapping errors end the pipeline without reading another chunk")
    func mappingError() async throws {
        let source = PullSource([.empty, .empty])
        let base = ChunkedDataFrame {
            AsyncThrowingStream(unfolding: { await source.next() })
        }
        let pipeline = base.mapChunk { _ in throw Failure.expected }
        var iterator = pipeline.makeAsyncIterator()
        do {
            _ = try await iterator.next()
            Issue.record("Expected the mapping error")
        } catch Failure.expected { }
        #expect(await source.requests == 1)
        #expect(try await iterator.next() == nil)
        #expect(await source.requests == 1)
    }

    @Test("Unsupported joins fail even on an empty upstream")
    func emptyUnsupportedJoin() async throws {
        let source = PullSource([])
        let base = ChunkedDataFrame {
            AsyncThrowingStream(unfolding: { await source.next() })
        }
        var iterator = base.join(.empty, on: "id", how: .outer).makeAsyncIterator()
        do {
            _ = try await iterator.next()
            Issue.record("An unsupported join must fail on an empty stream")
        } catch is SwiftMLError { }
        #expect(await source.requests == 0)
    }

    private enum Failure: Error { case expected }

    private actor PullSource {
        let frames: [DataFrame]
        private(set) var requests = 0
        init(_ frames: [DataFrame]) { self.frames = frames }
        func next() -> DataFrame? {
            let index = requests
            requests += 1
            return index < frames.count ? frames[index] : nil
        }
    }
}
