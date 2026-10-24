import Foundation

/// A sequence of `DataFrame` chunks for out-of-core and memory-efficient streaming data processing.
///
/// `ChunkedDataFrame` conforms to `AsyncSequence` and enables lazy streaming, batch filtering,
/// column selection, and transformations without loading the entire dataset into memory at once.
public struct ChunkedDataFrame: AsyncSequence, Sendable {
    /// The element type of each streamed chunk.
    public typealias Element = DataFrame

    private let iteratorProducer: @Sendable () -> AsyncIterator

    /// Initializes a `ChunkedDataFrame` from a stream generator closure.
    ///
    /// - Parameter streamProducer: A closure returning a new `AsyncThrowingStream` of `DataFrame` chunks.
    public init(streamProducer: @escaping @Sendable () -> AsyncThrowingStream<DataFrame, any Error>) {
        self.iteratorProducer = { AsyncIterator(iterator: streamProducer().makeAsyncIterator()) }
    }

    private init(iteratorProducer: @escaping @Sendable () -> AsyncIterator) {
        self.iteratorProducer = iteratorProducer
    }

    /// Initializes a `ChunkedDataFrame` from an in-memory array of `DataFrame` chunks.
    ///
    /// - Parameter chunks: An array of `DataFrame` chunks.
    public init(chunks: [DataFrame]) {
        self.init(streamProducer: {
            AsyncThrowingStream { continuation in
                for chunk in chunks {
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
        })
    }

    /// Initializes a `ChunkedDataFrame` from a single `DataFrame`.
    ///
    /// - Parameter dataFrame: The base DataFrame.
    public init(dataFrame: DataFrame) {
        self.init(chunks: [dataFrame])
    }

    // MARK: – AsyncSequence Conformance

    /// The async iterator for streaming DataFrame chunks.
    public struct AsyncIterator: AsyncIteratorProtocol {
        private indirect enum State {
            case source(AsyncThrowingStream<DataFrame, any Error>.Iterator)
            case transformed(AsyncIterator, @Sendable (DataFrame) async throws -> DataFrame?)
            case finished
        }

        private var state: State

        init(iterator: AsyncThrowingStream<DataFrame, any Error>.Iterator) {
            state = .source(iterator)
        }

        fileprivate init(upstream: AsyncIterator,
                         transform: @escaping @Sendable (DataFrame) async throws -> DataFrame?) {
            state = .transformed(upstream, transform)
        }

        /// Fetches the next DataFrame chunk in the sequence.
        /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
        /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
        public mutating func next() async throws -> DataFrame? {
            let current = state
            state = .finished
            try Task.checkCancellation()
            switch current {
            case .finished: return nil
            case .source(var iterator):
                let result = try await iterator.next()
                try Task.checkCancellation()
                if result != nil { state = .source(iterator) }
                return result
            case .transformed(var upstream, let transform):
                while let chunk = try await upstream.next() {
                    try Task.checkCancellation()
                    if let result = try await transform(chunk) {
                        try Task.checkCancellation()
                        state = .transformed(upstream, transform)
                        return result
                    }
                }
                return nil
            }
        }
    }

    /// Creates an asynchronous iterator over the DataFrame chunks.
    /// - Returns: The computed AsyncIterator result instance.
    public func makeAsyncIterator() -> AsyncIterator {
        iteratorProducer()
    }

    // MARK: – Factory Methods

    /// Creates a `ChunkedDataFrame` by streaming a CSV file in chunks.
    ///
    /// - Parameters:
    ///   - url: The file URL of the CSV file.
    ///   - chunkSize: The number of rows per chunk (default: 10,000).
    ///   - options: CSV parsing options.
    /// - Returns: A `ChunkedDataFrame` streaming the CSV file.
    public static func fromCSV(
        url: URL,
        chunkSize: Int = 10_000,
        options: CSVReadOptions = .default
    ) -> ChunkedDataFrame {
        ChunkedDataFrame {
            DataFrame.readCSVStream(contentsOf: url, chunkSize: chunkSize, options: options)
        }
    }

    // MARK: – Transformations

    /// Lazily filters rows across all chunks using a predicate.
    ///
    /// - Parameter predicate: A closure evaluating each row.
    /// - Returns: A new `ChunkedDataFrame` emitting filtered chunks.
    public func filter(_ predicate: @escaping @Sendable (DataFrameRow) -> Bool) -> ChunkedDataFrame {
        transformed { chunk in
            let filtered = chunk.filter(predicate)
            return filtered.rowCount > 0 ? filtered : nil
        }
    }

    /// Lazily selects a subset of columns from each chunk.
    ///
    /// - Parameter columns: The column names to retain.
    /// - Returns: A new `ChunkedDataFrame` emitting chunks with the selected columns.
    public func select(_ columns: [String]) -> ChunkedDataFrame {
        transformed { try $0.select(columns) }
    }

    /// Lazily selects a subset of columns from each chunk via variadic arguments.
    ///
    /// - Parameter columns: The column names to retain.
    /// - Returns: A new `ChunkedDataFrame` emitting chunks with the selected columns.
    public func select(_ columns: String...) -> ChunkedDataFrame {
        select(columns)
    }

    /// Transforms each chunk using an asynchronous mapping closure.
    ///
    /// - Parameter transform: An asynchronous closure transforming each `DataFrame` chunk.
    /// - Returns: A new `ChunkedDataFrame` emitting transformed chunks.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    public func mapChunk(
        _ transform: @escaping @Sendable (DataFrame) async throws -> DataFrame
    ) -> ChunkedDataFrame {
        transformed { try await transform($0) }
    }

    private func transformed(
        _ transform: @escaping @Sendable (DataFrame) async throws -> DataFrame?
    ) -> ChunkedDataFrame {
        let upstream = iteratorProducer
        return ChunkedDataFrame(iteratorProducer: {
            AsyncIterator(upstream: upstream(), transform: transform)
        })
    }

    // MARK: – Terminal Operations

    /// Collects and concatenates all chunks into a single unified `DataFrame`.
    ///
    /// > [!WARNING]
    /// > `collect()` is an eager in-memory materialization operation that loads all partitions into RAM.
    /// > For datasets exceeding available host memory, avoid `collect()` and process partitions via
    /// > `forEachChunk(_:)` or streaming sinks.
    ///
    /// - Returns: A single `DataFrame` containing all rows from all chunks.
    /// - Throws: Any error thrown during streaming or schema alignment.
    public func collect() async throws -> DataFrame {
        var collectedChunks: [DataFrame] = []
        for try await chunk in self {
            if chunk.rowCount > 0 {
                collectedChunks.append(chunk)
            }
        }
        return try DataFrame.concat(collectedChunks)
    }

    /// Computes the total number of rows across all chunks without retaining all chunks in memory.
    ///
    /// - Returns: Total row count.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    public func rowCount() async throws -> Int {
        var total = 0
        for try await chunk in self {
            let (newTotal, overflow) = total.addingReportingOverflow(chunk.rowCount)
            guard !overflow else {
                throw SwiftMLError.invalidParameter("ChunkedDataFrame total row count overflows Int")
            }
            total = newTotal
        }
        return total
    }

    /// Iterates over every chunk in the stream.
    ///
    /// - Parameter body: An asynchronous closure executed for each chunk.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    public func forEachChunk(
        _ body: @escaping @Sendable (DataFrame) async throws -> Void
    ) async throws {
        for try await chunk in self {
            try await body(chunk)
        }
    }
    /// Performs an out-of-core hash join between each chunk in this sequence and an in-memory right `DataFrame`.
    ///
    /// - Parameters:
    ///   - other: The right-side `DataFrame` (typically smaller or pre-loaded dimension table).
    ///   - key: The common join column name.
    ///   - how: The kind of join (.inner, .left).
    /// - Returns: A new `ChunkedDataFrame` yielding the joined partitions.
    public func join(
        _ other: DataFrame,
        on key: String,
        how: JoinKind = .inner
    ) -> ChunkedDataFrame {
        guard how == .inner || how == .left else {
            return ChunkedDataFrame {
                AsyncThrowingStream(unfolding: {
                    throw SwiftMLError.unsupportedOperation(
                        "Streaming join only supports .inner and .left against an in-memory dimension table. " +
                        "(.right and .outer require global tracking of right-side matches to avoid duplicating unmatched rows per chunk)"
                    )
                })
            }
        }
        return transformed { try $0.join(other, on: key, how: how) }
    }
}

extension DataFrame {
    /// Converts an eager `DataFrame` into a `ChunkedDataFrame` with a single chunk.
    /// - Returns: A new `DataFrame` containing the transformed columns and computed results.
    public func chunked() -> ChunkedDataFrame {
        ChunkedDataFrame(dataFrame: self)
    }

    /// Splits the `DataFrame` into a `ChunkedDataFrame` with batches of specified size.
    ///
    /// - Parameter chunkSize: The maximum number of rows per chunk.
    /// - Returns: A `ChunkedDataFrame` emitting the sliced chunks.
    public func chunked(by chunkSize: Int) -> ChunkedDataFrame {
        guard chunkSize > 0 else { return ChunkedDataFrame(dataFrame: self) }
        let total = self.rowCount
        guard total > 0 else { return ChunkedDataFrame(dataFrame: self) }

        var chunks: [DataFrame] = []
        var start = 0
        while start < total {
            let end = min(start + chunkSize, total)
            let indices = Array(start..<end)
            let slicedCols = self.columns.map { $0.gathered(at: indices) }
            if let slicedDF = try? DataFrame(columns: slicedCols) {
                chunks.append(slicedDF)
            }
            start = end
        }
        return ChunkedDataFrame(chunks: chunks)
    }

    /// Concatenates multiple DataFrames with compatible schemas into a single unified DataFrame.
    ///
    /// - Parameter dataFrames: An array of DataFrames to concatenate.
    /// - Returns: A single unified DataFrame containing all rows.
    /// - Throws: `SwiftMLError.emptySchema` if the list is empty, or `SwiftMLError.columnNotFound` if column schemas mismatch.
    public static func concat(_ dataFrames: [DataFrame]) throws -> DataFrame {
        guard let first = dataFrames.first(where: { $0.rowCount > 0 || $0.columnNames.count > 0 }) else {
            return DataFrame.empty
        }

        let nonEmpties = dataFrames.filter { $0.rowCount > 0 }
        guard !nonEmpties.isEmpty else { return first }
        if nonEmpties.count == 1 { return nonEmpties[0] }

        let schemaColumns = first.columnNames
        let totalRows = nonEmpties.reduce(0) { $0 + $1.rowCount }

        var combinedColumns: [any AnyColumn] = []
        combinedColumns.reserveCapacity(schemaColumns.count)

        for colName in schemaColumns {
            guard let firstCol = first[column: colName] else {
                throw SwiftMLError.columnNotFound(colName)
            }

            var mergedValues: [Any?] = []
            mergedValues.reserveCapacity(totalRows)

            for df in nonEmpties {
                guard let col = df[column: colName] else {
                    throw SwiftMLError.columnNotFound(colName)
                }
                for r in 0..<col.count {
                    mergedValues.append(col.value(at: r))
                }
            }

            let combinedCol = makeColumn(name: colName, dtype: firstCol.dtype, rawValues: mergedValues)
            combinedColumns.append(combinedCol)
        }

        return try DataFrame(columns: combinedColumns)
    }


}
