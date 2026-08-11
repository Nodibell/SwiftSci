import Foundation

/// A lazy-evaluated representation of a `DataFrame` query pipeline.
public struct LazyDataFrame: Sendable {
    /// The underlying evaluation query plan node sequence.
    public let plan: QueryPlan

    /// Initializes a lazy DataFrame wrapper around a query plan.
    /// - Parameter plan: The evaluation query plan.
    public init(plan: QueryPlan) {
        self.plan = plan
    }

    /// Initializes a lazy DataFrame from an abstract query source node.
    /// - Parameter source: The source data provider node.
    public init(source: QueryPlanNode.Source) {
        self.plan = QueryPlan(nodes: [.source(source)])
    }

    /// Initializes a lazy DataFrame from an eager memory-resident DataFrame.
    /// - Parameter dataFrame: The eager source DataFrame.
    public init(dataFrame: DataFrame) {
        self.init(source: .eager(dataFrame))
    }
    
    // MARK: – Lazy Transformations
    
    /// Lazily filters rows using a predicate. Returns a new `LazyDataFrame`.
    public func filter(_ predicate: @escaping @Sendable (DataFrameRow) -> Bool) -> LazyDataFrame {
        var updatedNodes = plan.nodes
        updatedNodes.append(.filter(predicate: predicate))
        return LazyDataFrame(plan: QueryPlan(nodes: updatedNodes))
    }
    
    /// Lazily selects a subset of columns by name. Returns a new `LazyDataFrame`.
    public func select(_ columns: String...) -> LazyDataFrame {
        select(columns)
    }
    
    /// Lazily selects a subset of columns by array of names. Returns a new `LazyDataFrame`.
    public func select(_ columns: [String]) -> LazyDataFrame {
        var updatedNodes = plan.nodes
        updatedNodes.append(.select(columns: columns))
        return LazyDataFrame(plan: QueryPlan(nodes: updatedNodes))
    }
    
    // MARK: – Execution / Evaluation
    
    /// Evaluates the optimized execution plan and returns an eager `DataFrame`.
    public func collect() async throws -> DataFrame {
        let optPlan = plan.optimized()
        var currentDF: DataFrame?
        
        for node in optPlan.nodes {
            switch node {
            case .source(let source):
                switch source {
                case .eager(let df):
                    currentDF = df
                case .csv(let url, let options):
                    currentDF = try await CSVReader.read(url: url, options: options)
                case .feather(let url):
                    currentDF = try await FeatherReader.read(url: url)
                }
            case .filter(let predicate):
                guard let df = currentDF else {
                    throw SwiftMLError.unsupportedOperation("Cannot apply filter on an uninitialized source")
                }
                currentDF = df.filter(predicate)
            case .select(let cols):
                guard let df = currentDF else {
                    throw SwiftMLError.unsupportedOperation("Cannot apply select on an uninitialized source")
                }
                currentDF = try df.select(cols)
            }
        }
        
        return currentDF ?? DataFrame.empty
    }
}

extension DataFrame {
    /// Converts an eager `DataFrame` into a `LazyDataFrame`.
    public func lazy() -> LazyDataFrame {
        LazyDataFrame(dataFrame: self)
    }
    
    /// Creates a `LazyDataFrame` reading from a CSV file.
    public static func lazyCSV(url: URL, options: CSVReadOptions = .default) -> LazyDataFrame {
        LazyDataFrame(source: .csv(url: url, options: options))
    }
    
    /// Creates a `LazyDataFrame` reading from a Feather file.
    public static func lazyFeather(url: URL) -> LazyDataFrame {
        LazyDataFrame(source: .feather(url: url))
    }
}
