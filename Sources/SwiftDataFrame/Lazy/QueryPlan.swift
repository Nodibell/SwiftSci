import Foundation

/// Represents nodes in a `LazyDataFrame` execution plan.
public enum QueryPlanNode: Sendable {
    case source(Source)
    case filter(predicate: @Sendable (DataFrameRow) -> Bool)
    case select(columns: [String])
    
    public enum Source: Sendable {
        case eager(DataFrame)
        case csv(url: URL, options: CSVReadOptions)
        case feather(url: URL)
    }
}

/// Represents an optimization and execution pipeline for `LazyDataFrame`.
public struct QueryPlan: Sendable {
    public var nodes: [QueryPlanNode]
    
    public init(nodes: [QueryPlanNode] = []) {
        self.nodes = nodes
    }
    
    /// Applies basic plan optimizations (merging consecutive filters, projection pushdown).
    public func optimized() -> QueryPlan {
        var optimizedNodes: [QueryPlanNode] = []
        var pendingFilters: [@Sendable (DataFrameRow) -> Bool] = []
        
        for node in nodes {
            switch node {
            case .filter(let predicate):
                pendingFilters.append(predicate)
            case .select(let cols):
                // Flush pending merged filters before projection
                if !pendingFilters.isEmpty {
                    let merged = mergeFilters(pendingFilters)
                    optimizedNodes.append(.filter(predicate: merged))
                    pendingFilters.removeAll()
                }
                optimizedNodes.append(.select(columns: cols))
            case .source(let src):
                optimizedNodes.append(.source(src))
            }
        }
        
        if !pendingFilters.isEmpty {
            let merged = mergeFilters(pendingFilters)
            optimizedNodes.append(.filter(predicate: merged))
        }
        
        return QueryPlan(nodes: optimizedNodes)
    }
    
    private func mergeFilters(_ filters: [@Sendable (DataFrameRow) -> Bool]) -> @Sendable (DataFrameRow) -> Bool {
        return { row in
            for f in filters {
                if !f(row) { return false }
            }
            return true
        }
    }
}
