import Foundation

/// Encodes categorical values into integer labels [0, C-1].
public final class LabelEncoder: @unchecked Sendable {
    public private(set) var classes: [String] = []
    
    public init() {}
    
    /// Fits the LabelEncoder on the input categories.
    public func fit(_ categories: [String]) {
        let uniqueSorted = Array(Set(categories)).sorted()
        self.classes = uniqueSorted
    }
    
    /// Transforms category strings into integer labels.
    public func transform(_ categories: [String]) throws -> [Int] {
        guard !classes.isEmpty else {
            throw PreprocessingError.fitNotCalled
        }
        
        var labels = [Int]()
        labels.reserveCapacity(categories.count)
        
        // Use a dictionary for fast lookup of classes
        let lookup = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($0.element, $0.offset) })
        
        for category in categories {
            guard let idx = lookup[category] else {
                throw PreprocessingError.unknownCategory(category)
            }
            labels.append(idx)
        }
        
        return labels
    }
    
    /// Reverses the transform mapping back to the original string labels.
    public func inverseTransform(_ labels: [Int]) throws -> [String] {
        guard !classes.isEmpty else {
            throw PreprocessingError.fitNotCalled
        }
        
        var categories = [String]()
        categories.reserveCapacity(labels.count)
        
        for label in labels {
            guard label >= 0 && label < classes.count else {
                throw PreprocessingError.unknownCategory(String(label))
            }
            categories.append(classes[label])
        }
        
        return categories
    }
    
    /// Fits to categories, then transforms it.
    public func fitTransform(_ categories: [String]) throws -> [Int] {
        fit(categories)
        return try transform(categories)
    }
}
