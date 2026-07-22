import Foundation

/// Generates interaction features $x_i \cdot x_j$ for numerical feature matrices.
public final class InteractionFeatures: PreprocessingTransformer, Sendable {
    public init() {}
    
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
    }
    
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let numFeatures = data[0].count
        
        return data.map { row in
            var newRow = row
            for i in 0..<numFeatures {
                for j in (i+1)..<numFeatures {
                    newRow.append(row[i] * row[j])
                }
            }
            return newRow
        }
    }
}

/// Feature transformer that extracts calendar date components (year, month, day, dayOfWeek, isWeekend) from Date objects.
public final class DateFeatures: Sendable {
    public init() {}
    
    /// Transforms array of Date objects into numerical feature matrix [year, month, day, dayOfWeek, isWeekend].
    public func transform(dates: [Date]) -> [[Double]] {
        let calendar = Calendar.current
        return dates.map { date in
            let components = calendar.dateComponents([.year, .month, .day, .weekday], from: date)
            let year = Double(components.year ?? 0)
            let month = Double(components.month ?? 0)
            let day = Double(components.day ?? 0)
            let weekday = Double(components.weekday ?? 0) // 1: Sun, 7: Sat
            let isWeekend = (weekday == 1 || weekday == 7) ? 1.0 : 0.0
            return [year, month, day, weekday, isWeekend]
        }
    }
}
