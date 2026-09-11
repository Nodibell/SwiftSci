import Foundation

// MARK: - Dataset Modality

/// The structural archetype or semantic modality of a DataFrame dataset.
public enum DatasetModality: String, Sendable, Codable, CaseIterable {
    /// Purely numeric features suitable for dense linear algebra, vector clustering, or numerical regression.
    case tabularNumeric
    /// Heterogeneous tabular data containing both numeric features and categorical/string fields.
    case tabularMixed
    /// Natural language corpus dominated by unstructured textual documents or high-cardinality text strings.
    case pureTextNLP
    /// Temporal sequence characterized by timestamps, dates, or continuous sequential time intervals.
    case timeSeries
}

// MARK: - Modality Profile

/// Detailed diagnostic metrics explaining the inferred structural modality of a DataFrame.
public struct ModalityProfile: Sendable, Codable, Equatable {
    /// Inferred dataset archetype.
    public let modality: DatasetModality
    /// Proportion of numeric columns relative to total columns (0.0 to 1.0).
    public let numericColumnRatio: Double
    /// Proportion of text / string columns relative to total columns (0.0 to 1.0).
    public let textColumnRatio: Double
    /// Average character length across string columns.
    public let averageTextLength: Double
    /// Name of detected temporal column, if any.
    public let temporalColumn: String?
    
    /// Initializes a modality profile.
    /// - Parameters:
    ///   - modality: Inferred dataset modality.
    ///   - numericColumnRatio: Ratio of numeric columns.
    ///   - textColumnRatio: Ratio of text columns.
    ///   - averageTextLength: Mean string character length.
    ///   - temporalColumn: Name of detected timestamp/date column.
    public init(
        modality: DatasetModality,
        numericColumnRatio: Double,
        textColumnRatio: Double,
        averageTextLength: Double,
        temporalColumn: String?
    ) {
        self.modality = modality
        self.numericColumnRatio = numericColumnRatio
        self.textColumnRatio = textColumnRatio
        self.averageTextLength = averageTextLength
        self.temporalColumn = temporalColumn
    }
}

// MARK: - DataFrame Modality Extension

extension DataFrame {
    /// Infers the dataset modality archetype based on column types, temporal indicators, and text distribution.
    ///
    /// - Returns: Inferred `DatasetModality` enum case.
    public func inferModality() -> DatasetModality {
        profileModality().modality
    }
    
    /// Generates detailed diagnostic metrics regarding the dataset's structural modality.
    ///
    /// - Returns: A `ModalityProfile` detailing column ratios, text lengths, and detected time columns.
    public func profileModality() -> ModalityProfile {
        let cols = columns
        guard !cols.isEmpty else {
            return ModalityProfile(
                modality: .tabularMixed,
                numericColumnRatio: 0.0,
                textColumnRatio: 0.0,
                averageTextLength: 0.0,
                temporalColumn: nil
            )
        }
        
        var numericCount = 0
        var textCount = 0
        var temporalColumn: String? = nil
        
        var totalTextLength = 0
        var totalTextSamples = 0
        
        let temporalKeywords = ["timestamp", "datetime", "date", "time", "epoch", "year", "month"]
        
        for col in cols {
            if col.dtype == .date32 {
                temporalColumn = col.name
            } else if temporalKeywords.contains(where: { col.name.lowercased().contains($0) }) {
                if temporalColumn == nil {
                    temporalColumn = col.name
                }
            }
            
            if col.dtype.isNumeric || col.dtype == .boolean {
                numericCount += 1
            } else if col.dtype == .utf8 {
                textCount += 1
                if let strCol = col as? TypedColumn<String> {
                    let sampleLimit = min(strCol.count, 200)
                    for i in 0..<sampleLimit {
                        if let str = strCol.values[i] {
                            totalTextLength += str.count
                            totalTextSamples += 1
                        }
                    }
                }
            }
        }
        
        let totalCols = Double(cols.count)
        let numRatio = Double(numericCount) / totalCols
        let textRatio = Double(textCount) / totalCols
        let avgTextLength = totalTextSamples > 0 ? (Double(totalTextLength) / Double(totalTextSamples)) : 0.0
        
        let modality: DatasetModality
        if temporalColumn != nil && (numRatio > 0.3 || cols.count <= 3) {
            modality = .timeSeries
        } else if avgTextLength > 35.0 || (textRatio >= 0.5 && avgTextLength > 20.0) {
            modality = .pureTextNLP
        } else if numRatio >= 0.99 {
            modality = .tabularNumeric
        } else {
            modality = .tabularMixed
        }
        
        return ModalityProfile(
            modality: modality,
            numericColumnRatio: numRatio,
            textColumnRatio: textRatio,
            averageTextLength: avgTextLength,
            temporalColumn: temporalColumn
        )
    }
}
