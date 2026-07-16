import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractFeatures(columns names: [String]) throws -> [[Double]] {
        let nRows = shape.rows
        guard nRows > 0 else { return [] }
        
        var features = [[Double]](repeating: [Double](repeating: 0.0, count: names.count), count: nRows)
        for (colIdx, colName) in names.enumerated() {
            guard let col = self[column: colName, as: Double.self] else {
                throw DataFrameError.columnNotFound(colName)
            }
            for rowIdx in 0..<nRows {
                features[rowIdx][colIdx] = col[rowIdx] ?? 0.0
            }
        }
        return features
    }

    /// Fits a StandardScaler on the specified columns.
    public func fitStandardScaler(columns names: [String]) throws -> StandardScaler {
        let features = try extractFeatures(columns: names)
        let scaler = StandardScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted StandardScaler, returning a new DataFrame.
    public func standardScale(columns names: [String], scaler: StandardScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using StandardScaler, returning the scaled DataFrame and scaler.
    public func standardScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: StandardScaler) {
        let scaler = try fitStandardScaler(columns: names)
        let scaledDf = try standardScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a MinMaxScaler on the specified columns.
    public func fitMinMaxScaler(columns names: [String]) throws -> MinMaxScaler {
        let features = try extractFeatures(columns: names)
        let scaler = MinMaxScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted MinMaxScaler, returning a new DataFrame.
    public func minMaxScale(columns names: [String], scaler: MinMaxScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using MinMaxScaler, returning the scaled DataFrame and scaler.
    public func minMaxScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: MinMaxScaler) {
        let scaler = try fitMinMaxScaler(columns: names)
        let scaledDf = try minMaxScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a LabelEncoder and encodes a category column into integers.
    public func labelEncode(column name: String) throws -> (encoded: DataFrame, encoder: LabelEncoder) {
        guard let col = self[column: name, as: String.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        let rawValues = col.values.map { $0 ?? "" }
        let encoder = LabelEncoder()
        encoder.fit(rawValues)
        let labels = try encoder.transform(rawValues)
        
        let intValues: [Int64?] = labels.map { Int64($0) }
        let newCol = TypedColumn<Int64>(name: name, values: intValues)
        let encodedDf = try withColumn(name, column: newCol)
        return (encodedDf, encoder)
    }
}
