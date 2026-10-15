import SwiftDataFrame

func indexedScalingPrecisionSource() throws -> PreparedNumericBatch {
    let powers: [Double] = [0x1p-400, 0x1p-20, 1, 0x1p20, 0x1p400]
    let names: [String] = (0..<9).map { "x\($0)" }
    let columns: [[Double]] = (0..<9).map { column -> [Double] in
        let scale = powers[column % powers.count]
        return (0..<97).map { row -> Double in
            if (row + column) % 17 == 0 { return .nan }
            let numerator = Double((row * 13 + column * 7) % 67 - 33)
            return numerator / 17 * scale
        }
    }
    return try PreparedNumericBatch(columnNames: names, columns: columns)
}
