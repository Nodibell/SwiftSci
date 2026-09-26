import Testing
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("Compact scaler compatibility")
struct CompactScalerTests {
    @Test(arguments: [1, 2, 17, 129], [1, 3, 8])
    func directColumnsMatchRows(rows: Int, width: Int) throws {
        let names = (0..<width).map { "x\($0)" }
        let columns: [any AnyColumn] = (0..<width).map { c in
            TypedColumn<Double>(name: names[c], values: (0..<rows).map { Double(($0 * 17 + c * 13) % 101) / 7 - 9 })
        }
        let frame = try DataFrame(columns: columns)
        let matrix = try frame.toFeatureMatrix(names)
        let batch = try frame.prepareNumericBatch(names)
        var reference = StandardScaler(), direct = StandardScaler()
        var minReference = MinMaxScaler(range: (-2, 3)), minDirect = MinMaxScaler(range: (-2, 3))
        let expected = try reference.fitTransform(matrix)
        let actual = try direct.fitTransform(batch)
        let minExpected = try minReference.fitTransform(matrix)
        let minActual = try minDirect.fitTransform(batch)
        #expect(reference.mean == direct.mean && reference.std == direct.std)
        #expect(minReference.dataMin == minDirect.dataMin && minReference.dataMax == minDirect.dataMax)
        for r in 0..<rows {
            for c in 0..<width {
                #expect(actual[r, c]!.bitPattern == expected[r][c].bitPattern)
                #expect(minActual[r, c]!.bitPattern == minExpected[r][c].bitPattern)
            }
        }
    }

    @Test func missingAndNonfiniteValues() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "x", values: [1, nil, 3, .nan]),
            TypedColumn<Double>(name: "y", values: [.infinity, 2, -.infinity, -0.0]),
            TypedColumn<Double>(name: "z", values: [Double?](repeating: nil, count: 4))
        ])
        let names = frame.columnNames, batch = try frame.prepareNumericBatch(frame.columnNames)
        var reference = StandardScaler(), direct = StandardScaler()
        var minReference = MinMaxScaler(), minDirect = MinMaxScaler()
        let expected = try reference.fitTransform(frame.toFeatureMatrix(names))
        let actual = try direct.fitTransform(batch)
        let minExpected = try minReference.fitTransform(frame.toFeatureMatrix(names))
        let minActual = try minDirect.fitTransform(batch)
        for c in 0..<3 {
            #expect(actual.nullCount(inColumn: c) == 0 && minActual.nullCount(inColumn: c) == 0)
            for r in 0..<4 {
                let a = actual[r,c]!, b = minActual[r,c]!
                #expect(a.bitPattern == expected[r][c].bitPattern || (a.isNaN && expected[r][c].isNaN))
                #expect(b.bitPattern == minExpected[r][c].bitPattern || (b.isNaN && minExpected[r][c].isNaN))
            }
        }
    }

    @Test func errorsAndSnapshotState() throws {
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1,2,3])])
        let batch = try frame.prepareNumericBatch(["x"])
        var scaler = StandardScaler()
        #expect(throws: PreprocessingError.fitNotCalled) { try scaler.transform(batch) }
        try scaler.fit(batch)
        let copy = scaler
        #expect(throws: PreprocessingError.emptyInput) { try scaler.fit(frame.prepareNumericBatch([])) }
        #expect(scaler.mean == copy.mean && scaler.std == copy.std)
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 1, got: 2)) { try scaler.transform(frame.prepareNumericBatch(["x","x"])) }
        let empty = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [Double]())]).prepareNumericBatch(["x","x"])
        #expect(try scaler.transform(empty).rowCount == 0)
    }
}
