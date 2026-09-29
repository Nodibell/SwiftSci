import Testing
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("Preprocessing conversion compatibility")
struct ConversionCompatibilityTests {
    @Test func frameAndMatrixPathsAgree() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "d", values: [1, 4, -2, 10]),
            TypedColumn<Int64>(name: "i", values: [1, 1, 1, 1]),
            TypedColumn<Bool>(name: "b", values: [true, false, true, false]),
            TypedColumn<String>(name: "label", values: ["a", "b", "c", "d"])
        ])
        let names = ["b", "d", "i", "d"]
        let matrix = try frame.toFeatureMatrix(names)
        var standard = StandardScaler(), minmax = MinMaxScaler()
        let expectedStandard = try standard.fitTransform(matrix)
        let expectedMinMax = try minmax.fitTransform(matrix)
        let actualStandard = try frame.standardScale(columns: names)
        let actualMinMax = try frame.minMaxScale(columns: names)
        #expect(try actualStandard.scaled.toFeatureMatrix(names) == expectedStandard)
        #expect(try actualMinMax.scaled.toFeatureMatrix(names) == expectedMinMax)
        #expect(actualStandard.scaler.mean == standard.mean)
        #expect(actualStandard.scaler.std == standard.std)
        #expect(actualMinMax.scaler.dataMin == minmax.dataMin)
        #expect(actualMinMax.scaler.dataMax == minmax.dataMax)
        #expect(actualStandard.scaled[column: "label", as: String.self]?.values == frame[column: "label", as: String.self]?.values)
        #expect(frame[column: "d", as: Double.self]?.values == [1, 4, -2, 10])
        var copy = standard
        try copy.fit([[100, 101, 102, 103], [200, 201, 202, 203]])
        #expect(try standard.transform(matrix) == expectedStandard)
    }

    @Test func genericAdapterPreservesFitThenTransformDispatch() throws {
        var transformer = DispatchProbe()
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [2, 4])])
        let result = try transformer.fitTransform(frame, columns: ["x"])
        #expect(transformer.fitted)
        #expect(result[column: "x", as: Double.self]?.values == [3, 5])
    }

    @Test func scratchDoesNotLeakBetweenRowsOrCalls() throws {
        let training = [[1.0, 8, 3], [4, 2, 3], [7, 5, 3]]
        let input = [[100.0, -20, 3], [1, 2, 3], [7, 8, 3]]
        var standard = StandardScaler(), minmax = MinMaxScaler(range: (-3, 2))
        try standard.fit(training); try minmax.fit(training)
        let a = try standard.transform(input), b = try minmax.transform(input)
        for r in input.indices {
            for c in input[r].indices {
                let s = (input[r][c] - standard.mean![c]) / standard.std![c]
                let span = minmax.dataMax![c] - minmax.dataMin![c]
                let m = (input[r][c] - minmax.dataMin![c]) * (span < 1e-12 ? 0 : 5 / span) - 3
                #expect(abs(a[r][c] - s) < 1e-12)
                #expect(abs(b[r][c] - m) < 1e-12)
            }
        }
        #expect(try standard.transform(input) == a)
        #expect(try minmax.transform(input) == b)
        #expect(try standard.transform([]).isEmpty)
        #expect(try minmax.transform([]).isEmpty)
        #expect(throws: (any Error).self) { try standard.transform([[1, 2, 3], [4]]) }
        #expect(throws: (any Error).self) { try minmax.transform([[1, 2, 3], [4]]) }
    }
}

private struct DispatchProbe: PreprocessingTransformer {
    var fitted = false
    mutating func fit(_ data: [[Double]]) throws { fitted = true }
    func transform(_ data: [[Double]]) throws -> [[Double]] { data.map { $0.map { $0 + 1 } } }
    mutating func fitTransform(_ data: [[Double]]) throws -> [[Double]] { data.map { $0.map { $0 + 100 } } }
}
