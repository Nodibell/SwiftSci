import Testing
@testable import SwiftDataFrame

@Suite("Feature conversion compatibility")
struct FeatureConversionTests {
    @Test func layoutsAndMissingValues() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "d", values: [-0.0, nil, .infinity, -.infinity]),
            TypedColumn<Int64>(name: "i", values: [Int64.max, nil, -3, 0]),
            TypedColumn<Bool>(name: "b", values: [false, true, nil, false])
        ])
        let names = ["b", "i", "d", "b"]
        let expected: [[Double]] = [
            [0, Double(Int64.max), -0.0, 0],
            [1, .nan, .nan, 1], [ .nan, -3, .infinity, .nan],
            [0, 0, -.infinity, 0]
        ]
        let nested = try frame.toFeatureMatrix(names)
        let flat = try frame.toFlatFeatureMatrix(names)
        #expect(flat.rows == 4 && flat.cols == 4)
        for r in expected.indices {
            for c in names.indices {
                let value = expected[r][c]
                #expect(nested[r][c].bitPattern == value.bitPattern || (nested[r][c].isNaN && value.isNaN))
                #expect(flat.flat[r * names.count + c].bitPattern == value.bitPattern || (flat.flat[r * names.count + c].isNaN && value.isNaN))
            }
        }
        for name in ["d", "i", "b"] {
            let target = try frame.toTargetVector(name)
            let reference = try frame.toFeatureMatrix([name])
            for r in target.indices {
                #expect(target[r].bitPattern == reference[r][0].bitPattern || (target[r].isNaN && reference[r][0].isNaN))
            }
        }
        var changed = nested
        changed[0][0] = 123
        #expect(changed[3][0] == 0)
        #expect(frame[column: "b", as: Bool.self]?[0] == false)
    }

    @Test func emptySelectionsAndColumns() throws {
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "d", values: [1, 2])])
        #expect(try frame.toFeatureMatrix([]) == [[], []])
        let flat = try frame.toFlatFeatureMatrix([])
        #expect(flat.flat.isEmpty && flat.rows == 2 && flat.cols == 0)
        let empty = try DataFrame(columns: [TypedColumn<Double>(name: "d", values: [Double]())])
        #expect(try empty.toFeatureMatrix(["d"]).isEmpty)
        #expect(try empty.toTargetVector("d").isEmpty)
    }

    @Test func errorPrecedenceAndAcceptedTypes() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<String>(name: "s", values: ["x"]),
            TypedColumn<Float>(name: "f", values: [1])
        ])
        for names in [["s", "missing"], ["missing", "s"]] {
            #expect(throws: SwiftMLError.columnNotFound("missing")) { try frame.toFeatureMatrix(names) }
            #expect(throws: SwiftMLError.columnNotFound("missing")) { try frame.toFlatFeatureMatrix(names) }
        }
        for name in ["s", "f"] {
            #expect(throws: SwiftMLError.castFailed(column: name, targetType: "Double")) { try frame.toTargetVector(name) }
        }
        #expect(throws: SwiftMLError.columnNotFound("missing")) { try frame.toTargetVector("missing") }
    }
}
