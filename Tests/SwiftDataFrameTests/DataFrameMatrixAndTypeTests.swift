import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame Matrix & Column Type Tests")
struct DataFrameMatrixAndTypeTests {
    @Test("toFeatureMatrix and toTargetVector extract numeric & boolean columns")
    func testToFeatureMatrix() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "age", values: [25.0, 30.0, 35.0]),
            TypedColumn<Int64>(name: "pclass", values: [1, 2, 3]),
            TypedColumn<Bool>(name: "survived", values: [true, false, true])
        ])

        let matrix = try df.toFeatureMatrix(["age", "pclass", "survived"])
        #expect(matrix.count == 3)
        #expect(matrix[0] == [25.0, 1.0, 1.0])
        #expect(matrix[1] == [30.0, 2.0, 0.0])
        #expect(matrix[2] == [35.0, 3.0, 1.0])

        let target = try df.toTargetVector("survived")
        #expect(target == [1.0, 0.0, 1.0])
    }

    @Test("CSVReadOptions columnTypeOverrides forces column type")
    func testCSVTypeOverrides() throws {
        let csv = """
        Survived,Pclass
        1,3
        0,1
        """
        var options = CSVReadOptions()
        options.columnTypeOverrides["Survived"] = .int64
        let df = try CSVReader.parse(csv, options: options)

        #expect(df[column: "Survived", as: Int64.self] != nil)
        #expect(df[column: "Survived", as: Bool.self] == nil)
    }

    @Test("CSVReadOptions columnTypes alias forces boolean column")
    func testCSVColumnTypesAlias() throws {
        let csv = """
        HasPool,Pclass
        1,3
        0,1
        """
        var options = CSVReadOptions()
        options.columnTypes["HasPool"] = .boolean
        let df = try CSVReader.parse(csv, options: options)

        let poolCol = df[column: "HasPool", as: Bool.self]
        #expect(poolCol != nil)
        #expect(poolCol?.values == [true, false])
    }

    @Test("CSVReader strictly infers 0 and 1 as Int64, and literals as Bool")
    func testCSVStrictBoolInference() throws {
        let csv = """
        binary_flag,bool_literal
        1,true
        0,false
        """
        let df = try CSVReader.parse(csv, options: CSVReadOptions())
        #expect(df[column: "binary_flag", as: Int64.self] != nil)
        #expect(df[column: "binary_flag", as: Bool.self] == nil)
        #expect(df[column: "bool_literal", as: Bool.self] != nil)
    }

    @Test("DataFrame.cast(column:to:) casts Int64 to boolean and numeric types")
    func testDataFrameCastToDType() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "flag", values: [1, 0, 1]),
            TypedColumn<Int64>(name: "count", values: [10, 20, 30])
        ])

        let boolDf = try df.cast(column: "flag", to: .boolean)
        let boolCol = boolDf[column: "flag", as: Bool.self]
        #expect(boolCol != nil)
        #expect(boolCol?.values == [true, false, true])

        let floatDf = try df.cast(column: "count", to: .float64)
        let floatCol = floatDf[column: "count", as: Double.self]
        #expect(floatCol != nil)
        #expect(floatCol?.values == [10.0, 20.0, 30.0])
    }
}
