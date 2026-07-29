import Testing
import Foundation
import SwiftDataFrame

@Suite("DataFrame Null Safety Tests (Phase 1)")
struct DataFrameNullSafetyTests {

    @Test("toFeatureMatrix maps Bool nil to NaN instead of 0.0")
    func testToFeatureMatrixBoolNilToNaN() throws {
        let boolCol = TypedColumn<Bool>(name: "flag", values: [true, false, nil])
        let df = try DataFrame(columns: [boolCol])
        
        let matrix = try df.toFeatureMatrix(["flag"])
        #expect(matrix.count == 3)
        #expect(matrix[0][0] == 1.0)
        #expect(matrix[1][0] == 0.0)
        #expect(matrix[2][0].isNaN)
    }
}
