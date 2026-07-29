import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("LabelEncoder Multi-Type Tests (Phase 2)")
struct LabelEncoderMultiTypeTests {

    @Test("labelEncode supports Int64 and Double columns")
    func testLabelEncodeInt64AndDoubleColumns() throws {
        let intCol = TypedColumn<Int64>(name: "code", values: [100, 200, 100, 300])
        let doubleCol = TypedColumn<Double>(name: "grade", values: [1.5, 2.5, 1.5, 3.5])
        let df = try DataFrame(columns: [intCol, doubleCol])
        
        let (dfEncodedInt, _) = try df.labelEncode(column: "code")
        guard let encodedIntCol = dfEncodedInt[column: "code", as: Int64.self] else {
            Issue.record("Encoded Int64 column missing")
            return
        }
        #expect(encodedIntCol[0] == encodedIntCol[2])
        #expect(encodedIntCol[0] != encodedIntCol[1])
        
        let (dfEncodedDouble, _) = try df.labelEncode(column: "grade")
        guard let encodedDoubleCol = dfEncodedDouble[column: "grade", as: Int64.self] else {
            Issue.record("Encoded Double column missing")
            return
        }
        #expect(encodedDoubleCol[0] == encodedDoubleCol[2])
        #expect(encodedDoubleCol[0] != encodedDoubleCol[1])
    }
}
