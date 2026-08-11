import Testing
import Foundation
@testable import SwiftDataFrame
@testable import SwiftML

@Suite("SwiftSci Error Consolidation Tests")
struct SwiftSciErrorConsolidationTests {

    @Test("Assert typealias equivalence for MLError and DataFrameError")
    func testTypealiasEquivalence() {
        #expect(MLError.self == SwiftMLError.self)
        #expect(DataFrameError.self == SwiftMLError.self)
    }

    @Test("SwiftMLError property descriptions")
    func testErrorDescriptions() {
        let err1 = SwiftMLError.modelNotFitted
        #expect(err1.description.contains("fitted"))

        let err2 = SwiftMLError.invalidInput("test input")
        #expect(err2.description.contains("test input"))
    }
}
