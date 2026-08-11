import Testing
import Foundation
@testable import SwiftML

@Suite("Shared Numerics Tests")
struct NumericsTests {

    @Test("Sigmoid numerical overflow stability")
    func testSigmoidOverflowStability() {
        // Extreme positive logit should return 1.0 without overflow or NaN
        let posSig = sigmoid(1000.0)
        #expect(posSig == 1.0)
        #expect(!posSig.isNaN)
        #expect(!posSig.isInfinite)

        // Extreme negative logit should return 0.0 without overflow or NaN
        let negSig = sigmoid(-1000.0)
        #expect(negSig < 1e-15)
        #expect(!negSig.isNaN)
        #expect(!negSig.isInfinite)

        // Standard sigmoid checks
        #expect(abs(sigmoid(0.0) - 0.5) < 1e-9)
    }

    @Test("Array argmax helper")
    func testArrayArgmax() {
        let arr = [1.5, 4.2, 3.1, 0.8]
        #expect(arr.argmax() == 1)

        let emptyArr: [Double] = []
        #expect(emptyArr.argmax() == 0)

        let singleArr = [99.0]
        #expect(singleArr.argmax() == 0)
    }
}
