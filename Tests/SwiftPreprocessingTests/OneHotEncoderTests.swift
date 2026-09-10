import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("OneHotEncoder Tests")
struct OneHotEncoderTests {
    
    @Test("OneHotEncoder basic encoding")
    func testOneHotEncoderBasic() throws {
        let encoder = OneHotEncoder()
        let data = [
            ["male", "US"],
            ["female", "Europe"],
            ["female", "US"]
        ]
        
        encoder.fit(data)
        
        #expect(encoder.categories.count == 2)
        #expect(encoder.categories[0] == ["female", "male"])
        #expect(encoder.categories[1] == ["Europe", "US"])
        
        let encoded = try encoder.transform([
            ["male", "US"],
            ["female", "Europe"]
        ])
        
        // male = [0, 1], US = [0, 1] -> [0, 1, 0, 1]
        // female = [1, 0], Europe = [1, 0] -> [1, 0, 1, 0]
        #expect(encoded.count == 2)
        #expect(encoded[0] == [0.0, 1.0, 0.0, 1.0])
        #expect(encoded[1] == [1.0, 0.0, 1.0, 0.0])
    }
    
    @Test("OneHotEncoder errors")
    func testOneHotEncoderErrors() throws {
        let encoder = OneHotEncoder()
        
        #expect(throws: PreprocessingError.self) {
            try encoder.transform([["male"]])
        }
        
        encoder.fit([["male", "US"], ["female", "Europe"]])
        
        // Dimension mismatch
        #expect(throws: PreprocessingError.self) {
            try encoder.transform([["male"]])
        }
        
        // Unknown category throws on default .error
        #expect(throws: PreprocessingError.self) {
            try encoder.transform([["male", "Asia"]])
        }
    }

    @Test("OneHotEncoder handleUnknown: .ignore encodes unseen categories as all zeros")
    func testOneHotEncoderHandleUnknownIgnore() throws {
        let encoder = OneHotEncoder(handleUnknown: .ignore)
        encoder.fit([["male", "US"], ["female", "Europe"]])
        // categories[0]: ["female", "male"] (count: 2)
        // categories[1]: ["Europe", "US"] (count: 2)

        let result = try encoder.transform([
            ["unknownGender", "US"],      // [0.0, 0.0] + [0.0, 1.0]
            ["male", "unknownRegion"],    // [0.0, 1.0] + [0.0, 0.0]
            ["alien", "mars"]             // [0.0, 0.0] + [0.0, 0.0]
        ])

        #expect(result.count == 3)
        #expect(result[0] == [0.0, 0.0, 0.0, 1.0])
        #expect(result[1] == [0.0, 1.0, 0.0, 0.0])
        #expect(result[2] == [0.0, 0.0, 0.0, 0.0])
    }
}
