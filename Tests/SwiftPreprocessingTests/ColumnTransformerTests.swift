import Testing
@testable import SwiftPreprocessing

@Suite("ColumnTransformer Tests")
struct ColumnTransformerTests {

    @Test("ColumnTransformer transforms selected column routes")
    func testColumnTransformerRoutes() throws {
        let features: [[Double]] = [
            [10.0, 100.0, 1.0],
            [20.0, 200.0, 2.0],
            [30.0, 300.0, 3.0]
        ]

        let route1 = ColumnTransformer.Route(
            name: "scaler1",
            transformer: StandardScaler(),
            columnIndices: [0, 2]
        )
        let route2 = ColumnTransformer.Route(
            name: "scaler2",
            transformer: MinMaxScaler(),
            columnIndices: [1]
        )

        var ct = ColumnTransformer(routes: [route1, route2])
        try ct.fit(features)
        let transformed = try ct.transform(features)

        #expect(transformed.count == 3)
        #expect(transformed[0].count == 3)
        // MinMaxScaler on column [100, 200, 300] yields 0.0 for first row
        #expect(abs(transformed[0][2] - 0.0) < 1e-5)
        #expect(abs(transformed[2][2] - 1.0) < 1e-5)
    }

    @Test("ColumnTransformer fit followed by separate transform on new data")
    func testColumnTransformerFitThenSeparateTransform() throws {
        let route = ColumnTransformer.Route(
            name: "scaler",
            transformer: MinMaxScaler(),
            columnIndices: [0]
        )
        let ct = ColumnTransformer(routes: [route])
        
        let trainData = [[10.0], [20.0]]
        let testData = [[15.0]]
        
        try ct.fit(trainData)
        let transformedTest = try ct.transform(testData)
        
        // Min = 10, Max = 20 -> (15 - 10)/(20 - 10) = 0.5
        #expect(abs(transformedTest[0][0] - 0.5) < 1e-5)
    }

    @Test("ColumnTransformer validates matrix rectangularity during fit and transform")
    func testColumnTransformerRaggedRowsValidation() {
        let route = ColumnTransformer.Route(
            name: "scaler",
            transformer: MinMaxScaler(),
            columnIndices: [0]
        )
        let ct = ColumnTransformer(routes: [route])

        // Ragged fit
        #expect(throws: PreprocessingError.self) {
            try ct.fit([[1.0, 2.0], [3.0]])
        }

        // Fit valid
        #expect(throws: Never.self) {
            try ct.fit([[1.0, 2.0], [3.0, 4.0]])
        }

        // Ragged transform
        #expect(throws: PreprocessingError.self) {
            _ = try ct.transform([[1.0, 2.0], [3.0]])
        }
    }

    @Test("ColumnTransformer remainder: .passthrough preserves unrouted columns")
    func testColumnTransformerRemainderPassthrough() throws {
        // Col 0: Scaled with MinMaxScaler, Col 1: Unrouted (should be passed through)
        let route = ColumnTransformer.Route(
            name: "scaler",
            transformer: MinMaxScaler(),
            columnIndices: [0]
        )
        let ct = ColumnTransformer(routes: [route], remainder: .passthrough)

        let data = [
            [10.0, 999.0],
            [20.0, 888.0]
        ]

        try ct.fit(data)
        let transformed = try ct.transform(data)

        #expect(transformed.count == 2)
        #expect(transformed[0].count == 2)
        // Col 0 scaled: 0.0 and 1.0; Col 1 passthrough: 999.0 and 888.0
        #expect(abs(transformed[0][0] - 0.0) < 1e-5)
        #expect(abs(transformed[0][1] - 999.0) < 1e-5)
        #expect(abs(transformed[1][0] - 1.0) < 1e-5)
        #expect(abs(transformed[1][1] - 888.0) < 1e-5)
    }
}
