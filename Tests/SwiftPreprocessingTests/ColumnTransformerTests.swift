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
}
