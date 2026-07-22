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

        let ct = ColumnTransformer(routes: [route1, route2])
        try ct.fit(features)
        let transformed = try ct.transform(features)

        #expect(transformed.count == 3)
        #expect(transformed[0].count == 3)
        // MinMaxScaler on column [100, 200, 300] yields 0.0 for first row
        #expect(abs(transformed[0][2] - 0.0) < 1e-5)
        #expect(abs(transformed[2][2] - 1.0) < 1e-5)
    }
}
