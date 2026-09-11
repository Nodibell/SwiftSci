import Testing
import SwiftDataFrame

@Suite("Dataset Modality Inference Tests")
struct ModalityTests {

    @Test("Pure numeric dataset infers .tabularNumeric")
    func testNumericModality() throws {
        let col1 = TypedColumn<Double>(name: "feat1", values: [1.0, 2.0, 3.0])
        let col2 = TypedColumn<Int>(name: "feat2", values: [10, 20, 30])
        let df = try DataFrame(columns: [col1, col2])
        
        #expect(df.inferModality() == .tabularNumeric)
        let profile = df.profileModality()
        #expect(profile.numericColumnRatio == 1.0)
    }

    @Test("Mixed categorical and numeric dataset infers .tabularMixed")
    func testMixedModality() throws {
        let col1 = TypedColumn<String>(name: "status", values: ["OK", "FAIL", "PENDING"])
        let col2 = TypedColumn<Double>(name: "amount", values: [100.0, 250.0, 15.0])
        let df = try DataFrame(columns: [col1, col2])
        
        #expect(df.inferModality() == .tabularMixed)
    }

    @Test("Long unstructured text documents infer .pureTextNLP")
    func testNLPModality() throws {
        let reviews = [
            "The quick brown fox jumps over the lazy dog and provides an exquisite natural language corpus sample.",
            "SwiftSci delivers zero-dependency high-throughput scientific computing on Apple Silicon with unified memory.",
            "Machine learning pipelines require rigorous validation, cross-entropy minimization, and gradient boosting."
        ]
        let col = TypedColumn<String>(name: "document", values: reviews)
        let df = try DataFrame(columns: [col])
        
        #expect(df.inferModality() == .pureTextNLP)
        let profile = df.profileModality()
        #expect(profile.averageTextLength > 50.0)
    }

    @Test("Timestamped data infers .timeSeries")
    func testTimeSeriesModality() throws {
        let timestamps = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"]
        let prices = [150.0, 155.5, 152.0]
        let col1 = TypedColumn<String>(name: "timestamp", values: timestamps)
        let col2 = TypedColumn<Double>(name: "close_price", values: prices)
        let df = try DataFrame(columns: [col1, col2])
        
        #expect(df.inferModality() == .timeSeries)
        let profile = df.profileModality()
        #expect(profile.temporalColumn == "timestamp")
    }
}
