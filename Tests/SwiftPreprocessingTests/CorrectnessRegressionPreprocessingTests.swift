import Testing
import SwiftDataFrame
@testable import SwiftPreprocessing

struct CorrectnessRegressionPreprocessingTests {
    @Test func noFeaturesSurvive() throws {
        var selector = VarianceThreshold()
        try selector.fit([[1, 2], [1, 2]])
        #expect(try selector.transform([[1, 2]]) == [[]])
    }
    @Test func negativeDegreeRejected() {
        // interactionOnly bounds the old recursion, safely reproducing missing validation.
        var transformer = PolynomialFeatures(degree: -1, interactionOnly: true)
        #expect(throws: (any Error).self) { try transformer.fit([[1, 2]]) }
    }
    @Test func smallSampleOOF() throws {
        var encoder = TargetEncoder(smoothing: 0)
        #expect(try encoder.fitTransform(categories: ["a", "b"], target: [0, 10]) == [10, 0])
        #expect(try encoder.transform(categories: ["a", "b"]) == [0, 10])
    }
    @Test func singletonOOFRejected() {
        var encoder = TargetEncoder()
        #expect(throws: (any Error).self) { try encoder.fitTransform(categories: ["a"], target: [10]) }
    }
    @Test func matrixOOF() throws {
        var encoder = TargetEncoder(smoothing: 0)
        #expect(try encoder.fitTransform([[1, 0], [2, 10]]) == [[10], [0]])
    }
    @Test func unfittedSelectorRejected() {
        let selector = VarianceThreshold()
        #expect(throws: PreprocessingError.self) { try selector.transform([[1]]) }
    }
    @Test func zeroDegreeBias() throws {
        var withBias = PolynomialFeatures(degree: 0)
        #expect(try withBias.fitTransform([[2, 3]]) == [[1]])
        var withoutBias = PolynomialFeatures(degree: 0, includeBias: false)
        #expect(try withoutBias.fitTransform([[2, 3]]) == [[]])
    }
    @Test func oofExcludesOwnTargetThroughProtocol() throws {
        var first: any PreprocessingTransformer = TargetEncoder(smoothing: 2)
        var second: any PreprocessingTransformer = TargetEncoder(smoothing: 2)
        let a = try first.fitTransform([[1, 0], [1, 10], [2, 20]])
        let b = try second.fitTransform([[1, 1000], [1, 10], [2, 20]])
        #expect(a[0] == b[0])
    }
    @Test func explicitOOFOptOut() throws {
        var encoder = TargetEncoder(smoothing: 0)
        #expect(try encoder.fitTransform(categories: ["a"], target: [7], cv: 1) == [7])
    }

    @Test func featureSelectionStabilityAndValidation() throws {
        // VarianceThreshold: shifted large offsets keep small real differences; ragged & NaN/Inf rejected
        var vt = VarianceThreshold(threshold: 0.5)
        let vtData = [[1e12 + 0.0, 5.0], [1e12 + 2.0, 5.0], [1e12 + 4.0, 5.0]]
        try vt.fit(vtData)
        #expect(try vt.transform(vtData) == [[1e12], [1e12 + 2.0], [1e12 + 4.0]])

        var badVt = VarianceThreshold(threshold: -0.1)
        #expect(throws: PreprocessingError.self) { try badVt.fit([[1, 2], [3, 4]]) }
        badVt = VarianceThreshold(threshold: .nan)
        #expect(throws: PreprocessingError.self) { try badVt.fit([[1, 2], [3, 4]]) }
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 2, got: 1)) { try vt.fit([[1, 2], [3]]) }
        #expect(throws: PreprocessingError.containsNaN) { try vt.fit([[1, .nan], [3, 4]]) }
        #expect(throws: PreprocessingError.containsInfinity) { try vt.transform([[1, .infinity]]) }
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 2, got: 3)) { try vt.transform([[1, 2, 3]]) }

        // SelectKBest: unfitted guard, negative k guard, k==0 empty output (no silent fallback), ragged & mismatch guards
        let skbUnfitted = SelectKBest(k: 2)
        #expect(throws: PreprocessingError.fitNotCalled) { try skbUnfitted.transform([[1, 2]]) }
        var skbNegative = SelectKBest(k: -1)
        #expect(throws: PreprocessingError.self) { try skbNegative.fit([[1, 2], [3, 4]]) }
        var skbZero = SelectKBest(k: 0)
        try skbZero.fit([[1, 2], [3, 4]])
        #expect(try skbZero.transform([[1, 2], [3, 4]]) == [[], []])

        // Shifted offset unsupervised selection picks variable feature (col 1) over constant huge col 0
        var skbShifted = SelectKBest(k: 1)
        let shiftedFeatures = [[1e12, 10.0], [1e12, 20.0], [1e12, 30.0]]
        try skbShifted.fit(shiftedFeatures)
        #expect(try skbShifted.transform(shiftedFeatures) == [[10.0], [20.0], [30.0]])

        // Supervised ANOVA selection & target/feature validations
        var skbSup = SelectKBest(k: 1)
        try skbSup.fit(features: [[1.0, 0.0], [1.0, 0.0], [2.0, 10.0], [2.0, 10.0]], targets: [0, 0, 1, 1])
        #expect(try skbSup.transform([[9.0, 5.0]]) == [[5.0]])
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 4, got: 3)) {
            try skbSup.fit(features: [[1, 2], [3, 4], [5, 6], [7, 8]], targets: [0, 1, 0])
        }
        #expect(throws: PreprocessingError.containsNaN) {
            try skbSup.fit(features: [[1, 2], [3, 4]], targets: [0, .nan])
        }
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 2, got: 1)) {
            try skbSup.transform([[1]])
        }

        // RecursiveFeatureElimination: unfitted guard, invalid params, importance/row mismatch, NaN, ranking & transform
        let rfeUnfitted = RecursiveFeatureElimination(nFeaturesToSelect: 1)
        #expect(throws: PreprocessingError.fitNotCalled) { try rfeUnfitted.transform([[1, 2]]) }
        var rfeBadK = RecursiveFeatureElimination(nFeaturesToSelect: 0)
        #expect(throws: PreprocessingError.self) { try rfeBadK.fit([[1, 2], [3, 4]]) }
        var rfeBadStep = RecursiveFeatureElimination(nFeaturesToSelect: 1, step: 0)
        #expect(throws: PreprocessingError.self) { try rfeBadStep.fit([[1, 2], [3, 4]]) }

        var rfe = RecursiveFeatureElimination(nFeaturesToSelect: 2, step: 1)
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 3, got: 2)) {
            try rfe.fit(features: [[1, 2, 3], [4, 5, 6]], featureImportances: [0.1, 0.9])
        }
        #expect(throws: PreprocessingError.containsNaN) {
            try rfe.fit(features: [[1, 2, 3], [4, 5, 6]], featureImportances: [0.1, .nan, 0.9])
        }
        try rfe.fit(features: [[1, 2, 3], [4, 5, 6]], featureImportances: [0.2, 0.9, 0.5])
        #expect(rfe.support == [false, true, true])
        #expect(rfe.ranking == [3, 1, 1])
        #expect(try rfe.transform([[10, 20, 30]]) == [[20, 30]])
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 3, got: 2)) {
            try rfe.transform([[10, 20]])
        }
    }

    @Test func polynomialFeaturesRaggedAndCombinations() throws {
        var poly = PolynomialFeatures(degree: 2, interactionOnly: false, includeBias: true)
        #expect(throws: PreprocessingError.dimensionMismatch(expected: 2, got: 1)) {
            try poly.fit([[1.0, 2.0], [3.0]])
        }
        let x: [[Double]] = [[2.0, 3.0, 5.0]]
        let full = try poly.fitTransform(x)
        // C(3 + 2, 2) = 10 terms: 1, x0, x1, x2, x0^2, x0x1, x0x2, x1^2, x1x2, x2^2
        #expect(full[0].count == 10)
        #expect(full[0] == [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 9.0, 15.0, 25.0])

        var inter = PolynomialFeatures(degree: 2, interactionOnly: true, includeBias: false)
        let interOut = try inter.fitTransform(x)
        #expect(interOut[0] == [2.0, 3.0, 5.0, 6.0, 10.0, 15.0])
    }

    @Test func targetEncoderValidationPipelineColumnTransformerAndDataFrame() throws {
        // Invalid smoothing, NaN/Inf targets, and signed zero category parity
        var badSmooth = TargetEncoder(smoothing: .nan)
        #expect(throws: PreprocessingError.self) { try badSmooth.fit(categories: ["a"], target: [1.0]) }
        var negSmooth = TargetEncoder(smoothing: -1.0)
        #expect(throws: PreprocessingError.self) { try negSmooth.fit(categories: ["a"], target: [1.0]) }
        var te = TargetEncoder(smoothing: 0.0)
        #expect(throws: PreprocessingError.containsNaN) { try te.fit(categories: ["a"], target: [.nan]) }
        #expect(throws: PreprocessingError.containsInfinity) { try te.fitTransform(categories: ["a", "b"], target: [1.0, .infinity]) }

        // Signed zero numeric categories share the same bucket across fit/fitTransform/transform
        _ = try te.fitTransform([[0.0, 10.0], [-0.0, 20.0], [1.0, 50.0]])
        #expect(try te.transform([[0.0]]) == [[15.0]])
        #expect(try te.transform([[-0.0]]) == [[15.0]])

        // Distinguish fit+transform (in-sample) vs fitTransform (OOF)
        var inSampleEncoder = TargetEncoder(smoothing: 0.0)
        try inSampleEncoder.fit([[1, 0], [1, 10], [2, 20]])
        #expect(try inSampleEncoder.transform([[1, 0], [1, 10], [2, 20]]) == [[5], [5], [20]])

        // Pipeline.fitTransform propagates OOF while Pipeline.fit+transform retains full-train inference
        var pipeA = Pipeline(steps: [TargetEncoder(smoothing: 0.0)])
        var pipeB = Pipeline(steps: [TargetEncoder(smoothing: 0.0)])
        let oofA = try pipeA.fitTransform([[1, 0], [1, 10], [2, 20]])
        let oofB = try pipeB.fitTransform([[1, 1000], [1, 10], [2, 20]])
        #expect(oofA[0] == [10.0] && oofB[0] == [10.0])
        #expect(try pipeA.transform([[1], [2]]) == [[5.0], [20.0]])

        // ColumnTransformer.fitTransform routes OOF cleanly without leaking row's own target
        let ctA = ColumnTransformer(
            routes: [
                .init(name: "te", transformer: TargetEncoder(smoothing: 0.0), columnIndices: [0, 2]),
                .init(name: "sc", transformer: StandardScaler(), columnIndices: [1])
            ],
            remainder: .passthrough
        )
        let ctB = ColumnTransformer(
            routes: [
                .init(name: "te", transformer: TargetEncoder(smoothing: 0.0), columnIndices: [0, 2]),
                .init(name: "sc", transformer: StandardScaler(), columnIndices: [1])
            ],
            remainder: .passthrough
        )
        let outA = try ctA.fitTransform([[1, 10, 0, 99], [1, 20, 10, 88], [2, 30, 20, 77]])
        let outB = try ctB.fitTransform([[1, 10, 1000, 99], [1, 20, 10, 88], [2, 30, 20, 77]])
        #expect(outA[0][0] == 10.0 && outB[0][0] == 10.0)
        #expect(outA[0][2] == 99.0)
        let ctTest = try ctA.transform([[1, 20, 999, 55]])
        #expect(abs(ctTest[0][0] - 5.0) < 1e-12 && abs(ctTest[0][1] - 0.0) < 1e-12 && ctTest[0][2] == 55.0)

        // DataFrame: explicit rejection when a transformer changes feature count in replacingFeatures
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "catNum", values: [1, 1, 2, 2]),
            TypedColumn<Double>(name: "x", values: [10, 20, 30, 40]),
            TypedColumn<Double>(name: "target", values: [0, 10, 20, 30])
        ])
        var polyDf = PolynomialFeatures(degree: 2)
        #expect(throws: PreprocessingError.self) { try polyDf.fitTransform(df, columns: ["catNum", "x"]) }
        var teDf = TargetEncoder(smoothing: 0)
        #expect(throws: PreprocessingError.self) { try teDf.fitTransform(df, columns: ["catNum", "target"]) }

        // DataFrame.targetEncode convenience API runs OOF and keeps fitted encoder for unseen data
        let strDf = try DataFrame(columns: [
            TypedColumn<String>(name: "cat", values: ["A", "A", "B", "B", "Z"]),
            TypedColumn<Double>(name: "target", values: [0.0, 0.0, 0.0, 0.0, 1.0])
        ])
        let (encodedDf, fittedDfEncoder) = try strDf.targetEncode(
            column: "cat", targetColumn: "target", outputColumn: "cat_enc", smoothing: 1.0, cv: 5
        )
        let encodedVals = encodedDf[column: "cat_enc", as: Double.self]?.values.compactMap { $0 } ?? []
        #expect(encodedVals.count == 5 && encodedVals[4] < 1.0)
        #expect(try fittedDfEncoder.transform(categories: ["Z"])[0] > 0.5)
    }
}
