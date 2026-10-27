import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftCluster

@Suite("Prepared PCA workflows")
struct PreparedPCATests {
    @Test(arguments: [SVDSolver.full, .randomized, .auto])
    func exactAnswersAndLayouts(_ solver: SVDSolver) async throws {
        let input = try PreparedPCAFixture.batch()
        let query = try PreparedPCAFixture.batch(PreparedPCAFixture.query)
        for count in [1, 2] {
            let model = try PCA(nComponents: count, svdSolver: solver, device: .cpu)
            let output = try await model.fitTransform(input)
            let scores = PreparedPCAFixture.scores.map { Array($0.prefix(count)) }
            PreparedPCAFixture.close(output, scores)
            #expect(output.columnNames == (1...count).map { "PC\($0)" })
            PreparedPCAFixture.close(try #require(await model.mean), [10, -2])
            PreparedPCAFixture.close(try #require(await model.explainedVariance), Array([200.0 / 3, 50.0 / 3].prefix(count)))
            PreparedPCAFixture.close(try #require(await model.explainedVarianceRatio), Array([0.8, 0.2].prefix(count)))
            let expected = PreparedPCAFixture.queryScores.map { Array($0.prefix(count)) }
            PreparedPCAFixture.close(try await model.transform(query), expected)
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let matrix = try input.matrix(order: order)
                let snapshot = matrix.values
                let projected = try await model.fitTransform(matrix)
                #expect(projected.order == order && matrix.values == snapshot)
                #expect(projected.columnNames == output.columnNames)
                for row in scores.indices {
                    for column in 0..<count { #expect(abs(projected[row, column] - scores[row][column]) < 1e-11) }
                }
                let heldOut = try await model.transform(query.matrix(order: order))
                for row in expected.indices {
                    for column in 0..<count { #expect(abs(heldOut[row, column] - expected[row][column]) < 1e-11) }
                }
            }
        }
    }

    @Test func preprocessingSelectionAndSnapshots() async throws {
        var source = try PreparedPCAFixture.batch()
        let snapshot = source
        let train = try source.selectingRows([3, 0, 2, 1])
        let heldOut = try source.selectingRows([2, 0, 2])
        var scaler = StandardScaler()
        try scaler.fit(train)
        let scaledTrain = try scaler.transform(train)
        let scaledQuery = try scaler.transform(heldOut)
        let model = try PCA(nComponents: 1, svdSolver: .full, device: .cpu)
        let trainingScores = try await model.fitTransform(scaledTrain)
        let scores = try await model.transform(scaledQuery)
        #expect(trainingScores.originalRowIndices == [3, 0, 2, 1])
        #expect(scores.originalRowIndices == [2, 0, 2])
        #expect(scores[0, 0] == scores[2, 0])
        let row = try scores.matrix(), column = try scores.matrix(order: .columnMajor)
        #expect(row.originalRowIndices == scores.originalRowIndices)
        #expect(column.originalRowIndices == scores.originalRowIndices)
        try source.updateColumn(at: 0, rows: [0], values: [999])
        #expect(snapshot[0, 0] == 16 && train[1, 0] == 16)
        #expect(try await model.transform(scaledQuery).columnValues(at: 0) == scores.columnValues(at: 0))
        let matrixScores = try await model.transform(scaledQuery.matrix(order: .columnMajor))
        #expect(matrixScores.originalRowIndices == [2, 0, 2])
        for i in 0..<scores.rowCount { #expect(matrixScores[i, 0] == scores[i, 0]) }
    }

    @Test func schemaAndRejectedRefit() async throws {
        let input = try PreparedPCAFixture.batch()
        let changed = try PreparedPCAFixture.batch(names: ["b", "a"])
        let model = try PCA(nComponents: 1, svdSolver: .full, device: .cpu)
        let scores = try await model.fitTransform(input)
        let mean = await model.mean, components = await model.components
        await #expect(throws: ClusterError.self) { _ = try await model.transform(changed) }
        await #expect(throws: ClusterError.self) { _ = try await model.transform(changed.matrix()) }
        let invalid = try PreparedPCAFixture.batch([[.nan, 1], [2, 3]])
        await #expect(throws: ClusterError.self) { try await model.fit(invalid) }
        let afterMean = await model.mean, afterComponents = await model.components
        #expect(afterMean == mean && afterComponents == components)
        #expect(await model.resolvedDevice == .cpu)
        #expect(try await model.transform(input).columnValues(at: 0) == scores.columnValues(at: 0))
        try await model.fit(PreparedPCAFixture.rows)
        _ = try await model.transform(changed)
        let duplicate = try PreparedPCAFixture.batch(names: ["x", "x"])
        try await model.fit(duplicate)
        _ = try await model.transform(duplicate)
        await #expect(throws: ClusterError.self) { _ = try await model.transform(input) }
    }

    @Test func emptyAndInvalidInputs() async throws {
        let model = try PCA(nComponents: 1, svdSolver: .randomized, device: .cpu)
        await #expect(throws: ClusterError.self) { try await model.fit([]) }
        await #expect(throws: ClusterError.self) { try await model.fit([[1, 2], [3]]) }
        let valid = try PreparedPCAFixture.batch()
        try await model.fit(valid)
        let empty = try valid.selectingRows([])
        let scores = try await model.transform(empty)
        #expect(scores.rowCount == 0 && scores.columnNames == ["PC1"])
        #expect(scores.originalRowIndices.isEmpty)
        let matrix = try await model.transform(empty.matrix(order: .columnMajor))
        #expect(matrix.rowCount == 0 && matrix.columnCount == 1 && matrix.values.isEmpty)
        await #expect(throws: ClusterError.self) { _ = try await model.transform([]) }
        await #expect(throws: ClusterError.self) { try await model.fit(empty) }
        for value in [Double.nan, .infinity, -.infinity] {
            let bad = try PreparedPCAFixture.batch([[value, 0], [1, 2]])
            await #expect(throws: ClusterError.self) { try await model.fit(bad) }
            await #expect(throws: ClusterError.self) { try await model.fit(bad.matrix()) }
            await #expect(throws: ClusterError.self) { _ = try await model.transform(bad) }
            await #expect(throws: ClusterError.self) { _ = try await model.transform(bad.matrix()) }
        }
        let missing = try DataFrame(columns: [
            TypedColumn<Double>(name: "a", values: [1, nil]),
            TypedColumn<Double>(name: "b", values: [2, 3])
        ]).prepareNumericBatch(["a", "b"])
        await #expect(throws: ClusterError.self) { try await model.fit(missing) }
        await #expect(throws: ClusterError.self) { _ = try await model.transform(missing) }
        let nanRows = try await model.transform([[.nan, 1]])
        #expect(nanRows[0][0].isNaN)
    }
}
