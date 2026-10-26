import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftCluster

@Suite("PCA finite mean overflow")
struct PCAMeanOverflowTests {
    @Test(arguments: [SVDSolver.full, .randomized])
    func rejectsOverflowAndPreservesFit(_ solver: SVDSolver) async throws {
        let input = try PreparedPCAFixture.batch()
        let model = try PCA(nComponents: 1, svdSolver: solver, device: .cpu)
        let scores = try await model.fitTransform(input)
        let mean = await model.mean
        let components = await model.components
        let variance = await model.explainedVariance
        let rows: [[Double]] = [[Double.greatestFiniteMagnitude, 0], [Double.greatestFiniteMagnitude, 0]]
        let overflow = try PreparedPCAFixture.batch(rows)
        for route in 0..<4 {
            await #expect(throws: ClusterError.self) {
                switch route {
                case 0: try await model.fit(rows)
                case 1: try await model.fit(overflow)
                case 2: try await model.fit(overflow.matrix(order: .rowMajor))
                default: try await model.fit(overflow.matrix(order: .columnMajor))
                }
            }
            #expect(await model.mean == mean)
            #expect(await model.components == components)
            #expect(await model.explainedVariance == variance)
            #expect(await model.resolvedDevice == .cpu)
            #expect(try await model.transform(input).columnValues(at: 0) == scores.columnValues(at: 0))
        }
    }
}
