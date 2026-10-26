import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftCluster

@Suite("Prepared PCA GPU ownership and parity")
struct PreparedPCAGPUTests {
    @Test func batchAndBothMatrixLayouts() async throws {
        let input = try PreparedPCAFixture.batch()
        let selected = try input.selectingRows([2, 0, 2])
        let reference = try PCA(nComponents: 2, device: .gpu)
        let expected = try await reference.fitTransform(PreparedPCAFixture.rows)
        let model = try PCA(nComponents: 2, device: .gpu)
        let actual = try await model.fitTransform(input)
        PreparedPCAFixture.close(actual, expected, tolerance: 1e-5)
        #expect(await model.resolvedDevice == .gpu)
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            try await model.fit(input.matrix(order: order))
            let scores = try await model.transform(selected.matrix(order: order))
            #expect(scores.order == order && scores.originalRowIndices == [2, 0, 2])
            for row in 0..<3 {
                let sourceRow = [2, 0, 2][row]
                for column in 0..<2 { #expect(abs(scores[row, column] - expected[sourceRow][column]) < 1e-5) }
            }
        }
        let mean = await model.mean
        let overflow = try PreparedPCAFixture.batch([[Double.greatestFiniteMagnitude, 1], [2, 3]])
        await #expect(throws: ClusterError.self) { try await model.fit(overflow) }
        #expect(await model.mean == mean)
    }
}
