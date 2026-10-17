import Testing
@testable import SwiftDataFrame

@Suite("Prepared row provenance storage")
struct PreparedProvenanceStorageTests {
    @Test func implicitRowsNeedNoProvenanceStorage() throws {
        let source = try PreparedNumericBatch(columnNames: ["x"], columns: [[10, 20, 30]])
        #expect(source.originalRowIndices == [0, 1, 2])
        #expect(source.provenanceStorageCapacity == 0)
        #expect(source.originalRowIndices == [0, 1, 2])
    }

    @Test func retainedCapacityIncludesSpareStorageAndPreservesDuplicateRows() throws {
        var provenance: [Int] = [7, 2, 7]
        provenance.reserveCapacity(4096)
        let source = PreparedNumericBatch(columnNames: ["x"],
            columns: [CompactNumericColumn([10, 20, 30])], rowCount: 3, sourceRows: provenance)
        #expect(source.provenanceStorageCapacity == provenance.capacity)
        #expect(source.provenanceStorageCapacity > source.rowCount)
        #expect(source.originalRowIndices == [7, 2, 7])
        let selected = try source.selectingRows([2, 0, 2, 1])
        #expect(selected.originalRowIndices == [7, 7, 7, 2])
        #expect(selected.provenanceStorageCapacity >= selected.rowCount)
        provenance[0] = 99
        #expect(source.originalRowIndices == [7, 2, 7])
    }
}
