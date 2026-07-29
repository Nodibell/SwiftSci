import Testing
@testable import SwiftDataFrame

@Suite("DataFrame SIMD Bitmask Filter & Primitive Sort Tests (Phase 1 & 2)")
struct DataFrameSIMDFilterTests {

    @Test("SIMD Double comparison filtering produces exact matching results across all operators")
    func testSIMDDoubleFilteringAllOperators() throws {
        let values: [Double] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        let col = TypedColumn<Double>(name: "val", values: values.map { $0 })

        // .greaterThan
        let gt = col.filteredIndices(matching: .greaterThan(30.0))
        #expect(gt == [3, 4, 5, 6])

        // .lessThan
        let lt = col.filteredIndices(matching: .lessThan(30.0))
        #expect(lt == [0, 1])

        // .greaterThanOrEqual
        let gte = col.filteredIndices(matching: .greaterThanOrEqual(30.0))
        #expect(gte == [2, 3, 4, 5, 6])

        // .lessThanOrEqual
        let lte = col.filteredIndices(matching: .lessThanOrEqual(30.0))
        #expect(lte == [0, 1, 2])

        // .equals
        let eq = col.filteredIndices(matching: .equals(30.0))
        #expect(eq == [2])

        // .notEquals
        let neq = col.filteredIndices(matching: .notEquals(30.0))
        #expect(neq == [0, 1, 3, 4, 5, 6])

        // Invalid RHS type returns nil
        let invalid = col.filteredIndices(matching: .greaterThan("invalid_string"))
        #expect(invalid == nil)
    }

    @Test("SIMD Int64 comparison filtering produces exact matching results across all operators")
    func testSIMDInt64FilteringAllOperators() throws {
        let values: [Int64] = [10, 20, 30, 40, 50, 60, 70]
        let col = TypedColumn<Int64>(name: "id", values: values.map { $0 })

        // .greaterThan
        let gt = col.filteredIndices(matching: .greaterThan(Int64(30)))
        #expect(gt == [3, 4, 5, 6])

        // .lessThan
        let lt = col.filteredIndices(matching: .lessThan(Int64(30)))
        #expect(lt == [0, 1])

        // .greaterThanOrEqual
        let gte = col.filteredIndices(matching: .greaterThanOrEqual(Int64(30)))
        #expect(gte == [2, 3, 4, 5, 6])

        // .lessThanOrEqual
        let lte = col.filteredIndices(matching: .lessThanOrEqual(Int64(30)))
        #expect(lte == [0, 1, 2])

        // .equals
        let eq = col.filteredIndices(matching: .equals(Int64(30)))
        #expect(eq == [2])

        // .notEquals
        let neq = col.filteredIndices(matching: .notEquals(Int64(30)))
        #expect(neq == [0, 1, 3, 4, 5, 6])

        // Invalid RHS type returns nil
        let invalid = col.filteredIndices(matching: .greaterThan("invalid_string"))
        #expect(invalid == nil)
    }

    @Test("SIMD filtering handles empty dataframes, odd lengths, and nulls")
    func testSIMDEmptyAndLimits() throws {
        let emptyCol = TypedColumn<Double>(name: "x", values: [])
        #expect(emptyCol.filteredIndices(matching: .greaterThan(10.0)) == [])

        let nullableCol = TypedColumn<Double>(name: "y", values: [1.0, nil, 3.0, 4.0, nil, 6.0])
        let gtNull = nullableCol.filteredIndices(matching: .greaterThan(2.0))
        #expect(gtNull == [2, 3, 5])

        let isNull = nullableCol.filteredIndices(matching: .isNull)
        #expect(isNull == [1, 4])

        let isNotNull = nullableCol.filteredIndices(matching: .isNotNull)
        #expect(isNotNull == [0, 2, 3, 5])
    }

    @Test("Primitive fast sorting operates correctly on Double, Float, Int64, and Int32")
    func testPrimitiveFastSorting() throws {
        // Double
        let colDouble = TypedColumn<Double>(name: "d", values: [5.0, 2.0, nil, 9.0, 1.0])
        #expect(colDouble.sortedIndices(ascending: true) == [4, 1, 0, 3, 2])
        #expect(colDouble.sortedIndices(ascending: false) == [3, 0, 1, 4, 2])

        // Float
        let colFloat = TypedColumn<Float>(name: "f", values: [5.0, 2.0, nil, 9.0, 1.0])
        #expect(colFloat.sortedIndices(ascending: true) == [4, 1, 0, 3, 2])
        #expect(colFloat.sortedIndices(ascending: false) == [3, 0, 1, 4, 2])

        // Int64
        let colInt64 = TypedColumn<Int64>(name: "i64", values: [5, 2, nil, 9, 1])
        #expect(colInt64.sortedIndices(ascending: true) == [4, 1, 0, 3, 2])
        #expect(colInt64.sortedIndices(ascending: false) == [3, 0, 1, 4, 2])

        // Int32
        let colInt32 = TypedColumn<Int32>(name: "i32", values: [5, 2, nil, 9, 1])
        #expect(colInt32.sortedIndices(ascending: true) == [4, 1, 0, 3, 2])
        #expect(colInt32.sortedIndices(ascending: false) == [3, 0, 1, 4, 2])
    }
}
