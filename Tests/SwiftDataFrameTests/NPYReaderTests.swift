import XCTest
@testable import SwiftDataFrame

final class NPYReaderTests: XCTestCase {

    func testSyntheticNPYArrayParse() throws {
        // Build valid NPY v1.0 buffer for shape (3, 2), dtype '<f8'
        var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]) // \x93NUMPY\x01\x00
        let dictStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 2), }\n"
        let paddedDict = dictStr.padding(toLength: 64, withPad: " ", startingAt: 0)
        let headerLen = UInt16(paddedDict.utf8.count)
        data.append(UInt8(headerLen & 0xFF))
        data.append(UInt8((headerLen >> 8) & 0xFF))
        data.append(Data(paddedDict.utf8))

        // Append 6 Double values
        let values: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for v in values {
            var bitVal = v.bitPattern
            withUnsafeBytes(of: &bitVal) { data.append(contentsOf: $0) }
        }

        let arr = try NPYReader.read(data: data)
        XCTAssertEqual(arr.shape, [3, 2])
        XCTAssertEqual(arr.descr, "<f8")
        XCTAssertFalse(arr.fortranOrder)
        XCTAssertEqual(arr.toDoubles(), values)

        let df = try arr.toDataFrame(columnPrefix: "feat")
        XCTAssertEqual(df.rowCount, 3)
        XCTAssertEqual(df.columnNames, ["feat_0", "feat_1"])
        let col0 = (df[column: "feat_0"] as? TypedColumn<Double>)?.values
        XCTAssertEqual(col0, [1.0, 3.0, 5.0])
    }

    func testReadMNISTMiniNPZArchive() throws {
        let npzURL = URL(fileURLWithPath: "/Users/oleksiichumak/Developer/Xcode.projects/Saura/sample_data/mnist_mini.npz")
        guard FileManager.default.fileExists(atPath: npzURL.path) else {
            print("mnist_mini.npz not found at path; skipping real file test")
            return
        }

        let arrays = try NPZReader.read(url: npzURL)
        XCTAssertTrue(arrays.keys.contains("x_train"))
        XCTAssertTrue(arrays.keys.contains("y_train"))

        let xArr = arrays["x_train"]!
        XCTAssertEqual(xArr.shape, [100, 8, 8])
        XCTAssertEqual(xArr.elementCount, 6400)

        let yArr = arrays["y_train"]!
        XCTAssertEqual(yArr.shape, [100])
        XCTAssertEqual(yArr.elementCount, 100)

        // Read directly via DataFrame(npz:)
        let df = try DataFrame(npz: npzURL)
        print("✅ MNIST Mini NPZ loaded into DataFrame! Shape: \(df.shape.rows) rows x \(df.shape.columns) columns")
        XCTAssertEqual(df.rowCount, 100)
        XCTAssertEqual(df.columnNames.count, 65) // 64 pixel columns + 1 label column
        XCTAssertTrue(df.columnNames.contains("pixel_0"))
        XCTAssertTrue(df.columnNames.contains("pixel_63"))
        XCTAssertTrue(df.columnNames.contains("label"))

        // Read specific array name
        let yDF = try DataFrame(npz: npzURL, arrayName: "y_train")
        XCTAssertEqual(yDF.rowCount, 100)
        XCTAssertEqual(yDF.columnNames, ["value"])
    }

    func testNPYTypesAndConversions() throws {
        // Test Float32 (<f4)
        var f4Data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let f4Dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, ), }\n"
        let f4Pad = f4Dict.padding(toLength: 64, withPad: " ", startingAt: 0)
        let f4Len = UInt16(f4Pad.utf8.count)
        f4Data.append(UInt8(f4Len & 0xFF))
        f4Data.append(UInt8((f4Len >> 8) & 0xFF))
        f4Data.append(Data(f4Pad.utf8))
        var f1: Float = 3.5
        var f2: Float = 7.25
        withUnsafeBytes(of: &f1) { f4Data.append(contentsOf: $0) }
        withUnsafeBytes(of: &f2) { f4Data.append(contentsOf: $0) }

        let arrF4 = try NPYReader.read(data: f4Data)
        XCTAssertEqual(arrF4.toDoubles().count, 2)
        XCTAssertEqual(arrF4.toInt64s(), [3, 7])

        // Test Int32 (<i4)
        var i4Data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let i4Dict = "{'descr': '<i4', 'fortran_order': False, 'shape': (2, ), }\n"
        let i4Pad = i4Dict.padding(toLength: 64, withPad: " ", startingAt: 0)
        let i4Len = UInt16(i4Pad.utf8.count)
        i4Data.append(UInt8(i4Len & 0xFF))
        i4Data.append(UInt8((i4Len >> 8) & 0xFF))
        i4Data.append(Data(i4Pad.utf8))
        var i1: Int32 = 42
        var i2: Int32 = -10
        withUnsafeBytes(of: &i1) { i4Data.append(contentsOf: $0) }
        withUnsafeBytes(of: &i2) { i4Data.append(contentsOf: $0) }

        let arrI4 = try NPYReader.read(data: i4Data)
        XCTAssertEqual(arrI4.toDoubles(), [42.0, -10.0])
        XCTAssertEqual(arrI4.toInt64s(), [42, -10])

        // Test Int64 (<i8) toInt64s and toDoubles
        var i8Data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let i8Dict = "{'descr': '<i8', 'fortran_order': False, 'shape': (2, ), }\n"
        let i8Pad = i8Dict.padding(toLength: 64, withPad: " ", startingAt: 0)
        let i8Len = UInt16(i8Pad.utf8.count)
        i8Data.append(UInt8(i8Len & 0xFF))
        i8Data.append(UInt8((i8Len >> 8) & 0xFF))
        i8Data.append(Data(i8Pad.utf8))
        var l1: Int64 = 1000
        var l2: Int64 = 2000
        withUnsafeBytes(of: &l1) { i8Data.append(contentsOf: $0) }
        withUnsafeBytes(of: &l2) { i8Data.append(contentsOf: $0) }

        let arrI8 = try NPYReader.read(data: i8Data)
        XCTAssertEqual(arrI8.toInt64s(), [1000, 2000])

        // Test UInt8 (|u1)
        var u1Data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let u1Dict = "{'descr': '|u1', 'fortran_order': False, 'shape': (3, ), }\n"
        let u1Pad = u1Dict.padding(toLength: 64, withPad: " ", startingAt: 0)
        let u1Len = UInt16(u1Pad.utf8.count)
        u1Data.append(UInt8(u1Len & 0xFF))
        u1Data.append(UInt8((u1Len >> 8) & 0xFF))
        u1Data.append(Data(u1Pad.utf8))
        u1Data.append(contentsOf: [10, 20, 30])

        let arrU1 = try NPYReader.read(data: u1Data)
        XCTAssertEqual(arrU1.toDoubles(), [10.0, 20.0, 30.0])
        XCTAssertEqual(arrU1.toInt64s(), [10, 20, 30])
    }

    func testNPYFileReadAndDataFrameConversions() throws {
        // Build 1D array
        var data1D = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
        let dict1D = "{'descr': '<f8', 'fortran_order': False, 'shape': (4, ), }\n"
        let pad1D = dict1D.padding(toLength: 64, withPad: " ", startingAt: 0)
        let len1D = UInt16(pad1D.utf8.count)
        data1D.append(UInt8(len1D & 0xFF))
        data1D.append(UInt8((len1D >> 8) & 0xFF))
        data1D.append(Data(pad1D.utf8))
        for v in [10.0, 20.0, 30.0, 40.0] {
            var b = v.bitPattern
            withUnsafeBytes(of: &b) { data1D.append(contentsOf: $0) }
        }

        // Save to temp file and read with DataFrame(npy:)
        let tempNPY = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".npy")
        defer { try? FileManager.default.removeItem(at: tempNPY) }
        try data1D.write(to: tempNPY)

        let df1D = try DataFrame(npy: tempNPY)
        XCTAssertEqual(df1D.rowCount, 4)
        XCTAssertEqual(df1D.columnNames, ["value"])

        let fromFileArr = try NPYReader.read(url: tempNPY)
        XCTAssertEqual(fromFileArr.elementCount, 4)

        // Test error handling
        XCTAssertThrowsError(try NPYReader.read(data: Data([0x00, 0x01, 0x02]))) { error in
            XCTAssertTrue(error is SwiftMLError)
        }

        let nonExistent = URL(fileURLWithPath: "/tmp/non_existent_\(UUID().uuidString).npy")
        XCTAssertThrowsError(try DataFrame(npy: nonExistent))
        XCTAssertThrowsError(try DataFrame(npz: nonExistent))
    }
}
