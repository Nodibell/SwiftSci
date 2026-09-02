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
    }
}
