import Foundation
import ArgumentParser
import SwiftDataFrame
import SwiftML

@main
struct SwiftSciCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "swiftsci",
        abstract: "SwiftSci Ecosystem Command-Line Utility for DataFrames, Conversions, and Models.",
        version: "3.5.2",
        subcommands: [Summary.self, Convert.self, ExportModel.self]
    )
}

extension SwiftSciCLI {
    struct Summary: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print dataset summary (rows, columns, dtypes)."
        )

        @Argument(help: "Path to input dataset file (.csv, .feather, .parquet, .npy, .npz).")
        var filePath: String

        func run() async throws {
            let url = URL(fileURLWithPath: filePath)
            let ext = url.pathExtension.lowercased()
            let df: DataFrame
            if ext == "feather" || ext == "arrow" {
                df = try await DataFrame(feather: url)
            } else if ext == "parquet" {
                df = try await DataFrame(parquet: url)
            } else if ext == "npz" {
                df = try DataFrame(npz: url)
            } else if ext == "npy" {
                df = try DataFrame(npy: url)
            } else {
                df = try await DataFrame(csv: url)
            }

            print("=== SwiftSci Dataset Summary ===")
            print("File: \(filePath)")
            print("Shape: \(df.shape.rows) rows × \(df.shape.columns) columns")
            print("Columns:")
            for col in df.columns {
                print("  - \(col.name): \(col.dtype)")
            }
            print("\nPreview:")
            df.debugPrint(maxRows: 5)
        }
    }

    struct Convert: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Convert dataset between CSV, Feather, and Parquet binary formats."
        )

        @Argument(help: "Input dataset file path.")
        var inputPath: String

        @Argument(help: "Output dataset file path.")
        var outputPath: String

        func run() async throws {
            let inURL = URL(fileURLWithPath: inputPath)
            let outURL = URL(fileURLWithPath: outputPath)

            let inExt = inURL.pathExtension.lowercased()
            let df: DataFrame
            if inExt == "feather" || inExt == "arrow" {
                df = try await DataFrame(feather: inURL)
            } else if inExt == "parquet" {
                df = try await DataFrame(parquet: inURL)
            } else if inExt == "npz" {
                df = try DataFrame(npz: inURL)
            } else if inExt == "npy" {
                df = try DataFrame(npy: inURL)
            } else {
                df = try await DataFrame(csv: inURL)
            }

            let outExt = outURL.pathExtension.lowercased()
            if outExt == "feather" || outExt == "arrow" {
                try await df.writeFeather(to: outURL)
            } else if outExt == "parquet" {
                try await df.writeParquet(to: outURL)
            } else {
                try await df.writeCSV(to: outURL)
            }

            print("Successfully converted '\(inputPath)' -> '\(outputPath)' (\(df.shape.rows) rows)")
        }
    }

    struct ExportModel: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Export model weights into binary ONNX format."
        )

        @Argument(help: "Output .onnx file path.")
        var outputPath: String

        func run() async throws {
            let onnxData = ONNXExporter.exportBinaryONNX(
                name: "SwiftSciModel",
                inputs: ["feature1", "feature2"],
                output: "prediction",
                weights: [1.0, 0.5],
                bias: 0.1
            )
            let outURL = URL(fileURLWithPath: outputPath)
            try onnxData.write(to: outURL)
            print("=== SwiftSci Model Exporter ===")
            print("Successfully exported binary ONNX model to '\(outputPath)' (\(onnxData.count) bytes).")
        }
    }
}
