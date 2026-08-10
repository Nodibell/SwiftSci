// VisionBenchmarks.swift
// Benchmarks for SwiftVision:
//   • YOLOv8Detector detect (640x640 letterbox + real GPU forward pass)
//   • YOLOPreprocessor preprocess (1024x768 → 640x640 letterbox)
//   • UNetSegmentationModel predict (image segmentation forward pass)

import Foundation
import SwiftVision

struct VisionBenchmarks: BenchmarkSuite {
    let module = "SwiftVision"

    func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // ── 1. YOLOv8Detector End-to-End Object Detection ────────────────
        let detector = YOLOv8Detector(confidenceThreshold: 0.25, iouThreshold: 0.45)
        let testImage = ImageDataset(
            width: 640,
            height: 480,
            channels: 3,
            data: [Double](repeating: 0.5, count: 640 * 480 * 3)
        )

        let yoloResult = await BenchmarkRunner.run(
            name: "YOLOv8Detector detect (640×640 real GPU inference)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            _ = try await detector.detect(image: testImage)
        }
        results.append(yoloResult)

        // ── 2. YOLOPreprocessor Letterbox Resizing ──────────────────────
        let prep = YOLOPreprocessor(targetWidth: 640, targetHeight: 640)
        let prepImage = ImageDataset(
            width: 1920,
            height: 1080,
            channels: 3,
            data: [Double](repeating: 0.8, count: 1920 * 1080 * 3)
        )

        let prepResult = await BenchmarkRunner.run(
            name: "YOLOPreprocessor letterbox (1920×1080 → 640×640)",
            module: module,
            warmup: 2,
            iterations: 10
        ) {
            _ = prep.preprocess(image: prepImage)
        }
        results.append(prepResult)

        // ── 3. UNet Segmentation Model Predict ──────────────────────────
        let unetModel = UNetSegmentationModel(inputChannels: 3, numClasses: 2)
        let unetImage = ImageDataset(
            width: 128,
            height: 128,
            channels: 3,
            data: [Double](repeating: 0.5, count: 128 * 128 * 3)
        )

        let unetResult = await BenchmarkRunner.run(
            name: "UNetSegmentationModel predict (128×128 image)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            _ = try await unetModel.predict(image: unetImage)
        }
        results.append(unetResult)

        return results
    }
}
