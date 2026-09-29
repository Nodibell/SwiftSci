import CoreFoundation
import Foundation
import MLX
import SwiftSciBenchmarkSupport
import SwiftVision

struct VisionLetterboxInput {
  let width: Int
  let height: Int
  let channels: Int
  let targetWidth: Int
  let targetHeight: Int
  let paddingColor: Double
  let pixels: [Double]

  static func decode(_ data: Data, rows: Int) throws -> VisionLetterboxInput {
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(object.keys) == ["operation", "device", "layout", "encoding", "width", "height", "channels",
        "target_width", "target_height", "padding_color", "pixels"],
      object["operation"] as? String == "vision-letterbox-cpu",
      object["device"] as? String == "cpu",
      object["layout"] as? String == "CHW",
      object["encoding"] as? String == "normalized-f32"
    else { throw BenchmarkFailure("Invalid vision letterbox input contract") }
    let width = try integer(object["width"], range: 1...32)
    let height = try integer(object["height"], range: 1...32)
    let targetWidth = try integer(object["target_width"], range: 1...32)
    let targetHeight = try integer(object["target_height"], range: 1...32)
    let channels = try integer(object["channels"], range: 1...3)
    let count = width * height
    guard [1, 3].contains(channels), rows == count,
      let rawPixels = object["pixels"] as? [Any], rawPixels.count == count * channels
    else { throw BenchmarkFailure("Invalid vision channel or buffer dimensions") }
    let pixels = try rawPixels.map { try normalized($0, exactFloat32: true) }
    let padding = try normalized(object["padding_color"], exactFloat32: false)
    let scale = min(Double(targetWidth) / Double(width), Double(targetHeight) / Double(height))
    let resizedWidth = min(targetWidth, max(1, Int((Double(width) * scale).rounded())))
    let resizedHeight = min(targetHeight, max(1, Int((Double(height) * scale).rounded())))
    if resizedWidth != width || resizedHeight != height {
      for channel in 0..<channels {
        let plane = pixels[(channel * count)..<((channel + 1) * count)]
        guard plane.allSatisfy({ $0 == pixels[channel * count] }) else {
          throw BenchmarkFailure("Vision resize contract requires constant planes")
        }
      }
    }
    return VisionLetterboxInput(width: width, height: height, channels: channels,
      targetWidth: targetWidth, targetHeight: targetHeight, paddingColor: padding, pixels: pixels)
  }

  private static func integer(_ value: Any?, range: ClosedRange<Int>) throws -> Int {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
      !["d", "f"].contains(String(cString: number.objCType)),
      number.doubleValue >= Double(range.lowerBound), number.doubleValue <= Double(range.upperBound),
      number.doubleValue.rounded(.towardZero) == number.doubleValue
    else { throw BenchmarkFailure("Vision dimensions must be bounded JSON integers") }
    return number.intValue
  }

  private static func normalized(_ value: Any?, exactFloat32: Bool) throws -> Double {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
      throw BenchmarkFailure("Vision values must be numbers")
    }
    let double = number.doubleValue
    guard double.isFinite, (0...1).contains(double),
      !exactFloat32 || Double(Float(double)) == double
    else { throw BenchmarkFailure("Invalid normalized vision value") }
    return double
  }
}

extension Worker {
  @inline(never) static func executeVisionLetterbox(_ input: VisionLetterboxInput) throws -> Output {
    try Device.withDefaultDevice(.cpu) {
      try Stream.withNewDefaultStream(device: .cpu) {
        let image = ImageDataset(width: input.width, height: input.height,
          channels: input.channels, data: input.pixels)
        let preprocessor = YOLOPreprocessor(targetWidth: input.targetWidth,
          targetHeight: input.targetHeight, paddingColor: input.paddingColor)
        let tensor = preprocessor.preprocess(image: image)
        guard tensor.dtype == .float32, tensor.shape == [1, input.targetHeight, input.targetWidth, 3] else {
          throw BenchmarkFailure("Invalid vision tensor shape or dtype")
        }
        eval(tensor)
        StreamOrDevice.default.stream.synchronize()
        let values = tensor.asArray(Float.self)
        guard values.count == input.targetHeight * input.targetWidth * 3,
          values.allSatisfy(\.isFinite)
        else { throw BenchmarkFailure("Invalid vision output values") }
        return .values(tensor.shape.map(Double.init) + values.map(Double.init))
      }
    }
  }
}
