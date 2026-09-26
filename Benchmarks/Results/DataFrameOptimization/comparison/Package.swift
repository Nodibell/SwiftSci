// swift-tools-version: 6.0
import PackageDescription
import Foundation
let kiraa = ProcessInfo.processInfo.environment["KIRAA_REPO"]!
let package = Package(name: "ProductionComparison", platforms: [.macOS(.v14)],
 dependencies: [.package(path: "../../../.."), .package(path: kiraa)],
 targets: [.executableTarget(name: "SciBench", dependencies: [
 .product(name: "SwiftDataFrame", package: "SwiftSci"),
 .product(name: "SwiftStats", package: "SwiftSci"),
 .product(name: "SwiftForecast", package: "SwiftSci")]),
 .executableTarget(name: "KiraaBench", dependencies: [
 .product(name: "SwiftPandas", package: "kiraa-swift-pandas")], swiftSettings: [.swiftLanguageMode(.v5)])])
