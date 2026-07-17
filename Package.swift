// swift-tools-version: 6.0

import PackageDescription

let globalSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
    .define("ACCELERATE_NEW_LAPACK"),
    .define("ACCELERATE_LAPACK_ILP64"),
    .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK", "-Xcc", "-DACCELERATE_LAPACK_ILP64"])
]

let package = Package(
    name: "SwiftAnalytics",
    platforms: [
        .macOS(.v14),
        .iOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "SwiftDataFrame", targets: ["SwiftDataFrame"]),
        .library(name: "SwiftStats",     targets: ["SwiftStats"]),
        .library(name: "SwiftPreprocessing", targets: ["SwiftPreprocessing"]),
        .library(name: "SwiftML",           targets: ["SwiftML"]),
        .library(name: "SwiftCluster",       targets: ["SwiftCluster"]),
        .library(name: "SwiftNLP",           targets: ["SwiftNLP"]),
        .library(name: "SwiftOptimize",      targets: ["SwiftOptimize"]),
        .library(name: "SwiftForecast",      targets: ["SwiftForecast"]),
        .library(name: "SwiftLLM",           targets: ["SwiftLLM"]),
        .library(name: "SwiftExplain",       targets: ["SwiftExplain"]),
        .library(name: "SwiftPrivacy",       targets: ["SwiftPrivacy"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apache/arrow-swift.git",
            from: "21.0.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            from: "0.32.0"
        ),
    ],
    targets: [
        // ── SwiftDataFrame ──────────────────────────────────────────────
        .target(
            name: "SwiftDataFrame",
            dependencies: [
                .product(name: "Arrow", package: "arrow-swift")
            ],
            path: "Sources/SwiftDataFrame",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftDataFrameTests",
            dependencies: ["SwiftDataFrame"],
            path: "Tests/SwiftDataFrameTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftStats ───────────────────────────────────────────────────
        .target(
            name: "SwiftStats",
            dependencies: ["SwiftDataFrame"],
            path: "Sources/SwiftStats",
            swiftSettings: globalSwiftSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftStatsTests",
            dependencies: ["SwiftStats"],
            path: "Tests/SwiftStatsTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftPreprocessing ────────────────────────────────────────────
        .target(
            name: "SwiftPreprocessing",
            dependencies: [
                "SwiftDataFrame",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftPreprocessing",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftPreprocessingTests",
            dependencies: ["SwiftPreprocessing"],
            path: "Tests/SwiftPreprocessingTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftML ──────────────────────────────────────────────────────
        .target(
            name: "SwiftML",
            dependencies: [
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftML",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftMLTests",
            dependencies: ["SwiftML"],
            path: "Tests/SwiftMLTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftCluster ─────────────────────────────────────────────────
        .target(
            name: "SwiftCluster",
            dependencies: [
                "SwiftDataFrame",
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftCluster",
            swiftSettings: [
                .define("ACCELERATE_NEW_LAPACK"),
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"]),
                .enableUpcomingFeature("ExistentialAny")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftClusterTests",
            dependencies: ["SwiftCluster", "SwiftPreprocessing"],
            path: "Tests/SwiftClusterTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftNLP ─────────────────────────────────────────────────────
        .target(
            name: "SwiftNLP",
            dependencies: [
                "SwiftDataFrame",
            ],
            path: "Sources/SwiftNLP",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftNLPTests",
            dependencies: ["SwiftNLP"],
            path: "Tests/SwiftNLPTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftOptimize ────────────────────────────────────────────────
        .target(
            name: "SwiftOptimize",
            dependencies: [
                "SwiftDataFrame",
                "SwiftML",
            ],
            path: "Sources/SwiftOptimize",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftOptimizeTests",
            dependencies: ["SwiftOptimize"],
            path: "Tests/SwiftOptimizeTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftForecast ────────────────────────────────────────────────
        .target(
            name: "SwiftForecast",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
            ],
            path: "Sources/SwiftForecast",
            swiftSettings: globalSwiftSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftForecastTests",
            dependencies: ["SwiftForecast"],
            path: "Tests/SwiftForecastTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftLLM ─────────────────────────────────────────────────────
        .target(
            name: "SwiftLLM",
            dependencies: [
                "SwiftNLP",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftLLM",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"],
            path: "Tests/SwiftLLMTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftExplain ─────────────────────────────────────────────────
        .target(
            name: "SwiftExplain",
            dependencies: [
                "SwiftML",
                "SwiftStats",
                "SwiftDataFrame",
                "SwiftPreprocessing",
            ],
            path: "Sources/SwiftExplain",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftExplainTests",
            dependencies: ["SwiftExplain"],
            path: "Tests/SwiftExplainTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftPrivacy ─────────────────────────────────────────────────
        .target(
            name: "SwiftPrivacy",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
            ],
            path: "Sources/SwiftPrivacy",
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftPrivacyTests",
            dependencies: ["SwiftPrivacy"],
            path: "Tests/SwiftPrivacyTests",
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftAnalyticsBenchmarks ──────────────────────────────────────
        .executableTarget(
            name: "SwiftAnalyticsBenchmarks",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
                "SwiftPreprocessing",
                "SwiftML",
                "SwiftCluster",
                "SwiftNLP",
                "SwiftOptimize",
                "SwiftForecast",
                "SwiftLLM",
                "SwiftExplain",
                "SwiftPrivacy",
            ],
            path: "Benchmarks/Swift",
            swiftSettings: globalSwiftSettings
        ),
    ]
)
