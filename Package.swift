// swift-tools-version: 6.0

import PackageDescription

let globalSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny")
]

let globalCSettings: [CSetting] = [
    .define("ACCELERATE_NEW_LAPACK")
]

let package = Package(
    name: "SwiftSci",
    platforms: [
        .macOS(.v14),
        .iOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "SwiftDataFrame",     targets: ["SwiftDataFrame"]),
        .library(name: "SwiftStats",         targets: ["SwiftStats"]),
        .library(name: "SwiftPreprocessing", targets: ["SwiftPreprocessing"]),
        .library(name: "SwiftML",            targets: ["SwiftML"]),
        .library(name: "SwiftCluster",       targets: ["SwiftCluster"]),
        .library(name: "SwiftNLP",           targets: ["SwiftNLP"]),
        .library(name: "SwiftOptimize",      targets: ["SwiftOptimize"]),
        .library(name: "SwiftForecast",      targets: ["SwiftForecast"]),
        .library(name: "SwiftLLM",           targets: ["SwiftLLM"]),
        .library(name: "SwiftExplain",       targets: ["SwiftExplain"]),
        .library(name: "SwiftVisualization", targets: ["SwiftVisualization"]),
        .library(name: "SwiftVision",        targets: ["SwiftVision"]),
        .library(name: "SwiftDatabase",      targets: ["SwiftDatabase"]),
        .library(name: "SwiftAgent",         targets: ["SwiftAgent"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apache/arrow-swift.git",
            from: "21.0.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.6"
        ),
        .package(
            url: "https://github.com/swiftlang/swift-docc-plugin",
            from: "1.3.0"
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
            resources: [
                .process("SwiftDataFrame.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftDataFrameTests",
            dependencies: ["SwiftDataFrame"],
            path: "Tests/SwiftDataFrameTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftStats ───────────────────────────────────────────────────
        .target(
            name: "SwiftStats",
            dependencies: ["SwiftDataFrame"],
            path: "Sources/SwiftStats",
            resources: [
                .process("SwiftStats.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]

        ),
        .testTarget(
            name: "SwiftStatsTests",
            dependencies: ["SwiftStats"],
            path: "Tests/SwiftStatsTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftPreprocessing ────────────────────────────────────────────
        .target(
            name: "SwiftPreprocessing",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftPreprocessing",
            resources: [
                .process("SwiftPreprocessing.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftPreprocessingTests",
            dependencies: ["SwiftPreprocessing"],
            path: "Tests/SwiftPreprocessingTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftML ──────────────────────────────────────────────────────
        .target(
            name: "SwiftML",
            dependencies: [
                "SwiftDataFrame",
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/SwiftML",
            resources: [
                .process("SwiftML.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftMLTests",
            dependencies: ["SwiftML"],
            path: "Tests/SwiftMLTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftCluster ─────────────────────────────────────────────────
        .target(
            name: "SwiftCluster",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
                "SwiftPreprocessing",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/SwiftCluster",
            resources: [
                .process("SwiftCluster.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftClusterTests",
            dependencies: ["SwiftCluster", "SwiftPreprocessing"],
            path: "Tests/SwiftClusterTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftNLP ─────────────────────────────────────────────────────
        .target(
            name: "SwiftNLP",
            dependencies: [
                "SwiftDataFrame",
            ],
            path: "Sources/SwiftNLP",
            resources: [
                .process("SwiftNLP.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftNLPTests",
            dependencies: ["SwiftNLP"],
            path: "Tests/SwiftNLPTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings,
        ),

        // ── SwiftOptimize ────────────────────────────────────────────────
        .target(
            name: "SwiftOptimize",
            dependencies: [
                "SwiftML",
            ],
            path: "Sources/SwiftOptimize",
            resources: [
                .process("SwiftOptimize.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftOptimizeTests",
            dependencies: ["SwiftOptimize"],
            path: "Tests/SwiftOptimizeTests",
            cSettings: globalCSettings,
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
            resources: [
                .process("SwiftForecast.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings,
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "SwiftForecastTests",
            dependencies: ["SwiftForecast"],
            path: "Tests/SwiftForecastTests",
            cSettings: globalCSettings,
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
            resources: [
                .process("SwiftLLM.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"],
            path: "Tests/SwiftLLMTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
            
        ),

        // ── SwiftExplain ─────────────────────────────────────────────────
        .target(
            name: "SwiftExplain",
            dependencies: [
                "SwiftML",
                "SwiftStats",
                "SwiftPreprocessing",
            ],
            path: "Sources/SwiftExplain",
            resources: [
                .process("SwiftExplain.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftExplainTests",
            dependencies: ["SwiftExplain"],
            path: "Tests/SwiftExplainTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftVisualization ─────────────────────────────────────────────
        .target(
            name: "SwiftVisualization",
            dependencies: [
                "SwiftDataFrame",
                "SwiftStats",
            ],
            path: "Sources/SwiftVisualization",
            resources: [
                .process("SwiftVisualization.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftVisualizationTests",
            dependencies: ["SwiftVisualization"],
            path: "Tests/SwiftVisualizationTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftVision ──────────────────────────────────────────────────
        .target(
            name: "SwiftVision",
            dependencies: [
                "SwiftDataFrame",
                "SwiftML",
            ],
            path: "Sources/SwiftVision",
            resources: [
                .process("SwiftVision.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftVisionTests",
            dependencies: ["SwiftVision"],
            path: "Tests/SwiftVisionTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftDatabase ────────────────────────────────────────────────
        .target(
            name: "SwiftDatabase",
            dependencies: [
                "SwiftDataFrame",
            ],
            path: "Sources/SwiftDatabase",
            resources: [
                .process("SwiftDatabase.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftDatabaseTests",
            dependencies: ["SwiftDatabase"],
            path: "Tests/SwiftDatabaseTests",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),

        // ── SwiftAgent ───────────────────────────────────────────────────
        .target(
            name: "SwiftAgent",
            dependencies: [
                "SwiftDataFrame",
                "SwiftLLM",
            ],
            path: "Sources/SwiftAgent",
            resources: [
                .process("SwiftAgent.docc")
            ],
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
        .testTarget(
            name: "SwiftAgentTests",
            dependencies: ["SwiftAgent"],
            path: "Tests/SwiftAgentTests",
            cSettings: globalCSettings,
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
                "SwiftVision",
                "SwiftDatabase",
                "SwiftAgent",
            ],
            path: "Benchmarks/Swift",
            cSettings: globalCSettings,
            swiftSettings: globalSwiftSettings
        ),
    ]
)
