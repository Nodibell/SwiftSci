#if canImport(SwiftUI)
import SwiftUI

/// Chart type for native `SwiftSciChartView` rendering.
public enum ChartType: Sendable {
    case line
    case bar
    case heatmap
}

/// Data series for native SwiftUI rendering.
public struct ChartSeries: Sendable, Identifiable {
    /// Unique identifier for the data series.
    public let id: String
    /// X-axis coordinate values.
    public let x: [Double]
    /// Y-axis coordinate values.
    public let y: [Double]
    /// Display color for the series.
    public let color: Color

    /// Creates a new data series for SwiftUI chart rendering.
    /// - Parameters:
    ///   - id: Unique identifier for the series. Defaults to a random UUID.
    ///   - x: Array of X-axis coordinates.
    ///   - y: Array of Y-axis coordinates matching `x` length.
    ///   - color: Rendering color for lines or bars. Defaults to blue.
    public init(id: String = UUID().uuidString, x: [Double], y: [Double], color: Color = .blue) {
        self.id = id
        self.x = x
        self.y = y
        self.color = color
    }
}

/// A native SwiftUI `Canvas` chart component for SwiftSci datasets.
///
/// `SwiftSciChartView` provides lightweight, zero-dependency 2D chart rendering
/// using SwiftUI `Canvas`, supporting line charts, bar charts, and heatmaps.
///
/// ```swift
/// let series = ChartSeries(x: [1.0, 2.0, 3.0], y: [10.0, 25.0, 18.0], color: .green)
/// let chartView = SwiftSciChartView(title: "Sales Growth", type: .line, series: [series])
/// ```
@MainActor
public struct SwiftSciChartView: View {
    /// Title header text displayed above the chart canvas.
    public let title: String
    /// Rendering mode for the dataset (line, bar, or heatmap).
    public let type: ChartType
    /// Collection of data series to plot on the canvas.
    public let series: [ChartSeries]

    /// Initializes a native SwiftUI chart view.
    /// - Parameters:
    ///   - title: Header title displayed above the chart.
    ///   - type: Rendering layout mode (`.line`, `.bar`, or `.heatmap`). Defaults to `.line`.
    ///   - series: Array of data series to draw.
    public init(title: String, type: ChartType = .line, series: [ChartSeries]) {
        self.title = title
        self.type = type
        self.series = series
    }

    /// The content and behavior of the SwiftUI view.
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)

            Canvas { context, size in
                guard !series.isEmpty else { return }

                let padding: CGFloat = 30.0
                let width = size.width - 2 * padding
                let height = size.height - 2 * padding

                let allX = series.flatMap { $0.x }
                let allY = series.flatMap { $0.y }

                guard let minX = allX.min(), let maxX = allX.max(),
                      let minY = allY.min(), let maxY = allY.max() else { return }

                let rangeX = maxX > minX ? maxX - minX : 1.0
                let rangeY = maxY > minY ? maxY - minY : 1.0

                // Draw axes
                var axisPath = Path()
                axisPath.move(to: CGPoint(x: padding, y: padding))
                axisPath.addLine(to: CGPoint(x: padding, y: size.height - padding))
                axisPath.addLine(to: CGPoint(x: size.width - padding, y: size.height - padding))
                context.stroke(axisPath, with: .color(.gray), lineWidth: 1)

                // Render series based on type
                for s in series {
                    guard s.x.count == s.y.count, !s.x.isEmpty else { continue }

                    switch type {
                    case .line:
                        var linePath = Path()
                        for (idx, (xVal, yVal)) in zip(s.x, s.y).enumerated() {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height

                            if idx == 0 {
                                linePath.move(to: CGPoint(x: px, y: py))
                            } else {
                                linePath.addLine(to: CGPoint(x: px, y: py))
                            }
                        }
                        context.stroke(linePath, with: .color(s.color), lineWidth: 2)

                    case .bar:
                        let barWidth = width / CGFloat(s.x.count) * 0.8
                        for (xVal, yVal) in zip(s.x, s.y) {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height
                            let rect = CGRect(x: px - barWidth / 2, y: py, width: barWidth, height: (size.height - padding) - py)
                            context.fill(Path(rect), with: .color(s.color))
                        }

                    case .heatmap:
                        let count = CGFloat(s.x.count)
                        let side = min(width, height) / count
                        for (idx, yVal) in s.y.enumerated() {
                            let normVal = CGFloat((yVal - minY) / rangeY)
                            let rect = CGRect(x: padding + CGFloat(idx) * side, y: padding, width: side, height: side)
                            context.fill(Path(rect), with: .color(s.color.opacity(Double(normVal))))
                        }
                    }
                }
            }
            .frame(minHeight: 200)
        }
        .padding()
    }
}
#endif
