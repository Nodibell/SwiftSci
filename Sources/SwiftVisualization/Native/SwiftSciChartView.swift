#if canImport(SwiftUI)
import SwiftUI

/// Chart type for native `SwiftSciChartView` rendering.
public enum ChartType: String, Sendable, CaseIterable {
    case line
    case bar
    case scatter
    case histogram
    case heatmap
}

/// Rendering options and customization for `SwiftSciChartView`.
public struct ChartOptions: Sendable {
    /// Whether to render coordinate axes and tick marks.
    public var showAxes: Bool
    /// Stroke line width for line and scatter plots.
    public var lineWidth: CGFloat
    /// Width ratio for bars in bar chart and histogram (0.1 to 1.0).
    public var barWidthRatio: CGFloat
    /// Optional label displayed on the horizontal X axis.
    public var xAxisLabel: String?
    /// Optional label displayed on the vertical Y axis.
    public var yAxisLabel: String?

    /// Creates a new chart options configuration.
    public init(
        showAxes: Bool = true,
        lineWidth: CGFloat = 2.0,
        barWidthRatio: CGFloat = 0.8,
        xAxisLabel: String? = nil,
        yAxisLabel: String? = nil
    ) {
        self.showAxes = showAxes
        self.lineWidth = lineWidth
        self.barWidthRatio = max(0.1, min(1.0, barWidthRatio))
        self.xAxisLabel = xAxisLabel
        self.yAxisLabel = yAxisLabel
    }

    /// Default configuration.
    public static let `default` = ChartOptions()
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
/// using SwiftUI `Canvas`, supporting line charts, bar charts, scatter plots, histograms, and 2D heatmaps.
@MainActor
public struct SwiftSciChartView: View {
    /// Title header text displayed above the chart canvas.
    public let title: String
    /// Rendering mode for the dataset (line, bar, scatter, histogram, or heatmap).
    public let type: ChartType
    /// Collection of data series to plot on the canvas.
    public let series: [ChartSeries]
    /// Visual customization options.
    public let options: ChartOptions

    /// Initializes a native SwiftUI chart view.
    /// - Parameters:
    ///   - title: Header title displayed above the chart.
    ///   - type: Rendering layout mode (`.line`, `.bar`, `.scatter`, `.histogram`, or `.heatmap`). Defaults to `.line`.
    ///   - series: Array of data series to draw.
    ///   - options: Custom chart display options.
    public init(
        title: String,
        type: ChartType = .line,
        series: [ChartSeries],
        options: ChartOptions = .default
    ) {
        self.title = title
        self.type = type
        self.series = series
        self.options = options
    }

    /// The content and behavior of the SwiftUI view.
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)

            Canvas { context, size in
                guard !series.isEmpty else { return }

                let padding: CGFloat = 35.0
                let width = max(1.0, size.width - 2 * padding)
                let height = max(1.0, size.height - 2 * padding)

                let allX = series.flatMap { $0.x }
                let allY = series.flatMap { $0.y }

                guard let minX = allX.min(), let maxX = allX.max(),
                      let minY = allY.min(), let maxY = allY.max() else { return }

                let rangeX = maxX > minX ? maxX - minX : 1.0
                let rangeY = maxY > minY ? maxY - minY : 1.0

                // Draw coordinate axes
                if options.showAxes {
                    var axisPath = Path()
                    axisPath.move(to: CGPoint(x: padding, y: padding))
                    axisPath.addLine(to: CGPoint(x: padding, y: size.height - padding))
                    axisPath.addLine(to: CGPoint(x: size.width - padding, y: size.height - padding))
                    context.stroke(axisPath, with: .color(.gray.opacity(0.6)), lineWidth: 1)

                    // Tick marks
                    var tickPath = Path()
                    let tickSteps = 4
                    for i in 0...tickSteps {
                        let frac = CGFloat(i) / CGFloat(tickSteps)
                        // X ticks
                        let tx = padding + frac * width
                        tickPath.move(to: CGPoint(x: tx, y: size.height - padding))
                        tickPath.addLine(to: CGPoint(x: tx, y: size.height - padding + 5))
                        // Y ticks
                        let ty = size.height - padding - frac * height
                        tickPath.move(to: CGPoint(x: padding - 5, y: ty))
                        tickPath.addLine(to: CGPoint(x: padding, y: ty))
                    }
                    context.stroke(tickPath, with: .color(.gray.opacity(0.5)), lineWidth: 1)
                }

                // Render series based on chart type
                switch type {
                case .line:
                    for s in series {
                        guard s.x.count == s.y.count, !s.x.isEmpty else { continue }
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
                        context.stroke(linePath, with: .color(s.color), lineWidth: options.lineWidth)
                    }

                case .scatter:
                    for s in series {
                        guard s.x.count == s.y.count, !s.x.isEmpty else { continue }
                        for (xVal, yVal) in zip(s.x, s.y) {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height
                            let r = options.lineWidth * 2.0
                            let circle = Path(ellipseIn: CGRect(x: px - r, y: py - r, width: r * 2, height: r * 2))
                            context.fill(circle, with: .color(s.color))
                        }
                    }

                case .bar:
                    for s in series {
                        guard s.x.count == s.y.count, !s.x.isEmpty else { continue }
                        let barWidth = width / CGFloat(max(1, s.x.count)) * options.barWidthRatio
                        for (xVal, yVal) in zip(s.x, s.y) {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height
                            let rect = CGRect(x: px - barWidth / 2, y: py, width: barWidth, height: (size.height - padding) - py)
                            context.fill(Path(rect), with: .color(s.color))
                        }
                    }

                case .histogram:
                    for s in series {
                        guard !s.y.isEmpty else { continue }
                        let numBins = min(20, max(5, s.y.count / 2))
                        var bins = [Int](repeating: 0, count: numBins)
                        for yVal in s.y {
                            let binIdx = min(numBins - 1, max(0, Int((yVal - minY) / rangeY * Double(numBins))))
                            bins[binIdx] += 1
                        }
                        let maxFreq = Double(bins.max() ?? 1)
                        let binWidth = width / CGFloat(numBins) * options.barWidthRatio
                        for (bIdx, count) in bins.enumerated() {
                            let binX = padding + CGFloat(bIdx) / CGFloat(numBins) * width
                            let barH = CGFloat(Double(count) / maxFreq) * height
                            let barY = size.height - padding - barH
                            let rect = CGRect(x: binX, y: barY, width: binWidth, height: barH)
                            context.fill(Path(rect), with: .color(s.color))
                        }
                    }

                case .heatmap:
                    // 2D grid matrix rendering across all series rows
                    let numRows = CGFloat(series.count)
                    let numCols = CGFloat(series.first?.y.count ?? 1)
                    let cellW = width / max(1, numCols)
                    let cellH = height / max(1, numRows)

                    for (rIdx, s) in series.enumerated() {
                        for (cIdx, val) in s.y.enumerated() {
                            let norm = CGFloat((val - minY) / rangeY)
                            let cellRect = CGRect(
                                x: padding + CGFloat(cIdx) * cellW,
                                y: padding + CGFloat(rIdx) * cellH,
                                width: cellW - 1,
                                height: cellH - 1
                            )
                            context.fill(Path(cellRect), with: .color(s.color.opacity(Double(max(0.1, norm)))))
                        }
                    }
                }
            }
            .frame(minHeight: 220)
        }
        .padding()
    }
}
#endif
