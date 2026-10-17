import Accelerate

func normalizePreprocessingTile<T: BinaryFloatingPoint>(buffer: UnsafeMutableBufferPointer<Double>,
    count: Int, width: Int, output: UnsafeMutableBufferPointer<T>, destinationRow: Int, stride: Int,
    negativeMeans: [Double], deviations: [Double], shifted: inout [Double], normalized: inout [Double]) {
    for r in 0..<count {
        let row = buffer.baseAddress!.advanced(by: r * width)
        vDSP_vaddD(row, 1, negativeMeans, 1, &shifted, 1, vDSP_Length(width))
        vDSP_vdivD(deviations, 1, shifted, 1, &normalized, 1, vDSP_Length(width))
        for c in 0..<width { output[(destinationRow + r) * stride + c] = T(normalized[c]) }
    }
}
