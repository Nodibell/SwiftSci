import Accelerate

func normalizePreprocessingTile<T: BinaryFloatingPoint>(buffer: UnsafeMutableBufferPointer<Double>,
    count: Int, width: Int, output: UnsafeMutableBufferPointer<T>, destinationRow: Int, stride: Int,
    negativeMeans: [Double], deviations: [Double], shifted: inout [Double], normalized: inout [Double]) {
    negativeMeans.withUnsafeBufferPointer { means in
        deviations.withUnsafeBufferPointer { deviations in
            shifted.withUnsafeMutableBufferPointer { shifted in
                normalized.withUnsafeMutableBufferPointer { normalized in
                    let length = vDSP_Length(width)
                    for r in 0..<count {
                        let row = buffer.baseAddress!.advanced(by: r * width)
                        vDSP_vaddD(row, 1, means.baseAddress!, 1, shifted.baseAddress!, 1, length)
                        vDSP_vdivD(deviations.baseAddress!, 1, shifted.baseAddress!, 1,
                            normalized.baseAddress!, 1, length)
                        let destination = output.baseAddress!.advanced(by: (destinationRow + r) * stride)
                        for c in 0..<width { destination[c] = T(normalized[c]) }
                    }
                }
            }
        }
    }
}
