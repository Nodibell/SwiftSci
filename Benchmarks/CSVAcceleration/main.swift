import Foundation
import Metal
import Darwin

@_silgen_name("classify_neon")
func classifyNEON(_ bytes: UnsafePointer<UInt8>, _ count: Int, _ output: UnsafeMutableRawPointer)

struct Marks { var comma: UInt32; var lf: UInt32; var quote: UInt32; var cr: UInt32 }
struct Trial: Codable { let bytes: Int; let method: String; let milliseconds: [Double]; let median: Double }
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let source = """
#include <metal_stdlib>
using namespace metal;
struct Marks { uint comma; uint lf; uint quote; uint cr; };
kernel void classify(device const uchar *bytes [[buffer(0)]],
                     device Marks *out [[buffer(1)]],
                     constant uint &count [[buffer(2)]],
                     uint tid [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uchar c = tid < count ? bytes[tid] : 0;
    uint comma = uint((ulong)simd_ballot(c == ','));
    uint lf = uint((ulong)simd_ballot(c == '\\n'));
    uint quote = uint((ulong)simd_ballot(c == '"'));
    uint cr = uint((ulong)simd_ballot(c == '\\r'));
    if (lane == 0) out[tid / 32] = {comma, lf, quote, cr};
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "classify")!)
precondition(pipeline.threadExecutionWidth == 32)
let parser = SystemsCSVParser()

@inline(never)
func classifyCPU(_ bytes: UnsafeBufferPointer<UInt8>, _ marks: UnsafeMutablePointer<Marks>) {
    let blocks = (bytes.count + 31) / 32
    for block in 0..<blocks {
        var comma: UInt32 = 0, lf: UInt32 = 0, quote: UInt32 = 0, cr: UInt32 = 0
        let start = block * 32
        for j in 0..<min(32, bytes.count - start) {
            let c = bytes[start + j], bit = UInt32(1) << j
            if c == 44 { comma |= bit }; if c == 10 { lf |= bit }
            if c == 34 { quote |= bit }; if c == 13 { cr |= bit }
        }
        marks[block] = Marks(comma: comma, lf: lf, quote: quote, cr: cr)
    }
}

@inline(never)
func materialize(_ bytes: UnsafeBufferPointer<UInt8>, _ marks: UnsafePointer<Marks>) -> CSVRecordIndex {
    let blocks = (bytes.count + 31) / 32
    var separators = 0, rows = 0
    for i in 0..<blocks {
        let m = marks[i]
        // Keep the exact quoted CSV semantics in the existing state machine.
        if m.quote != 0 { return parser.parseIndex(buffer: bytes) }
        separators += (m.comma | m.lf).nonzeroBitCount
        rows += m.lf.nonzeroBitCount
    }
    var fields = [CSVFieldOffset](); fields.reserveCapacity(separators + 1)
    var starts = [Int](); starts.reserveCapacity(rows + 2); starts.append(0)
    var fieldStart = 0
    for block in 0..<blocks {
        let m = marks[block]; var bits = m.comma | m.lf
        while bits != 0 {
            let bit = bits.trailingZeroBitCount, pos = block * 32 + bit
            let isLF = m.lf & (UInt32(1) << bit) != 0
            let end = isLF && pos > 0 && bytes[pos - 1] == 13 ? pos - 1 : pos
            fields.append(CSVFieldOffset(startOffset: fieldStart, length: max(0, end - fieldStart), escapedQuotesPresent: false))
            if isLF { starts.append(fields.count) }
            fieldStart = pos + 1
            bits &= bits &- 1
        }
    }
    if fieldStart < bytes.count {
        fields.append(CSVFieldOffset(startOffset: fieldStart, length: bytes.count - fieldStart, escapedQuotesPresent: false))
        starts.append(fields.count)
    } else {
        fields.removeSubrange(starts.last!..<fields.count)
    }
    return CSVRecordIndex(fields: fields, rowStarts: starts)
}

func encode(_ input: MTLBuffer, _ output: MTLBuffer, count: Int) {
    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
    var size = UInt32(count)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.dispatchThreads(MTLSize(width: ((count + 31) / 32) * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    precondition(command.status == .completed, "GPU failed")
}

func equivalent(_ a: CSVRecordIndex, _ b: CSVRecordIndex) -> Bool {
    a.rowStarts == b.rowStarts && a.fields.count == b.fields.count && zip(a.fields, b.fields).allSatisfy {
        $0.startOffset == $1.startOffset && $0.length == $1.length && $0.escapedQuotesPresent == $1.escapedQuotesPresent
    }
}

func gpuIndex(_ data: Data, copy: Bool) -> CSVRecordIndex {
    if data.isEmpty { return CSVRecordIndex(fields: [], rowStarts: [0]) }
    return data.withUnsafeBytes { raw in
        let bytes = raw.bindMemory(to: UInt8.self)
        let input: MTLBuffer
        if copy {
            input = device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)!
        } else {
            precondition(Int(bitPattern: bytes.baseAddress!) % Int(getpagesize()) == 0)
            let aligned = ((bytes.count + Int(getpagesize()) - 1) / Int(getpagesize())) * Int(getpagesize())
            input = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: bytes.baseAddress!), length: aligned, options: .storageModeShared, deallocator: nil)!
        }
        let out = device.makeBuffer(length: ((bytes.count + 31) / 32) * MemoryLayout<Marks>.stride, options: .storageModeShared)!
        encode(input, out, count: bytes.count)
        return materialize(bytes, out.contents().assumingMemoryBound(to: Marks.self))
    }
}

func cpuIndex(_ data: Data, neon: Bool = false, parallel: Bool = false) -> CSVRecordIndex {
    if data.isEmpty { return CSVRecordIndex(fields: [], rowStarts: [0]) }
    return data.withUnsafeBytes { raw in
        let bytes = raw.bindMemory(to: UInt8.self), n = (raw.count + 31) / 32
        let marks = UnsafeMutablePointer<Marks>.allocate(capacity: n)
        defer { marks.deallocate() }
        if neon {
            if parallel {
                let chunk = 1_048_576, parts = (bytes.count + chunk - 1) / chunk
                DispatchQueue.concurrentPerform(iterations: parts) { part in
                    let start = part * chunk, size = min(chunk, bytes.count - start)
                    classifyNEON(bytes.baseAddress!.advanced(by: start), size, marks.advanced(by: start / 32))
                }
            } else { classifyNEON(bytes.baseAddress!, bytes.count, marks) }
        } else { classifyCPU(bytes, marks) }
        return materialize(bytes, marks)
    }
}

var sink = 0
func consume(_ index: CSVRecordIndex) { sink &+= index.fields.count &+ index.rowStarts.count &+ (index.fields.last?.startOffset ?? 0) }
func timed(_ body: () -> Void) -> Double { let start = DispatchTime.now().uptimeNanoseconds; body(); return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

if CommandLine.arguments.contains("--check") {
    let cases = ["", "a,b\n1,2\n", "a,b\n\"hello\nthere\",3\n", "a,b\r\n\"hi\"\"!\",2\r\n", "a,b\n,\n", "a,b\n1,", "a,b\n1,2", "a,b\n\"unfinished,2"]
    for value in cases {
        let data = Data(value.utf8)
        let expected = data.withUnsafeBytes { parser.parseIndex(buffer: $0.bindMemory(to: UInt8.self)) }
        precondition(equivalent(expected, cpuIndex(data)))
        precondition(equivalent(expected, cpuIndex(data, neon: true)))
        precondition(equivalent(expected, gpuIndex(data, copy: true)))
    }
    var state: UInt64 = 1234
    for _ in 0..<100 {
        let chars: [UInt8] = [44,10,13,34,97,98,48,49]
        var bytes = [UInt8]()
        for _ in 0..<255 { state = state &* 6364136223846793005 &+ 1; bytes.append(chars[Int((state >> 32) % 8)]) }
        let data = Data(bytes), expected = data.withUnsafeBytes { parser.parseIndex(buffer: $0.bindMemory(to: UInt8.self)) }
        precondition(equivalent(expected, gpuIndex(data, copy: true)))
        precondition(equivalent(expected, cpuIndex(data, neon: true)))
        precondition(equivalent(expected, cpuIndex(data, neon: true, parallel: true)))
    }
    do {
        _ = try device.makeLibrary(source: "#include <metal_stdlib>\nusing namespace metal; kernel void tryDouble(device double *p [[buffer(0)]], uint i [[thread_position_in_grid]]) { p[i] = p[i] * 1.5; }", options: nil)
        print("double shader compiled")
    } catch { print("double shader unsupported: \(error.localizedDescription)") }
    print("PASS 108 parser equivalence cases on \(device.name)")
} else {
    guard CommandLine.arguments.count > 1 else {
        fputs("Usage: csv-gpu --check | path/to/input.csv [--verify-only]\n", stderr); exit(2)
    }
    let path = CommandLine.arguments[1]
    let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
    let blocks = (data.count + 31) / 32
    let reusableOutput = device.makeBuffer(length: blocks * MemoryLayout<Marks>.stride, options: .storageModeShared)!
    let reusableInput = data.withUnsafeBytes { raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)! }
    let cpuMarks = UnsafeMutablePointer<Marks>.allocate(capacity: blocks)
    defer { cpuMarks.deallocate() }
    let methods: [(String, () -> Void)] = [
        ("cpu_neon_classify_only", { data.withUnsafeBytes { raw in classifyNEON(raw.bindMemory(to: UInt8.self).baseAddress!, raw.count, cpuMarks); sink &+= Int(cpuMarks[blocks-1].lf) } }),
        ("gpu_resident_classify_only", { encode(reusableInput, reusableOutput, count: data.count); sink &+= Int(reusableOutput.contents().assumingMemoryBound(to: Marks.self)[blocks-1].lf) }),

        ("cpu_original_index", { consume(data.withUnsafeBytes { parser.parseIndex(buffer: $0.bindMemory(to: UInt8.self)) }) }),
        ("cpu_bitmask_index", { consume(cpuIndex(data)) }),
        ("cpu_neon_index", { consume(cpuIndex(data, neon: true)) }),
        ("cpu_parallel_neon_index", { consume(cpuIndex(data, neon: true, parallel: true)) }),
        ("gpu_copy_index", { consume(gpuIndex(data, copy: true)) }),
        ("gpu_no_copy_index", { consume(gpuIndex(data, copy: false)) })
    ]
    let expected = data.withUnsafeBytes { parser.parseIndex(buffer: $0.bindMemory(to: UInt8.self)) }
    precondition(equivalent(expected, gpuIndex(data, copy: true)))
    precondition(equivalent(expected, gpuIndex(data, copy: false)))
    precondition(equivalent(expected, cpuIndex(data)))
    precondition(equivalent(expected, cpuIndex(data, neon: true)))
    precondition(equivalent(expected, cpuIndex(data, neon: true, parallel: true)))
    if CommandLine.arguments.contains("--verify-only") { print("PASS all field-index methods agree on \(data.count) bytes"); exit(0) }
    for (_, work) in methods { work(); work() }
    var results = methods.map { _ in [Double]() }
    for iteration in 0..<9 {
        for offset in methods.indices {
            let i = (iteration + offset) % methods.count
            results[i].append(timed(methods[i].1))
        }
    }
    let trials = methods.indices.map { Trial(bytes: data.count, method: methods[$0].0, milliseconds: results[$0], median: median(results[$0])) }
    print(String(data: try JSONEncoder().encode(trials), encoding: .utf8)!)
    fputs("device=\(device.name) sink=\(sink)\n", stderr)
}
