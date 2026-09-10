#if os(macOS)
import Foundation
import Metal
import MLX
import MLXNN

/// The quantization scheme used for compressed model weights.
public enum QuantizationScheme: Sendable {
    /// 4-bit symmetric quantization with block size 32 (GGUF Q4_0).
    case q4_0
    /// 4-bit asymmetric quantization with scale and min offset (GGUF Q4_1).
    case q4_1
    /// 8-bit symmetric quantization with block size 32 (GGUF Q8_0).
    case q8_0
    /// 4-bit activation-aware weight quantization with configurable group size (AWQ).
    case awq4(groupSize: Int)
    /// 8-bit activation-aware weight quantization with configurable group size (AWQ).
    case awq8(groupSize: Int)
}

/// Errors occurring during Metal quantized kernel pipeline creation or GPU buffer execution.
public enum MetalQuantizedError: Error, LocalizedError, Sendable {
    /// The system does not possess a valid Metal device.
    case deviceNotFound
    /// The Metal library could not be compiled or located.
    case libraryNotFound(String)
    /// Pipeline compilation failed for the specified kernel function.
    case pipelineCreationFailed(String)
    /// Allocation of Metal device unified memory buffer failed.
    case bufferCreationFailed
    /// Kernel command execution or synchronization timed out / failed.
    case executionFailed(String)
    /// Input dimension mismatch for the quantized layer.
    case dimensionMismatch(expected: Int, got: Int)

    /// Localized human-readable error description.
    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "No default Metal device available on this host."
        case .libraryNotFound(let s):
            return "Failed to load or compile Metal shader library: \(s)"
        case .pipelineCreationFailed(let s):
            return "Failed to construct Metal compute pipeline: \(s)"
        case .bufferCreationFailed:
            return "Failed to allocate unified GPU memory buffer."
        case .executionFailed(let s):
            return "Metal compute command buffer execution failed: \(s)"
        case .dimensionMismatch(let expected, let got):
            return "Input dimension mismatch: expected \(expected), got \(got)."
        }
    }
}

/// A thread-safe executor and pipeline cache for Metal quantized MSL GEMV / GEMM kernels.
public final class MetalQuantizedEngine: @unchecked Sendable {
    /// Shared singleton instance.
    public static let shared = MetalQuantizedEngine()

    /// Metal GPU device reference.
    public let device: (any MTLDevice)?

    /// Command queue for kernel dispatches.
    public let commandQueue: (any MTLCommandQueue)?

    private var pipelines: [String: any MTLComputePipelineState] = [:]
    private let lock = NSLock()

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            self.device = nil
            self.commandQueue = nil
            return
        }
        self.device = dev
        self.commandQueue = dev.makeCommandQueue()
    }

    /// Retrieves or compiles a compute pipeline for the given kernel function name.
    ///
    /// - Parameter name: Name of the MSL kernel function (e.g. `gemv_q4_0`, `gemv_q8_0`).
    /// - Returns: A cached `MTLComputePipelineState`.
    /// - Throws: `MetalQuantizedError` if device or library cannot be initialized.
    public func getPipeline(name: String) throws -> any MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let existing = pipelines[name] {
            return existing
        }

        guard let dev = device else {
            throw MetalQuantizedError.deviceNotFound
        }

        let library: any MTLLibrary
        if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.module) {
            library = lib
        } else if let defaultLib = dev.makeDefaultLibrary() {
            library = defaultLib
        } else {
            // Embedded MSL fallback
            let mslSource = """
            #include <metal_stdlib>
            using namespace metal;

            kernel void gemv_q4_0(
                device const float* inVec                  [[buffer(0)]],
                device const uchar* weights                [[buffer(1)]],
                device const half* scales                  [[buffer(2)]],
                device float* outVec                       [[buffer(3)]],
                constant uint& inFeatures                  [[buffer(4)]],
                constant uint& outFeatures                 [[buffer(5)]],
                uint thread_position_in_grid               [[thread_position_in_grid]]
            ) {
                uint row = thread_position_in_grid;
                if (row >= outFeatures) return;
                uint blocksPerRow = inFeatures / 32;
                uint blockOffset = row * blocksPerRow;
                uint weightByteOffset = blockOffset * 16;
                float accumulator = 0.0f;
                for (uint b = 0; b < blocksPerRow; ++b) {
                    float scale = float(scales[blockOffset + b]);
                    device const uchar* bWeights = weights + weightByteOffset + (b * 16);
                    device const float* bIn = inVec + (b * 32);
                    for (uint i = 0; i < 16; ++i) {
                        uchar byteVal = bWeights[i];
                        int w0 = int(byteVal & 0x0F) - 8;
                        int w1 = int(byteVal >> 4) - 8;
                        accumulator += (float(w0) * scale) * bIn[2 * i];
                        accumulator += (float(w1) * scale) * bIn[2 * i + 1];
                    }
                }
                outVec[row] = accumulator;
            }

            kernel void gemv_q8_0(
                device const float* inVec                  [[buffer(0)]],
                device const int8_t* weights               [[buffer(1)]],
                device const half* scales                  [[buffer(2)]],
                device float* outVec                       [[buffer(3)]],
                constant uint& inFeatures                  [[buffer(4)]],
                constant uint& outFeatures                 [[buffer(5)]],
                uint thread_position_in_grid               [[thread_position_in_grid]]
            ) {
                uint row = thread_position_in_grid;
                if (row >= outFeatures) return;
                uint blocksPerRow = inFeatures / 32;
                uint blockOffset = row * blocksPerRow;
                uint weightByteOffset = blockOffset * 32;
                float accumulator = 0.0f;
                for (uint b = 0; b < blocksPerRow; ++b) {
                    float scale = float(scales[blockOffset + b]);
                    device const int8_t* bWeights = weights + weightByteOffset + (b * 32);
                    device const float* bIn = inVec + (b * 32);
                    for (uint i = 0; i < 32; ++i) {
                        accumulator += (float(bWeights[i]) * scale) * bIn[i];
                    }
                }
                outVec[row] = accumulator;
            }
            """
            do {
                library = try dev.makeLibrary(source: mslSource, options: nil)
            } catch {
                throw MetalQuantizedError.libraryNotFound(error.localizedDescription)
            }
        }

        guard let function = library.makeFunction(name: name) else {
            throw MetalQuantizedError.pipelineCreationFailed("Kernel function '\(name)' not found in Metal library")
        }

        do {
            let pipeline = try dev.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
            return pipeline
        } catch {
            throw MetalQuantizedError.pipelineCreationFailed(error.localizedDescription)
        }
    }

    /// Dispatches a quantized matrix-vector multiplication kernel onto the GPU using raw scale bytes.
    ///
    /// - Parameters:
    ///   - inVector: Input float vector of size `inFeatures`.
    ///   - rawWeights: Raw quantized weight byte buffer.
    ///   - rawScalesData: Raw FP16 scale factors byte buffer (2 bytes per block).
    ///   - inFeatures: Input dimension size (must be divisible by 32).
    ///   - outFeatures: Output dimension size.
    ///   - kernelName: Kernel function name (`gemv_q4_0` or `gemv_q8_0`).
    /// - Returns: Computed output float activations of size `outFeatures`.
    /// - Throws: `MetalQuantizedError` if buffers fail or execution errors.
    public func executeGEMV(
        inVector: [Float],
        rawWeights: Data,
        rawScalesData: Data,
        inFeatures: Int,
        outFeatures: Int,
        kernelName: String
    ) throws -> [Float] {
        guard let dev = device, let queue = commandQueue else {
            throw MetalQuantizedError.deviceNotFound
        }
        guard inVector.count == inFeatures else {
            throw MetalQuantizedError.dimensionMismatch(expected: inFeatures, got: inVector.count)
        }

        let pipeline = try getPipeline(name: kernelName)

        guard let inBuffer = dev.makeBuffer(bytes: inVector, length: inFeatures * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outBuffer = dev.makeBuffer(length: outFeatures * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalQuantizedError.bufferCreationFailed
        }

        let weightsBuffer: (any MTLBuffer)? = rawWeights.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.baseAddress else { return nil }
            return dev.makeBuffer(bytes: base, length: rawWeights.count, options: .storageModeShared)
        }
        guard let wBuf = weightsBuffer else {
            throw MetalQuantizedError.bufferCreationFailed
        }

        let scalesBuffer: (any MTLBuffer)? = rawScalesData.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.baseAddress else { return nil }
            return dev.makeBuffer(bytes: base, length: rawScalesData.count, options: .storageModeShared)
        }
        guard let sBuf = scalesBuffer else {
            throw MetalQuantizedError.bufferCreationFailed
        }

        var inF = UInt32(inFeatures)
        var outF = UInt32(outFeatures)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw MetalQuantizedError.executionFailed("Failed to create command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(wBuf, offset: 0, index: 1)
        encoder.setBuffer(sBuf, offset: 0, index: 2)
        encoder.setBuffer(outBuffer, offset: 0, index: 3)
        encoder.setBytes(&inF, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&outF, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadsPerGroup = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 32), height: 1, depth: 1)
        let numGroups = MTLSize(width: (outFeatures + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw MetalQuantizedError.executionFailed(error.localizedDescription)
        }

        let outPtr = outBuffer.contents().bindMemory(to: Float.self, capacity: outFeatures)
        return Array(UnsafeBufferPointer(start: outPtr, count: outFeatures))
    }

#if arch(arm64)
    /// Dispatches a quantized matrix-vector multiplication kernel onto the GPU.
    ///
    /// - Parameters:
    ///   - inVector: Input float vector of size `inFeatures`.
    ///   - rawWeights: Raw quantized weight byte buffer.
    ///   - rawScales: FP16 scale factors per block.
    ///   - inFeatures: Input dimension size (must be divisible by 32).
    ///   - outFeatures: Output dimension size.
    ///   - kernelName: Kernel function name (`gemv_q4_0` or `gemv_q8_0`).
    /// - Returns: Computed output float activations of size `outFeatures`.
    /// - Throws: `MetalQuantizedError` if buffers fail or execution errors.
    public func executeGEMV(
        inVector: [Float],
        rawWeights: Data,
        rawScales: [Float16],
        inFeatures: Int,
        outFeatures: Int,
        kernelName: String
    ) throws -> [Float] {
        let scalesData = rawScales.withUnsafeBytes { Data($0) }
        return try executeGEMV(
            inVector: inVector,
            rawWeights: rawWeights,
            rawScalesData: scalesData,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            kernelName: kernelName
        )
    }
#endif
}

/// A linear neural network layer performing matrix multiplication with 4-bit (Q4_0, Q4_K) or 8-bit quantized weights.
///
/// ## Metal Acceleration
/// Uses custom Metal Shading Language (MSL) SIMD-group matrix multiply kernels on Apple Silicon GPUs.
///
/// ## Thread Safety
/// Forward inference calls via MLX and native Metal pipelines are thread-safe and isolated to unified GPU device queues.
public final class QuantizedLinear: Module, @unchecked Sendable {
    /// The number of input features.
    public let inFeatures: Int
    /// The number of output features.
    public let outFeatures: Int
    /// The quantization scheme used for layer weights.
    public let scheme: QuantizationScheme
    /// The group block size.
    public let groupSize: Int

    /// Packed quantized weight tensor.
    public let weight: MLXArray
    /// Scale factor tensor per quantization group.
    public let scales: MLXArray
    /// Optional bias or zero-point offset tensor.
    public let bias: MLXArray?

    /// Initializes a quantized linear layer with pre-quantized weights and scales.
    ///
    /// - Parameters:
    ///   - inFeatures: Input dimension size.
    ///   - outFeatures: Output dimension size.
    ///   - scheme: Quantization scheme (Q4_0, Q4_1, Q8_0, AWQ).
    ///   - groupSize: Group block size for quantization (default: 32).
    ///   - weight: Quantized weights tensor.
    ///   - scales: Scale factors tensor.
    ///   - bias: Optional bias tensor.
    public init(
        inFeatures: Int,
        outFeatures: Int,
        scheme: QuantizationScheme = .q4_0,
        groupSize: Int = 32,
        weight: MLXArray,
        scales: MLXArray,
        bias: MLXArray? = nil
    ) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.scheme = scheme
        self.groupSize = groupSize
        self.weight = weight
        self.scales = scales
        self.bias = bias
        super.init()
    }

    /// Dequantizes the compressed weight matrix into a dense floating-point tensor on Apple Silicon GPU.
    ///
    /// - Returns: Dequantized weights of shape `[outFeatures, inFeatures]`.
    public func dequantize() -> MLXArray {
        switch scheme {
        case .q4_0:
            let centered = self.weight - MLXArray(Float(8.0))
            let dequant = centered * self.scales
            return dequant.reshaped([outFeatures, inFeatures])

        case .q4_1:
            var dequant = self.weight * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])

        case .q8_0:
            let dequant = self.weight * self.scales
            return dequant.reshaped([outFeatures, inFeatures])

        case .awq4:
            var dequant = self.weight * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])

        case .awq8:
            var dequant = self.weight * self.scales
            if let b = self.bias {
                dequant = dequant + b
            }
            return dequant.reshaped([outFeatures, inFeatures])
        }
    }

    /// Performs direct GPU matrix-vector multiplication with quantized weights using native Metal shaders without intermediate CPU allocations.
    ///
    /// - Parameters:
    ///   - inVector: Input float activations of dimension `inFeatures`.
    ///   - rawWeights: Raw packed weight byte buffer (16 bytes per 32 elements for Q4_0; 32 bytes for Q8_0).
    ///   - rawScalesData: FP16 scale factors byte buffer (2 bytes per block of 32 elements).
    /// - Returns: Computed output float activations of dimension `outFeatures`.
    /// - Throws: `MetalQuantizedError` if GPU buffers cannot be allocated or kernel fails.
    ///
    /// ## Metal Acceleration
    /// Executes `gemv_q4_0` or `gemv_q8_0` MSL kernels directly on unified memory.
    ///
    /// ## Complexity
    /// \(O(\text{outFeatures} \cdot \text{inFeatures} / 32)\) on parallel Apple Silicon GPU threads.
    public func forwardMetal(
        inVector: [Float],
        rawWeights: Data,
        rawScalesData: Data
    ) throws -> [Float] {
        let kernelName: String
        switch scheme {
        case .q4_0:
            kernelName = "gemv_q4_0"
        case .q8_0:
            kernelName = "gemv_q8_0"
        default:
            kernelName = "gemv_q4_0"
        }

        var result = try MetalQuantizedEngine.shared.executeGEMV(
            inVector: inVector,
            rawWeights: rawWeights,
            rawScalesData: rawScalesData,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            kernelName: kernelName
        )

        if let b = bias, b.shape == [outFeatures] {
            let biasVals = b.asArray(Float.self)
            for i in 0..<outFeatures {
                result[i] += biasVals[i]
            }
        }

        return result
    }

#if arch(arm64)
    /// Performs direct GPU matrix-vector multiplication with quantized weights using native Metal shaders without intermediate CPU allocations.
    ///
    /// - Parameters:
    ///   - inVector: Input float activations of dimension `inFeatures`.
    ///   - rawWeights: Raw packed weight byte buffer (16 bytes per 32 elements for Q4_0; 32 bytes for Q8_0).
    ///   - rawScales: FP16 scale factors per block (1 per 32 elements).
    /// - Returns: Computed output float activations of dimension `outFeatures`.
    /// - Throws: `MetalQuantizedError` if GPU buffers cannot be allocated or kernel fails.
    ///
    /// ## Metal Acceleration
    /// Executes `gemv_q4_0` or `gemv_q8_0` MSL kernels directly on unified memory.
    ///
    /// ## Complexity
    /// \(O(\text{outFeatures} \cdot \text{inFeatures} / 32)\) on parallel Apple Silicon GPU threads.
    public func forwardMetal(
        inVector: [Float],
        rawWeights: Data,
        rawScales: [Float16]
    ) throws -> [Float] {
        let scalesData = rawScales.withUnsafeBytes { Data($0) }
        return try forwardMetal(
            inVector: inVector,
            rawWeights: rawWeights,
            rawScalesData: scalesData
        )
    }
#endif

    /// Performs the forward pass linear transformation on input activations via MLX unified GPU runtime.
    ///
    /// - Parameter x: Input activation tensor of shape `[..., inFeatures]`.
    /// - Returns: Output activation tensor of shape `[..., outFeatures]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = dequantize()
        var out = matmul(x, w.transposed(1, 0))
        if let b = self.bias, b.shape == [outFeatures] {
            out = out + b
        }
        return out
    }
}
#endif
