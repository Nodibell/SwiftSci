import Metal

/// Retains the default device for shared-buffer allocations, without retaining buffers.
enum MetalDeviceOwner {
    private static let initialDevice = MTLCreateSystemDefaultDevice()

    static var defaultDevice: (any MTLDevice)? {
        // A failed initial lookup must not make later allocation attempts fail permanently.
        initialDevice ?? MTLCreateSystemDefaultDevice()
    }
}
