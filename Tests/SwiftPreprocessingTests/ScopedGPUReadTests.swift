import Testing
import Darwin
import Metal
import MLX
@testable import SwiftPreprocessing

@Suite("Scoped production GPU reads")
struct ScopedGPUReadTests {
    @Test func sharedCopyAndThrowingCompletion() throws {
        enum Failure: Error { case afterSubmission }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let page = Int(getpagesize()) / MemoryLayout<Float>.stride
        for count in [1,page,page+1] {
            for offset in [0,1,page] {
                for fail in [false,true] {
                    let buffer = try #require(device.makeBuffer(length:(offset+count)*4,options:.storageModeShared))
                    let pointer = buffer.contents().bindMemory(to:Float.self,capacity:offset+count)
                    for i in 0..<offset+count { pointer.advanced(by:i).initialize(to:Float(i)) }
                    do {
                        try ScopedGPURead.withBuffer(buffer,range:offset..<offset+count,shape:[count]) { input,shared in
                            #expect(shared == (offset % page == 0 && count % page == 0))
                            let output = multiply(input,Float(2),stream:.gpu)
                            asyncEval(output)
                            if fail { throw Failure.afterSubmission }
                            #expect(output.asArray(Float.self) == (offset..<offset+count).map { Float($0)*2 })
                        }
                        #expect(!fail)
                    } catch Failure.afterSubmission { #expect(fail) }
                    #expect((0..<offset+count).allSatisfy { pointer[$0] == Float($0) })
                }
            }
        }
        let buffer = try #require(device.makeBuffer(length:page*4,options:.storageModeShared))
        #expect(throws:(any Error).self) { try ScopedGPURead.withBuffer(buffer,range:0..<1,shape:[Int(Int32.max)+1]) { _,_ in } }
        #expect(throws:(any Error).self) { try ScopedGPURead.withMatrix(rows:Int(Int32.max)+1,columns:1,value:{ _,_ in 0 }) { _ in } }
        #expect(throws:(any Error).self) { try ScopedGPURead.withBuffer(buffer,range:0..<0,shape:[0]) { _,_ in } }
        #expect(throws:(any Error).self) { try ScopedGPURead.withBuffer(buffer,range:0..<page+1,shape:[page+1]) { _,_ in } }
    }
    @Test func constructedMatricesCompleteBeforeReturningOrThrowing() throws {
        enum Failure: Error { case afterSubmission }
        let page = Int(getpagesize()) / MemoryLayout<Float>.stride
        try Stream.withNewDefaultStream(device: .gpu) {
            for count in [1, page, page + 1] {
                for fail in [false, true] {
                    do {
                        let actual = try ScopedGPURead.withMatrix(rows: count, columns: 1, value: { r,_ in Double(r) }) { input in
                            let output = multiply(input, Float(3), stream: .gpu)
                            asyncEval(output)
                            if fail { throw Failure.afterSubmission }
                            return output.asArray(Float.self)
                        }
                        #expect(!fail)
                        #expect(actual == (0..<count).map { Float($0) * 3 })
                    } catch Failure.afterSubmission { #expect(fail) }
                }
            }
        }
    }

}
