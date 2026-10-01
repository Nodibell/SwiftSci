import CryptoKit
import Foundation
import Testing
@testable import SwiftNLP

@Suite("Pinned Llama tokenizer parity", .enabled(if: ProcessInfo.processInfo.environment["SWIFTSCI_LLAMA_TOKENIZER_JSON"] != nil))
struct LlamaTokenizerCheckpointTests {
    @Test("Checkpoint IDs, BOS processing and byte-exact decoding match the pinned reference")
    func checkpointParity() throws {
        let path = try #require(ProcessInfo.processInfo.environment["SWIFTSCI_LLAMA_TOKENIZER_JSON"])
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try #require(hash == "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4")
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: data)
        // Reference: tokenizers 0.23.2, Llama-3.2-1B-Instruct-bf16 revision
        // 863c846a9ac6fad4e49e1743d52984dff262e953, add_special_tokens=False.
        // Chat strings were rendered with transformers 5.17.0 and a fixed date.
        let cases: [(String, [Int])] = [
            ("Hello world!", [9906, 1917, 0]),
            ("hello", [15339]),
            (" hi\n", [15960, 198]),
            ("<|begin_of_text|>hello<|eot_id|>", [128000, 15339, 128009]),
            ("1234567890", [4513, 10961, 16474, 15]),
            ("I'm WE'RE can't", [40, 2846, 20255, 95253, 649, 956]),
            ("café é 世界 👩‍🔬 🇺🇦", [936, 59958, 384, 54939, 127365, 62904, 102, 102470, 9468, 242, 105, 11410, 229, 118, 9468, 229, 99]),
            ("\r\n  \t ", [319, 79199]),
            ("</w>", [524, 86, 29]),
            ("\u{0}\u{1}\u{1f}", [188, 189, 219]),
            ("Ａ１½Ⅻ⁴  ", [116531, 20713, 27154, 71567, 104, 53233, 112, 4194, 378, 101]),
            ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello world!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 882, 128007, 271, 9906, 1917, 0, 128009, 128006, 78191, 128007, 271]),
            ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n4.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 3513, 64694, 13, 128009, 128006, 882, 128007, 271, 3923, 374, 220, 17, 489, 220, 17, 30, 128009, 128006, 78191, 128007, 271, 19, 13, 128009, 128006, 882, 128007, 271, 10445, 30, 128009, 128006, 78191, 128007, 271]),
        ]
        for (text, ids) in cases {
            #expect(tokenizer.encode(text: text) == ids)
            #expect(tokenizer.encode(text: text, addSpecialTokens: true) == [128000] + ids)
            #expect(Array(tokenizer.decode(tokens: ids).utf8) == Array(text.utf8))
        }
    }
}
