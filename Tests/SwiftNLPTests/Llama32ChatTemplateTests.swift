import CryptoKit
import Foundation
import Testing
@testable import SwiftNLP

@Suite("Llama 3.2 Instruct chat formatting")
struct Llama32ChatTemplateTests {
    struct Fixture: Sendable {
        let name: String
        let messages: [ChatMessage]
        let generation: Bool
        let text: String
        let ids: [Int]
    }

    // Generated from transformers 5.17.0 and the pinned tokenizer config:
    // revision 863c846a9ac6fad4e49e1743d52984dff262e953,
    // SHA256 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e.
    // apply_chat_template(..., date_string="30 Sep 2026", tools=None).
    static let fixtures: [Fixture] = [
        Fixture(name: "user-only", messages: [.user("Hello world!")], generation: true, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello world!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 882, 128007, 271, 9906, 1917, 0, 128009, 128006, 78191, 128007, 271]),
        Fixture(name: "trim-system", messages: [.system(" \tBe concise.\n "), .user(" \n Why?\t "), .assistant("  Because.  ")], generation: false, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\nBe concise.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBecause.<|eot_id|>", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 3513, 64694, 13, 128009, 128006, 882, 128007, 271, 10445, 30, 128009, 128006, 78191, 128007, 271, 18433, 13, 128009]),
        Fixture(name: "tool-string", messages: [.user("Read a sensor."), .tool(" {\"value\": 2.5, \"unit\": \"°C\"}\n ")], generation: true, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRead a sensor.<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n\" {\\\"value\\\": 2.5, \\\"unit\\\": \\\"°C\\\"}\\n \"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 882, 128007, 271, 4518, 264, 12271, 13, 128009, 128006, 23799, 4690, 128007, 271, 1, 314, 2153, 970, 11955, 220, 17, 13, 20, 11, 7393, 3928, 11955, 7393, 11877, 34, 2153, 11281, 77, 330, 128009, 128006, 78191, 128007, 271]),
        Fixture(name: "unicode-trim", messages: [.system("\u{1c}\u{200b}retain\u{200b}　"), .user("\u{2028}世界 é\u{2029}")], generation: false, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n\u{200b}retain\u{200b}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n世界 é<|eot_id|>", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 16067, 56472, 16067, 128009, 128006, 882, 128007, 271, 102616, 384, 54939, 128009]),
        Fixture(name: "late-system", messages: [.user("first"), .system(" later ")], generation: false, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nfirst<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nlater<|eot_id|>", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 882, 128007, 271, 3983, 128009, 128006, 9125, 128007, 271, 68676, 128009]),
        Fixture(name: "system-only", messages: [.system("")], generation: true, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 78191, 128007, 271]),
        Fixture(name: "escaped-tool", messages: [.tool("\u{0}\u{8}\t\n\u{c}\r\u{1f}\\/\"<>&' café 👩‍🔬\u{2028}\u{2029}")], generation: false, text: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n\"\\u0000\\b\\t\\n\\f\\r\\u001f\\\\/\\\"<>&' café 👩‍🔬\u{2028}\u{2029}\"<|eot_id|>", ids: [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 966, 17907, 220, 2366, 21, 271, 128009, 128006, 23799, 4690, 128007, 271, 12200, 84, 931, 15, 44556, 5061, 1734, 65626, 12285, 3855, 4119, 69, 95653, 2153, 27, 5909, 6, 53050, 62904, 102, 102470, 9468, 242, 105, 378, 101, 378, 102, 1, 128009]),
    ]

    @Test("Rendered bytes match the pinned checkpoint template", arguments: fixtures)
    func referencePrompt(_ fixture: Fixture) {
        let template = ChatTemplate(style: .llama32Instruct(date: "30 Sep 2026"))
        let actual = template.render(messages: fixture.messages, addGenerationPrompt: fixture.generation)
        #expect(Array(actual.utf8) == Array(fixture.text.utf8), "\(fixture.name)")
    }

    @Test("The date is supplied by the caller and existing formatting stays unchanged")
    func explicitDateAndLegacy() {
        let template = ChatTemplate.llama32Instruct(date: "01 Jan 2025")
        #expect(template == ChatTemplate(style: .llama32Instruct(date: "01 Jan 2025")))
        #expect(template.render(messages: [.user("hello")]).contains("Today Date: 01 Jan 2025\n\n"))
        #expect(ChatTemplate.llama3.render(messages: [.user(" hello ")], addGenerationPrompt: false)
            == "<|start_header_id|>user<|end_header_id|>\n\n hello <|eot_id|>")
    }

    @Test("An empty conversation renders the system header without a user turn")
    func emptyConversation() {
        let template = ChatTemplate(style: .llama32Instruct(date: "30 Sep 2026"))
        #expect(template.render(messages: [], addGenerationPrompt: false)
            == "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Sep 2026\n\n<|eot_id|>")
    }

    @Test("Checkpoint chat IDs match without a duplicate BOS",
          .enabled(if: ProcessInfo.processInfo.environment["SWIFTSCI_LLAMA_TOKENIZER_JSON"] != nil),
          arguments: fixtures)
    func referenceIDs(_ fixture: Fixture) throws {
        let path = try #require(ProcessInfo.processInfo.environment["SWIFTSCI_LLAMA_TOKENIZER_JSON"])
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try #require(hash == "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4")
        let tokenizer = try BPETokenizer(llama3TokenizerJSON: data)
        let template = ChatTemplate(style: .llama32Instruct(date: "30 Sep 2026"))
        #expect(template.encode(messages: fixture.messages, using: tokenizer,
                                addGenerationPrompt: fixture.generation) == fixture.ids)
    }
}
