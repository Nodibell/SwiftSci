import Testing
import Foundation
@testable import SwiftNLP

@Suite("ChatTemplate 3.14.0 Tests")
struct ChatTemplate3140Tests {

    @Test("defaultStopSequences returns canonical turn-end tokens across all template families")
    func testDefaultStopSequencesAcrossFamilies() {
        #expect(ChatTemplate.llama3.defaultStopSequences == ["<|eot_id|>"])
        #expect(ChatTemplate.llama32Instruct(date: "26 Nov 2026").defaultStopSequences == ["<|eot_id|>"])
        #expect(ChatTemplate.chatML.defaultStopSequences == ["<|im_end|>"])
        #expect(ChatTemplate.mistral.defaultStopSequences == ["</s>"])
        #expect(ChatTemplate.gemma.defaultStopSequences == ["<end_of_turn>"])
        #expect(ChatTemplate.gemma4.defaultStopSequences == ["<turn|>"])
    }

    @Test("Mutating style on custom-formatted ChatTemplate updates defaultStopSequences")
    func testStyleMutationUpdatesStopSequences() {
        var template = ChatTemplate.gemma4
        #expect(template.defaultStopSequences == ["<turn|>"])
        template.style = .llama3
        #expect(template.defaultStopSequences == ["<|eot_id|>"])
        template.style = .chatML
        #expect(template.defaultStopSequences == ["<|im_end|>"])
        template.style = .mistral
        #expect(template.defaultStopSequences == ["</s>"])
    }
}
