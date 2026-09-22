import Testing
import Foundation
@testable import SwiftNLP

@Suite("ChatTemplate Tests")
struct ChatTemplateTests {

    @Test("Gate 9: Llama 3 ChatTemplate rendering")
    func testLlama3Rendering() {
        let template = ChatTemplate(style: .llama3)
        let messages = [
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Hello world!")
        ]

        let prompt = template.render(messages: messages, addGenerationPrompt: true)

        #expect(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"))
        #expect(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHello world!<|eot_id|>"))
        #expect(prompt.hasSuffix("<|start_header_id|>assistant<|end_header_id|>\n\n"))
    }

    @Test("Gate 9: ChatML ChatTemplate rendering")
    func testChatMLRendering() {
        let template = ChatTemplate(style: .chatML)
        let messages = [
            ChatMessage.system("System prompt."),
            ChatMessage.user("User question.")
        ]

        let prompt = template.render(messages: messages, addGenerationPrompt: true)

        #expect(prompt.contains("<|im_start|>system\nSystem prompt.<|im_end|>\n"))
        #expect(prompt.contains("<|im_start|>user\nUser question.<|im_end|>\n"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test("Gate 9: Mistral ChatTemplate rendering with system prompt")
    func testMistralRendering() {
        let template = ChatTemplate(style: .mistral)
        let messages = [
            ChatMessage.system("Be concise."),
            ChatMessage.user("What is Swift?")
        ]

        let prompt = template.render(messages: messages, addGenerationPrompt: false)
        #expect(prompt.contains("[INST] <<SYS>>\nBe concise.\n<</SYS>>\n\nWhat is Swift? [/INST]"))
    }

    @Test("Gate 9: ChatTemplate encode with Tokenizer")
    func testChatTemplateEncode() {
        let vocab: [String: Int] = [
            "<unk>": 0,
            "h": 1,
            "i": 2,
            "hi": 3
        ]
        let tokenizer = BPETokenizer(vocab: vocab, merges: ["h i"])
        let template = ChatTemplate(style: .chatML)

        let messages = [ChatMessage.user("hi")]
        let tokens = template.encode(messages: messages, using: tokenizer, addGenerationPrompt: false)

        #expect(!tokens.isEmpty)
    }
}
