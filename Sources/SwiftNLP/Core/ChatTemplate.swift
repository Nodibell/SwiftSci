import Foundation

/// Represents the participant role in a conversational dialogue turn.
public enum ChatRole: String, Sendable, Codable, Equatable {
    case system
    case user
    case assistant
    case tool
}

/// A structured conversational turn message.
public struct ChatMessage: Sendable, Codable, Equatable {
    /// Conversational role (system, user, assistant, tool).
    public var role: ChatRole
    /// Message textual content.
    public var content: String

    /// Creates a chat message.
    /// - Parameters:
    ///   - role: Conversational role.
    ///   - content: Textual content.
    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }

    /// Creates a system role message.
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }

    /// Creates a user role message.
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }

    /// Creates an assistant role message.
    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: content)
    }

    /// Creates a tool response message.
    public static func tool(_ content: String) -> ChatMessage {
        ChatMessage(role: .tool, content: content)
    }
}

/// Renders multi-turn chat messages into standardized model-specific prompts.
public struct ChatTemplate: Sendable, Equatable {
    /// Supported formatting styles for chat templates.
    public enum Style: Sendable, Equatable {
        /// Llama 3 format (`<|start_header_id|>...<|end_header_id|>\n\n...<|eot_id|>`).
        case llama3
        /// Llama 3.2 Instruct text chats with a caller-supplied date string.
        /// Includes the checkpoint's dated system header and string tool responses.
        /// Tool definitions, structured tool calls and multimodal content are unsupported.
        case llama32Instruct(date: String)
        /// ChatML format (`<|im_start|>role\ncontent<|im_end|>`).
        case chatML
        /// Mistral format (`[INST] ... [/INST]`).
        case mistral
    }

    /// The template formatting style.
    public var style: Style

    /// Creates a chat template with the given style.
    /// - Parameter style: Formatting style (default `.llama3`).
    public init(style: Style = .llama3) {
        self.style = style
    }

    /// Preconfigured template using Llama 3 format.
    public static let llama3 = ChatTemplate(style: .llama3)
    /// Creates the pinned Llama 3.2 Instruct text-chat template.
    /// - Parameter date: Date text such as `30 Sep 2026`. No clock or locale is consulted.
    /// - Returns: A template for plain-text messages and string tool responses.
    public static func llama32Instruct(date: String) -> ChatTemplate {
        ChatTemplate(style: .llama32Instruct(date: date))
    }

    /// Preconfigured template using ChatML format.
    public static let chatML = ChatTemplate(style: .chatML)
    /// Preconfigured template using Mistral format.
    public static let mistral = ChatTemplate(style: .mistral)

    /// Renders messages into a formatted prompt string.
    /// - Parameters:
    ///   - messages: Array of conversational chat messages.
    ///   - addGenerationPrompt: Whether to append the trailing assistant header to initiate generation.
    /// - Returns: Formatted prompt string.
    public func render(messages: [ChatMessage], addGenerationPrompt: Bool = true) -> String {
        switch style {
        case .llama32Instruct(let date):
            return Self.renderLlama32(messages: messages, date: date, addGenerationPrompt: addGenerationPrompt)

        case .llama3:
            var output = ""
            for msg in messages {
                output += "<|start_header_id|>\(msg.role.rawValue)<|end_header_id|>\n\n\(msg.content)<|eot_id|>"
            }
            if addGenerationPrompt {
                output += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            }
            return output

        case .chatML:
            var output = ""
            for msg in messages {
                output += "<|im_start|>\(msg.role.rawValue)\n\(msg.content)<|im_end|>\n"
            }
            if addGenerationPrompt {
                output += "<|im_start|>assistant\n"
            }
            return output

        case .mistral:
            var output = ""
            var systemPrompt: String? = nil
            var nonSystemMessages = [ChatMessage]()

            for msg in messages {
                if msg.role == .system && systemPrompt == nil {
                    systemPrompt = msg.content
                } else {
                    nonSystemMessages.append(msg)
                }
            }

            var isFirstUser = true
            for msg in nonSystemMessages {
                switch msg.role {
                case .user:
                    if isFirstUser, let sys = systemPrompt {
                        output += "[INST] <<SYS>>\n\(sys)\n<</SYS>>\n\n\(msg.content) [/INST]"
                        isFirstUser = false
                    } else {
                        output += "[INST] \(msg.content) [/INST]"
                    }
                case .assistant:
                    output += " \(msg.content) "
                default:
                    output += "[INST] \(msg.content) [/INST]"
                }
            }
            return output
        }
    }

    /// Encodes messages into token IDs using any `Tokenizer`.
    /// - Parameters:
    ///   - messages: Array of conversational chat messages.
    ///   - tokenizer: Target tokenizer conforming to `Tokenizer`.
    ///   - addGenerationPrompt: Whether to append the assistant generation trigger.
    /// - Returns: Array of token IDs.
    public func encode(messages: [ChatMessage], using tokenizer: any Tokenizer, addGenerationPrompt: Bool = true) -> [Int] {
        let text = render(messages: messages, addGenerationPrompt: addGenerationPrompt)
        return tokenizer.encode(text: text)
    }

    private static func renderLlama32(messages: [ChatMessage], date: String, addGenerationPrompt: Bool) -> String {
        let hasSystem = messages.first?.role == .system
        let system = hasSystem ? trimLlamaContent(messages[0].content) : ""
        var output = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        output += "Cutting Knowledge Date: December 2023\nToday Date: \(date)\n\n"
        output += system + "<|eot_id|>"
        for message in messages.dropFirst(hasSystem ? 1 : 0) {
            let role = message.role == .tool ? "ipython" : message.role.rawValue
            let content = message.role == .tool ? quoteLlamaToolResult(message.content) : trimLlamaContent(message.content)
            output += "<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>"
        }
        if addGenerationPrompt {
            output += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
        return output
    }

    private static func trimLlamaContent(_ text: String) -> String {
        // Jinja trim uses Python str.strip, including U+001C...U+001F but not U+200B.
        func whitespace(_ scalar: Unicode.Scalar) -> Bool {
            switch scalar.value {
            case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A,
                 0x2028, 0x2029, 0x202F, 0x205F, 0x3000: true
            default: false
            }
        }
        let scalars = text.unicodeScalars
        var start = scalars.startIndex
        var end = scalars.endIndex
        while start < end, whitespace(scalars[start]) { scalars.formIndex(after: &start) }
        while start < end, whitespace(scalars[scalars.index(before: end)]) { scalars.formIndex(before: &end) }
        return String(scalars[start..<end])
    }

    private static func quoteLlamaToolResult(_ text: String) -> String {
        // Match transformers' tojson filter: Unicode is literal and slashes are not escaped.
        var result = "\""
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x22: result += "\\\""
            case 0x5C: result += "\\\\"
            case 0x08: result += "\\b"
            case 0x09: result += "\\t"
            case 0x0A: result += "\\n"
            case 0x0C: result += "\\f"
            case 0x0D: result += "\\r"
            case 0x00...0x1F: result += String(format: "\\u%04x", scalar.value)
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result + "\""
    }

}
