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
}
