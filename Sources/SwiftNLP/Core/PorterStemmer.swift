import Foundation

/// Pure Swift implementation of the Porter Stemming Algorithm for suffix stripping.
public struct PorterStemmer: Sendable {
    public init() {}

    private static let vowels: Set<Character> = ["a", "e", "i", "o", "u"]

    private func isVowel(_ ch: Character) -> Bool {
        return Self.vowels.contains(ch)
    }

    private func containsVowel(_ word: String) -> Bool {
        return word.contains { isVowel($0) }
    }

    private func isConsonant(_ s: String, _ index: Int) -> Bool {
        let chars = Array(s)
        let ch = chars[index]
        if isVowel(ch) { return false }
        if ch == "y" {
            return index == 0 ? true : !isConsonant(s, index - 1)
        }
        return true
    }

    private func getMeasure(_ word: String) -> Int {
        let chars = Array(word)
        var n = 0
        var i = 0
        let len = chars.count

        while i < len {
            if !isConsonant(word, i) { break }
            i += 1
        }
        i += 1
        while i < len {
            while i < len {
                if isConsonant(word, i) { break }
                i += 1
            }
            if i < len {
                n += 1
                i += 1
                while i < len {
                    if !isConsonant(word, i) { break }
                    i += 1
                }
                i += 1
            }
        }
        return n
    }

    private func endsWithDoubleConsonant(_ word: String) -> Bool {
        let chars = Array(word)
        guard chars.count >= 2 else { return false }
        let last = chars[chars.count - 1]
        let prev = chars[chars.count - 2]
        return last == prev && isConsonant(word, chars.count - 1)
    }

    private func cvc(_ word: String) -> Bool {
        let chars = Array(word)
        let len = chars.count
        guard len >= 3 else { return false }
        let c1 = isConsonant(word, len - 3)
        let v = !isConsonant(word, len - 2)
        let c2 = isConsonant(word, len - 1)
        let last = chars[len - 1]
        return c1 && v && c2 && last != "w" && last != "x" && last != "y"
    }

    /// Stems a word to its canonical morphological root.
    public func stem(_ word: String) -> String {
        let lower = word.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard lower.count > 2 else { return lower }

        var w = lower

        // Step 1a
        if w.hasSuffix("sses") {
            w = String(w.dropLast(2))
        } else if w.hasSuffix("ies") {
            w = String(w.dropLast(2))
        } else if w.hasSuffix("ss") {
            // Keep ss
        } else if w.hasSuffix("s") {
            w = String(w.dropLast(1))
        }

        // Step 1b
        var step1bExtra = false
        if w.hasSuffix("eed") {
            let stemCandidate = String(w.dropLast(3))
            if getMeasure(stemCandidate) > 0 {
                w = stemCandidate + "ee"
            }
        } else if w.hasSuffix("ed") {
            let stemCandidate = String(w.dropLast(2))
            if containsVowel(stemCandidate) {
                w = stemCandidate
                step1bExtra = true
            }
        } else if w.hasSuffix("ing") {
            let stemCandidate = String(w.dropLast(3))
            if containsVowel(stemCandidate) {
                w = stemCandidate
                step1bExtra = true
            }
        }

        if step1bExtra {
            if w.hasSuffix("at") || w.hasSuffix("bl") || w.hasSuffix("iz") {
                w += "e"
            } else if endsWithDoubleConsonant(w) {
                let last = w.last
                if last != "l" && last != "s" && last != "z" {
                    w = String(w.dropLast(1))
                }
            } else if getMeasure(w) == 1 && cvc(w) {
                w += "e"
            }
        }

        // Step 1c
        if w.hasSuffix("y") {
            let stemCandidate = String(w.dropLast(1))
            if containsVowel(stemCandidate) {
                w = stemCandidate + "i"
            }
        }

        // Step 2
        let step2Map: [(suffix: String, replace: String)] = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")
        ]
        for item in step2Map {
            if w.hasSuffix(item.suffix) {
                let stemCandidate = String(w.dropLast(item.suffix.count))
                if getMeasure(stemCandidate) > 0 {
                    w = stemCandidate + item.replace
                }
                break
            }
        }

        // Step 3
        let step3Map: [(suffix: String, replace: String)] = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", "")
        ]
        for item in step3Map {
            if w.hasSuffix(item.suffix) {
                let stemCandidate = String(w.dropLast(item.suffix.count))
                if getMeasure(stemCandidate) > 0 {
                    w = stemCandidate + item.replace
                }
                break
            }
        }

        // Step 4
        if w.hasSuffix("ion") {
            let stemCandidate = String(w.dropLast(3))
            if getMeasure(stemCandidate) > 1 && (stemCandidate.hasSuffix("s") || stemCandidate.hasSuffix("t")) {
                w = stemCandidate
            }
        } else {
            let step4Suffixes = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
            for s in step4Suffixes {
                if w.hasSuffix(s) {
                    let stemCandidate = String(w.dropLast(s.count))
                    if getMeasure(stemCandidate) > 1 {
                        w = stemCandidate
                    }
                    break
                }
            }
        }

        // Step 5a
        if w.hasSuffix("e") {
            let stemCandidate = String(w.dropLast(1))
            let m = getMeasure(stemCandidate)
            if m > 1 || (m == 1 && !cvc(stemCandidate)) {
                w = stemCandidate
            }
        }

        // Step 5b
        if getMeasure(w) > 1 && endsWithDoubleConsonant(w) && w.hasSuffix("l") {
            w = String(w.dropLast(1))
        }

        return w
    }

    /// Stems an array of word tokens.
    public func stem(tokens: [String]) -> [String] {
        return tokens.map { stem($0) }
    }
}
