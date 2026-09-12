import Foundation

/// Provides static stop-words sets for multilingual text filtering.
public enum StopWords {
    /// Supported languages for stop-word filtering.
    public enum Language: String, Sendable, CaseIterable {
        /// English language.
        case english
        /// Ukrainian language.
        case ukrainian
        /// German language.
        case german
        /// French language.
        case french
        /// Spanish language.
        case spanish
    }

    /// English stop words set.
    public static let english: Set<String> = [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    ]

    /// Ukrainian stop words set.
    public static let ukrainian: Set<String> = [
        "і", "в", "на", "з", "до", "що", "як", "це", "про", "та", "по", "у", "за", "від", "для",
        "але", "чи", "не", "так", "він", "вона", "вони", "ми", "ви", "я", "його", "її", "їх",
        "той", "ця", "ці", "бути", "був", "була", "було", "були", "є", "ще", "вже", "після",
        "через", "під", "над", "при", "без", "щоб", "якщо", "де", "коли", "хто", "який", "яка",
        "яке", "які", "лише", "тільки", "навіть", "все", "всі", "весь", "вся", "те", "тут", "там",
        "тому", "ж", "же", "би", "б", "отож", "між", "перед", "проти", "крім", "серед", "понад",
        "проте", "однак", "хоч", "хоча", "мов", "наче", "немов", "ніби", "ні", "або", "також", "теж",
        "собі", "себе", "собою", "свій", "своя", "своє", "свої", "мене", "мені", "мною", "тобі",
        "тобою", "йому", "їм", "ними", "нами", "вами", "ким", "чим", "кого", "чого", "кому", "чому"
    ]

    /// German stop words set.
    public static let german: Set<String> = [
        "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander",
        "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders",
        "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "der", "den",
        "des", "dem", "die", "das", "daß", "dass", "derselbe", "derselben", "denselben", "desselben",
        "demselben", "dieselbe", "dieselben", "dasselbe", "dazu", "dein", "deine", "deinem", "deinen",
        "deiner", "deines", "denn", "derer", "dessen", "dich", "dir", "du", "dies", "diese", "diesem",
        "diesen", "dieser", "dieses", "doch", "dort", "durch", "ein", "eine", "einem", "einen",
        "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal",
        "er", "ihn", "ihm", "es", "etwas", "euer", "eure", "eurem", "euren", "eurer", "eures",
        "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin",
        "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "euch",
        "im", "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene",
        "jenem", "jenen", "jener", "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen",
        "keiner", "keines", "können", "könnte", "machen", "man", "manche", "manchem", "manchen",
        "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mit",
        "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", "oder", "ohne", "sehr",
        "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich", "sie", "sind",
        "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte", "sondern",
        "sonst", "über", "um", "und", "uns", "unser", "unsere", "unserem", "unseren", "unseres",
        "unter", "viel", "vom", "von", "vor", "während", "war", "waren", "warst", "was", "weg",
        "weil", "weiter", "welche", "welchem", "welchen", "welcher", "welches", "wenn", "wer",
        "werde", "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wolle", "wollte",
        "würde", "würden", "zu", "zum", "zur", "zwar", "zwischen"
    ]

    /// French stop words set.
    public static let french: Set<String> = [
        "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux",
        "il", "ils", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même", "mes",
        "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que",
        "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une",
        "vos", "votre", "vous", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "été", "étée",
        "étées", "étés", "étant", "étante", "étants", "étantes", "suis", "es", "est", "sommes",
        "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront", "serais", "serait",
        "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient", "fus", "fut",
        "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses",
        "fût", "fussions", "fussiez", "fussent", "ayant", "ayante", "ayantes", "ayants", "eu", "eue",
        "eues", "eus", "ai", "as", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons", "aurez",
        "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "avais", "avait", "avions",
        "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez",
        "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent"
    ]

    /// Spanish stop words set.
    public static let spanish: Set<String> = [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
        "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
        "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
        "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
        "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos",
        "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
        "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
        "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosostros", "vosostras", "os", "mío",
        "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
        "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
        "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés",
        "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán",
        "estaría", "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos",
        "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron",
        "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses",
        "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas",
        "estad", "he", "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayais",
        "hayan", "habré", "habrás", "habrá", "habremos", "habréis", "habrán", "habría", "habrías",
        "habríamos", "habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube",
        "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos",
        "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo",
        "habido", "habida", "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea",
        "seas", "seamos", "seáis", "sean", "seré", "serás", "será", "seremos", "seréis", "serán",
        "sería", "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos", "erais", "eran",
        "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais",
        "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "siendo", "sido", "tengo", "tienes",
        "tiene", "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan",
        "tendré", "tendrás", "tendrá", "tendremos", "tendréis", "tendrán", "tendría", "tendrías",
        "tendríamos", "tendríais", "tendrían", "tenía", "tenías", "teníamos", "teníais", "tenían",
        "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras", "tuviéramos",
        "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos", "tuvieseis", "tuviesen", "teniendo",
        "tenido", "tenida", "tenidos", "tenidas", "tened"
    ]

    /// Retrieves the static stop words set for a designated language.
    /// - Parameter language: Supported language identifier.
    /// - Returns: Set of stop words in lower-case format.
    public static func set(for language: Language) -> Set<String> {
        switch language {
        case .english:
            return english
        case .ukrainian:
            return ukrainian
        case .german:
            return german
        case .french:
            return french
        case .spanish:
            return spanish
        }
    }

    /// Filters out stop words from a token array for a given language.
    /// - Parameters:
    ///   - tokens: Array of segmented lexical token strings.
    ///   - language: Supported language identifier (default: `.english`).
    ///   - customStopWords: Optional additional set of stop words to prune.
    /// - Returns: Cleaned array of tokens excluding stop words.
    public static func filter(
        tokens: [String],
        language: Language = .english,
        customStopWords: Set<String>? = nil
    ) -> [String] {
        var stopSet = set(for: language)
        if let custom = customStopWords {
            stopSet.formUnion(custom.map { $0.lowercased() })
        }
        return tokens.filter { !stopSet.contains($0.lowercased()) }
    }
}
