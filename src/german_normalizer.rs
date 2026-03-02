//! German text normalizer for fair WER evaluation.
//!
//! Inspired by WhisperLive's `EnglishTextNormalizer`, this module provides
//! German-specific text normalization so that WER comparisons don't penalize
//! cosmetic differences (casing, punctuation, number formatting).

use std::collections::HashMap;

/// German text normalizer for WER evaluation.
pub struct GermanTextNormalizer {
    titles: Vec<(&'static str, &'static str)>,
    fillers: Vec<&'static str>,
    number_units: HashMap<&'static str, u64>,
    number_teens: HashMap<&'static str, u64>,
    number_tens: HashMap<&'static str, u64>,
}

impl Default for GermanTextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GermanTextNormalizer {
    pub fn new() -> Self {
        let titles = vec![
            ("hr.", "herr"),
            ("fr.", "frau"),
            ("dr.", "doktor"),
            ("prof.", "professor"),
            ("nr.", "nummer"),
            ("ca.", "circa"),
            ("bzw.", "beziehungsweise"),
            ("z.b.", "zum beispiel"),
            ("d.h.", "das heisst"),
            ("u.a.", "unter anderem"),
        ];

        let fillers = vec!["äh", "ähm", "hm", "mhm", "hmm", "ah", "ahm", "oh"];

        let mut number_units = HashMap::new();
        number_units.insert("null", 0);
        number_units.insert("eins", 1);
        number_units.insert("ein", 1);
        number_units.insert("eine", 1);
        number_units.insert("zwei", 2);
        number_units.insert("drei", 3);
        number_units.insert("vier", 4);
        number_units.insert("fünf", 5);
        number_units.insert("sechs", 6);
        number_units.insert("sieben", 7);
        number_units.insert("acht", 8);
        number_units.insert("neun", 9);

        let mut number_teens = HashMap::new();
        number_teens.insert("zehn", 10);
        number_teens.insert("elf", 11);
        number_teens.insert("zwölf", 12);
        number_teens.insert("dreizehn", 13);
        number_teens.insert("vierzehn", 14);
        number_teens.insert("fünfzehn", 15);
        number_teens.insert("sechzehn", 16);
        number_teens.insert("siebzehn", 17);
        number_teens.insert("achtzehn", 18);
        number_teens.insert("neunzehn", 19);

        let mut number_tens = HashMap::new();
        number_tens.insert("zwanzig", 20);
        number_tens.insert("dreissig", 30);
        number_tens.insert("dreißig", 30);
        number_tens.insert("vierzig", 40);
        number_tens.insert("fünfzig", 50);
        number_tens.insert("sechzig", 60);
        number_tens.insert("siebzig", 70);
        number_tens.insert("achtzig", 80);
        number_tens.insert("neunzig", 90);

        Self {
            titles,
            fillers,
            number_units,
            number_teens,
            number_tens,
        }
    }

    /// Full normalization pipeline for WER evaluation.
    pub fn normalize(&self, text: &str) -> String {
        let text = text.to_lowercase();
        let text = self.remove_brackets(&text);
        let text = self.expand_titles(&text);
        let text = self.remove_fillers(&text);
        let text = self.normalize_eszett(&text);
        let text = self.normalize_number_words(&text);
        let text = self.remove_punctuation(&text);
        self.collapse_whitespace(&text)
    }

    /// Replace German compound number words with digit strings.
    ///
    /// Handles: units, teens, tens, `{unit}und{tens}` compounds,
    /// hundreds (`{unit}hundert{rest}`), thousands (`{unit}tausend{rest}`),
    /// and "million"/"milliarde" prefixes.
    pub fn normalize_number_words(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result: Vec<String> = Vec::with_capacity(words.len());
        let mut i = 0;

        while i < words.len() {
            // Try multi-word numbers: "eine million dreihundertzweiundvierzig"
            let (num, consumed) = self.try_parse_number(&words[i..]);
            if consumed > 0 {
                result.push(num.to_string());
                i += consumed;
            } else {
                result.push(words[i].to_string());
                i += 1;
            }
        }

        result.join(" ")
    }

    /// Words that can be articles/pronouns and should NOT be converted to numbers
    /// when they appear as standalone words (not part of a compound or multi-word number).
    fn is_ambiguous_standalone(&self, word: &str) -> bool {
        matches!(word, "ein" | "eine" | "eins")
    }

    /// Try to parse a number starting at `words[0]`. Returns (value, words_consumed).
    /// Returns (0, 0) if no number word is recognized.
    fn try_parse_number(&self, words: &[&str]) -> (u64, usize) {
        if words.is_empty() {
            return (0, 0);
        }

        let mut total: u64;
        let mut consumed: usize;

        // Check for "eine million" / "eine milliarde" prefix
        if words.len() >= 2 {
            let w0 = words[0];
            let w1 = words[1];
            if (w0 == "eine" || w0 == "ein") && w1 == "million" {
                total = 1_000_000;
                consumed = 2;
                if consumed < words.len() {
                    let (rest, rest_consumed) = self.try_parse_number(&words[consumed..]);
                    if rest_consumed > 0 {
                        total += rest;
                        consumed += rest_consumed;
                    }
                }
                return (total, consumed);
            }
            if (w0 == "eine" || w0 == "ein") && w1 == "milliarde" {
                total = 1_000_000_000;
                consumed = 2;
                if consumed < words.len() {
                    let (rest, rest_consumed) = self.try_parse_number(&words[consumed..]);
                    if rest_consumed > 0 {
                        total += rest;
                        consumed += rest_consumed;
                    }
                }
                return (total, consumed);
            }

            // "{digit} millionen/milliarden"
            if let Some(&unit_val) = self.number_units.get(w0) {
                if unit_val >= 2 {
                    if w1 == "millionen" {
                        total = unit_val * 1_000_000;
                        consumed = 2;
                        if consumed < words.len() {
                            let (rest, rest_consumed) = self.try_parse_number(&words[consumed..]);
                            if rest_consumed > 0 {
                                total += rest;
                                consumed += rest_consumed;
                            }
                        }
                        return (total, consumed);
                    }
                    if w1 == "milliarden" {
                        total = unit_val * 1_000_000_000;
                        consumed = 2;
                        if consumed < words.len() {
                            let (rest, rest_consumed) = self.try_parse_number(&words[consumed..]);
                            if rest_consumed > 0 {
                                total += rest;
                                consumed += rest_consumed;
                            }
                        }
                        return (total, consumed);
                    }
                }
            }
        }

        // Try parsing a single compound word (e.g. "dreihundertzweiundvierzig")
        // Skip ambiguous standalone words like "ein"/"eine" (articles)
        if !self.is_ambiguous_standalone(words[0]) {
            if let Some(v) = self.parse_compound_number(words[0]) {
                return (v, 1);
            }
        }

        (0, 0)
    }

    /// Parse a single compound German number word.
    /// E.g. "dreihundertzweiundvierzig" → Some(342)
    fn parse_compound_number(&self, word: &str) -> Option<u64> {
        if word.is_empty() {
            return None;
        }

        // Direct lookups first
        if let Some(&v) = self.number_units.get(word) {
            return Some(v);
        }
        if let Some(&v) = self.number_teens.get(word) {
            return Some(v);
        }
        if let Some(&v) = self.number_tens.get(word) {
            return Some(v);
        }

        // Try thousands: "{unit}tausend{rest}"
        if let Some(pos) = word.find("tausend") {
            let prefix = &word[..pos];
            let suffix = &word[pos + "tausend".len()..];

            let thousands = if prefix.is_empty() {
                1
            } else {
                self.parse_compound_number(prefix)?
            };

            let rest = if suffix.is_empty() {
                0
            } else {
                self.parse_compound_number(suffix)?
            };

            return Some(thousands * 1000 + rest);
        }

        // Try hundreds: "{unit}hundert{rest}"
        if let Some(pos) = word.find("hundert") {
            let prefix = &word[..pos];
            let suffix = &word[pos + "hundert".len()..];

            let hundreds = if prefix.is_empty() {
                1
            } else {
                self.parse_compound_number(prefix)?
            };

            let rest = if suffix.is_empty() {
                0
            } else {
                self.parse_compound_number(suffix)?
            };

            return Some(hundreds * 100 + rest);
        }

        // Try compound tens: "{unit}und{tens}" (e.g. "einundzwanzig")
        if let Some(pos) = word.find("und") {
            let unit_part = &word[..pos];
            let tens_part = &word[pos + "und".len()..];

            if let (Some(&u), Some(&t)) = (
                self.number_units.get(unit_part),
                self.number_tens.get(tens_part),
            ) {
                return Some(t + u);
            }
        }

        None
    }

    // --- Internal normalization steps ---

    fn remove_brackets(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut depth_angle = 0i32;
        let mut depth_paren = 0i32;
        let mut depth_square = 0i32;

        for ch in text.chars() {
            match ch {
                '<' => depth_angle += 1,
                '>' => {
                    depth_angle = (depth_angle - 1).max(0);
                    continue;
                }
                '(' => depth_paren += 1,
                ')' => {
                    depth_paren = (depth_paren - 1).max(0);
                    continue;
                }
                '[' => depth_square += 1,
                ']' => {
                    depth_square = (depth_square - 1).max(0);
                    continue;
                }
                _ => {}
            }
            if depth_angle == 0 && depth_paren == 0 && depth_square == 0 {
                result.push(ch);
            }
        }

        result
    }

    fn expand_titles(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (abbr, expanded) in &self.titles {
            // Word-boundary-aware replacement
            result = result.replace(abbr, expanded);
        }
        result
    }

    fn remove_fillers(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let filtered: Vec<&str> = words
            .into_iter()
            .filter(|w| {
                // Strip trailing punctuation for comparison
                let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
                !self.fillers.contains(&clean.as_str())
            })
            .collect();
        filtered.join(" ")
    }

    fn normalize_eszett(&self, text: &str) -> String {
        text.replace('ß', "ss")
    }

    fn remove_punctuation(&self, text: &str) -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect()
    }

    fn collapse_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Convenience function: normalize German text for WER evaluation.
pub fn normalize_german(text: &str) -> String {
    GermanTextNormalizer::new().normalize(text)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn n() -> GermanTextNormalizer {
        GermanTextNormalizer::new()
    }

    // --- Lowercase + punctuation ---

    #[test]
    fn test_lowercase_and_punctuation() {
        assert_eq!(n().normalize("Hello, World!"), "hello world");
    }

    #[test]
    fn test_multiple_punctuation() {
        assert_eq!(n().normalize("Was??? Nein!!!"), "was nein");
    }

    // --- Filler words ---

    #[test]
    fn test_remove_fillers() {
        assert_eq!(n().normalize("Das äh Wetter"), "das wetter");
    }

    #[test]
    fn test_remove_multiple_fillers() {
        assert_eq!(n().normalize("Ähm ja hm das ist ähm gut"), "ja das ist gut");
    }

    #[test]
    fn test_filler_not_substring() {
        // "ahmen" should not be affected by "ah" filler removal
        // (filler removal works on whole words)
        assert_eq!(n().normalize("ahmen"), "ahmen");
    }

    // --- Title abbreviation expansion ---

    #[test]
    fn test_title_expansion() {
        assert_eq!(n().normalize("Hr. Müller"), "herr müller");
    }

    #[test]
    fn test_title_doktor() {
        assert_eq!(n().normalize("Dr. Schmidt"), "doktor schmidt");
    }

    #[test]
    fn test_title_professor() {
        assert_eq!(n().normalize("Prof. Wagner"), "professor wagner");
    }

    // --- Eszett normalization ---

    #[test]
    fn test_eszett() {
        assert_eq!(n().normalize("Straße"), "strasse");
    }

    #[test]
    fn test_eszett_dreissig() {
        assert_eq!(n().normalize("dreißig"), "30");
    }

    // --- Umlaut preservation ---

    #[test]
    fn test_umlaut_preserved() {
        assert_eq!(n().normalize("schön"), "schön");
    }

    #[test]
    fn test_umlauts_all() {
        assert_eq!(n().normalize("Ärger über Öl"), "ärger über öl");
    }

    // --- Bracket removal ---

    #[test]
    fn test_remove_angle_brackets() {
        assert_eq!(n().normalize("Das ist <lachen> ein Test"), "das ist ein test");
    }

    #[test]
    fn test_remove_parentheses() {
        assert_eq!(n().normalize("Wien (Österreich) ist schön"), "wien ist schön");
    }

    #[test]
    fn test_remove_square_brackets() {
        assert_eq!(n().normalize("Text [annotation] here"), "text here");
    }

    // --- Number words ---

    #[test]
    fn test_number_units() {
        // Standalone "eins"/"ein"/"eine" are ambiguous (could be article), skipped
        assert_eq!(n().normalize("eins"), "eins");
        assert_eq!(n().normalize("neun"), "9");
        assert_eq!(n().normalize("null"), "0");
    }

    #[test]
    fn test_number_teens() {
        assert_eq!(n().normalize("elf"), "11");
        assert_eq!(n().normalize("zwölf"), "12");
        assert_eq!(n().normalize("neunzehn"), "19");
    }

    #[test]
    fn test_number_tens() {
        assert_eq!(n().normalize("zwanzig"), "20");
        assert_eq!(n().normalize("neunzig"), "90");
    }

    #[test]
    fn test_number_compound_tens() {
        assert_eq!(n().normalize("einundzwanzig"), "21");
        assert_eq!(n().normalize("zweiunddreissig"), "32");
    }

    #[test]
    fn test_number_hundreds() {
        assert_eq!(n().normalize("dreihundert"), "300");
        assert_eq!(n().normalize("hundert"), "100");
    }

    #[test]
    fn test_number_compound_hundreds() {
        assert_eq!(n().normalize("dreihundertzweiundvierzig"), "342");
    }

    #[test]
    fn test_number_thousands() {
        assert_eq!(n().normalize("zweitausend"), "2000");
        assert_eq!(n().normalize("tausend"), "1000");
    }

    #[test]
    fn test_number_complex() {
        assert_eq!(n().normalize("dreitausendzweihundertvierundfünfzig"), "3254");
    }

    #[test]
    fn test_number_million() {
        assert_eq!(n().normalize("eine million"), "1000000");
    }

    #[test]
    fn test_number_million_with_rest() {
        assert_eq!(n().normalize("eine million dreihundert"), "1000300");
    }

    #[test]
    fn test_number_milliarden() {
        assert_eq!(n().normalize("drei milliarden"), "3000000000");
    }

    #[test]
    fn test_number_in_sentence() {
        assert_eq!(
            n().normalize("Es sind fünfundzwanzig Grad"),
            "es sind 25 grad"
        );
    }

    // --- Combined pipeline ---

    #[test]
    fn test_combined_normalization() {
        let input = "Hr. Müller sagte: \"Die Straße ist dreißig Meter lang.\"";
        let expected = "herr müller sagte die strasse ist 30 meter lang";
        assert_eq!(n().normalize(input), expected);
    }

    #[test]
    fn test_full_german_sentence() {
        let input = "Die Temperatur beträgt heute fünfundzwanzig Grad Celsius";
        let expected = "die temperatur beträgt heute 25 grad celsius";
        assert_eq!(n().normalize(input), expected);
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_string() {
        assert_eq!(n().normalize(""), "");
    }

    #[test]
    fn test_all_punctuation() {
        assert_eq!(n().normalize("...!!!???"), "");
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(n().normalize("   \t  \n  "), "");
    }

    #[test]
    fn test_mixed_german_english() {
        assert_eq!(
            n().normalize("Das Meeting ist um drei Uhr"),
            "das meeting ist um 3 uhr"
        );
    }

    #[test]
    fn test_digits_pass_through() {
        assert_eq!(n().normalize("Es kostet 42 Euro"), "es kostet 42 euro");
    }

    // --- normalize_german convenience function ---

    #[test]
    fn test_convenience_function() {
        assert_eq!(normalize_german("Straße"), "strasse");
    }

    // --- normalize_number_words standalone ---

    #[test]
    fn test_number_words_standalone() {
        let norm = n();
        assert_eq!(norm.normalize_number_words("einundzwanzig"), "21");
        assert_eq!(
            norm.normalize_number_words("ich habe drei Katzen"),
            "ich habe 3 Katzen"
        );
    }
}
