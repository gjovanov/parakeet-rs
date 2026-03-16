//! FormattingTranscriber decorator — wraps any StreamingTranscriber to format FINAL segments
//!
//! PARTIALs pass through unchanged (too fast-changing for formatting).
//! FINALs get `raw_text` set to original ASR output and `text` replaced with formatted version.

use crate::error::Result;
use crate::streaming_transcriber::{
    ModelInfo, StreamingChunkResult, StreamingTranscriber,
};
use crate::text_formatter::{FormattingContext, TextFormatter};
use std::time::Duration;

/// Decorator that wraps a StreamingTranscriber and formats FINAL segments
pub struct FormattingTranscriber {
    inner: Box<dyn StreamingTranscriber>,
    formatter: Box<dyn TextFormatter>,
    context: FormattingContext,
    timeout: Duration,
}

impl FormattingTranscriber {
    pub fn new(
        inner: Box<dyn StreamingTranscriber>,
        formatter: Box<dyn TextFormatter>,
        context: FormattingContext,
        timeout: Duration,
    ) -> Self {
        eprintln!(
            "[FormattingTranscriber] Wrapping with {} formatter (tone: {:?}, lang: {})",
            formatter.name(),
            context.tone,
            context.language
        );
        Self {
            inner,
            formatter,
            context,
            timeout,
        }
    }

    /// Format FINAL segments in-place, set raw_text on all segments
    fn format_result(&mut self, result: &mut StreamingChunkResult) {
        for segment in &mut result.segments {
            // Always preserve original text as raw_text
            segment.raw_text = Some(segment.text.clone());

            if segment.is_final && !segment.text.trim().is_empty() {
                let start = std::time::Instant::now();
                let formatted = self.formatter.format(&segment.text, &self.context);
                let elapsed = start.elapsed();

                if elapsed > self.timeout {
                    eprintln!(
                        "[FormattingTranscriber] Formatter exceeded timeout ({:?} > {:?}), using raw text",
                        elapsed, self.timeout
                    );
                    // Keep original text on timeout
                    continue;
                }

                // Sanity check: don't use formatting result if it's empty or suspiciously long
                if formatted.is_empty() || formatted.len() > segment.text.len() * 3 {
                    eprintln!(
                        "[FormattingTranscriber] Formatter output rejected (empty or >3x input length)"
                    );
                    continue;
                }

                segment.text = formatted;

                // Update recent context for future formatting
                self.context.recent_text.push(segment.text.clone());
                if self.context.recent_text.len() > 5 {
                    self.context.recent_text.remove(0);
                }
            }
        }
    }
}

impl StreamingTranscriber for FormattingTranscriber {
    fn model_info(&self) -> ModelInfo {
        let mut info = self.inner.model_info();
        info.display_name = format!("{} +Formatted", info.display_name);
        info
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let mut result = self.inner.push_audio(samples)?;
        self.format_result(&mut result);
        Ok(result)
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let mut result = self.inner.finalize()?;
        self.format_result(&mut result);
        Ok(result)
    }

    fn reset(&mut self) {
        self.context.recent_text.clear();
        self.inner.reset();
    }

    fn buffer_duration(&self) -> f32 {
        self.inner.buffer_duration()
    }

    fn total_duration(&self) -> f32 {
        self.inner.total_duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming_transcriber::TranscriptionSegment;
    use crate::text_formatter::RuleBasedFormatter;

    /// Mock transcriber that returns predetermined segments
    struct MockTranscriber {
        segments: Vec<TranscriptionSegment>,
    }

    impl StreamingTranscriber for MockTranscriber {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                id: "mock".to_string(),
                display_name: "Mock".to_string(),
                description: "Test".to_string(),
                supports_diarization: false,
                languages: vec![],
                is_loaded: true,
            }
        }

        fn push_audio(&mut self, _samples: &[f32]) -> Result<StreamingChunkResult> {
            Ok(StreamingChunkResult {
                segments: self.segments.clone(),
                buffer_duration: 1.0,
                total_duration: 1.0,
            })
        }

        fn finalize(&mut self) -> Result<StreamingChunkResult> {
            Ok(StreamingChunkResult::empty())
        }

        fn reset(&mut self) {}
        fn buffer_duration(&self) -> f32 { 0.0 }
        fn total_duration(&self) -> f32 { 0.0 }
    }

    #[test]
    fn test_final_segments_get_formatted() {
        let mock = MockTranscriber {
            segments: vec![TranscriptionSegment {
                text: "Ähm das ist gut".to_string(),
                raw_text: None,
                start_time: 0.0,
                end_time: 1.0,
                speaker: None,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            }],
        };

        let mut ft = FormattingTranscriber::new(
            Box::new(mock),
            Box::new(RuleBasedFormatter::new()),
            FormattingContext::default(),
            Duration::from_millis(200),
        );

        let result = ft.push_audio(&[0.0]).unwrap();
        let seg = &result.segments[0];

        // raw_text should have the original
        assert_eq!(seg.raw_text.as_deref(), Some("Ähm das ist gut"));
        // text should be formatted (filler removed)
        assert_eq!(seg.text, "Das ist gut");
    }

    #[test]
    fn test_partial_segments_not_formatted() {
        let mock = MockTranscriber {
            segments: vec![TranscriptionSegment {
                text: "Ähm das ist gut".to_string(),
                raw_text: None,
                start_time: 0.0,
                end_time: 1.0,
                speaker: None,
                confidence: None,
                is_final: false, // PARTIAL
                inference_time_ms: None,
            }],
        };

        let mut ft = FormattingTranscriber::new(
            Box::new(mock),
            Box::new(RuleBasedFormatter::new()),
            FormattingContext::default(),
            Duration::from_millis(200),
        );

        let result = ft.push_audio(&[0.0]).unwrap();
        let seg = &result.segments[0];

        // raw_text set but text NOT formatted (partial)
        assert_eq!(seg.raw_text.as_deref(), Some("Ähm das ist gut"));
        assert_eq!(seg.text, "Ähm das ist gut");
    }

    #[test]
    fn test_model_info_shows_formatted() {
        let mock = MockTranscriber { segments: vec![] };
        let ft = FormattingTranscriber::new(
            Box::new(mock),
            Box::new(RuleBasedFormatter::new()),
            FormattingContext::default(),
            Duration::from_millis(200),
        );

        let info = ft.model_info();
        assert!(info.display_name.contains("+Formatted"));
    }

    #[test]
    fn test_reset_clears_context() {
        let mock = MockTranscriber { segments: vec![] };
        let mut ft = FormattingTranscriber::new(
            Box::new(mock),
            Box::new(RuleBasedFormatter::new()),
            FormattingContext {
                recent_text: vec!["some context".to_string()],
                ..Default::default()
            },
            Duration::from_millis(200),
        );

        ft.reset();
        assert!(ft.context.recent_text.is_empty());
    }
}
