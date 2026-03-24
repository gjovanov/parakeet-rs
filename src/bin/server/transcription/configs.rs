//! Transcription configuration factories for different latency modes

use crate::api::sessions::PauseConfig;

/// Create RealtimeCanaryConfig based on latency mode
pub fn create_canary_config(
    mode: &str,
    language: String,
) -> parakeet_rs::realtime_canary::RealtimeCanaryConfig {
    use parakeet_rs::realtime_canary::RealtimeCanaryConfig;

    match mode {
        "speedy" => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 0.5,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
            emit_full_text: false,
            min_stable_count: None,
        },
        "growing_segments" => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.5,
            silence_energy_threshold: 0.008,
            emit_full_text: true,
            min_stable_count: None,
        },
        _ => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
            emit_full_text: false,
            min_stable_count: None,
        },
    }
}

/// Create RealtimeTDTConfig based on latency mode with optional pause config override
pub fn create_transcription_config(
    mode: &str,
    pause_config: Option<&PauseConfig>,
) -> parakeet_rs::RealtimeTDTConfig {
    use parakeet_rs::RealtimeTDTConfig;

    let (pause_threshold, silence_energy) = match pause_config {
        Some(pc) => (
            pc.pause_threshold_ms as f32 / 1000.0,
            pc.silence_energy_threshold,
        ),
        None => match mode {
            "speedy" => (0.6, 0.008),
            _ => (0.5, 0.008),
        },
    };

    match mode {
        "speedy" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "growing_segments" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.15,
            confirm_threshold_secs: 0.3,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        _ => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
    }
}

/// Create RealtimeWhisperConfig based on latency mode
#[cfg(feature = "whisper")]
pub fn create_whisper_config(
    mode: &str,
    language: String,
) -> parakeet_rs::RealtimeWhisperConfig {
    use parakeet_rs::RealtimeWhisperConfig;

    match mode {
        "speedy" => RealtimeWhisperConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 0.5,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
            emit_full_text: false,
            min_stable_count: None,
            beam_size: 5,
            n_threads: 4,
            use_gpu: false,
        },
        "growing_segments" => RealtimeWhisperConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.5,
            silence_energy_threshold: 0.008,
            emit_full_text: true,
            min_stable_count: None,
            beam_size: 5,
            n_threads: 4,
            use_gpu: false,
        },
        _ => RealtimeWhisperConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
            emit_full_text: false,
            min_stable_count: None,
            beam_size: 5,
            n_threads: 4,
            use_gpu: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canary_speedy() {
        let config = create_canary_config("speedy", "de".to_string());
        assert_eq!(config.language, "de");
        assert_eq!(config.buffer_size_secs, 8.0);
        assert!(config.pause_based_confirm);
        assert_eq!(config.process_interval_secs, 0.5);
    }

    #[test]
    fn test_canary_growing_segments() {
        let config = create_canary_config("growing_segments", "de".to_string());
        assert_eq!(config.buffer_size_secs, 8.0);
        assert_eq!(config.process_interval_secs, 1.0);
        assert!(config.emit_full_text);
    }

    #[test]
    fn test_canary_default_fallback() {
        let config = create_canary_config("unknown_mode", "de".to_string());
        assert_eq!(config.buffer_size_secs, 8.0);
        assert!(!config.emit_full_text);
    }

    #[test]
    fn test_tdt_speedy() {
        let config = create_transcription_config("speedy", None);
        assert_eq!(config.buffer_size_secs, 8.0);
        assert_eq!(config.process_interval_secs, 0.2);
        assert!(config.pause_based_confirm);
        assert_eq!(config.pause_threshold_secs, 0.6);
    }

    #[test]
    fn test_tdt_growing_segments() {
        let config = create_transcription_config("growing_segments", None);
        assert_eq!(config.buffer_size_secs, 10.0);
        assert_eq!(config.process_interval_secs, 0.15);
    }

    #[test]
    fn test_tdt_with_pause_config_override() {
        let pause = PauseConfig {
            pause_threshold_ms: 800,
            silence_energy_threshold: 0.01,
            max_segment_secs: 10.0,
            context_segments: 1,
            min_segment_secs: None,
            partial_interval_secs: None,
        };
        let config = create_transcription_config("speedy", Some(&pause));
        assert_eq!(config.pause_threshold_secs, 0.8);
        assert_eq!(config.silence_energy_threshold, 0.01);
    }

    #[test]
    fn test_tdt_default_fallback() {
        let config = create_transcription_config("unknown_mode", None);
        assert_eq!(config.buffer_size_secs, 8.0);
        assert!(config.pause_based_confirm);
    }

    use crate::api::sessions::GrowingSegmentsConfig;

    fn apply_gs_canary(mode: &str, gs: Option<&GrowingSegmentsConfig>) -> parakeet_rs::realtime_canary::RealtimeCanaryConfig {
        let mut c = create_canary_config(mode, "de".to_string());
        if mode == "growing_segments" {
            if let Some(gs) = gs {
                if let Some(v) = gs.buffer_size_secs { c.buffer_size_secs = v; }
                if let Some(v) = gs.process_interval_secs { c.process_interval_secs = v; }
                if let Some(v) = gs.pause_threshold_ms { c.pause_threshold_secs = v as f32 / 1000.0; }
                if let Some(v) = gs.silence_energy_threshold { c.silence_energy_threshold = v; }
            }
        }
        c
    }

    fn apply_gs_tdt(mode: &str, gs: Option<&GrowingSegmentsConfig>) -> parakeet_rs::RealtimeTDTConfig {
        let mut c = create_transcription_config(mode, None);
        if mode == "growing_segments" {
            if let Some(gs) = gs {
                if let Some(v) = gs.buffer_size_secs { c.buffer_size_secs = v; }
                if let Some(v) = gs.process_interval_secs { c.process_interval_secs = v; }
                if let Some(v) = gs.pause_threshold_ms { c.pause_threshold_secs = v as f32 / 1000.0; }
                if let Some(v) = gs.silence_energy_threshold { c.silence_energy_threshold = v; }
            }
        }
        c
    }

    #[test]
    fn test_gs_canary_partial_override() {
        let gs = GrowingSegmentsConfig {
            buffer_size_secs: Some(12.0),
            process_interval_secs: None,
            pause_threshold_ms: Some(600),
            silence_energy_threshold: None,
            ..Default::default()
        };
        let config = apply_gs_canary("growing_segments", Some(&gs));
        assert_eq!(config.buffer_size_secs, 12.0);
        assert_eq!(config.process_interval_secs, 1.0);
        assert_eq!(config.pause_threshold_secs, 0.6);
    }

    #[test]
    fn test_gs_canary_non_growing_ignores_config() {
        let gs = GrowingSegmentsConfig {
            buffer_size_secs: Some(99.0),
            ..Default::default()
        };
        let config = apply_gs_canary("speedy", Some(&gs));
        assert_eq!(config.buffer_size_secs, 8.0);
    }

    #[test]
    fn test_gs_tdt_partial_override() {
        let gs = GrowingSegmentsConfig {
            buffer_size_secs: Some(15.0),
            pause_threshold_ms: Some(700),
            ..Default::default()
        };
        let config = apply_gs_tdt("growing_segments", Some(&gs));
        assert_eq!(config.buffer_size_secs, 15.0);
        assert_eq!(config.pause_threshold_secs, 0.7);
    }

    #[test]
    fn test_gs_none_config_uses_defaults() {
        let config = apply_gs_canary("growing_segments", None);
        assert_eq!(config.buffer_size_secs, 8.0);
        assert_eq!(config.process_interval_secs, 1.0);
    }
}
