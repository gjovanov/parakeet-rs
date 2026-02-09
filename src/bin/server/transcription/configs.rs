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
        },
        "pause_based" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        "low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.5,
            language,
            pause_based_confirm: false,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        "ultra_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 6.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.5,
            silence_energy_threshold: 0.008,
        },
        "extreme_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 4.0,
            min_audio_secs: 0.5,
            process_interval_secs: 0.5,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.4,
            silence_energy_threshold: 0.008,
        },
        "lookahead" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        _ => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
    }
}

/// Create RealtimeTDTConfig based on latency mode with optional pause config override
pub fn create_transcription_config(
    mode: &str,
    pause_config: Option<&PauseConfig>,
) -> parakeet_rs::RealtimeTDTConfig {
    use parakeet_rs::RealtimeTDTConfig;

    // Get pause config values or use mode defaults (0.6s for better sentence boundaries)
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
        "pause_based" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "ultra_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 1.0,
            confirm_threshold_secs: 1.5,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "extreme_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 5.0,
            process_interval_secs: 0.5,
            confirm_threshold_secs: 0.8,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "lookahead" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: true,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Canary config
    // ========================================================================

    #[test]
    fn test_canary_speedy() {
        let config = create_canary_config("speedy", "de".to_string());
        assert_eq!(config.language, "de");
        assert_eq!(config.buffer_size_secs, 8.0);
        assert!(config.pause_based_confirm);
        assert_eq!(config.process_interval_secs, 0.5);
    }

    #[test]
    fn test_canary_pause_based() {
        let config = create_canary_config("pause_based", "en".to_string());
        assert_eq!(config.buffer_size_secs, 10.0);
        assert_eq!(config.min_audio_secs, 2.0);
        assert!(config.pause_based_confirm);
    }

    #[test]
    fn test_canary_low_latency() {
        let config = create_canary_config("low_latency", "de".to_string());
        assert!(!config.pause_based_confirm);
        assert_eq!(config.process_interval_secs, 1.5);
    }

    #[test]
    fn test_canary_ultra_low_latency() {
        let config = create_canary_config("ultra_low_latency", "de".to_string());
        assert_eq!(config.buffer_size_secs, 6.0);
        assert!(config.pause_based_confirm);
    }

    #[test]
    fn test_canary_extreme_low_latency() {
        let config = create_canary_config("extreme_low_latency", "de".to_string());
        assert_eq!(config.buffer_size_secs, 4.0);
        assert_eq!(config.min_audio_secs, 0.5);
        assert_eq!(config.pause_threshold_secs, 0.4);
    }

    #[test]
    fn test_canary_lookahead() {
        let config = create_canary_config("lookahead", "de".to_string());
        assert_eq!(config.buffer_size_secs, 10.0);
        assert!(config.pause_based_confirm);
    }

    #[test]
    fn test_canary_default_fallback() {
        let config = create_canary_config("unknown_mode", "de".to_string());
        assert_eq!(config.buffer_size_secs, 8.0);
        assert!(config.pause_based_confirm);
    }

    #[test]
    fn test_canary_all_modes_have_valid_intervals() {
        let modes = ["speedy", "pause_based", "low_latency", "ultra_low_latency",
                      "extreme_low_latency", "lookahead"];
        for mode in modes {
            let config = create_canary_config(mode, "de".to_string());
            assert!(config.process_interval_secs > 0.0, "Mode {} has invalid interval", mode);
            assert!(config.buffer_size_secs > 0.0, "Mode {} has invalid buffer", mode);
            assert!(config.min_audio_secs > 0.0, "Mode {} has invalid min_audio", mode);
        }
    }

    // ========================================================================
    // TDT config
    // ========================================================================

    #[test]
    fn test_tdt_speedy() {
        let config = create_transcription_config("speedy", None);
        assert_eq!(config.buffer_size_secs, 8.0);
        assert_eq!(config.process_interval_secs, 0.2);
        assert!(config.pause_based_confirm);
        assert!(!config.lookahead_mode);
        assert_eq!(config.pause_threshold_secs, 0.6);
    }

    #[test]
    fn test_tdt_lookahead() {
        let config = create_transcription_config("lookahead", None);
        assert!(config.lookahead_mode);
        assert_eq!(config.lookahead_segments, 2);
    }

    #[test]
    fn test_tdt_with_pause_config_override() {
        let pause = PauseConfig {
            pause_threshold_ms: 800,
            silence_energy_threshold: 0.01,
            max_segment_secs: 10.0,
            context_buffer_secs: 3.0,
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
        assert_eq!(config.pause_threshold_secs, 0.5);
    }

    #[test]
    fn test_tdt_all_modes_consistency() {
        let modes = ["speedy", "pause_based", "low_latency", "ultra_low_latency",
                      "extreme_low_latency", "lookahead"];
        for mode in modes {
            let config = create_transcription_config(mode, None);
            assert!(config.process_interval_secs > 0.0, "Mode {} has invalid interval", mode);
            assert!(config.buffer_size_secs > 0.0, "Mode {} has invalid buffer", mode);
            assert_eq!(config.lookahead_segments, 2, "Mode {} should have 2 lookahead segments", mode);
        }
    }

    #[test]
    fn test_tdt_latency_ordering() {
        let speedy = create_transcription_config("speedy", None);
        let extreme = create_transcription_config("extreme_low_latency", None);
        assert!(extreme.buffer_size_secs <= speedy.buffer_size_secs);
    }
}
