//! Transcriber factory - creates the appropriate StreamingTranscriber based on mode and model

use crate::api::sessions::{ParallelConfig, PauseConfig};
use parakeet_rs::streaming_transcriber::StreamingTranscriber;
use std::path::PathBuf;

use super::configs::{create_canary_config, create_transcription_config};

/// Parameters for creating a transcriber
pub struct TranscriberParams {
    pub session_id: String,
    pub model_path: PathBuf,
    pub diar_path: Option<PathBuf>,
    pub exec_config: parakeet_rs::ExecutionConfig,
    pub is_canary: bool,
    pub is_canary_flash: bool,
    pub is_vad_mode: bool,
    pub mode: String,
    pub vad_base_mode: String,
    pub vad_model_path: String,
    pub language: String,
    pub parallel_config: Option<ParallelConfig>,
    pub pause_config: Option<PauseConfig>,
}

/// Create the appropriate transcriber based on model type and mode.
/// Returns None if creation fails (error is logged).
pub fn create_transcriber(params: TranscriberParams) -> Option<Box<dyn StreamingTranscriber>> {
    let TranscriberParams {
        session_id,
        model_path,
        diar_path,
        exec_config,
        is_canary,
        is_canary_flash,
        is_vad_mode,
        mode,
        vad_base_mode,
        vad_model_path,
        language,
        parallel_config,
        pause_config,
    } = params;

    let is_parallel_mode = mode == "parallel";
    let is_pause_parallel_mode = mode == "pause_parallel";

    if is_pause_parallel_mode {
        create_pause_parallel(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            is_canary, &language, &parallel_config, &pause_config,
        )
    } else if is_parallel_mode {
        create_parallel(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            is_canary, &language, &parallel_config,
        )
    } else if is_vad_mode {
        create_vad(
            &session_id, &model_path, diar_path, exec_config,
            is_canary, &vad_base_mode, &vad_model_path, &language,
        )
    } else if is_canary_flash {
        create_canary_flash(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &mode, &language,
        )
    } else if is_canary {
        create_canary(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &mode, &language,
        )
    } else {
        create_tdt(
            &session_id, &model_path, diar_path, exec_config,
            &mode, pause_config.as_ref(),
        )
    }
}

/// Get the display name for the model type
pub fn model_type_name(
    is_pause_parallel_mode: bool,
    is_parallel_mode: bool,
    is_vad_mode: bool,
    is_canary_flash: bool,
    is_canary: bool,
) -> &'static str {
    if is_pause_parallel_mode {
        if is_canary { "PauseParallelCanary" } else { "PauseParallelTDT" }
    } else if is_parallel_mode {
        if is_canary { "ParallelCanary" } else { "ParallelTDT" }
    } else {
        match (is_vad_mode, is_canary_flash, is_canary) {
            (true, _, true) => "VAD+Canary",
            (true, _, false) => "VAD+TDT",
            (false, true, _) => "CanaryFlash",
            (false, false, true) => "Canary",
            (false, false, false) => "TDT",
        }
    }
}

fn create_pause_parallel(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    is_canary: bool,
    language: &str,
    parallel_config: &Option<ParallelConfig>,
    pause_config: &Option<PauseConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    let num_threads = match parallel_config {
        Some(cfg) => cfg.num_threads,
        None => if is_canary { 8 } else { 4 },
    };

    let pause_threshold_secs = pause_config.as_ref()
        .map(|p| p.pause_threshold_ms as f32 / 1000.0)
        .unwrap_or(0.5);
    let silence_energy = pause_config.as_ref()
        .map(|p| p.silence_energy_threshold)
        .unwrap_or(0.008);
    let max_segment_secs = pause_config.as_ref()
        .map(|p| p.max_segment_secs)
        .unwrap_or(6.0);
    let context_buffer = pause_config.as_ref()
        .map(|p| p.context_buffer_secs)
        .unwrap_or(2.0);

    if is_canary {
        use parakeet_rs::pause_parallel_canary::{PauseParallelCanary, PauseParallelConfig};

        let config = PauseParallelConfig {
            num_threads,
            language: language.to_string(),
            intra_threads: 1,
            pause_threshold_secs,
            silence_energy_threshold: silence_energy,
            max_segment_duration_secs: max_segment_secs,
            context_buffer_secs: context_buffer,
        };

        eprintln!(
            "[Session {}] Creating PauseParallelCanary transcriber with {} threads, {}ms pause, {}s context (diar: {:?})",
            session_id, config.num_threads, (config.pause_threshold_secs * 1000.0) as u32, config.context_buffer_secs, diar_path
        );

        #[cfg(feature = "sortformer")]
        let result = PauseParallelCanary::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(config),
        );
        #[cfg(not(feature = "sortformer"))]
        let result = PauseParallelCanary::new(model_path, Some(exec_config), Some(config));

        match result {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create PauseParallelCanary transcriber: {}", session_id, e);
                None
            }
        }
    } else {
        use parakeet_rs::pause_parallel_tdt::{PauseParallelTDT, PauseParallelTDTConfig};

        let config = PauseParallelTDTConfig {
            num_threads,
            intra_threads: 2,
            pause_threshold_secs,
            silence_energy_threshold: silence_energy,
            max_segment_duration_secs: max_segment_secs,
            context_buffer_secs: context_buffer,
        };

        eprintln!(
            "[Session {}] Creating PauseParallelTDT transcriber with {} threads, {}ms pause, {}s context (diar: {:?})",
            session_id, config.num_threads, (config.pause_threshold_secs * 1000.0) as u32, config.context_buffer_secs, diar_path
        );

        #[cfg(feature = "sortformer")]
        let result = if diar_path.is_some() {
            PauseParallelTDT::new_with_diarization(
                model_path, diar_path, Some(exec_config), Some(config),
            )
        } else {
            PauseParallelTDT::new(model_path, Some(exec_config), Some(config))
        };

        #[cfg(not(feature = "sortformer"))]
        let result = PauseParallelTDT::new(model_path, Some(exec_config), Some(config));

        match result {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create PauseParallelTDT transcriber: {}", session_id, e);
                None
            }
        }
    }
}

fn create_parallel(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    is_canary: bool,
    language: &str,
    parallel_config: &Option<ParallelConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    let (num_threads, buffer_size) = match parallel_config {
        Some(cfg) => (cfg.num_threads, cfg.buffer_size_secs),
        None => if is_canary { (8, 6) } else { (4, 6) },
    };

    if is_canary {
        use parakeet_rs::parallel_canary::{ParallelCanary, ParallelCanaryConfig};

        let config = ParallelCanaryConfig {
            num_threads,
            buffer_size_chunks: buffer_size,
            chunk_duration_secs: 1.0,
            language: language.to_string(),
            intra_threads: 1,
        };

        eprintln!(
            "[Session {}] Creating ParallelCanary transcriber with {} threads, {}s buffer (diar: {:?})",
            session_id, config.num_threads, config.buffer_size_chunks, diar_path
        );

        #[cfg(feature = "sortformer")]
        let result = ParallelCanary::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(config),
        );
        #[cfg(not(feature = "sortformer"))]
        let result = ParallelCanary::new(model_path, Some(exec_config), Some(config));

        match result {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create ParallelCanary transcriber: {}", session_id, e);
                None
            }
        }
    } else {
        use parakeet_rs::parallel_tdt::{ParallelTDT, ParallelTDTConfig};

        let config = ParallelTDTConfig {
            num_threads,
            buffer_size_chunks: buffer_size,
            chunk_duration_secs: 1.0,
            intra_threads: 2,
        };

        eprintln!(
            "[Session {}] Creating ParallelTDT transcriber with {} threads, {}s buffer (diar: {:?})",
            session_id, config.num_threads, config.buffer_size_chunks, diar_path
        );

        #[cfg(feature = "sortformer")]
        let result = if diar_path.is_some() {
            ParallelTDT::new_with_diarization(
                model_path, diar_path, Some(exec_config), Some(config),
            )
        } else {
            ParallelTDT::new(model_path, Some(exec_config), Some(config))
        };

        #[cfg(not(feature = "sortformer"))]
        let result = ParallelTDT::new(model_path, Some(exec_config), Some(config));

        match result {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create ParallelTDT transcriber: {}", session_id, e);
                None
            }
        }
    }
}

fn create_vad(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    is_canary: bool,
    vad_base_mode: &str,
    vad_model_path: &str,
    language: &str,
) -> Option<Box<dyn StreamingTranscriber>> {
    if is_canary {
        use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};

        let config = if vad_base_mode == "sliding_window" {
            RealtimeCanaryVadConfig::sliding_window(language.to_string())
        } else {
            RealtimeCanaryVadConfig::buffered(language.to_string())
        };

        eprintln!(
            "[Session {}] Creating VAD+Canary transcriber from {:?} (language: {}, vad_mode: {}, diar: {:?})",
            session_id, model_path, language, vad_base_mode, diar_path
        );

        #[cfg(feature = "sortformer")]
        let result = RealtimeCanaryVad::new(
            model_path, diar_path.as_ref(), vad_model_path,
            Some(exec_config), Some(config),
        );
        #[cfg(not(feature = "sortformer"))]
        let result = RealtimeCanaryVad::new(
            model_path, vad_model_path, Some(exec_config), Some(config),
        );

        match result {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create VAD+Canary transcriber: {}", session_id, e);
                None
            }
        }
    } else {
        #[cfg(feature = "sortformer")]
        {
            use parakeet_rs::realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig};

            let diar_path = match diar_path {
                Some(p) => p,
                None => {
                    eprintln!("[Session {}] No diarization model configured for TDT", session_id);
                    return None;
                }
            };

            let config = if vad_base_mode == "sliding_window" {
                RealtimeTdtVadConfig::sliding_window()
            } else {
                RealtimeTdtVadConfig::from_mode(vad_base_mode)
            };

            eprintln!(
                "[Session {}] Creating VAD+TDT transcriber from {:?} (vad_mode: {})",
                session_id, model_path, vad_base_mode
            );

            match RealtimeTdtVad::new(
                model_path, Some(&diar_path), vad_model_path,
                Some(exec_config), Some(config),
            ) {
                Ok(t) => Some(Box::new(t)),
                Err(e) => {
                    eprintln!("[Session {}] Failed to create VAD+TDT transcriber: {}", session_id, e);
                    None
                }
            }
        }
        #[cfg(not(feature = "sortformer"))]
        {
            eprintln!("[Session {}] VAD+TDT mode requires sortformer feature", session_id);
            None
        }
    }
}

fn create_canary_flash(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    mode: &str,
    language: &str,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::realtime_canary_flash::{RealtimeCanaryFlash, RealtimeCanaryFlashConfig};

    let flash_config = RealtimeCanaryFlashConfig {
        buffer_size_secs: 8.0,
        min_audio_secs: 1.0,
        process_interval_secs: 0.5,
        language: language.to_string(),
    };

    eprintln!(
        "[Session {}] Creating Canary Flash transcriber from {:?} (language: {}, mode: {}, diar: {:?})",
        session_id, model_path, language, mode, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        RealtimeCanaryFlash::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(flash_config),
        )
    } else {
        RealtimeCanaryFlash::new(model_path, Some(exec_config), Some(flash_config))
    };

    #[cfg(not(feature = "sortformer"))]
    let result = RealtimeCanaryFlash::new(model_path, Some(exec_config), Some(flash_config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create Canary Flash transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_canary(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    mode: &str,
    language: &str,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::realtime_canary::RealtimeCanary;

    let canary_config = create_canary_config(mode, language.to_string());

    eprintln!(
        "[Session {}] Creating Canary transcriber from {:?} (language: {}, mode: {}, diar: {:?})",
        session_id, model_path, language, mode, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        RealtimeCanary::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(canary_config),
        )
    } else {
        RealtimeCanary::new(model_path, Some(exec_config), Some(canary_config))
    };

    #[cfg(not(feature = "sortformer"))]
    let result = RealtimeCanary::new(model_path, Some(exec_config), Some(canary_config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create Canary transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_tdt(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    mode: &str,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    #[cfg(feature = "sortformer")]
    {
        use parakeet_rs::RealtimeTDTDiarized;

        let diar_path = match diar_path {
            Some(p) => p,
            None => {
                eprintln!("[Session {}] No diarization model configured for TDT", session_id);
                return None;
            }
        };

        let config = create_transcription_config(mode, pause_config);

        eprintln!(
            "[Session {}] Creating TDT transcriber from {:?} (pause: {}ms)",
            session_id, model_path, (config.pause_threshold_secs * 1000.0) as u32
        );

        match RealtimeTDTDiarized::new(model_path, &diar_path, Some(exec_config), Some(config)) {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create TDT transcriber: {}", session_id, e);
                None
            }
        }
    }
    #[cfg(not(feature = "sortformer"))]
    {
        use parakeet_rs::parallel_tdt::{ParallelTDT, ParallelTDTConfig};

        let config = ParallelTDTConfig {
            num_threads: 1,
            buffer_size_chunks: 6,
            chunk_duration_secs: 1.0,
            intra_threads: 4,
        };

        eprintln!(
            "[Session {}] Creating TDT transcriber from {:?} (single-thread fallback, no diarization)",
            session_id, model_path
        );

        match ParallelTDT::new(model_path, Some(exec_config), Some(config)) {
            Ok(t) => Some(Box::new(t)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create TDT transcriber: {}", session_id, e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_name_pause_parallel_canary() {
        assert_eq!(model_type_name(true, false, false, false, true), "PauseParallelCanary");
    }

    #[test]
    fn test_model_type_name_pause_parallel_tdt() {
        assert_eq!(model_type_name(true, false, false, false, false), "PauseParallelTDT");
    }

    #[test]
    fn test_model_type_name_parallel_canary() {
        assert_eq!(model_type_name(false, true, false, false, true), "ParallelCanary");
    }

    #[test]
    fn test_model_type_name_parallel_tdt() {
        assert_eq!(model_type_name(false, true, false, false, false), "ParallelTDT");
    }

    #[test]
    fn test_model_type_name_vad_canary() {
        assert_eq!(model_type_name(false, false, true, false, true), "VAD+Canary");
    }

    #[test]
    fn test_model_type_name_vad_tdt() {
        assert_eq!(model_type_name(false, false, true, false, false), "VAD+TDT");
    }

    #[test]
    fn test_model_type_name_canary_flash() {
        assert_eq!(model_type_name(false, false, false, true, false), "CanaryFlash");
    }

    #[test]
    fn test_model_type_name_canary() {
        assert_eq!(model_type_name(false, false, false, false, true), "Canary");
    }

    #[test]
    fn test_model_type_name_tdt() {
        assert_eq!(model_type_name(false, false, false, false, false), "TDT");
    }
}
