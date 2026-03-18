//! Transcriber factory - creates the appropriate StreamingTranscriber based on mode and model

use crate::api::sessions::{GrowingSegmentsConfig, ParallelConfig, PauseConfig};
use parakeet_rs::streaming_transcriber::StreamingTranscriber;
use std::path::PathBuf;

use super::configs::{create_canary_config, create_canary_qwen_config, create_transcription_config};

/// Parameters for creating a transcriber
pub struct TranscriberParams {
    pub session_id: String,
    pub model_path: PathBuf,
    pub diar_path: Option<PathBuf>,
    pub exec_config: parakeet_rs::ExecutionConfig,
    pub is_canary: bool,
    pub is_canary_flash: bool,
    pub is_canary_qwen: bool,
    pub is_voxtral: bool,
    pub is_vad_mode: bool,
    pub mode: String,
    pub vad_base_mode: String,
    pub vad_model_path: String,
    pub language: String,
    pub parallel_config: Option<ParallelConfig>,
    pub pause_config: Option<PauseConfig>,
    pub growing_segments_config: Option<GrowingSegmentsConfig>,
    /// Enable text formatting on FINAL segments
    pub enable_formatting: bool,
    /// Formatting tone hint
    pub formatting_tone: String,
    /// User vocabulary for formatting
    pub formatting_vocabulary: Vec<String>,
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
        is_canary_qwen,
        is_voxtral,
        is_vad_mode,
        mode,
        vad_base_mode,
        vad_model_path,
        language,
        parallel_config,
        pause_config,
        growing_segments_config,
        enable_formatting,
        formatting_tone,
        formatting_vocabulary,
    } = params;

    let is_parallel_mode = mode == "parallel";
    let is_pause_parallel_mode = mode == "pause_parallel";

    let transcriber = if is_canary_qwen && is_pause_parallel_mode {
        create_canary_qwen_pause_parallel(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &language, &parallel_config, &pause_config,
        )
    } else if is_canary_qwen && is_parallel_mode {
        create_canary_qwen_parallel(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &language, &parallel_config,
        )
    } else if is_canary_qwen && is_vad_mode {
        create_canary_qwen_vad(
            &session_id, &model_path, diar_path, exec_config,
            &vad_base_mode, &vad_model_path, &language,
        )
    } else if is_canary_qwen && mode == "pause_segmented" {
        create_pause_segmented_canary_qwen(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &language, pause_config.as_ref(),
        )
    } else if is_canary_qwen {
        create_canary_qwen(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &mode, &language, growing_segments_config.as_ref(),
        )
    } else if is_pause_parallel_mode {
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
    } else if mode == "pause_segmented" && is_canary {
        create_pause_segmented_canary(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &language, pause_config.as_ref(),
        )
    } else if mode == "voxtral_streaming" && is_voxtral {
        #[cfg(feature = "voxtral")]
        {
            create_voxtral_streaming(
                &session_id, &model_path, exec_config, &language,
            )
        }
        #[cfg(not(feature = "voxtral"))]
        { eprintln!("[Session {}] Voxtral feature not enabled", session_id); None }
    } else if mode == "pause_segmented" && is_voxtral {
        #[cfg(feature = "voxtral")]
        {
            create_pause_segmented_voxtral(
                &session_id, &model_path, diar_path.as_ref(), exec_config,
                &language, pause_config.as_ref(),
            )
        }
        #[cfg(not(feature = "voxtral"))]
        {
            eprintln!("[Session {}] Voxtral feature not enabled", session_id);
            None
        }
    } else if mode == "pause_segmented" {
        create_pause_segmented_tdt(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            pause_config.as_ref(),
        )
    } else if is_canary_flash {
        create_canary_flash(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &mode, &language,
        )
    } else if is_canary {
        create_canary(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &mode, &language, growing_segments_config.as_ref(),
        )
    } else {
        create_tdt(
            &session_id, &model_path, diar_path, exec_config,
            &mode, pause_config.as_ref(), growing_segments_config.as_ref(),
        )
    };

    // Optionally wrap with FormattingTranscriber
    if enable_formatting {
        if let Some(inner) = transcriber {
            use parakeet_rs::text_formatter::{FormattingContext, FormattingTone, LlmFormatter, RuleBasedFormatter};
            use parakeet_rs::formatting_transcriber::FormattingTranscriber;

            let context = FormattingContext {
                tone: FormattingTone::from_str(&formatting_tone),
                language: language.clone(),
                vocabulary: formatting_vocabulary,
                recent_text: vec![],
            };

            // Use LlmFormatter when FORMATTER_MODEL_PATH is set and the path exists,
            // otherwise fall back to RuleBasedFormatter
            let formatter: Box<dyn parakeet_rs::text_formatter::TextFormatter> =
                match std::env::var("FORMATTER_MODEL_PATH") {
                    Ok(path) if std::path::Path::new(&path).exists() => {
                        eprintln!(
                            "[Session {}] Formatter LLM model found at {}, using LlmFormatter",
                            session_id, path
                        );
                        Box::new(LlmFormatter::new(path))
                    }
                    _ => {
                        Box::new(RuleBasedFormatter::new())
                    }
                };

            eprintln!(
                "[Session {}] Wrapping transcriber with {} formatter (tone: {})",
                session_id, formatter.name(), formatting_tone
            );

            Some(Box::new(FormattingTranscriber::new(
                inner,
                formatter,
                context,
                std::time::Duration::from_millis(
                    std::env::var("FORMATTER_TIMEOUT_MS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(5000u64),
                ),
            )))
        } else {
            None
        }
    } else {
        transcriber
    }
}

/// Get the display name for the model type
pub fn model_type_name(
    is_pause_parallel_mode: bool,
    is_parallel_mode: bool,
    is_vad_mode: bool,
    is_canary_qwen: bool,
    is_canary_flash: bool,
    is_canary: bool,
) -> &'static str {
    if is_canary_qwen {
        if is_pause_parallel_mode { "PauseParallelCanaryQwen" }
        else if is_parallel_mode { "ParallelCanaryQwen" }
        else if is_vad_mode { "VAD+CanaryQwen" }
        else { "CanaryQwen" }
    } else if is_pause_parallel_mode {
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
    gs_config: Option<&GrowingSegmentsConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::realtime_canary::RealtimeCanary;

    let mut canary_config = create_canary_config(mode, language.to_string());

    // Apply growing segments overrides
    if mode == "growing_segments" {
        if let Some(gs) = gs_config {
            if let Some(v) = gs.buffer_size_secs { canary_config.buffer_size_secs = v; }
            if let Some(v) = gs.process_interval_secs { canary_config.process_interval_secs = v; }
            if let Some(v) = gs.pause_threshold_ms { canary_config.pause_threshold_secs = v as f32 / 1000.0; }
            if let Some(v) = gs.silence_energy_threshold { canary_config.silence_energy_threshold = v; }
            if let Some(v) = gs.emit_full_text { canary_config.emit_full_text = v; }
            if let Some(v) = gs.min_stable_count { canary_config.min_stable_count = Some(v); }
        }
    }

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
    gs_config: Option<&GrowingSegmentsConfig>,
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

        let mut config = create_transcription_config(mode, pause_config);

        // Apply growing segments overrides (TDT uses buffer_size_secs, not buffer_duration_secs)
        if mode == "growing_segments" {
            if let Some(gs) = gs_config {
                if let Some(v) = gs.buffer_size_secs { config.buffer_size_secs = v; }
                if let Some(v) = gs.process_interval_secs { config.process_interval_secs = v; }
                if let Some(v) = gs.pause_threshold_ms { config.pause_threshold_secs = v as f32 / 1000.0; }
                if let Some(v) = gs.silence_energy_threshold { config.silence_energy_threshold = v; }
            }
        }

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

fn create_canary_qwen(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    mode: &str,
    language: &str,
    gs_config: Option<&GrowingSegmentsConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::realtime_canary_qwen::RealtimeCanaryQwen;

    let mut qwen_config = create_canary_qwen_config(mode, language.to_string());

    // Apply growing segments overrides
    if mode == "growing_segments" {
        if let Some(gs) = gs_config {
            if let Some(v) = gs.buffer_size_secs { qwen_config.buffer_size_secs = v; }
            if let Some(v) = gs.process_interval_secs { qwen_config.process_interval_secs = v; }
            if let Some(v) = gs.pause_threshold_ms { qwen_config.pause_threshold_secs = v as f32 / 1000.0; }
            if let Some(v) = gs.silence_energy_threshold { qwen_config.silence_energy_threshold = v; }
        }
    }

    eprintln!(
        "[Session {}] Creating CanaryQwen transcriber from {:?} (language: {}, mode: {}, diar: {:?})",
        session_id, model_path, language, mode, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        RealtimeCanaryQwen::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(qwen_config),
        )
    } else {
        RealtimeCanaryQwen::new(model_path, Some(exec_config), Some(qwen_config))
    };

    #[cfg(not(feature = "sortformer"))]
    let result = RealtimeCanaryQwen::new(model_path, Some(exec_config), Some(qwen_config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create CanaryQwen transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_canary_qwen_vad(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    vad_base_mode: &str,
    vad_model_path: &str,
    language: &str,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::realtime_canary_qwen_vad::{RealtimeCanaryQwenVad, RealtimeCanaryQwenVadConfig};

    let config = if vad_base_mode == "sliding_window" {
        RealtimeCanaryQwenVadConfig::sliding_window(language.to_string())
    } else {
        RealtimeCanaryQwenVadConfig::buffered(language.to_string())
    };

    eprintln!(
        "[Session {}] Creating VAD+CanaryQwen transcriber from {:?} (language: {}, vad_mode: {}, diar: {:?})",
        session_id, model_path, language, vad_base_mode, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = RealtimeCanaryQwenVad::new(
        model_path, diar_path.as_ref(), vad_model_path,
        Some(exec_config), Some(config),
    );
    #[cfg(not(feature = "sortformer"))]
    let result = RealtimeCanaryQwenVad::new(
        model_path, vad_model_path, Some(exec_config), Some(config),
    );

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create VAD+CanaryQwen transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_canary_qwen_parallel(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    parallel_config: &Option<ParallelConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::parallel_canary_qwen::{ParallelCanaryQwen, ParallelCanaryQwenConfig};

    let (num_threads, buffer_size) = match parallel_config {
        Some(cfg) => (cfg.num_threads, cfg.buffer_size_secs),
        None => (4, 6),
    };

    let config = ParallelCanaryQwenConfig {
        num_threads,
        buffer_size_chunks: buffer_size,
        chunk_duration_secs: 1.0,
        language: language.to_string(),
        intra_threads: 1,
    };

    eprintln!(
        "[Session {}] Creating ParallelCanaryQwen transcriber with {} threads, {}s buffer (diar: {:?})",
        session_id, config.num_threads, config.buffer_size_chunks, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = ParallelCanaryQwen::new_with_diarization(
        model_path, diar_path, Some(exec_config), Some(config),
    );
    #[cfg(not(feature = "sortformer"))]
    let result = ParallelCanaryQwen::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create ParallelCanaryQwen transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_canary_qwen_pause_parallel(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    parallel_config: &Option<ParallelConfig>,
    pause_config: &Option<PauseConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::pause_parallel_canary_qwen::{PauseParallelCanaryQwen, PauseParallelCanaryQwenConfig};

    let num_threads = match parallel_config {
        Some(cfg) => cfg.num_threads,
        None => 4,
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

    let config = PauseParallelCanaryQwenConfig {
        num_threads,
        language: language.to_string(),
        intra_threads: 1,
        pause_threshold_secs,
        silence_energy_threshold: silence_energy,
        max_segment_duration_secs: max_segment_secs,
        context_buffer_secs: context_buffer,
    };

    eprintln!(
        "[Session {}] Creating PauseParallelCanaryQwen transcriber with {} threads, {}ms pause (diar: {:?})",
        session_id, config.num_threads, (config.pause_threshold_secs * 1000.0) as u32, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = PauseParallelCanaryQwen::new_with_diarization(
        model_path, diar_path, Some(exec_config), Some(config),
    );
    #[cfg(not(feature = "sortformer"))]
    let result = PauseParallelCanaryQwen::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create PauseParallelCanaryQwen transcriber: {}", session_id, e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_name_canary_qwen() {
        assert_eq!(model_type_name(false, false, false, true, false, false), "CanaryQwen");
    }

    #[test]
    fn test_model_type_name_canary_qwen_vad() {
        assert_eq!(model_type_name(false, false, true, true, false, false), "VAD+CanaryQwen");
    }

    #[test]
    fn test_model_type_name_canary_qwen_parallel() {
        assert_eq!(model_type_name(false, true, false, true, false, false), "ParallelCanaryQwen");
    }

    #[test]
    fn test_model_type_name_canary_qwen_pause_parallel() {
        assert_eq!(model_type_name(true, false, false, true, false, false), "PauseParallelCanaryQwen");
    }

    #[test]
    fn test_model_type_name_pause_parallel_canary() {
        assert_eq!(model_type_name(true, false, false, false, false, true), "PauseParallelCanary");
    }

    #[test]
    fn test_model_type_name_pause_parallel_tdt() {
        assert_eq!(model_type_name(true, false, false, false, false, false), "PauseParallelTDT");
    }

    #[test]
    fn test_model_type_name_parallel_canary() {
        assert_eq!(model_type_name(false, true, false, false, false, true), "ParallelCanary");
    }

    #[test]
    fn test_model_type_name_parallel_tdt() {
        assert_eq!(model_type_name(false, true, false, false, false, false), "ParallelTDT");
    }

    #[test]
    fn test_model_type_name_vad_canary() {
        assert_eq!(model_type_name(false, false, true, false, false, true), "VAD+Canary");
    }

    #[test]
    fn test_model_type_name_vad_tdt() {
        assert_eq!(model_type_name(false, false, true, false, false, false), "VAD+TDT");
    }

    #[test]
    fn test_model_type_name_canary_flash() {
        assert_eq!(model_type_name(false, false, false, false, true, false), "CanaryFlash");
    }

    #[test]
    fn test_model_type_name_canary() {
        assert_eq!(model_type_name(false, false, false, false, false, true), "Canary");
    }

    #[test]
    fn test_model_type_name_tdt() {
        assert_eq!(model_type_name(false, false, false, false, false, false), "TDT");
    }
}

fn create_pause_segmented_canary(
    session_id: &str,
    model_path: &std::path::PathBuf,
    diar_path: Option<&std::path::PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn parakeet_rs::streaming_transcriber::StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::{PauseSegmentedCanary, PauseSegmentedConfig};

    let mut config = PauseSegmentedConfig {
        language: language.to_string(),
        ..Default::default()
    };

    // Apply pause config overrides
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 {
            config.context_segments = pc.context_segments;
        }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented Canary (language: {}, pause: {:.0}ms, max_seg: {:.0}s, ctx_seg: {}, diar: {:?})",
        session_id, language, config.pause_threshold_secs * 1000.0, config.max_segment_secs, config.context_segments, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        PauseSegmentedCanary::new_with_diarization(
            model_path, diar_path, Some(exec_config), Some(config),
        )
    } else {
        PauseSegmentedCanary::new(model_path, Some(exec_config), Some(config))
    };

    #[cfg(not(feature = "sortformer"))]
    let result = PauseSegmentedCanary::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create Pause-Segmented Canary: {}", session_id, e);
            None
        }
    }
}

fn create_pause_segmented_canary_qwen(
    session_id: &str,
    model_path: &std::path::PathBuf,
    diar_path: Option<&std::path::PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn parakeet_rs::streaming_transcriber::StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::PauseSegmentedConfig;
    use parakeet_rs::pause_segmented_canary_qwen::PauseSegmentedCanaryQwen;

    let mut config = PauseSegmentedConfig {
        language: language.to_string(),
        ..Default::default()
    };
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 { config.context_segments = pc.context_segments; }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented Canary-Qwen (language: {}, pause: {:.0}ms, ctx_seg: {}, diar: {:?})",
        session_id, language, config.pause_threshold_secs * 1000.0, config.context_segments, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        PauseSegmentedCanaryQwen::new_with_diarization(model_path, diar_path, Some(exec_config), Some(config))
    } else {
        PauseSegmentedCanaryQwen::new(model_path, Some(exec_config), Some(config))
    };
    #[cfg(not(feature = "sortformer"))]
    let result = PauseSegmentedCanaryQwen::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => { eprintln!("[Session {}] Failed to create Pause-Segmented Canary-Qwen: {}", session_id, e); None }
    }
}

fn create_pause_segmented_tdt(
    session_id: &str,
    model_path: &std::path::PathBuf,
    diar_path: Option<&std::path::PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn parakeet_rs::streaming_transcriber::StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::PauseSegmentedConfig;
    use parakeet_rs::pause_segmented_tdt::PauseSegmentedTDT;

    let mut config = PauseSegmentedConfig {
        context_segments: 1, // TDT benefits less from context
        ..Default::default()
    };
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 { config.context_segments = pc.context_segments; }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented TDT (pause: {:.0}ms, ctx_seg: {}, diar: {:?})",
        session_id, config.pause_threshold_secs * 1000.0, config.context_segments, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = if diar_path.is_some() {
        PauseSegmentedTDT::new_with_diarization(model_path, diar_path, Some(exec_config), Some(config))
    } else {
        PauseSegmentedTDT::new(model_path, Some(exec_config), Some(config))
    };
    #[cfg(not(feature = "sortformer"))]
    let result = PauseSegmentedTDT::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => { eprintln!("[Session {}] Failed to create Pause-Segmented TDT: {}", session_id, e); None }
    }
}

#[cfg(feature = "voxtral")]
fn create_pause_segmented_voxtral(
    session_id: &str,
    model_path: &std::path::PathBuf,
    diar_path: Option<&std::path::PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn parakeet_rs::streaming_transcriber::StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::PauseSegmentedConfig;
    use parakeet_rs::pause_segmented_voxtral::PauseSegmentedVoxtral;

    let mut config = PauseSegmentedConfig {
        language: language.to_string(),
        ..Default::default()
    };
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 { config.context_segments = pc.context_segments; }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented Voxtral (language: {}, pause: {:.0}ms, ctx_seg: {})",
        session_id, language, config.pause_threshold_secs * 1000.0, config.context_segments
    );

    let result = PauseSegmentedVoxtral::new(model_path, Some(exec_config), Some(config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => { eprintln!("[Session {}] Failed to create Pause-Segmented Voxtral: {}", session_id, e); None }
    }
}

#[cfg(feature = "voxtral")]
fn create_voxtral_streaming(
    session_id: &str,
    model_path: &std::path::PathBuf,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
) -> Option<Box<dyn parakeet_rs::streaming_transcriber::StreamingTranscriber>> {
    use parakeet_rs::voxtral_streaming::{VoxtralStreaming, VoxtralStreamingConfig};

    let config = VoxtralStreamingConfig {
        language: language.to_string(),
        ..Default::default()
    };

    eprintln!(
        "[Session {}] Creating Voxtral Streaming (language: {}, min_audio_tok: {}, delay: {})",
        session_id, language, config.min_audio_tokens, config.num_delay_tokens
    );

    match VoxtralStreaming::new(model_path, Some(exec_config), Some(config)) {
        Ok(t) => Some(Box::new(t)),
        Err(e) => { eprintln!("[Session {}] Failed: {}", session_id, e); None }
    }
}
