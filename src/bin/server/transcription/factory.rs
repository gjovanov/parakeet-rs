//! Transcriber factory - creates the appropriate StreamingTranscriber based on mode and model

use crate::api::sessions::{GrowingSegmentsConfig, PauseConfig};
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
    #[cfg(feature = "whisper")]
    pub is_whisper: bool,
    #[cfg(feature = "whisper")]
    pub whisper_model_id: Option<String>,
    pub mode: String,
    pub language: String,
    pub pause_config: Option<PauseConfig>,
    pub growing_segments_config: Option<GrowingSegmentsConfig>,
}

/// Create the appropriate transcriber based on mode and model
pub fn create_transcriber(params: TranscriberParams) -> Option<Box<dyn StreamingTranscriber>> {
    let session_id = params.session_id.clone();
    let mode = params.mode.clone();

    #[cfg(feature = "whisper")]
    let is_whisper = params.is_whisper;
    #[cfg(not(feature = "whisper"))]
    let is_whisper = false;

    let is_canary = params.is_canary;
    let model_type = if is_whisper { "Whisper" } else { model_type_name(is_canary) };

    eprintln!(
        "[Session {}] Using transcription mode: {} (model: {})",
        session_id, mode, model_type
    );

    // Dispatch: Whisper first, then pause_segmented, then standard modes
    #[cfg(feature = "whisper")]
    if is_whisper {
        let transcriber = if mode == "pause_segmented" {
            create_pause_segmented_whisper(&params)
        } else {
            create_whisper_realtime(&params)
        };
        if transcriber.is_none() {
            eprintln!("[Session {}] Failed to create Whisper transcriber", session_id);
        }
        return transcriber;
    }

    let TranscriberParams {
        session_id,
        model_path,
        diar_path,
        exec_config,
        is_canary,
        mode,
        language,
        pause_config,
        growing_segments_config,
        ..
    } = params;

    let transcriber = if mode == "pause_segmented" && is_canary {
        create_pause_segmented_canary(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            &language, pause_config.as_ref(),
        )
    } else if mode == "pause_segmented" {
        create_pause_segmented_tdt(
            &session_id, &model_path, diar_path.as_ref(), exec_config,
            pause_config.as_ref(),
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

    if transcriber.is_none() {
        eprintln!("[Session {}] Failed to create {} transcriber", session_id, model_type);
    }
    transcriber
}

pub fn model_type_name(is_canary: bool) -> &'static str {
    if is_canary { "Canary" } else { "TDT" }
}

// ============================================================================
// Canary-1B modes
// ============================================================================

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

fn create_pause_segmented_canary(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    language: &str,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::{PauseSegmentedCanary, PauseSegmentedConfig};

    let mut config = PauseSegmentedConfig {
        language: language.to_string(),
        ..Default::default()
    };
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 { config.context_segments = pc.context_segments; }
        if let Some(v) = pc.min_segment_secs { config.min_segment_secs = v; }
        if let Some(v) = pc.partial_interval_secs { config.partial_interval_secs = v; }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented Canary (language: {}, pause: {:.0}ms, energy: {:.4}, max_seg: {:.0}s, min_seg: {:.1}s, partial: {:.1}s, ctx_seg: {}, diar: {:?})",
        session_id, language, config.pause_threshold_secs * 1000.0, config.silence_energy_threshold,
        config.max_segment_secs, config.min_segment_secs, config.partial_interval_secs,
        config.context_segments, diar_path
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

// ============================================================================
// TDT modes
// ============================================================================

fn create_tdt(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    mode: &str,
    pause_config: Option<&PauseConfig>,
    gs_config: Option<&GrowingSegmentsConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    let mut tdt_config = create_transcription_config(mode, pause_config);

    if mode == "growing_segments" {
        if let Some(gs) = gs_config {
            if let Some(v) = gs.buffer_size_secs { tdt_config.buffer_size_secs = v; }
            if let Some(v) = gs.process_interval_secs { tdt_config.process_interval_secs = v; }
            if let Some(v) = gs.pause_threshold_ms { tdt_config.pause_threshold_secs = v as f32 / 1000.0; }
            if let Some(v) = gs.silence_energy_threshold { tdt_config.silence_energy_threshold = v; }
        }
    }

    eprintln!(
        "[Session {}] Creating TDT transcriber from {:?} (mode: {}, diar: {:?})",
        session_id, model_path, mode, diar_path
    );

    #[cfg(feature = "sortformer")]
    let result = {
        let diar_path = match diar_path {
            Some(p) => p,
            None => {
                eprintln!("[Session {}] TDT requires diarization model (--diar-model)", session_id);
                return None;
            }
        };
        parakeet_rs::RealtimeTDTDiarized::new(
            model_path, &diar_path, Some(exec_config), Some(tdt_config),
        )
    };

    #[cfg(not(feature = "sortformer"))]
    let result = parakeet_rs::RealtimeTDT::new(model_path, Some(exec_config), Some(tdt_config));

    match result {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create TDT transcriber: {}", session_id, e);
            None
        }
    }
}

fn create_pause_segmented_tdt(
    session_id: &str,
    model_path: &PathBuf,
    diar_path: Option<&PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    pause_config: Option<&PauseConfig>,
) -> Option<Box<dyn StreamingTranscriber>> {
    use parakeet_rs::pause_segmented::PauseSegmentedConfig;
    use parakeet_rs::pause_segmented_tdt::PauseSegmentedTDT;

    let mut config = PauseSegmentedConfig {
        context_segments: 1,
        ..Default::default()
    };
    if let Some(pc) = pause_config {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 { config.context_segments = pc.context_segments; }
        if let Some(v) = pc.min_segment_secs { config.min_segment_secs = v; }
        if let Some(v) = pc.partial_interval_secs { config.partial_interval_secs = v; }
    }

    eprintln!(
        "[Session {}] Creating Pause-Segmented TDT (pause: {:.0}ms, energy: {:.4}, max_seg: {:.0}s, min_seg: {:.1}s, partial: {:.1}s, ctx_seg: {}, diar: {:?})",
        session_id, config.pause_threshold_secs * 1000.0, config.silence_energy_threshold,
        config.max_segment_secs, config.min_segment_secs, config.partial_interval_secs,
        config.context_segments, diar_path
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

// ============================================================================
// Whisper modes
// ============================================================================

#[cfg(feature = "whisper")]
fn create_whisper_realtime(params: &TranscriberParams) -> Option<Box<dyn StreamingTranscriber>> {
    use super::configs::create_whisper_config;

    let config = create_whisper_config(&params.mode, params.language.clone());

    // Apply growing segments overrides
    let config = if params.mode == "growing_segments" {
        if let Some(gs) = params.growing_segments_config.as_ref() {
            let mut c = config;
            if let Some(v) = gs.buffer_size_secs { c.buffer_size_secs = v; }
            if let Some(v) = gs.process_interval_secs { c.process_interval_secs = v; }
            if let Some(v) = gs.pause_threshold_ms { c.pause_threshold_secs = v as f32 / 1000.0; }
            if let Some(v) = gs.silence_energy_threshold { c.silence_energy_threshold = v; }
            if let Some(v) = gs.emit_full_text { c.emit_full_text = v; }
            if let Some(v) = gs.min_stable_count { c.min_stable_count = Some(v); }
            c
        } else {
            config
        }
    } else {
        config
    };

    // Determine GPU from execution config
    let use_gpu = {
        #[cfg(feature = "cuda")]
        { matches!(params.exec_config.execution_provider, parakeet_rs::ExecutionProvider::Cuda) }
        #[cfg(not(feature = "cuda"))]
        { false }
    };
    let config = parakeet_rs::RealtimeWhisperConfig {
        use_gpu,
        ..config
    };

    let model_id = params.whisper_model_id.clone();

    eprintln!(
        "[Session {}] Creating Whisper transcriber from {:?} (language: {}, mode: {}, gpu: {})",
        params.session_id, params.model_path, params.language, params.mode, use_gpu
    );

    match parakeet_rs::RealtimeWhisper::new(&params.model_path, Some(config), model_id) {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create Whisper transcriber: {}", params.session_id, e);
            None
        }
    }
}

#[cfg(feature = "whisper")]
fn create_pause_segmented_whisper(params: &TranscriberParams) -> Option<Box<dyn StreamingTranscriber>> {
    let mut config = parakeet_rs::PauseSegmentedConfig {
        language: params.language.clone(),
        ..Default::default()
    };
    if let Some(pc) = params.pause_config.as_ref() {
        config.pause_threshold_secs = pc.pause_threshold_ms as f32 / 1000.0;
        config.silence_energy_threshold = pc.silence_energy_threshold;
        config.max_segment_secs = pc.max_segment_secs;
        if pc.context_segments >= 1 {
            config.context_segments = pc.context_segments;
        }
        if let Some(v) = pc.min_segment_secs { config.min_segment_secs = v; }
        if let Some(v) = pc.partial_interval_secs { config.partial_interval_secs = v; }
    }

    let use_gpu = {
        #[cfg(feature = "cuda")]
        { matches!(params.exec_config.execution_provider, parakeet_rs::ExecutionProvider::Cuda) }
        #[cfg(not(feature = "cuda"))]
        { false }
    };

    let model_id = params.whisper_model_id.clone();

    eprintln!(
        "[Session {}] Creating Pause-Segmented Whisper (language: {}, pause: {:.0}ms, gpu: {})",
        params.session_id, params.language, config.pause_threshold_secs * 1000.0, use_gpu
    );

    match parakeet_rs::PauseSegmentedWhisper::new(
        &params.model_path,
        Some(config),
        model_id,
        Some(5), // beam_size
        Some(4), // n_threads
        use_gpu,
    ) {
        Ok(t) => Some(Box::new(t)),
        Err(e) => {
            eprintln!("[Session {}] Failed to create Pause-Segmented Whisper: {}", params.session_id, e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_name() {
        assert_eq!(model_type_name(true), "Canary");
        assert_eq!(model_type_name(false), "TDT");
    }
}
