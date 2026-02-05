//! FAB Live Transcription Forwarder
//!
//! Subscribes to a session's subtitle broadcast channel and forwards
//! finalized transcription text to a FAB endpoint via HTTP GET.
//!
//! Includes deduplication and 500ms debounce to reduce request volume.

use std::collections::HashSet;
use tokio::sync::broadcast;
use tokio::time::{Duration, Instant};

const DEBOUNCE_MS: u64 = 500;
const SIMILARITY_THRESHOLD: f64 = 0.8;

/// Compute Jaccard word-overlap similarity between two strings.
fn text_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();
    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}

/// Send text to the FAB endpoint via HTTP GET.
async fn send_to_fab(
    client: &reqwest::Client,
    fab_url: &str,
    session_id: &str,
    text: &str,
) -> bool {
    let url = match reqwest::Url::parse_with_params(
        fab_url,
        &[("language", "Default"), ("text", text)],
    ) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[FAB] [{}] URL parse error: {}", session_id, e);
            return false;
        }
    };

    eprintln!("[FAB] [{}] Sending text: \"{}\"", session_id, &text.chars().take(120).collect::<String>());
    match client.get(url).send().await {
        Ok(resp) => {
            eprintln!("[FAB] [{}] Sent, status={}", session_id, resp.status());
            true
        }
        Err(e) => {
            eprintln!("[FAB] [{}] Send error: {}", session_id, e);
            false
        }
    }
}

/// Spawn an async task that forwards finalized subtitles to the FAB endpoint.
///
/// The task listens on `subtitle_rx` for JSON subtitle messages, filters for
/// `is_final: true`, extracts `text`, applies deduplication, and sends via
/// a 500ms debounce window to coalesce rapid-fire finals into single sends.
///
/// The task automatically stops when the broadcast channel is closed (session ends).
pub fn spawn_fab_forwarder(
    session_id: String,
    fab_url: String,
    mut subtitle_rx: broadcast::Receiver<String>,
    client: reqwest::Client,
) {
    tokio::spawn(async move {
        eprintln!("[FAB] Forwarder started for session {}", session_id);

        let mut last_sent_text = String::new();
        let mut last_full_transcript_len: usize = 0;
        let mut sent_count: u64 = 0;
        let mut skipped_count: u64 = 0;

        let mut pending_text: Option<String> = None;
        let mut debounce_deadline: Option<Instant> = None;

        loop {
            let sleep_fut = async {
                match debounce_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            };

            tokio::select! {
                result = subtitle_rx.recv() => {
                    match result {
                        Ok(json_str) => {
                            let msg: serde_json::Value = match serde_json::from_str(&json_str) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            if msg["is_final"].as_bool() != Some(true) {
                                continue;
                            }

                            // Use growing_text (cumulative), fall back to text
                            let text = msg["growing_text"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .or_else(|| msg["text"].as_str())
                                .unwrap_or("");
                            if text.is_empty() {
                                continue;
                            }

                            let candidate = text.trim().to_string();
                            if candidate.is_empty() {
                                continue;
                            }

                            // Track full_transcript length for forward-progress check
                            let full_transcript_len = msg["full_transcript"]
                                .as_str()
                                .map(|s| s.len())
                                .or_else(|| msg["growing_text"].as_str().map(|s| s.len()))
                                .unwrap_or(0);

                            // Dedup check 1: exact duplicate
                            if candidate == last_sent_text {
                                skipped_count += 1;
                                continue;
                            }

                            // Dedup check 2: subset of last sent text
                            if !last_sent_text.is_empty() && last_sent_text.contains(&candidate) {
                                skipped_count += 1;
                                continue;
                            }

                            // Dedup check 3: near-duplicate with no forward progress
                            if full_transcript_len > 0
                                && full_transcript_len <= last_full_transcript_len
                                && !last_sent_text.is_empty()
                                && text_similarity(&candidate, &last_sent_text) > SIMILARITY_THRESHOLD
                            {
                                skipped_count += 1;
                                continue;
                            }

                            // Update transcript length tracking
                            if full_transcript_len > last_full_transcript_len {
                                last_full_transcript_len = full_transcript_len;
                            }

                            // Buffer this text and (re)set debounce timer
                            pending_text = Some(candidate);
                            debounce_deadline = Some(Instant::now() + Duration::from_millis(DEBOUNCE_MS));
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            eprintln!("[FAB] [{}] Lagged {} messages, continuing", session_id, n);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            // Flush pending text before exit
                            if let Some(text) = pending_text.take() {
                                if send_to_fab(&client, &fab_url, &session_id, &text).await {
                                    sent_count += 1;
                                }
                            }
                            eprintln!(
                                "[FAB] Forwarder stopped for session {} (sent: {}, skipped: {})",
                                session_id, sent_count, skipped_count
                            );
                            break;
                        }
                    }
                }
                _ = sleep_fut => {
                    // Debounce timer fired — send the buffered text
                    debounce_deadline = None;
                    if let Some(text) = pending_text.take() {
                        if send_to_fab(&client, &fab_url, &session_id, &text).await {
                            last_sent_text = text;
                            sent_count += 1;
                        }
                    }
                }
            }
        }
    });
}
