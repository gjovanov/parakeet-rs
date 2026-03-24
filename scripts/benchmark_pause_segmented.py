#!/usr/bin/env python3
"""
Benchmark pause_segmented mode with different silence energy thresholds.

Tests against broadcast_1.wav for 5 minutes, comparing results across configurations.
Runs each config sequentially, collects FINALs, and computes WER/CER/recall.

Usage:
  python3 scripts/benchmark_pause_segmented.py [--server URL] [--model MODEL] [--duration SECS]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
REFERENCE_FILE = PROJECT_DIR / "media" / "broadcast_1.txt"

# Key phrases from broadcast_1.txt first 5-10 minutes
KEY_PHRASES = [
    "Bischofshofen", "ORF", "Manuel Rubay", "Simon Schwarz", "Hallwang",
    "Alighieri", "Domquartier", "Salzachblume", "Nussdorf", "Wirtschaftskammer",
    "EU Kommission", "Kartellverfahren", "Red Bull", "Walsberg",
    "Videoüberwachung", "Grenzkontrollen", "Salzburg", "Österreich",
]

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# Configurations to test
CONFIGS = [
    {
        "name": "baseline (0.008 / 300ms)",
        "silence_energy": 0.008,
        "pause_threshold_ms": 300,
        "max_segment_secs": 15.0,
    },
    {
        "name": "lower energy (0.005 / 300ms)",
        "silence_energy": 0.005,
        "pause_threshold_ms": 300,
        "max_segment_secs": 15.0,
    },
    {
        "name": "very low energy (0.003 / 300ms)",
        "silence_energy": 0.003,
        "pause_threshold_ms": 300,
        "max_segment_secs": 15.0,
    },
    {
        "name": "low energy + longer pause (0.005 / 500ms)",
        "silence_energy": 0.005,
        "pause_threshold_ms": 500,
        "max_segment_secs": 15.0,
    },
    {
        "name": "low energy + short max (0.005 / 300ms / 5s max)",
        "silence_energy": 0.005,
        "pause_threshold_ms": 300,
        "max_segment_secs": 5.0,
    },
]


def normalize_text(text: str) -> str:
    text = text.lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else float("inf")

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else float("inf")

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def key_phrase_recall(transcript: str, phrases: list) -> tuple:
    text_lower = normalize_text(transcript)
    found = [p for p in phrases if normalize_text(p) in text_lower]
    return len(found) / len(phrases) if phrases else 0.0, found


def load_reference(duration_secs: int) -> str:
    """Load reference transcript, truncate approximately to duration."""
    text = REFERENCE_FILE.read_text(encoding="utf-8").strip()
    # Reference is ~10 min. For 5 min, take roughly first half by lines
    lines = text.strip().split("\n")
    if duration_secs < 600:
        ratio = duration_secs / 600.0
        n_lines = max(1, int(len(lines) * ratio))
        lines = lines[:n_lines]
    return "\n".join(lines)


async def run_single_benchmark(server: str, model: str, config: dict, duration: int, language: str) -> dict:
    """Run a single pause_segmented session and collect results."""
    try:
        import websockets
    except ImportError:
        print(f"{RED}Error: websockets not installed. Run: pip install websockets{NC}")
        sys.exit(1)

    # Create session
    body = json.dumps({
        "model_id": model,
        "media_id": "broadcast_1",
        "mode": "pause_segmented",
        "language": language,
        "noise_cancellation": "none",
        "diarization": False,
        "sentence_completion": "off",
        "pause_config": {
            "pause_threshold_ms": config["pause_threshold_ms"],
            "silence_energy_threshold": config["silence_energy"],
            "max_segment_secs": config["max_segment_secs"],
            "context_segments": 1,
        },
    }).encode()

    req = urllib.request.Request(
        f"{server}/api/sessions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    session = json.loads(resp.read())["data"]
    session_id = session["id"]
    print(f"  Session {session_id} created")

    # Start session
    req = urllib.request.Request(
        f"{server}/api/sessions/{session_id}/start",
        data=b"",
        method="POST",
    )
    urllib.request.urlopen(req)

    # Connect WebSocket and collect results
    ws_url = f"ws://{server.split('//')[1]}/ws/{session_id}"
    finals = []
    partials = 0
    full_transcript = ""
    inference_times = []
    start_time = time.time()
    max_segment_duration = 0.0
    forced_cuts = 0

    try:
        async with websockets.connect(ws_url, ping_interval=30, ping_timeout=60) as ws:
            # Send ready
            await ws.send(json.dumps({"type": "ready"}))

            while True:
                elapsed = time.time() - start_time
                if elapsed > duration + 30:  # grace period
                    break

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(msg)

                    if data.get("type") == "subtitle":
                        text = data.get("text", "").strip()
                        is_final = data.get("is_final", False)
                        inf_ms = data.get("inference_time_ms")

                        if is_final and text:
                            finals.append({
                                "text": text,
                                "start_time": data.get("start_time", 0),
                                "end_time": data.get("end_time", 0),
                                "inference_time_ms": inf_ms,
                            })
                            seg_dur = data.get("end_time", 0) - data.get("start_time", 0)
                            max_segment_duration = max(max_segment_duration, seg_dur)
                            if inf_ms:
                                inference_times.append(inf_ms)
                        elif not is_final:
                            partials += 1

                    elif data.get("type") == "growing_text":
                        full_transcript = data.get("full_text", full_transcript)

                    elif data.get("type") == "end":
                        break

                except asyncio.TimeoutError:
                    if time.time() - start_time > duration + 15:
                        break
                    continue

    except Exception as e:
        print(f"  {RED}WebSocket error: {e}{NC}")

    # Stop session
    try:
        req = urllib.request.Request(
            f"{server}/api/sessions/{session_id}",
            method="DELETE",
        )
        urllib.request.urlopen(req)
    except Exception:
        pass

    # Wait for cleanup
    await asyncio.sleep(2)

    # Build transcript from FINALs
    finals_text = " ".join(f["text"] for f in finals)

    return {
        "config_name": config["name"],
        "silence_energy": config["silence_energy"],
        "pause_threshold_ms": config["pause_threshold_ms"],
        "max_segment_secs": config["max_segment_secs"],
        "final_count": len(finals),
        "partial_count": partials,
        "finals_text": finals_text,
        "full_transcript": full_transcript,
        "max_segment_duration": max_segment_duration,
        "avg_inference_ms": sum(inference_times) / len(inference_times) if inference_times else 0,
        "finals": finals,
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark pause_segmented configurations")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--model", default=None)
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (default: 300 = 5 min)")
    parser.add_argument("--language", default="de")
    args = parser.parse_args()

    # Auto-detect model if not specified
    if not args.model:
        resp = urllib.request.urlopen(f"{args.server}/api/models")
        models = json.loads(resp.read())["data"]
        available = [m for m in models if m["is_loaded"]]
        if not available:
            print(f"{RED}No models available{NC}")
            sys.exit(1)
        # Prefer whisper > canary > tdt for German
        for preferred in ["whisper", "canary-1b", "parakeet-tdt"]:
            for m in available:
                if m["id"].startswith(preferred):
                    args.model = m["id"]
                    break
            if args.model:
                break
        if not args.model:
            args.model = available[0]["id"]

    # Load reference
    ref_text = load_reference(args.duration)
    ref_words = len(normalize_text(ref_text).split())

    print(f"\n{BOLD}{'='*70}{NC}")
    print(f"{BOLD}  Pause-Segmented Benchmark{NC}")
    print(f"{BOLD}{'='*70}{NC}")
    print(f"  Model:     {CYAN}{args.model}{NC}")
    print(f"  Duration:  {args.duration}s ({args.duration // 60}m)")
    print(f"  Language:  {args.language}")
    print(f"  Reference: {ref_words} words")
    print(f"  Configs:   {len(CONFIGS)}")
    print(f"{BOLD}{'='*70}{NC}\n")

    results = []

    for i, config in enumerate(CONFIGS):
        print(f"{BOLD}[{i+1}/{len(CONFIGS)}] {config['name']}{NC}")
        print(f"  energy={config['silence_energy']}, pause={config['pause_threshold_ms']}ms, max_seg={config['max_segment_secs']}s")

        result = await run_single_benchmark(
            args.server, args.model, config, args.duration, args.language
        )

        # Compute metrics
        transcript = result["finals_text"]
        wer = compute_wer(ref_text, transcript)
        cer = compute_cer(ref_text, transcript)
        recall, found_phrases = key_phrase_recall(transcript, KEY_PHRASES)
        hyp_words = len(normalize_text(transcript).split())

        result["wer"] = wer
        result["cer"] = cer
        result["key_phrase_recall"] = recall
        result["found_phrases"] = found_phrases
        result["hyp_words"] = hyp_words
        result["ref_words"] = ref_words

        # Analyze segment durations
        seg_durations = [f["end_time"] - f["start_time"] for f in result["finals"] if f["end_time"] > f["start_time"]]
        avg_seg = sum(seg_durations) / len(seg_durations) if seg_durations else 0
        min_seg = min(seg_durations) if seg_durations else 0
        max_seg = max(seg_durations) if seg_durations else 0

        result["avg_segment_secs"] = avg_seg
        result["min_segment_secs"] = min_seg
        result["max_segment_secs_actual"] = max_seg

        results.append(result)

        # Print results
        wer_color = GREEN if wer < 0.5 else YELLOW if wer < 1.0 else RED
        print(f"  FINALs: {BOLD}{result['final_count']}{NC}  PARTIALs: {result['partial_count']}")
        print(f"  WER: {wer_color}{wer*100:.1f}%{NC}  CER: {cer*100:.1f}%  Recall: {recall*100:.0f}%")
        print(f"  Words: {hyp_words} hyp / {ref_words} ref ({hyp_words/ref_words*100:.0f}%)")
        print(f"  Segments: avg={avg_seg:.1f}s  min={min_seg:.1f}s  max={max_seg:.1f}s")
        print(f"  Avg inference: {result['avg_inference_ms']:.0f}ms")
        print()

    # Summary table
    print(f"\n{BOLD}{'='*100}{NC}")
    print(f"{BOLD}  COMPARISON TABLE{NC}")
    print(f"{BOLD}{'='*100}{NC}")
    print(f"{'Config':<40} {'FINALs':>7} {'WER':>8} {'CER':>8} {'Recall':>8} {'AvgSeg':>8} {'MaxSeg':>8} {'InfMs':>8}")
    print(f"{'-'*100}")
    for r in results:
        wer_str = f"{r['wer']*100:.1f}%"
        cer_str = f"{r['cer']*100:.1f}%"
        recall_str = f"{r['key_phrase_recall']*100:.0f}%"
        print(f"{r['config_name']:<40} {r['final_count']:>7} {wer_str:>8} {cer_str:>8} {recall_str:>8} {r['avg_segment_secs']:>7.1f}s {r['max_segment_secs_actual']:>7.1f}s {r['avg_inference_ms']:>7.0f}ms")
    print(f"{'='*100}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"pause_segmented_benchmark_{args.model}_{timestamp}.json"

    save_data = {
        "model": args.model,
        "duration_secs": args.duration,
        "language": args.language,
        "timestamp": timestamp,
        "ref_words": ref_words,
        "results": [{k: v for k, v in r.items() if k != "finals"} for r in results],
    }
    outfile.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {outfile}")


if __name__ == "__main__":
    asyncio.run(main())
