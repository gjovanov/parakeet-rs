#!/usr/bin/env python3
"""
Benchmark transcription quality against broadcast_1.wav reference transcript.

Compares ASR output (FINALs + full_transcript) with the human-verified reference
in media/broadcast_1.txt using WER, CER, key-phrase recall, and timing metrics.

Usage:
  python3 scripts/benchmark_broadcast1.py [options]

Options:
  --server URL        Server URL (default: http://localhost:8080)
  --model MODEL       Model ID (default: auto-detect first available)
  --mode MODE         Latency mode (default: growing_segments)
  --language LANG     Language code (default: de)
  --duration SECS     Duration in seconds to benchmark (default: 600)
  --enable-formatting Enable text formatting
  --tone TONE         Tone setting (default: casual)
  --compare-baseline  Compare against most recent baseline result

Requirements:
  pip install websockets
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colors (same approach as test_growing_segments.py)
# ---------------------------------------------------------------------------
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# ---------------------------------------------------------------------------
# Key phrases from broadcast_1.txt first 10 minutes
# ---------------------------------------------------------------------------
KEY_PHRASES = [
    "Bischofshofen", "ORF", "Manuel Rubay", "Simon Schwarz", "Hallwang",
    "Alighieri", "Domquartier", "Salzachblume", "Nussdorf", "Wirtschaftskammer",
    "EU Kommission", "Kartellverfahren", "Red Bull", "Walsberg",
    "Videoüberwachung", "Grenzkontrollen", "Salzburg", "Österreich", "Mara",
    "Kulturzentrum", "November", "Filialen", "Arzneimittel", "Eigenmarke",
    "Rezeptfreie", "Wille", "Marke", "Pointe", "Restaurant", "Societat",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
REFERENCE_FILE = PROJECT_DIR / "media" / "broadcast_1.txt"

# ---------------------------------------------------------------------------
# Text normalization (mirrors integration_transcription.rs approach)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Remove punctuation but keep letters, digits, whitespace
    # Keep unicode letters (umlauts etc.)
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        # Replace punctuation with space to avoid word merging
        else:
            out.append(" ")
    result = "".join(out)
    # Collapse whitespace
    return " ".join(result.split())


def normalize_words(text: str) -> list:
    """Normalize and split into word tokens."""
    return normalize_text(text).split()


# ---------------------------------------------------------------------------
# Levenshtein distance (word-level and char-level)
# ---------------------------------------------------------------------------

def levenshtein(a: list, b: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (S + D + I) / N, where N = words in reference."""
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein(ref_words, hyp_words)
    return dist / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars)."""
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = levenshtein(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def key_phrase_recall(hypothesis: str, phrases: list) -> tuple:
    """Return (recall_fraction, found_list, missed_list)."""
    hyp_lower = hypothesis.lower()
    found = []
    missed = []
    for phrase in phrases:
        if phrase.lower() in hyp_lower:
            found.append(phrase)
        else:
            missed.append(phrase)
    recall = len(found) / len(phrases) if phrases else 1.0
    return recall, found, missed


# ---------------------------------------------------------------------------
# Reference transcript parser
# ---------------------------------------------------------------------------

def parse_reference(path: Path, max_seconds: int) -> str:
    """
    Parse broadcast_1.txt format: 4-line groups (SPK_N, mm:ss, text, blank).
    Filter to segments starting within max_seconds.
    Return concatenated reference text.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    segments = []
    i = 0
    while i < len(lines):
        # Skip blank lines
        if not lines[i].strip():
            i += 1
            continue
        # Expect: SPK_N
        if not lines[i].strip().startswith("SPK_"):
            i += 1
            continue
        # speaker = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        # Timestamp mm:ss
        ts_line = lines[i].strip()
        i += 1
        seconds = parse_timestamp(ts_line)
        if seconds is None:
            # Not a valid timestamp, skip
            continue
        if i >= len(lines):
            break
        # Text line
        text = lines[i].strip()
        i += 1
        # Skip trailing blank line
        if i < len(lines) and not lines[i].strip():
            i += 1
        if seconds <= max_seconds:
            segments.append(text)
    return " ".join(segments)


def parse_timestamp(ts: str) -> int | None:
    """Parse mm:ss timestamp, return total seconds or None."""
    m = re.match(r"^(\d+):(\d{2})$", ts.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


# ---------------------------------------------------------------------------
# HTTP helpers (matching test_growing_segments.py style)
# ---------------------------------------------------------------------------

def api_get(server: str, path: str):
    """HTTP GET, return parsed JSON."""
    url = f"{server}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def api_post(server: str, path: str, body: dict = None):
    """HTTP POST JSON, return parsed JSON."""
    url = f"{server}{path}"
    if body:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
    else:
        req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Git commit SHA
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    """Return short git commit SHA or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def find_baseline(model: str, mode: str) -> dict | None:
    """
    Find the most recent baseline result (enable_formatting=false) for the
    given model and mode. Returns parsed JSON or None.
    """
    if not RESULTS_DIR.is_dir():
        return None
    # Pattern: broadcast1_{model}_{mode}_noformat_*.json
    candidates = []
    for f in RESULTS_DIR.iterdir():
        if not f.name.endswith(".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            data.get("model") == model
            and data.get("mode") == mode
            and not data.get("enable_formatting", False)
        ):
            candidates.append((f.stat().st_mtime, data, f.name))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def print_delta(label: str, current: float, baseline: float, unit: str = "%",
                lower_is_better: bool = True, width: int = 28):
    """Print a metric with delta from baseline, colored."""
    delta = current - baseline
    if abs(delta) < 0.001:
        color = DIM
        sign = " "
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        color = GREEN
        sign = ""
    else:
        color = RED
        sign = "+"
    print(
        f"  {label:<{width}} {current:>8.1f}{unit}  "
        f"{color}({sign}{delta:+.1f}{unit}){NC}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, baseline: dict | None):
    """Print a formatted summary table with optional baseline comparison."""
    print()
    print(f"{CYAN}{'=' * 72}{NC}")
    print(f"{CYAN}{BOLD}  BENCHMARK RESULTS{NC}")
    print(f"{CYAN}{'=' * 72}{NC}")
    print()

    # Configuration
    print(f"  {DIM}Model:{NC}          {metrics['model']}")
    print(f"  {DIM}Mode:{NC}           {metrics['mode']}")
    print(f"  {DIM}Language:{NC}        {metrics['language']}")
    print(f"  {DIM}Duration:{NC}        {metrics['duration']}s")
    print(f"  {DIM}Formatting:{NC}      {'enabled' if metrics['enable_formatting'] else 'disabled'}")
    print(f"  {DIM}Tone:{NC}            {metrics['tone']}")
    print(f"  {DIM}Git SHA:{NC}         {metrics['git_sha']}")
    print(f"  {DIM}Timestamp:{NC}       {metrics['timestamp']}")
    print()

    # Counts
    print(f"{YELLOW}  Message Counts{NC}")
    print(f"  {'FINAL count':<28} {metrics['final_count']:>8d}")
    print(f"  {'PARTIAL count':<28} {metrics['partial_count']:>8d}")
    ratio_str = f"{metrics['partial_final_ratio']:.1f}x"
    print(f"  {'Partial/Final ratio':<28} {ratio_str:>8s}")
    print()

    # Timing
    print(f"{YELLOW}  Timing{NC}")
    print(f"  {'Wall clock time':<28} {metrics['wall_clock_secs']:>8.1f}s")
    print(f"  {'Avg inference time':<28} {metrics['avg_inference_time_ms']:>8.1f}ms")
    print()

    # Quality
    wer_pct = metrics["wer"] * 100
    cer_pct = metrics["cer"] * 100
    recall_pct = metrics["key_phrase_recall"] * 100
    ref_words = metrics["reference_word_count"]
    hyp_words = metrics["hypothesis_word_count"]

    print(f"{YELLOW}  Quality Metrics{NC}")
    if baseline:
        bwer = baseline["wer"] * 100
        bcer = baseline["cer"] * 100
        brecall = baseline["key_phrase_recall"] * 100
        print_delta("WER", wer_pct, bwer, "%", lower_is_better=True)
        print_delta("CER", cer_pct, bcer, "%", lower_is_better=True)
        print_delta("Key Phrase Recall", recall_pct, brecall, "%", lower_is_better=False)
    else:
        wer_color = GREEN if wer_pct < 40 else (YELLOW if wer_pct < 60 else RED)
        cer_color = GREEN if cer_pct < 30 else (YELLOW if cer_pct < 50 else RED)
        recall_color = GREEN if recall_pct > 70 else (YELLOW if recall_pct > 50 else RED)
        print(f"  {'WER':<28} {wer_color}{wer_pct:>8.1f}%{NC}")
        print(f"  {'CER':<28} {cer_color}{cer_pct:>8.1f}%{NC}")
        print(f"  {'Key Phrase Recall':<28} {recall_color}{recall_pct:>8.1f}%{NC}")

    print(f"  {'Reference words':<28} {ref_words:>8d}")
    print(f"  {'Hypothesis words':<28} {hyp_words:>8d}")
    print()

    # Key phrase details
    found = metrics.get("key_phrases_found", [])
    missed = metrics.get("key_phrases_missed", [])
    if missed:
        print(f"{YELLOW}  Key Phrases Missed ({len(missed)}/{len(KEY_PHRASES)}):{NC}")
        for phrase in missed:
            print(f"    {RED}- {phrase}{NC}")
        print()

    if baseline:
        print(f"{DIM}  Baseline: {baseline.get('timestamp', 'unknown')} "
              f"(SHA: {baseline.get('git_sha', '?')}){NC}")
        print()

    print(f"{CYAN}{'=' * 72}{NC}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    import websockets

    print(f"{CYAN}{'=' * 72}{NC}")
    print(f"{CYAN}{BOLD}  BROADCAST_1 TRANSCRIPTION BENCHMARK{NC}")
    print(f"{CYAN}{'=' * 72}{NC}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Parse reference transcript
    # -----------------------------------------------------------------------
    print(f"{YELLOW}[1/7] Parsing reference transcript...{NC}")
    if not REFERENCE_FILE.exists():
        print(f"{RED}Error: Reference file not found: {REFERENCE_FILE}{NC}")
        sys.exit(1)

    reference_text = parse_reference(REFERENCE_FILE, args.duration)
    ref_word_count = len(normalize_words(reference_text))
    print(f"  Reference: {ref_word_count} words (first {args.duration}s)")

    # -----------------------------------------------------------------------
    # Step 2: Check server health
    # -----------------------------------------------------------------------
    print(f"\n{YELLOW}[2/7] Checking server...{NC}")
    try:
        modes_resp = api_get(args.server, "/api/modes")
    except Exception as e:
        print(f"{RED}Error: Server not reachable at {args.server}: {e}{NC}")
        sys.exit(1)

    mode_ids = [m["id"] for m in modes_resp.get("data", [])]
    if args.mode not in mode_ids:
        print(f"{RED}Error: Mode '{args.mode}' not available. Available: {mode_ids}{NC}")
        sys.exit(1)
    print(f"  {GREEN}Server is running. Mode '{args.mode}' available.{NC}")

    # -----------------------------------------------------------------------
    # Step 3: Detect model
    # -----------------------------------------------------------------------
    print(f"\n{YELLOW}[3/7] Detecting model...{NC}")
    model = args.model
    if not model:
        models_resp = api_get(args.server, "/api/models")
        models = models_resp.get("data", [])
        if not models:
            print(f"{RED}Error: No models available.{NC}")
            sys.exit(1)
        model = models[0]["id"]
    print(f"  {GREEN}Using model: {model}{NC}")

    # -----------------------------------------------------------------------
    # Step 4: Create and start session
    # -----------------------------------------------------------------------
    print(f"\n{YELLOW}[4/7] Creating session...{NC}")
    session_body = {
        "model_id": model,
        "mode": args.mode,
        "language": args.language,
        "media_id": "broadcast_1",
        "enable_formatting": args.enable_formatting,
        "formatting_tone": args.tone,
    }
    session_resp = api_post(args.server, "/api/sessions", session_body)
    if not session_resp.get("success"):
        print(f"{RED}Error creating session: {session_resp.get('error')}{NC}")
        sys.exit(1)
    session_id = session_resp["data"]["id"]
    print(f"  Session: {session_id}")

    print(f"\n{YELLOW}[5/7] Starting session...{NC}")
    start_resp = api_post(args.server, f"/api/sessions/{session_id}/start")
    if not start_resp.get("success"):
        print(f"{RED}Error starting session: {start_resp.get('error')}{NC}")
        sys.exit(1)
    print(f"  {GREEN}Session started.{NC}")

    # -----------------------------------------------------------------------
    # Step 5: Connect WebSocket and collect messages
    # -----------------------------------------------------------------------
    print(f"\n{YELLOW}[6/7] Collecting transcription via WebSocket...{NC}")
    ws_url = f"{args.server.replace('http', 'ws')}/ws/{session_id}"

    finals = []
    partials = []
    inference_times = []
    last_full_transcript = ""
    wall_start = time.monotonic()
    timeout_secs = args.duration + 120

    try:
        async with websockets.connect(ws_url, close_timeout=5) as ws:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_secs)
                except asyncio.TimeoutError:
                    print(f"  {YELLOW}Timeout after {timeout_secs}s{NC}")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "subtitle":
                    is_final = msg.get("is_final", False)
                    text = msg.get("text", "")
                    ft = msg.get("full_transcript", "")
                    if ft:
                        last_full_transcript = ft
                    inf_time = msg.get("inference_time_ms")
                    if inf_time is not None:
                        inference_times.append(inf_time)

                    if is_final:
                        finals.append(msg)
                        # Truncate long texts for display
                        display = text[:100] + ("..." if len(text) > 100 else "")
                        print(f"  {GREEN}[FINAL {len(finals):>3d}]{NC} {display}")
                    else:
                        partials.append(msg)
                        if len(partials) % 50 == 0:
                            display = (msg.get("growing_text", "") or text)[:80]
                            print(
                                f"  {DIM}[PARTIAL {len(partials):>4d}]{NC} {display}"
                            )

                elif msg_type == "end":
                    print(f"  {CYAN}[END]{NC} Session complete.")
                    break

                elif msg_type == "status":
                    status = msg.get("status", "")
                    if status in ("completed", "finished", "stopped"):
                        print(f"  {CYAN}[STATUS]{NC} {status}")
                        break

    except Exception as e:
        print(f"{RED}WebSocket error: {e}{NC}")
        if not finals:
            sys.exit(1)

    wall_secs = time.monotonic() - wall_start

    # -----------------------------------------------------------------------
    # Step 6: Compute metrics
    # -----------------------------------------------------------------------
    print(f"\n{YELLOW}[7/7] Computing metrics...{NC}")

    # Build hypothesis from FINALs text concatenation
    finals_text = " ".join(f.get("text", "") for f in finals)

    # Use full_transcript if available as it includes the complete growing buffer
    hypothesis = last_full_transcript if last_full_transcript else finals_text

    wer = word_error_rate(reference_text, hypothesis)
    cer = char_error_rate(reference_text, hypothesis)
    recall, found_phrases, missed_phrases = key_phrase_recall(hypothesis, KEY_PHRASES)

    avg_inf_time = (
        sum(inference_times) / len(inference_times) if inference_times else 0.0
    )
    partial_final_ratio = (
        len(partials) / len(finals) if finals else 0.0
    )

    hyp_word_count = len(normalize_words(hypothesis))

    git_sha = get_git_sha()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Also compute metrics for finals-only text (separate from full_transcript)
    finals_wer = word_error_rate(reference_text, finals_text)
    finals_cer = char_error_rate(reference_text, finals_text)
    finals_recall, _, _ = key_phrase_recall(finals_text, KEY_PHRASES)

    formatting_tag = "format" if args.enable_formatting else "noformat"

    metrics = {
        "model": model,
        "mode": args.mode,
        "language": args.language,
        "duration": args.duration,
        "enable_formatting": args.enable_formatting,
        "tone": args.tone,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "final_count": len(finals),
        "partial_count": len(partials),
        "partial_final_ratio": round(partial_final_ratio, 2),
        "avg_inference_time_ms": round(avg_inf_time, 1),
        "wall_clock_secs": round(wall_secs, 1),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "key_phrase_recall": round(recall, 4),
        "key_phrases_found": found_phrases,
        "key_phrases_missed": missed_phrases,
        "reference_word_count": ref_word_count,
        "hypothesis_word_count": hyp_word_count,
        "finals_only_wer": round(finals_wer, 4),
        "finals_only_cer": round(finals_cer, 4),
        "finals_only_key_phrase_recall": round(finals_recall, 4),
        "hypothesis_source": "full_transcript" if last_full_transcript else "finals_concat",
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename (replace / with -)
    model_safe = model.replace("/", "-").replace(" ", "_")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"broadcast1_{model_safe}_{args.mode}_{formatting_tag}_{ts_file}.json"
    result_path = RESULTS_DIR / filename

    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {GREEN}Results saved: {result_path}{NC}")

    # -----------------------------------------------------------------------
    # Baseline comparison
    # -----------------------------------------------------------------------
    baseline = None
    if args.compare_baseline:
        baseline = find_baseline(model, args.mode)
        if baseline:
            print(f"  {GREEN}Baseline loaded: {baseline.get('timestamp', 'unknown')}{NC}")
        else:
            print(f"  {YELLOW}No baseline found for {model}/{args.mode} (noformat).{NC}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print_summary(metrics, baseline)

    # Print secondary (finals-only) metrics if hypothesis came from full_transcript
    if last_full_transcript:
        print()
        print(f"{DIM}  Finals-only metrics (for comparison):{NC}")
        print(f"  {'  WER':<28} {finals_wer * 100:>8.1f}%")
        print(f"  {'  CER':<28} {finals_cer * 100:>8.1f}%")
        print(f"  {'  Key Phrase Recall':<28} {finals_recall * 100:>8.1f}%")
        print()

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transcription quality against broadcast_1.wav"
    )
    parser.add_argument(
        "--server", default="http://localhost:8080",
        help="Server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model", default="",
        help="Model ID (default: auto-detect first available)",
    )
    parser.add_argument(
        "--mode", default="growing_segments",
        help="Latency mode (default: growing_segments)",
    )
    parser.add_argument(
        "--language", default="de",
        help="Language code (default: de)",
    )
    parser.add_argument(
        "--duration", type=int, default=600,
        help="Duration in seconds to benchmark (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--enable-formatting", action="store_true", default=False,
        help="Enable text formatting",
    )
    parser.add_argument(
        "--tone", default="casual",
        help="Tone setting (default: casual)",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true", default=False,
        help="Compare against most recent baseline result (enable_formatting=false)",
    )
    args = parser.parse_args()

    # Validate duration
    if args.duration < 10:
        print(f"{RED}Error: Duration must be at least 10 seconds.{NC}")
        sys.exit(1)

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
