#!/usr/bin/env python3
"""
Compare transcription output between two voxtral-server runs.

Usage:
    # Phase 1: Record from container (3 min)
    python compare_transcripts.py record ws://localhost:8091 SESSION_ID container.jsonl

    # Phase 2: Record from native (3 min)
    python compare_transcripts.py record ws://localhost:80 SESSION_ID native.jsonl

    # Phase 3: Compare
    python compare_transcripts.py compare container.jsonl native.jsonl
"""

import asyncio
import json
import sys
import time

RECORD_DURATION = 180  # 3 minutes


async def record(ws_url: str, session_id: str, output_file: str):
    """Connect to a voxtral-server WebSocket, record subtitles for 3 minutes."""
    import websockets

    full_url = f"{ws_url}/ws/{session_id}"
    print(f"Connecting to {full_url}...")

    async with websockets.connect(full_url, max_size=10 * 1024 * 1024) as ws:
        # Send ready
        await ws.send(json.dumps({"type": "ready"}))
        print(f"Connected. Recording for {RECORD_DURATION}s to {output_file}...")

        start = time.monotonic()
        count = 0

        with open(output_file, "w") as f:
            while time.monotonic() - start < RECORD_DURATION:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                msg = json.loads(raw)
                if msg.get("type") != "subtitle":
                    if msg.get("type") == "welcome":
                        print(f"  Welcome: {msg.get('message')}")
                    continue

                elapsed = time.monotonic() - start
                record = {
                    "elapsed_s": round(elapsed, 3),
                    "is_final": msg.get("is_final", False),
                    "text": msg.get("text", ""),
                    "growing_text": msg.get("growing_text"),
                    "inference_time_ms": msg.get("inference_time_ms"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                tag = "FINAL" if record["is_final"] else "PARTIAL"
                print(f"  [{elapsed:6.1f}s] [{tag:7s}] {record['text'][:80]}")

        print(f"\nDone. Recorded {count} messages in {RECORD_DURATION}s.")


def compare(file_a: str, file_b: str):
    """Compare two transcript recordings."""
    def load(path):
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        return records

    a = load(file_a)
    b = load(file_b)

    finals_a = [r for r in a if r["is_final"]]
    finals_b = [r for r in b if r["is_final"]]
    partials_a = [r for r in a if not r["is_final"]]
    partials_b = [r for r in b if not r["is_final"]]

    print(f"=== Comparison: {file_a} vs {file_b} ===\n")
    print(f"{'Metric':<30} {'A':>10} {'B':>10}")
    print("-" * 52)
    print(f"{'Total messages':<30} {len(a):>10} {len(b):>10}")
    print(f"{'Final segments':<30} {len(finals_a):>10} {len(finals_b):>10}")
    print(f"{'Partial updates':<30} {len(partials_a):>10} {len(partials_b):>10}")

    # Average inference time for finals
    inf_a = [r["inference_time_ms"] for r in finals_a if r.get("inference_time_ms") is not None]
    inf_b = [r["inference_time_ms"] for r in finals_b if r.get("inference_time_ms") is not None]
    avg_a = sum(inf_a) / len(inf_a) if inf_a else 0
    avg_b = sum(inf_b) / len(inf_b) if inf_b else 0
    print(f"{'Avg inference_time_ms':<30} {avg_a:>10.1f} {avg_b:>10.1f}")

    # Average segment length
    len_a = sum(len(r["text"]) for r in finals_a) / len(finals_a) if finals_a else 0
    len_b = sum(len(r["text"]) for r in finals_b) / len(finals_b) if finals_b else 0
    print(f"{'Avg segment chars':<30} {len_a:>10.1f} {len_b:>10.1f}")

    # Total text
    text_a = " ".join(r["text"] for r in finals_a)
    text_b = " ".join(r["text"] for r in finals_b)
    print(f"{'Total chars (finals)':<30} {len(text_a):>10} {len(text_b):>10}")
    print(f"{'Total words (finals)':<30} {len(text_a.split()):>10} {len(text_b.split()):>10}")

    # Time of first and last final
    if finals_a and finals_b:
        print(f"{'First final at (s)':<30} {finals_a[0]['elapsed_s']:>10.1f} {finals_b[0]['elapsed_s']:>10.1f}")
        print(f"{'Last final at (s)':<30} {finals_a[-1]['elapsed_s']:>10.1f} {finals_b[-1]['elapsed_s']:>10.1f}")

    # Word overlap (rough similarity)
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if words_a and words_b:
        overlap = len(words_a & words_b)
        jaccard = overlap / len(words_a | words_b)
        print(f"{'Word overlap (Jaccard)':<30} {jaccard:>10.2%} {'':>10}")

    # Print first 5 finals side by side
    print(f"\n--- First 5 final segments ---\n")
    for i in range(min(5, max(len(finals_a), len(finals_b)))):
        ta = finals_a[i]["text"][:60] if i < len(finals_a) else "(none)"
        tb = finals_b[i]["text"][:60] if i < len(finals_b) else "(none)"
        print(f"  A: {ta}")
        print(f"  B: {tb}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "record" and len(sys.argv) == 5:
        ws_url, session_id, output = sys.argv[2], sys.argv[3], sys.argv[4]
        asyncio.run(record(ws_url, session_id, output))
    elif cmd == "compare" and len(sys.argv) == 4:
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
