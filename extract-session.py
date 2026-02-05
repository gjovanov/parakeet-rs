#!/usr/bin/env python3
"""Extract growing and confirmed sentences from a parakeet-rs session log.

Usage:
    ./extract-session.py <session_id> [log_file]

Examples:
    ./extract-session.py 537355ab
    ./extract-session.py 537355ab /tmp/parakeet-server.log
    ./extract-session.py 537355ab server.log -o results/

Output:
    session-<id>.json in the current directory (or -o dir)
"""

import argparse
import json
import re
import sys

def extract_session(session_id, log_file, output_dir="."):
    pattern = re.compile(
        r'\[Session ' + re.escape(session_id)
        + r' \| (partial|FINAL) \| Speaker ([^\]]+)\] "(.+?)" '
        + r'\[(\d+\.\d+)s-(\d+\.\d+)s\] \(inference: (\d+)ms, receivers: (\d+)\)(.*)'
    )

    entries = []
    seq = 0

    with open(log_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            entries.append({
                "seq": seq,
                "type": "confirmed" if m.group(1) == "FINAL" else "growing",
                "text": m.group(3),
                "speaker": m.group(2).strip(),
                "start_time": float(m.group(4)),
                "end_time": float(m.group(5)),
                "inference_ms": int(m.group(6)),
                "tail_changed": "TAIL CHANGED" in m.group(8),
            })
            seq += 1

    if not entries:
        print(f"No entries found for session {session_id} in {log_file}", file=sys.stderr)
        sys.exit(1)

    out_path = f"{output_dir}/session-{session_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    confirmed = sum(1 for e in entries if e["type"] == "confirmed")
    growing = len(entries) - confirmed
    print(f"{len(entries)} entries ({confirmed} confirmed, {growing} growing) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract session transcript from parakeet-rs logs")
    parser.add_argument("session_id", help="Session ID to extract")
    parser.add_argument("log_file", nargs="?", default="/tmp/parakeet-server.log", help="Log file path (default: /tmp/parakeet-server.log)")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory (default: current dir)")
    args = parser.parse_args()

    extract_session(args.session_id, args.log_file, args.output_dir)
