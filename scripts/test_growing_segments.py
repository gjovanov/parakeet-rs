#!/usr/bin/env python3
"""
Test script for Growing Segments ASR mode.

Validates that the server produces PARTIAL and FINAL subtitle messages
in the expected growing-segment pattern:
  [PARTIAL] wie
  [PARTIAL] wie jene bei
  ...
  [FINAL]   wie jene bei Paleo Kastriza oder die Zwillingsbucht Porto Timoni.

Usage:
  python3 scripts/test_growing_segments.py [SERVER_URL] [MEDIA_ID] [LANGUAGE] [MODEL]

Defaults:
  SERVER_URL = http://localhost:8080
  MEDIA_ID   = broadcast_1
  LANGUAGE   = de
  MODEL      = (auto-detect first available)

Requirements:
  pip install websockets
"""

import asyncio
import json
import sys
import urllib.request
import urllib.error

# --- Configuration ---
SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
MEDIA_ID   = sys.argv[2] if len(sys.argv) > 2 else "broadcast_1"
LANGUAGE   = sys.argv[3] if len(sys.argv) > 3 else "de"
MODEL      = sys.argv[4] if len(sys.argv) > 4 else ""

# Colors
RED    = "\033[0;31m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
NC     = "\033[0m"

TIMEOUT_SECS = 180  # max wait for WebSocket messages


def api_get(path: str):
    """HTTP GET and return parsed JSON."""
    url = f"{SERVER_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def api_post(path: str, body: dict = None):
    """HTTP POST JSON and return parsed JSON."""
    url = f"{SERVER_URL}{path}"
    if body:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


async def run_test():
    import websockets

    print(f"{CYAN}=== Growing Segments ASR Mode Test ==={NC}")
    print(f"Server:   {SERVER_URL}")
    print(f"Media:    {MEDIA_ID}")
    print(f"Language: {LANGUAGE}")

    # 1. Check server
    print(f"\n{YELLOW}[1/6] Checking server health...{NC}")
    try:
        modes_resp = api_get("/api/modes")
    except Exception as e:
        print(f"{RED}Error: Server not reachable at {SERVER_URL}: {e}{NC}")
        sys.exit(1)
    print(f"{GREEN}Server is running.{NC}")

    # 2. Verify growing_segments mode
    print(f"\n{YELLOW}[2/6] Verifying growing_segments mode...{NC}")
    mode_ids = [m["id"] for m in modes_resp.get("data", [])]
    if "growing_segments" not in mode_ids:
        print(f"{RED}Error: 'growing_segments' not in available modes: {mode_ids}{NC}")
        sys.exit(1)
    print(f"{GREEN}Mode 'growing_segments' is registered.{NC}")

    # 3. Auto-detect model
    global MODEL
    if not MODEL:
        print(f"\n{YELLOW}[3/6] Auto-detecting model...{NC}")
        models_resp = api_get("/api/models")
        models = models_resp.get("data", [])
        if not models:
            print(f"{RED}Error: No models available.{NC}")
            sys.exit(1)
        MODEL = models[0]["id"]
    print(f"{GREEN}Using model: {MODEL}{NC}")

    # 4. Create session
    print(f"\n{YELLOW}[4/6] Creating session...{NC}")
    session_resp = api_post("/api/sessions", {
        "model_id": MODEL,
        "mode": "growing_segments",
        "language": LANGUAGE,
        "media_id": MEDIA_ID,
    })
    if not session_resp.get("success"):
        print(f"{RED}Error creating session: {session_resp.get('error')}{NC}")
        sys.exit(1)
    session_id = session_resp["data"]["id"]
    print(f"{GREEN}Session created: {session_id}{NC}")

    # 5. Start session
    print(f"\n{YELLOW}[5/6] Starting session...{NC}")
    start_resp = api_post(f"/api/sessions/{session_id}/start")
    if not start_resp.get("success"):
        print(f"{RED}Error starting session: {start_resp.get('error')}{NC}")
        sys.exit(1)
    print(f"{GREEN}Session started.{NC}")

    # 6. Connect WebSocket and capture
    print(f"\n{YELLOW}[6/6] Connecting WebSocket and capturing subtitles...{NC}")
    ws_url = f"{SERVER_URL.replace('http', 'ws')}/ws/{session_id}"

    messages = []
    partials = []
    finals = []
    last_final_group_partials = []

    try:
        async with websockets.connect(ws_url, close_timeout=5) as ws:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    print(f"{YELLOW}Timeout waiting for messages (>{TIMEOUT_SECS}s){NC}")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                messages.append(msg)
                msg_type = msg.get("type", "")

                if msg_type == "subtitle":
                    is_final = msg.get("is_final", False)
                    text = msg.get("growing_text", "") or msg.get("text", "")
                    raw_text = msg.get("text", "")

                    if is_final:
                        finals.append(msg)
                        print(f"{GREEN}[FINAL]  {NC} {raw_text[:120]}")
                        last_final_group_partials = []
                    else:
                        partials.append(msg)
                        last_final_group_partials.append(msg)
                        print(f"{YELLOW}[PARTIAL]{NC} {text[:120]}")

                elif msg_type == "status":
                    status = msg.get("status", "")
                    print(f"{CYAN}[STATUS] {NC} {status}")
                    if status in ("completed", "finished", "stopped"):
                        break

    except Exception as e:
        print(f"{RED}WebSocket error: {e}{NC}")

    # --- Validation ---
    print(f"\n{CYAN}=== Results ==={NC}")
    total = len(partials) + len(finals)
    print(f"Total subtitle messages: {total}")
    print(f"  PARTIAL: {len(partials)}")
    print(f"  FINAL:   {len(finals)}")

    passed = True
    warnings = []

    # Check: we got messages at all
    if total == 0:
        print(f"{RED}FAIL: No subtitle messages received.{NC}")
        passed = False

    # Check: we got FINAL messages
    if len(finals) == 0:
        print(f"{RED}FAIL: No FINAL messages received.{NC}")
        passed = False

    # Check: we got PARTIAL messages (expected for growing segments)
    if len(partials) == 0:
        warnings.append("No PARTIAL messages received (may indicate too-slow processing).")

    # Check: growing pattern — partials should have is_final=false
    bad_partials = [m for m in partials if m.get("is_final") is not False]
    if bad_partials:
        print(f"{RED}FAIL: {len(bad_partials)} PARTIAL messages have is_final != false{NC}")
        passed = False

    # Check: finals should have is_final=true
    bad_finals = [m for m in finals if m.get("is_final") is not True]
    if bad_finals:
        print(f"{RED}FAIL: {len(bad_finals)} FINAL messages have is_final != true{NC}")
        passed = False

    # Check: ratio — growing segments should produce more partials than finals
    if len(partials) > 0 and len(finals) > 0:
        ratio = len(partials) / len(finals)
        print(f"  Partial/Final ratio: {ratio:.1f}x")
        if ratio < 1.0:
            warnings.append(f"Low partial/final ratio ({ratio:.1f}x). Expected > 1.0 for growing segments.")

    # Show reference comparison (first few finals vs broadcast_1.txt)
    if finals:
        print(f"\n{CYAN}=== First 5 FINAL texts ==={NC}")
        for i, f in enumerate(finals[:5]):
            print(f"  {i+1}. {f.get('text', '')[:120]}")

    for w in warnings:
        print(f"{YELLOW}WARN: {w}{NC}")

    if passed:
        print(f"\n{GREEN}PASS: Growing segments mode produced expected PARTIAL/FINAL pattern.{NC}")
    else:
        print(f"\n{RED}FAIL: Growing segments validation failed.{NC}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_test())
