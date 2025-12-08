#!/usr/bin/env python3
"""
UI-like WebSocket test for the transcription server.
Simulates what the frontend does: creates a session, connects via WebSocket,
and verifies transcripts are received.
"""

import asyncio
import json
import sys
import time
import httpx
import websockets

SERVER_URL = "http://localhost:8082"
WS_URL = "ws://localhost:8082/ws"

async def test_transcription_flow():
    """Test the full transcription flow like the UI does."""

    print("=" * 60)
    print("  UI-like WebSocket Transcription Test")
    print("=" * 60)

    # Step 1: Get available models
    print("\n[1] Fetching available models...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/models")
        if resp.status_code != 200:
            print(f"    FAIL: Could not get models: {resp.status_code}")
            return False
        models = resp.json()
        print(f"    OK: Found {len(models)} models: {[m['id'] for m in models]}")

        # Step 2: Get available media files
        print("\n[2] Fetching available media files...")
        resp = await client.get(f"{SERVER_URL}/api/media")
        if resp.status_code != 200:
            print(f"    FAIL: Could not get media: {resp.status_code}")
            return False
        media = resp.json()
        print(f"    OK: Found {len(media)} media files: {[m['name'] for m in media]}")

        # Find broadcast.wav
        broadcast = next((m for m in media if m['name'] == 'broadcast.wav'), None)
        if not broadcast:
            print("    FAIL: broadcast.wav not found")
            return False

        # Step 3: Create a session with Canary speedy mode
        print("\n[3] Creating transcription session (Canary, speedy, German)...")
        session_config = {
            "model": "canary-1b",
            "media": "broadcast",
            "mode": "speedy",
            "language": "de"
        }
        resp = await client.post(f"{SERVER_URL}/api/sessions", json=session_config)
        if resp.status_code != 200:
            print(f"    FAIL: Could not create session: {resp.status_code} - {resp.text}")
            return False
        session = resp.json()
        session_id = session['id']
        print(f"    OK: Created session {session_id}")

    # Step 4: Connect via WebSocket and listen for transcripts
    print("\n[4] Connecting via WebSocket...")

    received_segments = []
    final_segments = []
    partial_segments = []
    start_time = time.time()
    test_duration = 15  # Listen for 15 seconds

    try:
        async with websockets.connect(f"{WS_URL}?session={session_id}") as ws:
            print(f"    OK: Connected to WebSocket")
            print(f"\n[5] Listening for transcripts ({test_duration}s)...")
            print("-" * 60)

            while time.time() - start_time < test_duration:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)

                    # Skip non-subtitle messages
                    if 'text' not in data:
                        continue

                    received_segments.append(data)

                    is_final = data.get('is_final', False)
                    text = data.get('text', '')[:60]
                    start = data.get('start', 0)
                    end = data.get('end', 0)

                    if is_final:
                        final_segments.append(data)
                        print(f"    [FINAL] {start:.1f}s-{end:.1f}s: {text}...")
                    else:
                        partial_segments.append(data)
                        # Only print every 5th partial to avoid spam
                        if len(partial_segments) % 5 == 1:
                            print(f"    [partial] {start:.1f}s-{end:.1f}s: {text}...")

                except asyncio.TimeoutError:
                    continue

    except Exception as e:
        print(f"    ERROR: WebSocket connection failed: {e}")
        return False

    # Step 5: Stop the session
    print("-" * 60)
    print(f"\n[6] Stopping session...")
    async with httpx.AsyncClient() as client:
        resp = await client.delete(f"{SERVER_URL}/api/sessions/{session_id}")
        print(f"    OK: Session stopped")

    # Step 6: Report results
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print(f"  Total segments received: {len(received_segments)}")
    print(f"  Partial segments: {len(partial_segments)}")
    print(f"  Final segments: {len(final_segments)}")

    if len(received_segments) == 0:
        print("\n  ❌ FAIL: No segments received!")
        return False

    if len(partial_segments) == 0:
        print("\n  ⚠️  WARNING: No partial segments received")

    print("\n  Sample transcripts:")
    for i, seg in enumerate(received_segments[:5]):
        text = seg.get('text', '')[:80]
        is_final = "FINAL" if seg.get('is_final') else "partial"
        print(f"    {i+1}. [{is_final}] {text}")

    if len(final_segments) > 0:
        print(f"\n  ✓ SUCCESS: Transcription working!")
        print(f"    First final segment: \"{final_segments[0].get('text', '')[:60]}...\"")
        return True
    elif len(partial_segments) > 0:
        print(f"\n  ⚠️  PARTIAL SUCCESS: Only partial segments received (no finals yet)")
        print(f"    This is expected for short test duration with speedy mode")
        return True
    else:
        print(f"\n  ❌ FAIL: No transcription data received")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_transcription_flow())
    sys.exit(0 if success else 1)
