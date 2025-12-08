#!/usr/bin/env python3
"""
WebSocket test for Canary VAD pause-based transcription.
Simulates the frontend behavior.
"""

import asyncio
import json
import subprocess
import websockets
import struct

async def test_canary_vad():
    print("=== WebSocket Canary VAD Test ===\n")

    # Extract audio using ffmpeg
    print("Extracting audio...")
    result = subprocess.run([
        "ffmpeg", "-i", "./media/broadcast.wav",
        "-ss", "60", "-t", "15",  # 15 seconds from position 60s
        "-ar", "16000", "-ac", "1",
        "-f", "s16le", "-"
    ], capture_output=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr.decode()}")
        return False

    audio_bytes = result.stdout
    print(f"Audio extracted: {len(audio_bytes)} bytes ({len(audio_bytes) / 32000:.2f}s)\n")

    # Connect to WebSocket
    uri = "ws://localhost:8082/ws"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        print("Connected!")

        # Read initial messages (models, config)
        for _ in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(msg)
            print(f"  Received: {data.get('type', 'unknown')}")

        # Configure for Canary with VAD pause-based mode
        config = {
            "type": "config",
            "config": {
                "model": "canary-1b",
                "mode": "pause_based",
                "language": "en"
            }
        }
        print(f"\nSending config: {json.dumps(config)}")
        await ws.send(json.dumps(config))

        # Wait for config confirmation
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)
        print(f"Config response: {data}")

        # Start transcription
        start_msg = {"type": "start"}
        print(f"\nSending start command...")
        await ws.send(json.dumps(start_msg))

        # Wait for start confirmation
        msg = await asyncio.wait_for(ws.recv(), timeout=5)
        data = json.loads(msg)
        print(f"Start response: {data}")

        # Stream audio in chunks (100ms chunks = 3200 bytes at 16kHz, 16-bit)
        chunk_size = 3200
        chunks_sent = 0
        transcripts = []

        print(f"\nStreaming audio...")

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await ws.send(chunk)
            chunks_sent += 1

            # Check for incoming messages (non-blocking)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("type") == "transcript":
                        for seg in data.get("segments", []):
                            text = seg.get("text", "")
                            if text:
                                print(f"\n  [TRANSCRIPT] {seg.get('start_time', 0):.2f}s: \"{text}\"")
                                transcripts.append(text)
            except asyncio.TimeoutError:
                pass

            # Progress indicator
            if chunks_sent % 10 == 0:
                print(f"  Sent {chunks_sent * 100}ms of audio...", end="\r")

            # Simulate real-time pace (slightly faster)
            await asyncio.sleep(0.05)

        print(f"\n\nFinished streaming {chunks_sent} chunks")

        # Send stop to finalize
        stop_msg = {"type": "stop"}
        print("Sending stop command...")
        await ws.send(json.dumps(stop_msg))

        # Collect remaining transcripts
        print("Collecting final transcripts...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                data = json.loads(msg)
                if data.get("type") == "transcript":
                    for seg in data.get("segments", []):
                        text = seg.get("text", "")
                        if text:
                            print(f"  [FINAL] {seg.get('start_time', 0):.2f}s: \"{text}\"")
                            transcripts.append(text)
                elif data.get("type") == "stopped":
                    print("  Received stopped confirmation")
                    break
        except asyncio.TimeoutError:
            print("  Timeout waiting for final messages")

        print(f"\n=== Results ===")
        print(f"Total transcripts received: {len(transcripts)}")

        if transcripts:
            print(f"\nFull transcript:")
            print(" ".join(transcripts))
            print("\n SUCCESS: Canary VAD WebSocket transcription working!")
            return True
        else:
            print("\n FAILURE: No transcripts received!")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_canary_vad())
    exit(0 if success else 1)
