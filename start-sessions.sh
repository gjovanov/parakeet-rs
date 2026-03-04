#!/bin/bash
# Start audio-only (without transcription) sessions for all SRT channels
# Reads SRT_CHANNELS from .env and creates one session per channel via the API

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
API_BASE="http://localhost:80"

# Load .env
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found"
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

if [ -z "$SRT_CHANNELS" ]; then
  echo "ERROR: SRT_CHANNELS not set in .env"
  exit 1
fi

# Parse channel count
CHANNEL_COUNT=$(echo "$SRT_CHANNELS" | jq 'length')
echo "Found $CHANNEL_COUNT SRT channels"
echo ""

for i in $(seq 0 $((CHANNEL_COUNT - 1))); do
  NAME=$(echo "$SRT_CHANNELS" | jq -r ".[$i].name")

  echo -n "Creating audio-only session for $NAME (channel $i)... "

  RESPONSE=$(curl -s -X POST "$API_BASE/api/sessions" \
    -H "Content-Type: application/json" \
    -d "{\"srt_channel_id\": $i, \"without_transcription\": true}")

  SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
  if [ "$SUCCESS" = "true" ]; then
    SESSION_ID=$(echo "$RESPONSE" | jq -r '.data.id')
    echo "created ($SESSION_ID)"

    # Start the session
    START_RESPONSE=$(curl -s -X POST "$API_BASE/api/sessions/$SESSION_ID/start")
    START_OK=$(echo "$START_RESPONSE" | jq -r '.success')
    if [ "$START_OK" = "true" ]; then
      echo "  -> started"
    else
      ERROR=$(echo "$START_RESPONSE" | jq -r '.error // "unknown error"')
      echo "  -> start failed: $ERROR"
    fi
  else
    ERROR=$(echo "$RESPONSE" | jq -r '.error // "unknown error"')
    echo "failed: $ERROR"
  fi
done

echo ""
echo "Done. All sessions started."
