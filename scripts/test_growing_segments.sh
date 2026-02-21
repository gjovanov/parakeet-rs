#!/usr/bin/env bash
# Test script for Growing Segments ASR mode
# Validates that the server produces PARTIAL and FINAL subtitle messages
# in the expected growing-segment pattern.
#
# Usage:
#   ./scripts/test_growing_segments.sh [SERVER_URL] [MEDIA_ID] [LANGUAGE] [MODEL]
#
# Defaults:
#   SERVER_URL=http://localhost:3000
#   MEDIA_ID=broadcast_1
#   LANGUAGE=de
#   MODEL=(first available model from /api/models)

set -euo pipefail

SERVER_URL="${1:-http://localhost:3000}"
MEDIA_ID="${2:-broadcast_1}"
LANGUAGE="${3:-de}"
MODEL="${4:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== Growing Segments ASR Mode Test ===${NC}"
echo "Server: $SERVER_URL"
echo "Media:  $MEDIA_ID"
echo "Language: $LANGUAGE"

# Check dependencies
for cmd in curl jq websocat; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error: '$cmd' is required but not installed.${NC}"
        exit 1
    fi
done

# 1. Verify server is running
echo -e "\n${YELLOW}[1/5] Checking server health...${NC}"
if ! curl -sf "$SERVER_URL/api/modes" >/dev/null 2>&1; then
    echo -e "${RED}Error: Server not reachable at $SERVER_URL${NC}"
    echo "Start the server first, then re-run this script."
    exit 1
fi
echo -e "${GREEN}Server is running.${NC}"

# 2. Verify growing_segments mode is available
echo -e "\n${YELLOW}[2/5] Verifying growing_segments mode...${NC}"
MODES=$(curl -sf "$SERVER_URL/api/modes")
if ! echo "$MODES" | jq -e '.data[] | select(.id == "growing_segments")' >/dev/null 2>&1; then
    echo -e "${RED}Error: 'growing_segments' mode not found in /api/modes${NC}"
    echo "Available modes:"
    echo "$MODES" | jq -r '.data[].id'
    exit 1
fi
echo -e "${GREEN}Mode 'growing_segments' is registered.${NC}"

# 3. Auto-detect model if not provided
if [ -z "$MODEL" ]; then
    echo -e "\n${YELLOW}[3/5] Auto-detecting model...${NC}"
    MODEL=$(curl -sf "$SERVER_URL/api/models" | jq -r '.data[0].id // empty')
    if [ -z "$MODEL" ]; then
        echo -e "${RED}Error: No models available.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}Using model: $MODEL${NC}"

# 4. Create a session
echo -e "\n${YELLOW}[4/5] Creating session...${NC}"
SESSION_RESP=$(curl -sf -X POST "$SERVER_URL/api/sessions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"$MODEL\",
        \"mode\": \"growing_segments\",
        \"language\": \"$LANGUAGE\",
        \"media_id\": \"$MEDIA_ID\"
    }")

SESSION_ID=$(echo "$SESSION_RESP" | jq -r '.data.id // .data.session_id // empty')
if [ -z "$SESSION_ID" ]; then
    echo -e "${RED}Error: Failed to create session.${NC}"
    echo "$SESSION_RESP" | jq .
    exit 1
fi
echo -e "${GREEN}Session created: $SESSION_ID${NC}"

# 5. Connect via WebSocket and capture subtitles
echo -e "\n${YELLOW}[5/5] Connecting WebSocket and capturing subtitles...${NC}"

WS_URL="${SERVER_URL/http/ws}/ws/$SESSION_ID"
TMPFILE=$(mktemp /tmp/growing_segments_XXXXXX.jsonl)
trap "rm -f $TMPFILE" EXIT

# Capture messages for up to 120 seconds
timeout 120 websocat -t "$WS_URL" 2>/dev/null | while IFS= read -r line; do
    echo "$line" >> "$TMPFILE"
    # Check if it's a subtitle message
    TYPE=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)
    if [ "$TYPE" = "subtitle" ]; then
        IS_FINAL=$(echo "$line" | jq -r '.is_final // false' 2>/dev/null)
        TEXT=$(echo "$line" | jq -r '.text // empty' 2>/dev/null)
        if [ "$IS_FINAL" = "true" ]; then
            echo -e "${GREEN}[FINAL]${NC}   $TEXT"
        else
            echo -e "${YELLOW}[PARTIAL]${NC} $TEXT"
        fi
    elif [ "$TYPE" = "status" ]; then
        STATUS=$(echo "$line" | jq -r '.status // empty' 2>/dev/null)
        echo -e "${CYAN}[STATUS]${NC}  $STATUS"
        if [ "$STATUS" = "completed" ] || [ "$STATUS" = "finished" ]; then
            break
        fi
    fi
done || true

echo ""

# Validation
TOTAL=$(jq -r 'select(.type == "subtitle")' "$TMPFILE" 2>/dev/null | wc -l)
PARTIALS=$(jq -r 'select(.type == "subtitle" and .is_final == false)' "$TMPFILE" 2>/dev/null | wc -l)
FINALS=$(jq -r 'select(.type == "subtitle" and .is_final == true)' "$TMPFILE" 2>/dev/null | wc -l)

echo -e "${CYAN}=== Results ===${NC}"
echo "Total subtitle messages: $TOTAL"
echo "  PARTIAL: $PARTIALS"
echo "  FINAL:   $FINALS"

PASS=true
if [ "$TOTAL" -eq 0 ]; then
    echo -e "${RED}FAIL: No subtitle messages received.${NC}"
    PASS=false
fi
if [ "$FINALS" -eq 0 ]; then
    echo -e "${RED}FAIL: No FINAL messages received.${NC}"
    PASS=false
fi
if [ "$PARTIALS" -eq 0 ]; then
    echo -e "${YELLOW}WARN: No PARTIAL messages received (may indicate too-slow processing).${NC}"
fi

if [ "$PASS" = true ]; then
    echo -e "\n${GREEN}PASS: Growing segments mode produced expected PARTIAL/FINAL pattern.${NC}"
else
    echo -e "\n${RED}FAIL: Growing segments validation failed.${NC}"
    exit 1
fi
