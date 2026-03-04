#!/bin/bash
# Start coturn TURN server in Docker using coturn/turnserver.conf
# Uses shared-secret (ephemeral) auth matching TURN_SHARED_SECRET in .env

set -e

CONTAINER_NAME="parakeet-coturn"
CONF_FILE="$(cd "$(dirname "$0")" && pwd)/coturn/turnserver.conf"

if [ ! -f "$CONF_FILE" ]; then
  echo "ERROR: $CONF_FILE not found"
  exit 1
fi

# Stop existing container if running
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting coturn with config: $CONF_FILE"

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  -v "$CONF_FILE":/etc/turnserver.conf:ro \
  coturn/coturn:latest

if [ $? -eq 0 ]; then
  echo ""
  echo "Coturn started (container: $CONTAINER_NAME)"
  echo "  Listening:  0.0.0.0:3478"
  echo "  Relay:      10.84.17.28 ports 10000-13000"
  echo "  Auth:       shared-secret (ephemeral credentials)"
  echo ""
  echo "  Logs:   docker logs -f $CONTAINER_NAME"
  echo "  Stop:   docker stop $CONTAINER_NAME"
else
  echo "Failed to start coturn"
  exit 1
fi
