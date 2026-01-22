#!/bin/bash
# Start a local coturn TURN server in Docker for WebRTC relay

# Get WSL2 IP address
WSL_IP=$(ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
echo "WSL2 IP: $WSL_IP"

# Configuration
TURN_PORT=3478
TURN_USER="localuser"
TURN_PASS="localpass"
REALM="local"

# Stop existing container if running
docker rm -f local-coturn 2>/dev/null

# Start coturn container with explicit port mapping
docker run -d \
  --name local-coturn \
  -p 3478:3478 \
  -p 3478:3478/udp \
  -p 49152-49252:49152-49252/udp \
  coturn/coturn:latest \
  -n \
  --log-file=stdout \
  --external-ip="$WSL_IP" \
  --listening-ip="0.0.0.0" \
  --listening-port=$TURN_PORT \
  --min-port=49152 \
  --max-port=49252 \
  --realm=$REALM \
  --user="$TURN_USER:$TURN_PASS" \
  --lt-cred-mech \
  --fingerprint \
  --no-tls \
  --no-dtls \
  --no-multicast-peers \
  --server-relay \
  --verbose

if [ $? -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "  Coturn TURN server started!"
  echo "=========================================="
  echo ""
  echo "TURN URL:      turn:$WSL_IP:$TURN_PORT?transport=udp"
  echo "               turn:$WSL_IP:$TURN_PORT?transport=tcp"
  echo "Username:      $TURN_USER"
  echo "Password:      $TURN_PASS"
  echo ""
  echo "Update your .env file:"
  echo "  TURN_SERVER=turn:$WSL_IP:$TURN_PORT?transport=udp"
  echo "  TURN_USERNAME=$TURN_USER"
  echo "  TURN_PASSWORD=$TURN_PASS"
  echo ""
  echo "View logs: docker logs -f local-coturn"
  echo "Stop:      docker stop local-coturn"
  echo "=========================================="
else
  echo "Failed to start coturn container"
  exit 1
fi
