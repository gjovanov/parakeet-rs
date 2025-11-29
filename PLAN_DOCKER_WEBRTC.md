# Docker Containerization with coturn TURN Server

## Problem Statement

The WebRTC implementation is complete but audio isn't playing due to ICE connectivity issues between Windows browser and WSL2 server. Even with `set_nat_1to1_ips(vec!["127.0.0.1"])`, the WebRTC connection doesn't reach "connected" state.

**Root Cause**: In WSL2/Docker environments, direct peer-to-peer connections often fail due to NAT traversal issues. A TURN server is needed to relay media when direct connections aren't possible.

---

## Proposed Solution

Create a Docker-based deployment with:
1. **coturn** - TURN/STUN server for reliable WebRTC connectivity
2. **parakeet-webrtc** - Containerized transcriber service
3. **nginx** - Static file server for frontend
4. **docker compose** - Single-command deployment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Machine (WSL2)                       │
├─────────────────────────────────────────────────────────────────┤
│  docker compose                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    coturn       │  │ parakeet-webrtc │  │     nginx       │  │
│  │  (TURN/STUN)    │  │  (transcriber)  │  │   (frontend)    │  │
│  │                 │  │                 │  │                 │  │
│  │  UDP 3478       │  │  TCP 8080 (WS)  │  │  TCP 80 (HTTP)  │  │
│  │  UDP 49152-65535│  │  UDP 50000-60000│  │                 │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
├───────────┴────────────────────┴────────────────────┴───────────┤
│                      Docker Network (webrtc-net)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Exposed Ports:
                    - 3478/udp (STUN/TURN)
                    - 5349/tcp (TURN TLS)
                    - 8080/tcp (WebSocket signaling)
                    - 80/tcp (Frontend)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Windows Browser                             │
│  - Connects to http://localhost (nginx)                         │
│  - WebRTC signaling via ws://localhost:8080/ws                  │
│  - ICE uses TURN at turn:localhost:3478                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: coturn Container Configuration

**File: `docker/coturn/turnserver.conf`**
```conf
# TURN server configuration
listening-port=3478
tls-listening-port=5349
fingerprint
lt-cred-mech
realm=parakeet.local
server-name=parakeet.local

# Static user credentials
user=parakeet:parakeet123

# Relay ports
min-port=49152
max-port=65535

# External IP (will be overridden by docker compose)
external-ip=127.0.0.1

# Logging
log-file=stdout
verbose

# No TLS for local development
no-tls
no-dtls
```

### Phase 2: Transcriber Container (Multi-stage Build)

**File: `docker/Dockerfile.transcriber`**
```dockerfile
# Stage 1: Build
FROM rust:1.75-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    clang \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY examples/ ./examples/

# Build release binary
RUN cargo build --release --example webrtc_transcriber

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/examples/webrtc_transcriber /app/

# Create volume mount points for models
VOLUME ["/app/models/tdt", "/app/models/diarization"]

# Expose ports
EXPOSE 8080

# Environment variables
ENV RUST_LOG=info
ENV TDT_MODEL_PATH=/app/models/tdt
ENV DIARIZATION_MODEL_PATH=/app/models/diarization

ENTRYPOINT ["/app/webrtc_transcriber"]
CMD ["--tdt-model", "/app/models/tdt", "--low-latency"]
```

### Phase 3: Frontend Container

**File: `docker/Dockerfile.frontend`**
```dockerfile
FROM nginx:alpine

# Copy frontend files
COPY frontend/ /usr/share/nginx/html/

# Copy nginx config
COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

**File: `docker/nginx/default.conf`**
```nginx
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index-webrtc.html;

    location / {
        try_files $uri $uri/ /index-webrtc.html;
    }

    # WebSocket proxy to transcriber
    location /ws {
        proxy_pass http://transcriber:8080/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### Phase 4: Docker Compose

**File: `docker compose.yml`**
```yaml
version: '3.8'

services:
  coturn:
    image: coturn/coturn:4.6
    container_name: parakeet-turn
    restart: unless-stopped
    network_mode: host
    volumes:
      - ./docker/coturn/turnserver.conf:/etc/coturn/turnserver.conf:ro
    command: -c /etc/coturn/turnserver.conf

  transcriber:
    build:
      context: .
      dockerfile: docker/Dockerfile.transcriber
    container_name: parakeet-transcriber
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./tdt:/app/models/tdt:ro
      - ./diar_sortformer_4spk-v1.onnx:/app/models/diarization/model.onnx:ro
    environment:
      - RUST_LOG=info
      - TURN_SERVER=turn:127.0.0.1:3478
      - TURN_USERNAME=parakeet
      - TURN_PASSWORD=parakeet123
    depends_on:
      - coturn
    stdin_open: true  # For piping audio

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    container_name: parakeet-frontend
    restart: unless-stopped
    ports:
      - "80:80"
    depends_on:
      - transcriber

networks:
  default:
    name: parakeet-webrtc
```

### Phase 5: Rust Code Updates

**Update `examples/webrtc_transcriber.rs`** to use TURN server:

1. Read TURN credentials from environment variables
2. Add TURN server to ICE configuration
3. Use relay candidates in addition to host candidates

```rust
// Add to webrtc_transcriber.rs - ICE configuration
let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();

let ice_servers = if !turn_server.is_empty() {
    vec![
        RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_owned()],
            ..Default::default()
        },
        RTCIceServer {
            urls: vec![turn_server],
            username: turn_username,
            credential: turn_password,
            ..Default::default()
        },
    ]
} else {
    vec![RTCIceServer {
        urls: vec!["stun:stun.l.google.com:19302".to_owned()],
        ..Default::default()
    }]
};
```

**Update `frontend/js/modules/webrtc.js`** to use TURN server:

```javascript
// Update ICE configuration in WebRTCClient
this.options = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    {
      urls: 'turn:localhost:3478',
      username: 'parakeet',
      credential: 'parakeet123'
    }
  ],
  ...options,
};
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `docker/coturn/turnserver.conf` | TURN server configuration |
| `docker/Dockerfile.transcriber` | Multi-stage build for transcriber |
| `docker/Dockerfile.frontend` | Nginx frontend container |
| `docker/nginx/default.conf` | Nginx configuration with WS proxy |
| `docker compose.yml` | Orchestration configuration |
| `.dockerignore` | Exclude unnecessary files from build |

## Files to Modify

| File | Changes |
|------|---------|
| `examples/webrtc_transcriber.rs` | Add TURN server configuration from env vars |
| `frontend/js/modules/webrtc.js` | Add TURN server to ICE configuration |
| `frontend/js/config.js` | Add TURN server config |

---

## Usage

### Development (without Docker)

```bash
# Start coturn in Docker only
docker run -d --network host coturn/coturn:4.6 \
  -n --log-file=stdout --lt-cred-mech \
  --user=parakeet:parakeet123 --realm=parakeet.local

# Run transcriber locally
ffmpeg -i audio.wav -f s16le -ar 16000 -ac 1 - | \
  TURN_SERVER=turn:127.0.0.1:3478 \
  TURN_USERNAME=parakeet \
  TURN_PASSWORD=parakeet123 \
  cargo run --release --example webrtc_transcriber -- --tdt-model ./tdt --low-latency

# Serve frontend
python3 -m http.server 3000 -d frontend
```

### Production (full Docker)

```bash
# Build and start all services
docker compose up -d --build

# Stream audio to transcriber
docker exec -i parakeet-transcriber sh -c 'cat > /tmp/audio.raw' < audio.raw
# OR use named pipe
mkfifo /tmp/audio_pipe
docker run -i --rm -v /tmp/audio_pipe:/tmp/audio_pipe alpine cat /tmp/audio_pipe | \
  docker exec -i parakeet-transcriber /app/webrtc_transcriber --tdt-model /app/models/tdt
```

### Testing

```bash
# Open browser
xdg-open http://localhost

# Check TURN connectivity
turnutils_uclient -u parakeet -w parakeet123 localhost

# View logs
docker compose logs -f
```

---

## Model Volume Mapping

Models are mounted as volumes rather than baked into the image:

| Model | Host Path | Container Path | Size |
|-------|-----------|----------------|------|
| TDT | `./tdt/` | `/app/models/tdt/` | ~2.5GB |
| Diarization | `./diar_sortformer_4spk-v1.onnx` | `/app/models/diarization/model.onnx` | ~500MB |

This approach:
- Keeps Docker images small
- Allows model updates without rebuilding
- Supports different model versions

---

## Implementation Order

1. **Create Docker directory structure**
2. **Create coturn configuration**
3. **Create Dockerfile.transcriber** (multi-stage build)
4. **Create Dockerfile.frontend** (nginx)
5. **Create nginx config** (with WS proxy)
6. **Create docker compose.yml**
7. **Update webrtc_transcriber.rs** (TURN support)
8. **Update frontend webrtc.js** (TURN ICE servers)
9. **Create .dockerignore**
10. **Test and verify**

---

## Success Criteria

1. `docker compose up` starts all services
2. Browser connects via WebRTC with TURN relay
3. Audio plays without interruption
4. Subtitles appear in real-time
5. RTT/Jitter stats displayed in UI

---

## Alternative: Simple coturn Fix First

Before full containerization, we could test if coturn alone fixes the issue:

```bash
# Quick test with coturn only
docker run -d --name coturn --network host coturn/coturn:4.6 \
  -n --log-file=stdout --lt-cred-mech \
  --user=test:test123 --realm=test.local

# Update frontend to use TURN and test
```

This would validate the TURN approach before investing in full containerization.
