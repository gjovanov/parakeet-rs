# Frontend Guide

> **Navigation**: [Index](./README.md) | [Architecture](./architecture.md) | [API Reference](./api-reference.md) | [Latency Modes](./latency-modes.md) | Frontend | [Deployment](./deployment.md)

The frontend is a multi-session web application built with vanilla JavaScript modules.

## File Structure

```
frontend/
├── index.html              # Main HTML with multi-session UI
├── css/
│   └── styles.css          # Dark theme with speaker colors
└── js/
    ├── config.js           # Dynamic config from /api/config
    ├── main-sessions.js    # Multi-session entry point
    ├── main-webrtc.js      # Single-session entry point (alternative)
    └── modules/
        ├── webrtc.js       # WebRTC client implementation
        ├── session-manager.js  # Session API client
        ├── subtitles.js    # Subtitle rendering
        ├── websocket.js    # WebSocket signaling
        ├── audio.js        # Audio utilities
        └── utils.js        # Helper functions
```

---

## UI Components

### 1. Session Panel

Tabbed interface with three views:

- **Sessions**: List of active sessions with status and progress
- **Create New**: Form to create sessions (model, media, mode, language)
- **Media Files**: Upload zone and file list

### 2. Player Controls

- Play/pause button
- Progress bar with current time / total duration
- Volume control
- Join/leave session button

### 3. Live Subtitle Display

- Current speaker with color-coded text
- Real-time updates as transcription progresses
- Fade animation for new subtitles

### 4. Transcript View

- Full transcript with timestamps
- Speaker labels with colors
- Click to seek functionality
- Export options (TXT, JSON, timestamps)

---

## JavaScript Modules

### Session Manager (`session-manager.js`)

Handles all session API interactions:

```javascript
import { SessionManager } from './modules/session-manager.js';

const sessionManager = new SessionManager();

// Fetch initial data
await sessionManager.fetchModels();
await sessionManager.fetchMedia();
await sessionManager.fetchModes();
await sessionManager.fetchSessions();

// Create and start a session
const session = await sessionManager.createSession(
  'tdt-en',      // model_id
  'media-uuid',  // media_id
  'speedy',      // mode
  'en'           // language
);
await sessionManager.startSession(session.id);

// Stop a session
await sessionManager.stopSession(session.id);

// Upload media
const file = document.querySelector('input[type="file"]').files[0];
await sessionManager.uploadMedia(file);

// Delete media
await sessionManager.deleteMedia('media-uuid');

// Get WebSocket URL for session
const wsUrl = sessionManager.getWebSocketUrl(session.id);
// Returns: ws://host:port/ws/{session_id}

// Event listeners
sessionManager.on('modelsLoaded', (models) => { ... });
sessionManager.on('mediaLoaded', (files) => { ... });
sessionManager.on('modesLoaded', (modes) => { ... });
sessionManager.on('sessionsUpdated', (sessions) => { ... });
sessionManager.on('sessionCreated', (session) => { ... });
sessionManager.on('sessionStarted', (session) => { ... });
sessionManager.on('sessionStopped', (sessionId) => { ... });
sessionManager.on('mediaUploaded', (file) => { ... });
sessionManager.on('mediaDeleted', (mediaId) => { ... });

// Start polling for session updates
sessionManager.startPolling(3000); // 3 second interval
sessionManager.stopPolling();
```

### WebRTC Client (`webrtc.js`)

Handles WebRTC connection and audio playback:

```javascript
import { WebRTCClient } from './modules/webrtc.js';
import { getConfig } from './config.js';

const config = getConfig();
const wsUrl = sessionManager.getWebSocketUrl(session.id);

const webrtcClient = new WebRTCClient(wsUrl, {
  iceServers: config.iceServers
});

// Connect to session
const audioElement = document.getElementById('audio-player');
await webrtcClient.connect(audioElement);

// Audio controls
await webrtcClient.play();
webrtcClient.pause();
webrtcClient.setVolume(0.8);

// Get current playback state
const currentTime = webrtcClient.currentTime;
const duration = webrtcClient.duration;
const isPlaying = webrtcClient.playing;

// Get connection stats
const stats = await webrtcClient.getStats();
// { bytesReceived, packetsReceived, packetsLost, jitter, roundTripTime }

// Disconnect
webrtcClient.disconnect();
```

### WebRTC Client Events

```javascript
// Session info received
webrtcClient.on('welcome', (msg) => {
  console.log('Client ID:', msg.client_id);
  console.log('Session:', msg.session);
});

// WebRTC connection established
webrtcClient.on('connected', () => {
  console.log('WebRTC connected');
});

// Audio track received and ready
webrtcClient.on('trackReceived', (stream) => {
  console.log('Audio stream ready');
});

// Browser blocked autoplay
webrtcClient.on('autoplayBlocked', () => {
  // Show play button to user
});

// Transcription segment received
webrtcClient.on('subtitle', (segment) => {
  console.log('Speaker', segment.speaker, ':', segment.text);
  console.log('Time:', segment.start, '-', segment.end);
  console.log('Final:', segment.isFinal);
});

// Progress update
webrtcClient.on('status', ({ bufferTime, totalDuration }) => {
  console.log('Progress:', bufferTime, '/', totalDuration);
});

// Stream completed
webrtcClient.on('end', ({ totalDuration }) => {
  console.log('Finished:', totalDuration, 'seconds');
});

// Connection closed
webrtcClient.on('disconnect', ({ code, reason }) => {
  console.log('Disconnected:', code, reason);
});

// Connection failed
webrtcClient.on('connectionFailed', () => {
  console.error('WebRTC connection failed');
});

// Server error
webrtcClient.on('serverError', ({ message }) => {
  console.error('Server error:', message);
});

// General error
webrtcClient.on('error', (error) => {
  console.error('Error:', error);
});
```

### Subtitle Renderer (`subtitles.js`)

Renders subtitles in the UI:

```javascript
import { SubtitleRenderer } from './modules/subtitles.js';

const subtitleRenderer = new SubtitleRenderer(
  document.getElementById('live-subtitle'),    // Live subtitle element
  document.getElementById('transcript-content'), // Transcript container
  {
    maxSegments: 1000,
    autoScroll: true,
    showTimestamps: true,
    speakerColors: config.speakerColors
  }
);

// Add a subtitle segment
subtitleRenderer.addSegment({
  text: 'Hello world',
  speaker: 0,
  start: 1.5,
  end: 2.3,
  isFinal: true
});

// Update current playback time (for highlighting)
subtitleRenderer.updateTime(currentTime);

// Configuration
subtitleRenderer.setAutoScroll(true);
subtitleRenderer.setShowTimestamps(true);

// Export
const plainText = subtitleRenderer.getTranscript();
const withTimestamps = subtitleRenderer.getTranscriptWithTimestamps();
const jsonExport = subtitleRenderer.exportJSON();

// Clear all
subtitleRenderer.clear();
subtitleRenderer.clearCurrent(); // Just clear live subtitle

// Seek event (when user clicks timestamp)
subtitleRenderer.on('seek', (time) => {
  audioElement.currentTime = time;
});
```

### Config Loader (`config.js`)

Loads runtime configuration from server:

```javascript
import { loadConfig, getConfig } from './config.js';

// Load config from server (call once at startup)
await loadConfig();

// Get cached config
const config = getConfig();
// {
//   wsUrl: 'ws://...',
//   iceServers: [...],
//   speakerColors: [...],
//   subtitles: { maxSegments, autoScroll, showTimestamps }
// }
```

---

## Supported Languages

The frontend supports 25 European languages for Canary model:

| Code | Language | Code | Language |
|------|----------|------|----------|
| de | German | en | English |
| fr | French | es | Spanish |
| it | Italian | pt | Portuguese |
| nl | Dutch | pl | Polish |
| ru | Russian | uk | Ukrainian |
| cs | Czech | sk | Slovak |
| hu | Hungarian | ro | Romanian |
| bg | Bulgarian | hr | Croatian |
| sl | Slovenian | et | Estonian |
| lv | Latvian | lt | Lithuanian |
| fi | Finnish | sv | Swedish |
| da | Danish | el | Greek |
| mt | Maltese | | |

---

## Complete Integration Example

```javascript
import { loadConfig, getConfig } from './config.js';
import { WebRTCClient } from './modules/webrtc.js';
import { SubtitleRenderer } from './modules/subtitles.js';
import { SessionManager } from './modules/session-manager.js';

async function main() {
  // 1. Load configuration
  const config = await loadConfig();

  // 2. Initialize managers
  const sessionManager = new SessionManager();
  const subtitleRenderer = new SubtitleRenderer(
    document.getElementById('live-subtitle'),
    document.getElementById('transcript'),
    { speakerColors: config.speakerColors }
  );

  // 3. Load initial data
  await Promise.all([
    sessionManager.fetchModels(),
    sessionManager.fetchMedia(),
    sessionManager.fetchModes(),
    sessionManager.fetchSessions()
  ]);

  // 4. Create and start a session
  const session = await sessionManager.createSession(
    'tdt-en',
    sessionManager.mediaFiles[0].id,
    'speedy',
    'en'
  );
  await sessionManager.startSession(session.id);

  // 5. Connect via WebRTC
  const webrtcClient = new WebRTCClient(
    sessionManager.getWebSocketUrl(session.id),
    { iceServers: config.iceServers }
  );

  webrtcClient.on('subtitle', (segment) => {
    subtitleRenderer.addSegment(segment);
  });

  webrtcClient.on('status', ({ bufferTime }) => {
    subtitleRenderer.updateTime(bufferTime);
  });

  const audioElement = document.getElementById('audio');
  await webrtcClient.connect(audioElement);

  // 6. Handle playback
  document.getElementById('play').onclick = () => webrtcClient.play();
  document.getElementById('pause').onclick = () => webrtcClient.pause();

  // 7. Handle export
  document.getElementById('export').onclick = () => {
    const transcript = subtitleRenderer.exportJSON();
    // Download transcript...
  };

  // 8. Cleanup on page unload
  window.onbeforeunload = () => {
    webrtcClient.disconnect();
    sessionManager.stopPolling();
  };
}

main();
```

---

## Styling

The frontend uses CSS custom properties for theming:

```css
:root {
  /* Theme colors */
  --primary: #1a1a2e;
  --secondary: #16213e;
  --accent: #e94560;
  --text: #eaeaea;
  --text-muted: #8892b0;

  /* Speaker colors */
  --speaker-0: #4A90D9;
  --speaker-1: #50C878;
  --speaker-2: #E9967A;
  --speaker-3: #DDA0DD;
  --speaker-4: #F0E68C;
  --speaker-5: #87CEEB;
  --speaker-6: #FFB6C1;
  --speaker-7: #98FB98;
}
```

---

## Related Documentation

- [API Reference](./api-reference.md) - Backend API endpoints
- [Architecture](./architecture.md) - System design
- [Deployment](./deployment.md) - Server configuration
