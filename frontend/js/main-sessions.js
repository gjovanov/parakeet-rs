/**
 * Main entry point for multi-session mode
 */
import { loadConfig, getConfig } from './config.js';
import { WebRTCClient } from './modules/webrtc.js';
import { SubtitleRenderer } from './modules/subtitles.js';
import { formatTime } from './modules/utils.js';
import { SessionManager, formatDuration, formatFileSize } from './modules/session-manager.js';

// Application state
const state = {
  connected: false,
  totalDuration: 0,
  selectedSessionId: null,
  subtitleCount: 0,  // Counter for received subtitles
};

// Modules
let webrtcClient = null;
let subtitleRenderer = null;
let sessionManager = null;

// DOM elements
const elements = {};

/**
 * Initialize the application
 */
async function init() {
  // Load configuration from server
  const config = await loadConfig();

  // Cache DOM elements
  cacheElements();

  // Initialize session manager
  sessionManager = new SessionManager();
  setupSessionManagerListeners();

  // Initialize subtitle renderer
  subtitleRenderer = new SubtitleRenderer(
    elements.liveSubtitle,
    elements.transcriptContent,
    {
      maxSegments: config.subtitles.maxSegments,
      autoScroll: config.subtitles.autoScroll,
      showTimestamps: config.subtitles.showTimestamps,
      speakerColors: config.speakerColors,
    }
  );

  // Set up UI event listeners
  setupEventListeners();

  // Load initial data
  await Promise.all([
    sessionManager.fetchModels(),
    sessionManager.fetchMedia(),
    sessionManager.fetchModes(),
    sessionManager.fetchSessions(),
  ]);

  // Start polling for session updates
  sessionManager.startPolling(3000);

  console.log('Multi-session application initialized');
}

function cacheElements() {
  elements.connectionStatus = document.getElementById('connection-status');
  elements.bufferInfo = document.getElementById('buffer-info');
  elements.durationInfo = document.getElementById('duration-info');
  elements.playPauseBtn = document.getElementById('play-pause');
  elements.currentTime = document.getElementById('current-time');
  elements.totalDuration = document.getElementById('total-duration');
  elements.progressBuffered = document.getElementById('progress-buffered');
  elements.progressPlayed = document.getElementById('progress-played');
  elements.progressContainer = document.getElementById('progress-container');
  elements.connectBtn = document.getElementById('connect-btn');
  elements.liveSubtitle = document.getElementById('live-subtitle');
  elements.transcriptContent = document.getElementById('transcript-content');
  elements.autoScrollCheckbox = document.getElementById('auto-scroll');
  elements.showTimestampsCheckbox = document.getElementById('show-timestamps');
  elements.exportBtn = document.getElementById('export-btn');
  elements.clearBtn = document.getElementById('clear-btn');
  elements.exportModal = document.getElementById('export-modal');
  elements.audioPlayer = document.getElementById('audio-player');
  elements.latencyInfo = document.getElementById('latency-info');

  // Session panel elements
  elements.sessionTabs = document.querySelectorAll('.session-tab');
  elements.sessionContents = document.querySelectorAll('.session-content');
  elements.sessionsList = document.getElementById('sessions-list');
  elements.modelSelect = document.getElementById('model-select');
  elements.mediaSelect = document.getElementById('media-select');
  elements.modeSelect = document.getElementById('mode-select');
  elements.languageSelect = document.getElementById('language-select');
  elements.createSessionBtn = document.getElementById('create-session-btn');
  elements.uploadZone = document.getElementById('upload-zone');
  elements.fileInput = document.getElementById('file-input');
  elements.mediaList = document.getElementById('media-list');
}

function setupSessionManagerListeners() {
  sessionManager.on('modelsLoaded', renderModelSelect);
  sessionManager.on('mediaLoaded', () => {
    renderMediaSelect();
    renderMediaList();
  });
  sessionManager.on('modesLoaded', renderModeSelect);
  sessionManager.on('sessionsUpdated', renderSessionsList);
  sessionManager.on('sessionCreated', (session) => {
    selectSession(session.id);
    showTab('sessions');
  });
  sessionManager.on('mediaUploaded', () => {
    elements.bufferInfo.textContent = 'File uploaded!';
  });
}

function setupEventListeners() {
  // Tab switching
  elements.sessionTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      showTab(tab.dataset.tab);
    });
  });

  // Create session
  elements.createSessionBtn.addEventListener('click', async () => {
    const modelId = elements.modelSelect.value;
    const mediaId = elements.mediaSelect.value;
    const mode = elements.modeSelect?.value || 'speedy';
    const language = elements.languageSelect?.value || 'de';

    if (!modelId || !mediaId) {
      alert('Please select a model and media file');
      return;
    }

    try {
      elements.createSessionBtn.disabled = true;
      elements.createSessionBtn.textContent = 'Creating...';
      const session = await sessionManager.createSession(modelId, mediaId, mode, language);
      // Auto-start the session
      await sessionManager.startSession(session.id);
    } catch (e) {
      alert('Failed to create session: ' + e.message);
    } finally {
      elements.createSessionBtn.disabled = false;
      elements.createSessionBtn.textContent = 'Create Session';
    }
  });

  // File upload
  elements.uploadZone.addEventListener('click', () => {
    elements.fileInput.click();
  });

  elements.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.add('dragover');
  });

  elements.uploadZone.addEventListener('dragleave', () => {
    elements.uploadZone.classList.remove('dragover');
  });

  elements.uploadZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) await uploadFile(file);
  });

  elements.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await uploadFile(file);
    e.target.value = '';
  });

  // Connect button
  elements.connectBtn.addEventListener('click', () => {
    if (state.connected) {
      disconnect();
    } else if (state.selectedSessionId) {
      connect(state.selectedSessionId);
    }
  });

  // Play/Pause button
  elements.playPauseBtn.addEventListener('click', async () => {
    if (webrtcClient?.playing) {
      webrtcClient.pause();
      elements.playPauseBtn.querySelector('.icon-play').style.display = 'inline';
      elements.playPauseBtn.querySelector('.icon-pause').style.display = 'none';
    } else if (webrtcClient) {
      await webrtcClient.play();
      elements.playPauseBtn.querySelector('.icon-play').style.display = 'none';
      elements.playPauseBtn.querySelector('.icon-pause').style.display = 'inline';
    }
  });

  // Audio element events
  elements.audioPlayer.addEventListener('play', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'none';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'inline';
  });

  elements.audioPlayer.addEventListener('pause', () => {
    elements.playPauseBtn.querySelector('.icon-play').style.display = 'inline';
    elements.playPauseBtn.querySelector('.icon-pause').style.display = 'none';
  });

  elements.audioPlayer.addEventListener('timeupdate', () => {
    const currentTime = elements.audioPlayer.currentTime;
    elements.currentTime.textContent = formatTime(currentTime);
    subtitleRenderer.updateTime(currentTime);
    updateProgressBar();
  });

  // Auto-scroll toggle
  elements.autoScrollCheckbox.addEventListener('change', (e) => {
    subtitleRenderer.setAutoScroll(e.target.checked);
  });

  // Show timestamps toggle
  elements.showTimestampsCheckbox.addEventListener('change', (e) => {
    subtitleRenderer.setShowTimestamps(e.target.checked);
  });

  // Export buttons
  elements.exportBtn.addEventListener('click', () => {
    elements.exportModal.style.display = 'flex';
  });

  document.getElementById('export-txt').addEventListener('click', () => {
    downloadTranscript('transcript.txt', subtitleRenderer.getTranscript());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('export-timestamps').addEventListener('click', () => {
    downloadTranscript('transcript_timestamps.txt', subtitleRenderer.getTranscriptWithTimestamps());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('export-json').addEventListener('click', () => {
    downloadTranscript('transcript.json', subtitleRenderer.exportJSON());
    elements.exportModal.style.display = 'none';
  });

  document.getElementById('close-modal').addEventListener('click', () => {
    elements.exportModal.style.display = 'none';
  });

  // Clear button
  elements.clearBtn.addEventListener('click', () => {
    if (confirm('Clear all transcript data?')) {
      subtitleRenderer.clear();
      state.totalDuration = 0;
      updateUI();
    }
  });

  // Close modal on outside click
  elements.exportModal.addEventListener('click', (e) => {
    if (e.target === elements.exportModal) {
      elements.exportModal.style.display = 'none';
    }
  });

  // Subtitle renderer events
  subtitleRenderer.on('seek', (time) => {
    if (elements.audioPlayer) {
      elements.audioPlayer.currentTime = time;
    }
  });

  // Update stats periodically
  setInterval(updateStats, 1000);
}

function showTab(tabName) {
  elements.sessionTabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });
  elements.sessionContents.forEach(content => {
    content.classList.toggle('active', content.id === `${tabName}-content`);
  });
}

function renderModelSelect() {
  const models = sessionManager.models;
  elements.modelSelect.innerHTML = models.length
    ? models.map(m => `<option value="${m.id}">${m.display_name}</option>`).join('')
    : '<option value="">No models available</option>';
}

function renderMediaSelect() {
  const files = sessionManager.mediaFiles;
  elements.mediaSelect.innerHTML = files.length
    ? files.map(f => `<option value="${f.id}">${f.filename} (${formatDuration(f.duration_secs)})</option>`).join('')
    : '<option value="">No media files</option>';
}

function renderModeSelect() {
  if (!elements.modeSelect) return;
  const modes = sessionManager.modes;
  elements.modeSelect.innerHTML = modes.length
    ? modes.map(m => `<option value="${m.id}" title="${m.description}">${m.name}</option>`).join('')
    : '<option value="speedy">Speedy (~0.3-1.5s)</option>';
}

function renderMediaList() {
  const files = sessionManager.mediaFiles;
  if (files.length === 0) {
    elements.mediaList.innerHTML = '<div class="empty-state"><p>No media files uploaded</p></div>';
    return;
  }

  elements.mediaList.innerHTML = files.map(f => `
    <div class="media-item">
      <div>
        <div class="media-name">${f.filename}</div>
        <div class="media-meta">${formatDuration(f.duration_secs)} | ${formatFileSize(f.size_bytes)}</div>
      </div>
      <button class="btn-small btn-danger" onclick="window.deleteMedia('${f.id}')">Delete</button>
    </div>
  `).join('');
}

// Expose delete function globally
window.deleteMedia = async (mediaId) => {
  if (confirm('Delete this media file?')) {
    await sessionManager.deleteMedia(mediaId);
  }
};

function renderSessionsList() {
  const sessions = sessionManager.sessions;
  if (sessions.length === 0) {
    elements.sessionsList.innerHTML = `
      <div class="empty-state">
        <h3>No active sessions</h3>
        <p>Create a new session to start transcribing</p>
      </div>
    `;
    return;
  }

  elements.sessionsList.innerHTML = sessions.map(s => {
    const progress = s.duration_secs > 0 ? (s.progress_secs / s.duration_secs) * 100 : 0;
    const isSelected = state.selectedSessionId === s.id;
    const modeLabel = s.mode ? s.mode.replace(/_/g, '-') : 'speedy';

    return `
      <div class="session-card ${isSelected ? 'selected' : ''}" onclick="window.selectSession('${s.id}')">
        <div class="session-info">
          <div class="session-title">${s.media_filename}</div>
          <div class="session-meta">${s.model_name} | <span class="mode-badge">${modeLabel}</span> | ${formatDuration(s.progress_secs)} / ${formatDuration(s.duration_secs)}</div>
          <div class="session-progress">
            <div class="session-progress-bar" style="width: ${progress}%"></div>
          </div>
        </div>
        <div class="session-actions">
          <span class="session-status ${s.state}">${s.state}</span>
          ${s.state === 'running' ? `<button class="btn-small btn-danger" onclick="event.stopPropagation(); window.stopSession('${s.id}')">Stop</button>` : ''}
        </div>
      </div>
    `;
  }).join('');
}

function selectSession(sessionId) {
  state.selectedSessionId = sessionId;
  elements.connectBtn.disabled = !sessionId;
  elements.bufferInfo.textContent = sessionId ? 'Ready to join' : 'Select a session';
  renderSessionsList();
}

// Expose functions globally
window.selectSession = selectSession;

window.stopSession = async (sessionId) => {
  if (confirm('Stop this session?')) {
    await sessionManager.stopSession(sessionId);
    if (state.selectedSessionId === sessionId) {
      selectSession(null);
    }
  }
};

async function uploadFile(file) {
  if (!file.name.match(/\.(wav|mp3)$/i)) {
    alert('Please upload a WAV or MP3 file');
    return;
  }

  try {
    elements.bufferInfo.textContent = 'Uploading...';
    await sessionManager.uploadMedia(file);
    elements.bufferInfo.textContent = 'Upload complete!';
  } catch (e) {
    alert('Upload failed: ' + e.message);
    elements.bufferInfo.textContent = 'Upload failed';
  }
}

async function connect(sessionId) {
  const config = getConfig();
  const wsUrl = sessionManager.getWebSocketUrl(sessionId);

  // Create WebRTC client
  webrtcClient = new WebRTCClient(wsUrl, {
    iceServers: config.iceServers
  });

  // WebRTC events
  webrtcClient.on('welcome', (msg) => {
    console.log('Server welcome:', msg);
    if (msg.session) {
      state.totalDuration = msg.session.duration_secs || 0;
      elements.totalDuration.textContent = formatTime(state.totalDuration);
    }
  });

  webrtcClient.on('connected', () => {
    state.connected = true;
    updateConnectionStatus('connected');
    elements.connectBtn.textContent = 'Leave Session';
    elements.playPauseBtn.disabled = false;
    elements.bufferInfo.textContent = 'WebRTC Connected';
    console.log('WebRTC connected');
  });

  webrtcClient.on('trackReceived', () => {
    console.log('Audio track received');
    elements.bufferInfo.textContent = 'Audio streaming';
  });

  webrtcClient.on('autoplayBlocked', () => {
    elements.bufferInfo.textContent = 'Click Play to start';
  });

  webrtcClient.on('disconnect', ({ code, reason }) => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.connectBtn.textContent = 'Join Session';
    subtitleRenderer.clearCurrent();
    console.log('Disconnected:', code, reason);
  });

  webrtcClient.on('connectionFailed', () => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
    subtitleRenderer.clearCurrent();
  });

  webrtcClient.on('error', (error) => {
    console.error('WebRTC error:', error);
  });

  webrtcClient.on('subtitle', (segment) => {
    state.subtitleCount++;
    elements.bufferInfo.textContent = `Subtitles: ${state.subtitleCount}`;
    subtitleRenderer.addSegment(segment);
  });

  webrtcClient.on('status', ({ bufferTime, totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.durationInfo.textContent = `Duration: ${formatTime(totalDuration)}`;
    elements.totalDuration.textContent = formatTime(totalDuration);
    updateProgressBar();
  });

  webrtcClient.on('end', ({ totalDuration }) => {
    state.totalDuration = totalDuration;
    elements.totalDuration.textContent = formatTime(totalDuration);
    console.log('Stream ended, total duration:', formatTime(totalDuration));
  });

  webrtcClient.on('serverError', ({ message }) => {
    console.error('Server error:', message);
    alert(`Server error: ${message}`);
  });

  // Reset subtitle counter
  state.subtitleCount = 0;

  // Connect
  updateConnectionStatus('connecting');
  try {
    await webrtcClient.connect(elements.audioPlayer);
  } catch (error) {
    console.error('Failed to connect:', error);
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
  }
}

function disconnect() {
  if (webrtcClient) {
    webrtcClient.disconnect();
    webrtcClient = null;
  }
  state.connected = false;
  updateConnectionStatus('disconnected');
  elements.connectBtn.textContent = 'Join Session';
}

function updateConnectionStatus(status) {
  elements.connectionStatus.className = `status ${status}`;

  switch (status) {
    case 'connected':
      elements.connectionStatus.textContent = 'Connected';
      break;
    case 'disconnected':
      elements.connectionStatus.textContent = 'Disconnected';
      break;
    case 'connecting':
      elements.connectionStatus.textContent = 'Connecting...';
      break;
  }
}

function updateProgressBar() {
  if (state.totalDuration === 0) {
    elements.progressBuffered.style.width = '0%';
    elements.progressPlayed.style.width = '0%';
    return;
  }

  const currentTime = elements.audioPlayer?.currentTime || 0;
  const playedPercent = (currentTime / state.totalDuration) * 100;

  elements.progressBuffered.style.width = '100%';
  elements.progressPlayed.style.width = `${Math.min(playedPercent, 100)}%`;
}

function updateUI() {
  const currentTime = elements.audioPlayer?.currentTime || 0;
  elements.currentTime.textContent = formatTime(currentTime);
  elements.totalDuration.textContent = formatTime(state.totalDuration);
  updateProgressBar();
}

async function updateStats() {
  if (!webrtcClient || !state.connected) {
    return;
  }

  try {
    const stats = await webrtcClient.getStats();
    if (stats && elements.latencyInfo) {
      const rtt = (stats.roundTripTime * 1000).toFixed(0);
      const jitter = (stats.jitter * 1000).toFixed(1);
      elements.latencyInfo.textContent = `RTT: ${rtt}ms | Jitter: ${jitter}ms`;
    }
  } catch (e) {
    // Ignore stats errors
  }
}

function downloadTranscript(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
