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
  sourceType: 'media', // 'media' or 'srt'
};

// Pause-related modes that show pause config
const PAUSE_MODES = ['speedy', 'pause_based', 'lookahead', 'vad_speedy', 'vad_pause_based', 'pause_parallel'];

// Map slider value (1-5) to actual silence_energy_threshold
function getSilenceEnergyThreshold(sliderValue) {
  const values = [0.003, 0.005, 0.008, 0.012, 0.02];
  return values[sliderValue - 1];
}

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

  // Populate FAB defaults from config
  if (elements.fabEnabledSelect && config.fabEnabled !== undefined) {
    const defaultLabel = config.fabEnabled ? 'Default (enabled)' : 'Default (disabled)';
    elements.fabEnabledSelect.options[0].textContent = defaultLabel;
  }
  if (elements.fabUrlInput && config.fabUrl) {
    elements.fabUrlInput.value = config.fabUrl;
    elements.fabUrlInput.placeholder = config.fabUrl || 'No server default';
  }
  if (elements.fabSendTypeSelect && config.fabSendType) {
    const defaultLabel = `Default (${config.fabSendType})`;
    elements.fabSendTypeSelect.options[0].textContent = defaultLabel;
  }

  // Load initial data
  await Promise.all([
    sessionManager.fetchModels(),
    sessionManager.fetchMedia(),
    sessionManager.fetchModes(),
    sessionManager.fetchNoiseCancellation(),
    sessionManager.fetchDiarization(),
    sessionManager.fetchSrtStreams(),
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

  // Parallel config elements
  elements.parallelConfig = document.getElementById('parallel-config');
  elements.parallelThreads = document.getElementById('parallel-threads');
  elements.parallelBuffer = document.getElementById('parallel-buffer');
  elements.threadsValue = document.getElementById('threads-value');
  elements.bufferValue = document.getElementById('buffer-value');

  // Noise cancellation and diarization elements
  elements.noiseSelect = document.getElementById('noise-select');
  elements.diarizationSelect = document.getElementById('diarization-select');
  elements.sentenceCompletionSelect = document.getElementById('sentence-completion-select');

  // Pause config elements
  elements.pauseConfig = document.getElementById('pause-config');
  elements.pauseThreshold = document.getElementById('pause-threshold');
  elements.silenceEnergy = document.getElementById('silence-energy');
  elements.maxSegment = document.getElementById('max-segment');
  elements.pauseThresholdValue = document.getElementById('pause-threshold-value');
  elements.silenceEnergyValue = document.getElementById('silence-energy-value');
  elements.maxSegmentValue = document.getElementById('max-segment-value');
  elements.contextBufferGroup = document.getElementById('context-buffer-group');
  elements.contextBuffer = document.getElementById('context-buffer');
  elements.contextBufferValue = document.getElementById('context-buffer-value');

  // FAB config elements
  elements.fabEnabledSelect = document.getElementById('fab-enabled-select');
  elements.fabUrlGroup = document.getElementById('fab-url-group');
  elements.fabUrlInput = document.getElementById('fab-url-input');
  elements.fabSendTypeGroup = document.getElementById('fab-send-type-group');
  elements.fabSendTypeSelect = document.getElementById('fab-send-type-select');

  // Source type tabs (Media Files vs Live Streams)
  elements.sourceTabs = document.querySelectorAll('.source-tab');
  elements.mediaSourceContent = document.getElementById('media-source-content');
  elements.srtSourceContent = document.getElementById('srt-source-content');
  elements.srtSelect = document.getElementById('srt-select');
}

function setupSessionManagerListeners() {
  sessionManager.on('modelsLoaded', renderModelSelect);
  sessionManager.on('mediaLoaded', () => {
    renderMediaSelect();
    renderMediaList();
  });
  sessionManager.on('modesLoaded', renderModeSelect);
  sessionManager.on('noiseCancellationLoaded', renderNoiseSelect);
  sessionManager.on('diarizationLoaded', renderDiarizationSelect);
  sessionManager.on('srtStreamsLoaded', ({ streams, configured }) => {
    renderSrtSelect(streams, configured);
  });
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

  // Source type tab switching (Media Files vs Live Streams)
  if (elements.sourceTabs) {
    elements.sourceTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const sourceType = tab.dataset.source;
        state.sourceType = sourceType;

        // Update tab active state
        elements.sourceTabs.forEach(t => t.classList.toggle('active', t.dataset.source === sourceType));

        // Show/hide source content
        if (elements.mediaSourceContent) {
          elements.mediaSourceContent.style.display = sourceType === 'media' ? 'block' : 'none';
        }
        if (elements.srtSourceContent) {
          elements.srtSourceContent.style.display = sourceType === 'srt' ? 'block' : 'none';
        }
      });
    });
  }

  // Mode select - show/hide parallel config and pause config
  if (elements.modeSelect) {
    elements.modeSelect.addEventListener('change', () => {
      const mode = elements.modeSelect.value;
      const isParallelMode = mode === 'parallel' || mode === 'pause_parallel';
      const isPauseMode = PAUSE_MODES.includes(mode);
      const isPauseParallel = mode === 'pause_parallel';

      // Show/hide parallel config
      if (elements.parallelConfig) {
        elements.parallelConfig.style.display = isParallelMode ? 'block' : 'none';
        // Hide buffer size for pause_parallel (not applicable)
        const bufferRow = elements.parallelBuffer?.parentElement;
        if (bufferRow) {
          bufferRow.style.display = mode === 'parallel' ? 'flex' : 'none';
        }
      }

      // Show/hide pause config
      if (elements.pauseConfig) {
        elements.pauseConfig.style.display = isPauseMode ? 'block' : 'none';
      }

      // Show context buffer only for pause_parallel mode
      if (elements.contextBufferGroup) {
        elements.contextBufferGroup.style.display = isPauseParallel ? 'flex' : 'none';
      }
    });
  }

  // Parallel sliders - update displayed values
  if (elements.parallelThreads) {
    elements.parallelThreads.addEventListener('input', () => {
      elements.threadsValue.textContent = elements.parallelThreads.value;
    });
  }
  if (elements.parallelBuffer) {
    elements.parallelBuffer.addEventListener('input', () => {
      elements.bufferValue.textContent = elements.parallelBuffer.value;
    });
  }

  // Pause config sliders - update displayed values
  if (elements.pauseThreshold) {
    elements.pauseThreshold.addEventListener('input', () => {
      elements.pauseThresholdValue.textContent = elements.pauseThreshold.value;
    });
  }
  if (elements.silenceEnergy) {
    elements.silenceEnergy.addEventListener('input', () => {
      const labels = ['Very High', 'High', 'Medium', 'Low', 'Very Low'];
      elements.silenceEnergyValue.textContent = labels[elements.silenceEnergy.value - 1];
    });
  }
  if (elements.maxSegment) {
    elements.maxSegment.addEventListener('input', () => {
      elements.maxSegmentValue.textContent = elements.maxSegment.value;
    });
  }
  if (elements.contextBuffer) {
    elements.contextBuffer.addEventListener('input', () => {
      elements.contextBufferValue.textContent = elements.contextBuffer.value;
    });
  }

  // FAB select - show/hide URL input and send type
  if (elements.fabEnabledSelect) {
    elements.fabEnabledSelect.addEventListener('change', () => {
      const val = elements.fabEnabledSelect.value;
      const show = val === 'enabled';
      if (elements.fabUrlGroup) {
        elements.fabUrlGroup.style.display = show ? 'flex' : 'none';
      }
      if (elements.fabSendTypeGroup) {
        elements.fabSendTypeGroup.style.display = show ? 'flex' : 'none';
      }
    });
  }

  // Create session
  elements.createSessionBtn.addEventListener('click', async () => {
    const modelId = elements.modelSelect.value;
    const mode = elements.modeSelect?.value || 'speedy';
    const language = elements.languageSelect?.value || 'de';
    const noiseCancellation = elements.noiseSelect?.value || 'none';
    const diarization = elements.diarizationSelect?.value !== 'none';
    const sentenceCompletion = elements.sentenceCompletionSelect?.value || 'minimal';

    // Determine source based on active tab
    let mediaId = null;
    let srtChannelId = null;

    if (state.sourceType === 'srt') {
      // SRT stream source
      const srtValue = elements.srtSelect?.value;
      if (!srtValue || srtValue === '') {
        alert('Please select an SRT stream');
        return;
      }
      srtChannelId = parseInt(srtValue, 10);
    } else {
      // Media file source
      mediaId = elements.mediaSelect.value;
      if (!mediaId) {
        alert('Please select a media file');
        return;
      }
    }

    if (!modelId) {
      alert('Please select a model');
      return;
    }

    // Get parallel config if in parallel or pause_parallel mode
    let parallelConfig = null;
    if ((mode === 'parallel' || mode === 'pause_parallel') && elements.parallelThreads) {
      parallelConfig = {
        num_threads: parseInt(elements.parallelThreads.value, 10),
        buffer_size_secs: mode === 'parallel' ? parseInt(elements.parallelBuffer?.value || 6, 10) : 6
      };
    }

    // Get FAB config
    const fabEnabled = elements.fabEnabledSelect?.value || 'default';
    const fabUrl = elements.fabUrlInput?.value?.trim() || '';
    const fabSendType = elements.fabSendTypeSelect?.value || 'default';

    // Get pause config for pause-related modes
    let pauseConfig = null;
    if (PAUSE_MODES.includes(mode) && elements.pauseThreshold) {
      pauseConfig = {
        pause_threshold_ms: parseInt(elements.pauseThreshold.value, 10),
        silence_energy_threshold: getSilenceEnergyThreshold(parseInt(elements.silenceEnergy?.value || 3, 10)),
        max_segment_secs: parseFloat(elements.maxSegment?.value || 5),
        context_buffer_secs: parseFloat(elements.contextBuffer?.value || 0)
      };
    }

    try {
      elements.createSessionBtn.disabled = true;
      elements.createSessionBtn.textContent = 'Creating...';
      const session = await sessionManager.createSession(modelId, {
        mediaId,
        srtChannelId,
        mode,
        language,
        parallelConfig,
        noiseCancellation,
        diarization,
        pauseConfig,
        sentenceCompletion,
        fabEnabled,
        fabUrl,
        fabSendType
      });
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
    console.log('[Click] Connect button clicked, connected:', state.connected, 'selectedSessionId:', state.selectedSessionId);
    if (state.connected) {
      disconnect();
    } else if (state.selectedSessionId) {
      connect(state.selectedSessionId);
    } else {
      console.log('[Click] No session selected!');
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

function renderNoiseSelect() {
  if (!elements.noiseSelect) return;
  const options = sessionManager.noiseCancellationOptions;
  elements.noiseSelect.innerHTML = options.length
    ? options.filter(o => o.available).map(o =>
        `<option value="${o.id}" title="${o.description}">${o.name}</option>`
      ).join('')
    : '<option value="none">None</option>';
}

function renderDiarizationSelect() {
  if (!elements.diarizationSelect) return;
  const options = sessionManager.diarizationOptions;
  elements.diarizationSelect.innerHTML = options.length
    ? options.filter(o => o.available).map(o =>
        `<option value="${o.id}" title="${o.description}">${o.name}</option>`
      ).join('')
    : '<option value="none">None</option>';
}

function renderSrtSelect(streams, configured) {
  if (!elements.srtSelect) return;

  if (!configured || streams.length === 0) {
    elements.srtSelect.innerHTML = '<option value="">SRT not configured</option>';
    elements.srtSelect.disabled = true;
    // Hide SRT tab if not configured
    if (elements.sourceTabs) {
      elements.sourceTabs.forEach(tab => {
        if (tab.dataset.source === 'srt') {
          tab.style.display = 'none';
        }
      });
    }
    return;
  }

  // Show SRT tab if configured
  if (elements.sourceTabs) {
    elements.sourceTabs.forEach(tab => {
      if (tab.dataset.source === 'srt') {
        tab.style.display = '';
      }
    });
  }

  elements.srtSelect.disabled = false;
  elements.srtSelect.innerHTML = streams
    .map(s => `<option value="${s.id}">${s.display}</option>`)
    .join('');
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
    const isLive = s.source_type === 'srt_stream';
    const isVod = s.mode === 'vod';
    const progress = !isLive && s.duration_secs > 0 ? (s.progress_secs / s.duration_secs) * 100 : 0;
    const isSelected = state.selectedSessionId === s.id;
    const modeLabel = s.mode ? s.mode.replace(/_/g, '-') : 'speedy';

    // Build badges for noise cancellation and diarization
    const noiseLabel = s.noise_cancellation && s.noise_cancellation !== 'none'
      ? ` | <span class="noise-badge">${s.noise_cancellation}</span>`
      : '';
    const diarLabel = s.diarization
      ? ` | <span class="diar-badge">${s.diarization_model || 'Diar'}</span>`
      : '';
    // Sentence completion badge
    const sentenceLabel = s.sentence_completion && s.sentence_completion !== 'off'
      ? ` | <span class="sentence-badge" title="Sentence completion: ${s.sentence_completion}">${s.sentence_completion}</span>`
      : '';

    // VoD badge for VoD mode
    const vodLabel = isVod
      ? '<span class="vod-badge">VoD</span> '
      : '';

    // LIVE badge for SRT streams
    const liveLabel = isLive
      ? '<span class="live-badge">LIVE</span> '
      : '';

    // Duration display - varies by mode
    let durationDisplay;
    if (isLive) {
      durationDisplay = `${formatDuration(s.progress_secs)} elapsed`;
    } else if (isVod && s.vod_progress) {
      // VoD progress: show chunk progress
      durationDisplay = `<span class="vod-progress-text">Chunk ${s.vod_progress.completed_chunks}/${s.vod_progress.total_chunks} (${Math.round(s.vod_progress.percent)}%)</span>`;
    } else if (isVod && s.state === 'completed') {
      durationDisplay = `${formatDuration(s.duration_secs)} - Completed`;
    } else {
      durationDisplay = `${formatDuration(s.progress_secs)} / ${formatDuration(s.duration_secs)}`;
    }

    // Progress bar - varies by mode
    let progressBar;
    if (isLive) {
      progressBar = '<div class="session-progress-bar live-progress" style="width: 100%"></div>';
    } else if (isVod && s.vod_progress) {
      progressBar = `<div class="session-progress-bar" style="width: ${s.vod_progress.percent}%"></div>`;
    } else if (isVod && s.state === 'completed') {
      progressBar = '<div class="session-progress-bar" style="width: 100%"></div>';
    } else {
      progressBar = `<div class="session-progress-bar" style="width: ${progress}%"></div>`;
    }

    // Action buttons - include download for completed VoD sessions
    let actionButtons = '';
    if (s.state === 'running') {
      actionButtons = `<button class="btn-small btn-danger" onclick="event.stopPropagation(); window.stopSession('${s.id}')">Stop</button>`;
    } else if (isVod && s.state === 'completed' && s.transcript_available) {
      actionButtons = `<button class="btn-small btn-download" onclick="event.stopPropagation(); window.downloadTranscript('${s.id}')">Download</button>`;
    }

    return `
      <div class="session-card ${isSelected ? 'selected' : ''}" onclick="window.selectSession('${s.id}')">
        <div class="session-info">
          <div class="session-title">${vodLabel}${liveLabel}${s.media_filename}</div>
          <div class="session-meta">${s.model_name} | <span class="mode-badge">${modeLabel}</span>${sentenceLabel}${noiseLabel}${diarLabel} | ${durationDisplay}</div>
          <div class="session-progress">
            ${progressBar}
          </div>
        </div>
        <div class="session-actions">
          <span class="session-status ${s.state}">${s.state}</span>
          ${actionButtons}
        </div>
      </div>
    `;
  }).join('');
}

function selectSession(sessionId) {
  console.log('[SelectSession] Selecting session:', sessionId);
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

window.downloadTranscript = async (sessionId) => {
  try {
    await sessionManager.downloadTranscript(sessionId);
  } catch (e) {
    alert('Failed to download transcript: ' + e.message);
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
  console.log('[Connect] Starting connection to session:', sessionId);
  const config = getConfig();
  console.log('[Connect] Config:', config);
  const wsUrl = sessionManager.getWebSocketUrl(sessionId);
  console.log('[Connect] WebSocket URL:', wsUrl);

  // Create WebRTC client with ICE transport policy from server config
  webrtcClient = new WebRTCClient(wsUrl, {
    iceServers: config.iceServers,
    iceTransportPolicy: config.iceTransportPolicy || 'all'
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

  webrtcClient.on('trackReceived', async () => {
    console.log('Audio track received');
    elements.bufferInfo.textContent = 'Audio streaming';
    // Start stats logging and run debug after short delay
    webrtcClient.startStatsLogging(3000);
    setTimeout(async () => {
      await webrtcClient.debugStatus();
    }, 2000);
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

  webrtcClient.on('reconnecting', ({ attempt, delay }) => {
    state.connected = false;
    updateConnectionStatus('reconnecting');
    elements.bufferInfo.textContent = `Reconnecting... (attempt ${attempt})`;
    console.log(`Reconnecting in ${delay}ms (attempt ${attempt})`);
  });

  webrtcClient.on('reconnectFailed', () => {
    state.connected = false;
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Reconnection failed';
    elements.connectBtn.textContent = 'Join Session';
    console.error('Max reconnection attempts reached');
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

  webrtcClient.on('end', ({ totalDuration, is_live }) => {
    state.totalDuration = totalDuration;
    elements.totalDuration.textContent = formatTime(totalDuration);
    if (is_live) {
      console.log('Live stream stopped');
    } else {
      console.log('Stream ended, total duration:', formatTime(totalDuration));
    }
  });

  webrtcClient.on('serverError', ({ message }) => {
    console.error('Server error:', message);
    alert(`Server error: ${message}`);
  });

  // SRT stream reconnection events
  webrtcClient.on('srtReconnecting', ({ attempt, maxAttempts, delayMs }) => {
    elements.bufferInfo.textContent = `SRT reconnecting... (${attempt}/${maxAttempts})`;
    elements.connectionStatus.textContent = 'Reconnecting...';
    elements.connectionStatus.className = 'status reconnecting';
    console.log(`SRT reconnecting in ${delayMs}ms (attempt ${attempt}/${maxAttempts})`);
  });

  webrtcClient.on('srtReconnected', () => {
    elements.bufferInfo.textContent = 'SRT reconnected';
    elements.connectionStatus.textContent = 'Connected';
    elements.connectionStatus.className = 'status connected';
    console.log('SRT stream reconnected');
  });

  // VoD progress events
  webrtcClient.on('vodProgress', ({ totalChunks, completedChunks, currentChunk, percent }) => {
    elements.bufferInfo.textContent = `VoD: Chunk ${completedChunks}/${totalChunks} (${Math.round(percent)}%)`;
    console.log(`VoD progress: ${completedChunks}/${totalChunks} (${percent}%)`);
    // Refresh sessions to update progress display
    sessionManager.fetchSessions();
  });

  webrtcClient.on('vodComplete', ({ transcriptAvailable, durationSecs, segmentCount }) => {
    elements.bufferInfo.textContent = `VoD complete: ${segmentCount} segments`;
    console.log(`VoD complete: ${segmentCount} segments, ${durationSecs}s, transcript: ${transcriptAvailable}`);
    // Refresh sessions to show download button
    sessionManager.fetchSessions();
  });

  // Reset subtitle counter
  state.subtitleCount = 0;

  // Connect
  updateConnectionStatus('connecting');
  try {
    await webrtcClient.connect(elements.audioPlayer);
    // Expose for debugging (call window._webrtcClient.debugStatus() in console)
    window._webrtcClient = webrtcClient;
  } catch (error) {
    console.error('Failed to connect:', error);
    updateConnectionStatus('disconnected');
    elements.bufferInfo.textContent = 'Connection failed';
  }
}

function disconnect() {
  if (webrtcClient) {
    webrtcClient.stopStatsLogging();
    webrtcClient.disconnect();
    webrtcClient = null;
    window._webrtcClient = null;
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
    case 'reconnecting':
      elements.connectionStatus.textContent = 'Reconnecting...';
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
