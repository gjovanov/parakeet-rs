/**
 * Subtitle renderer module
 */
import { formatTime, escapeHtml, createEventEmitter } from './utils.js';

/**
 * Detect if text contains excessive repetition (hallucination)
 * @param {string} text - Text to check
 * @param {number} threshold - Max allowed consecutive repetitions (default: 3)
 * @returns {boolean} True if text appears to be hallucinated
 */
function isHallucinated(text, threshold = 3) {
  if (!text || text.length < 10) return false;

  // Split into words
  const words = text.toLowerCase().trim().split(/\s+/);
  if (words.length < threshold * 2) return false;

  // Check for consecutive word repetitions
  let consecutiveCount = 1;
  for (let i = 1; i < words.length; i++) {
    if (words[i] === words[i - 1] && words[i].length > 1) {
      consecutiveCount++;
      if (consecutiveCount >= threshold) {
        console.warn('[Subtitles] Hallucination detected: repeated word "' + words[i] + '" ' + consecutiveCount + ' times');
        return true;
      }
    } else {
      consecutiveCount = 1;
    }
  }

  // Check for repeated phrases (2-3 word patterns)
  for (let patternLen = 2; patternLen <= 3; patternLen++) {
    if (words.length < patternLen * threshold) continue;

    for (let i = 0; i <= words.length - patternLen * threshold; i++) {
      const pattern = words.slice(i, i + patternLen).join(' ');
      let patternCount = 1;

      for (let j = i + patternLen; j <= words.length - patternLen; j += patternLen) {
        const candidate = words.slice(j, j + patternLen).join(' ');
        if (candidate === pattern) {
          patternCount++;
          if (patternCount >= threshold) {
            console.warn('[Subtitles] Hallucination detected: repeated phrase "' + pattern + '" ' + patternCount + ' times');
            return true;
          }
        } else {
          break;
        }
      }
    }
  }

  return false;
}

/**
 * Subtitle segment
 * @typedef {Object} Segment
 * @property {string} text - Segment text
 * @property {number|null} speaker - Speaker ID (null for unknown)
 * @property {number} start - Start time in seconds
 * @property {number} end - End time in seconds
 * @property {boolean} isFinal - Whether segment is finalized
 * @property {number|null} inferenceTimeMs - Model inference time in milliseconds
 */

export class SubtitleRenderer {
  /**
   * @param {HTMLElement} liveElement - Element for current subtitle display
   * @param {HTMLElement} transcriptElement - Element for full transcript
   * @param {Object} options - Configuration options
   */
  constructor(liveElement, transcriptElement, options = {}) {
    this.liveElement = liveElement;
    this.transcriptElement = transcriptElement;

    this.options = {
      maxSegments: 1000,
      autoScroll: true,
      showTimestamps: true,
      speakerColors: [
        '#4A90D9', '#50C878', '#E9967A', '#DDA0DD',
        '#F0E68C', '#87CEEB', '#FFB6C1', '#98FB98',
      ],
      ...options,
    };

    /** @type {Segment[]} */
    this.segments = [];

    /** @type {Segment|null} */
    this.currentSegment = null;

    // Current playback time
    this.currentTime = 0;

    // Stale subtitle timeout (clear display after 5 seconds of no updates)
    this.staleTimeoutMs = 5000;
    this.staleTimer = null;
    this.lastSubtitleTime = 0;
    this.isStale = false;  // Track if content is stale (should not display)

    // Queue for segments that arrived ahead of playback time
    /** @type {Segment[]} */
    this.pendingSegments = [];

    // Sync mode: if true, hold FINAL segments that are too far ahead of playback
    // DISABLED: The sync logic causes issues when WebRTC reconnects and audio time resets
    // Transcripts should display immediately as they arrive
    this.syncWithPlayback = false;
    this.maxAheadSecs = 30.0;  // Not used when syncWithPlayback is false

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);

    // Initialize display
    this.initializeDisplay();
  }

  /**
   * Initialize the display elements
   */
  initializeDisplay() {
    // Ensure live element has proper structure
    if (this.liveElement) {
      this.liveElement.innerHTML = `
        <div class="speaker-label" data-speaker="?">Speaker ?</div>
        <div class="subtitle-text"></div>
        <div class="inference-time"></div>
      `;
      this.liveSpeakerEl = this.liveElement.querySelector('.speaker-label');
      this.liveTextEl = this.liveElement.querySelector('.subtitle-text');
      this.liveInferenceTimeEl = this.liveElement.querySelector('.inference-time');
    }

    // Clear transcript
    if (this.transcriptElement) {
      this.transcriptElement.innerHTML = '';
    }
  }

  /**
   * Get color for speaker
   * @param {number|null} speaker - Speaker ID
   * @returns {string} CSS color
   */
  getSpeakerColor(speaker) {
    if (speaker === null || speaker === undefined) {
      return '#888888';
    }
    return this.options.speakerColors[speaker % this.options.speakerColors.length];
  }

  /**
   * Get speaker display name
   * @param {number|null} speaker - Speaker ID
   * @returns {string} Display name
   */
  getSpeakerName(speaker) {
    if (speaker === null || speaker === undefined) {
      return ''; // Hide speaker label when not available (e.g., Canary model)
    }
    return `Speaker ${speaker}`;
  }

  /**
   * Add a new segment
   * @param {Segment} segment - Segment to add
   */
  addSegment(segment) {
    console.log('[Subtitles] addSegment called:', {
      text: segment.text?.substring(0, 50),
      isFinal: segment.isFinal,
      start: segment.start,
      end: segment.end
    });

    // Filter out hallucinated transcripts (excessive repetition)
    if (isHallucinated(segment.text)) {
      console.warn('[Subtitles] Filtering hallucinated segment:', segment.text?.substring(0, 100));
      this.emit('hallucination', segment);
      return;
    }

    // Reset stale timer on any new segment
    this.resetStaleTimer();

    // If sync mode is enabled, check if FINAL segment should be queued
    // Partial segments always display immediately (they're the live preview)
    // Only queue FINAL segments that are WAY too far ahead of playback
    if (this.syncWithPlayback && segment.isFinal && segment.start > this.currentTime + this.maxAheadSecs) {
      // Add to pending queue (will be processed when playback catches up)
      this.pendingSegments.push(segment);
      // Sort by start time
      this.pendingSegments.sort((a, b) => a.start - b.start);
      return;
    }

    this._processSegment(segment);
  }

  /**
   * Internal method to process a segment (display it)
   * @param {Segment} segment - Segment to process
   */
  _processSegment(segment) {
    if (segment.isFinal) {
      // Add to finalized segments
      this.segments.push(segment);

      // Trim if over max
      while (this.segments.length > this.options.maxSegments) {
        this.segments.shift();
      }

      // Add to transcript
      this.appendToTranscript(segment);

      // Clear current if it matches
      if (this.currentSegment &&
          this.currentSegment.start === segment.start) {
        this.currentSegment = null;
      }
    } else {
      // Update current (partial) segment
      this.currentSegment = segment;
    }

    // Update live display
    this.updateLiveDisplay();

    // Emit event
    this.emit('segment', segment);
  }

  /**
   * Reset the stale subtitle timer
   * Clears display if no new messages arrive within timeout
   */
  resetStaleTimer() {
    // Clear existing timer
    if (this.staleTimer) {
      clearTimeout(this.staleTimer);
    }

    this.lastSubtitleTime = Date.now();
    this.isStale = false;  // New content arrived, not stale

    // Set new timer to mark content as stale and clear display
    this.staleTimer = setTimeout(() => {
      console.log('[Subtitles] Marking content as stale after timeout');
      this.isStale = true;
      this.currentSegment = null;
      this.updateLiveDisplay();
    }, this.staleTimeoutMs);
  }

  /**
   * Clear the current (partial) subtitle
   * Called on disconnect or stale timeout
   */
  clearCurrent() {
    this.currentSegment = null;
    this.isStale = true;  // Mark as stale to hide display
    if (this.staleTimer) {
      clearTimeout(this.staleTimer);
      this.staleTimer = null;
    }
    this.updateLiveDisplay();
  }

  /**
   * Append segment to transcript view
   * @param {Segment} segment - Segment to append
   */
  appendToTranscript(segment) {
    if (!this.transcriptElement) return;

    const el = document.createElement('div');
    el.className = 'transcript-segment';
    el.dataset.start = segment.start;
    el.dataset.end = segment.end;
    el.dataset.speaker = segment.speaker ?? '?';

    const color = this.getSpeakerColor(segment.speaker);
    const speakerName = this.getSpeakerName(segment.speaker);

    let html = '';

    // Only show speaker label if available (TDT has diarization, Canary doesn't)
    if (speakerName) {
      html += `<span class="segment-speaker" style="color: ${color}">[${speakerName}]</span>`;
    }

    if (this.options.showTimestamps) {
      html += `<span class="segment-time">${formatTime(segment.start)}</span>`;
    }

    html += `<span class="segment-text">${escapeHtml(segment.text)}</span>`;

    // Show inference time if available
    if (segment.inferenceTimeMs != null) {
      const inferenceTime = segment.inferenceTimeMs >= 1000
        ? `${(segment.inferenceTimeMs / 1000).toFixed(1)}s`
        : `${segment.inferenceTimeMs}ms`;
      html += `<span class="segment-inference-time" title="Model inference time">${inferenceTime}</span>`;
    }

    el.innerHTML = html;

    // Add click handler to seek
    el.addEventListener('click', () => {
      this.emit('seek', segment.start);
    });

    // Insert at top (newest first) instead of appending at bottom
    this.transcriptElement.insertBefore(el, this.transcriptElement.firstChild);

    // Auto-scroll to top when new segments arrive
    if (this.options.autoScroll) {
      this.scrollToTop();
    }
  }

  /**
   * Update the live subtitle display
   */
  updateLiveDisplay() {
    if (!this.liveElement) {
      console.warn('[Subtitles] liveElement is null, cannot update live display');
      return;
    }

    // Show current partial segment, or the most recent final segment
    // But don't show anything if content is stale (no updates for a while)
    let segment = null;

    if (!this.isStale) {
      segment = this.currentSegment;

      // If no partial segment, show the most recent final segment temporarily
      if (!segment && this.segments.length > 0) {
        segment = this.segments[this.segments.length - 1];
      }
    }

    if (segment) {
      const color = this.getSpeakerColor(segment.speaker);
      const speakerName = this.getSpeakerName(segment.speaker);

      // Only show speaker label if available (TDT has diarization, Canary doesn't)
      if (speakerName) {
        this.liveSpeakerEl.textContent = speakerName;
        this.liveSpeakerEl.style.color = color;
        this.liveSpeakerEl.style.display = '';
      } else {
        this.liveSpeakerEl.textContent = '';
        this.liveSpeakerEl.style.display = 'none';
      }
      this.liveSpeakerEl.dataset.speaker = segment.speaker ?? '?';
      // Split text by sentence-ending punctuation into separate paragraphs
      const sentences = segment.text.split(/([.!?])/);
      let formattedHtml = '';
      let currentSentence = '';

      for (let i = 0; i < sentences.length; i++) {
        currentSentence += sentences[i];
        // If this is a sentence-ending punctuation mark, end the sentence
        if (/^[.!?]$/.test(sentences[i])) {
          const trimmed = currentSentence.trim();
          // Skip empty paragraphs or dot-only paragraphs
          if (trimmed && trimmed !== '.') {
            formattedHtml += `<p>${escapeHtml(trimmed)}</p>`;
          }
          currentSentence = '';
        }
      }

      // Add any remaining text that doesn't end with punctuation
      const remaining = currentSentence.trim();
      if (remaining && remaining !== '.') {
        formattedHtml += `<p>${escapeHtml(remaining)}</p>`;
      }

      this.liveTextEl.innerHTML = formattedHtml || '';

      // Show inference time if available
      if (this.liveInferenceTimeEl && segment.inferenceTimeMs != null) {
        const inferenceTime = segment.inferenceTimeMs >= 1000
          ? `${(segment.inferenceTimeMs / 1000).toFixed(1)}s`
          : `${segment.inferenceTimeMs}ms`;
        this.liveInferenceTimeEl.textContent = `Inference: ${inferenceTime}`;
        this.liveInferenceTimeEl.style.display = '';
      } else if (this.liveInferenceTimeEl) {
        this.liveInferenceTimeEl.textContent = '';
        this.liveInferenceTimeEl.style.display = 'none';
      }

      this.liveElement.classList.add('active');
      console.log('[Subtitles] Live display updated, added .active class, text:', segment.text?.substring(0, 50));
    } else {
      this.liveSpeakerEl.textContent = '';
      this.liveSpeakerEl.style.display = 'none';
      this.liveTextEl.textContent = '';
      if (this.liveInferenceTimeEl) {
        this.liveInferenceTimeEl.textContent = '';
        this.liveInferenceTimeEl.style.display = 'none';
      }
      this.liveElement.classList.remove('active');
    }
  }

  /**
   * Update current playback time
   * @param {number} time - Current time in seconds
   */
  updateTime(time) {
    this.currentTime = time;

    // Process any pending segments that playback has caught up to
    this._processPendingSegments();

    this.updateLiveDisplay();
    this.highlightCurrentSegment(time);
  }

  /**
   * Process pending segments that are now ready (within maxAhead window)
   * Only FINAL segments are queued, so we check their start time
   */
  _processPendingSegments() {
    while (this.pendingSegments.length > 0) {
      const segment = this.pendingSegments[0];

      // Release segment if it's now within the allowed ahead window
      if (segment.start <= this.currentTime + this.maxAheadSecs) {
        // Remove from queue and process
        this.pendingSegments.shift();
        this._processSegment(segment);
      } else {
        // Remaining segments are too far in the future
        break;
      }
    }
  }

  /**
   * Highlight segment at current time in transcript
   * @param {number} time - Current time in seconds
   */
  highlightCurrentSegment(time) {
    if (!this.transcriptElement) return;

    // Remove existing highlights
    this.transcriptElement.querySelectorAll('.current').forEach(el => {
      el.classList.remove('current');
    });

    // Find and highlight current
    const segments = this.transcriptElement.querySelectorAll('.transcript-segment');
    for (const el of segments) {
      const start = parseFloat(el.dataset.start);
      const end = parseFloat(el.dataset.end);

      if (time >= start && time <= end) {
        el.classList.add('current');

        // Scroll into view if auto-scroll enabled
        if (this.options.autoScroll) {
          el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        break;
      }
    }
  }

  /**
   * Get segment at a specific time
   * @param {number} time - Time in seconds
   * @returns {Segment|null}
   */
  getSegmentAtTime(time) {
    // Check current segment first
    if (this.currentSegment &&
        time >= this.currentSegment.start &&
        time <= this.currentSegment.end) {
      return this.currentSegment;
    }

    // Search finalized segments
    for (let i = this.segments.length - 1; i >= 0; i--) {
      const seg = this.segments[i];
      if (time >= seg.start && time <= seg.end) {
        return seg;
      }
    }

    return null;
  }

  /**
   * Get full transcript text
   * @returns {string}
   */
  getTranscript() {
    return this.segments.map(seg => {
      const speaker = this.getSpeakerName(seg.speaker);
      return `[${speaker}] ${seg.text}`;
    }).join('\n');
  }

  /**
   * Get transcript with timestamps
   * @returns {string}
   */
  getTranscriptWithTimestamps() {
    return this.segments.map(seg => {
      const speaker = this.getSpeakerName(seg.speaker);
      const time = formatTime(seg.start);
      return `[${time}] [${speaker}] ${seg.text}`;
    }).join('\n');
  }

  /**
   * Export segments as JSON
   * @returns {string}
   */
  exportJSON() {
    return JSON.stringify(this.segments, null, 2);
  }

  /**
   * Scroll transcript to top (for newest-first display)
   */
  scrollToTop() {
    if (this.transcriptElement) {
      this.transcriptElement.scrollTop = 0;
    }
  }

  /**
   * Scroll transcript to bottom
   */
  scrollToBottom() {
    if (this.transcriptElement) {
      this.transcriptElement.scrollTop = this.transcriptElement.scrollHeight;
    }
  }

  /**
   * Clear all segments
   */
  clear() {
    this.segments = [];
    this.currentSegment = null;
    this.currentTime = 0;
    this.pendingSegments = [];
    this.isStale = false;  // Reset stale state

    // Clear stale timer
    if (this.staleTimer) {
      clearTimeout(this.staleTimer);
      this.staleTimer = null;
    }

    if (this.transcriptElement) {
      this.transcriptElement.innerHTML = '';
    }

    this.updateLiveDisplay();
    this.emit('clear');
  }

  /**
   * Enable/disable sync with playback
   * @param {boolean} enabled - Whether to sync subtitles with playback time
   */
  setSyncWithPlayback(enabled) {
    this.syncWithPlayback = enabled;
    // If disabling sync, process all pending segments immediately
    if (!enabled) {
      while (this.pendingSegments.length > 0) {
        this._processSegment(this.pendingSegments.shift());
      }
    }
  }

  /**
   * Set the lookahead window for transcript display
   * @param {number} seconds - How many seconds ahead to show transcripts
   */
  setLookahead(seconds) {
    this.lookaheadSecs = seconds;
    // Process any segments that are now within the new lookahead window
    this._processPendingSegments();
  }

  /**
   * Set speaker colors
   * @param {string[]} colors - Array of CSS colors
   */
  setSpeakerColors(colors) {
    this.options.speakerColors = colors;
  }

  /**
   * Toggle auto-scroll
   * @param {boolean} enabled
   */
  setAutoScroll(enabled) {
    this.options.autoScroll = enabled;
  }

  /**
   * Toggle timestamps
   * @param {boolean} show
   */
  setShowTimestamps(show) {
    this.options.showTimestamps = show;
    // Re-render transcript
    this.rerender();
  }

  /**
   * Re-render transcript from segments
   * Renders in reverse order (newest first)
   */
  rerender() {
    if (!this.transcriptElement) return;

    this.transcriptElement.innerHTML = '';
    // Render oldest first so newest ends up at top after insertBefore
    for (const segment of this.segments) {
      this.appendToTranscript(segment);
    }
  }
}
