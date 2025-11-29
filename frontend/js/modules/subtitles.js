/**
 * Subtitle renderer module
 */
import { formatTime, escapeHtml, createEventEmitter } from './utils.js';

/**
 * Subtitle segment
 * @typedef {Object} Segment
 * @property {string} text - Segment text
 * @property {number|null} speaker - Speaker ID (null for unknown)
 * @property {number} start - Start time in seconds
 * @property {number} end - End time in seconds
 * @property {boolean} isFinal - Whether segment is finalized
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
      `;
      this.liveSpeakerEl = this.liveElement.querySelector('.speaker-label');
      this.liveTextEl = this.liveElement.querySelector('.subtitle-text');
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
      return 'Speaker ?';
    }
    return `Speaker ${speaker}`;
  }

  /**
   * Add a new segment
   * @param {Segment} segment - Segment to add
   */
  addSegment(segment) {
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

    let html = `<span class="segment-speaker" style="color: ${color}">[${this.getSpeakerName(segment.speaker)}]</span>`;

    if (this.options.showTimestamps) {
      html += `<span class="segment-time">${formatTime(segment.start)}</span>`;
    }

    html += `<span class="segment-text">${escapeHtml(segment.text)}</span>`;

    el.innerHTML = html;

    // Add click handler to seek
    el.addEventListener('click', () => {
      this.emit('seek', segment.start);
    });

    this.transcriptElement.appendChild(el);

    // Auto-scroll
    if (this.options.autoScroll) {
      this.scrollToBottom();
    }
  }

  /**
   * Update the live subtitle display
   */
  updateLiveDisplay() {
    if (!this.liveElement) return;

    const segment = this.currentSegment || this.getSegmentAtTime(this.currentTime);

    if (segment) {
      const color = this.getSpeakerColor(segment.speaker);
      this.liveSpeakerEl.textContent = this.getSpeakerName(segment.speaker);
      this.liveSpeakerEl.style.color = color;
      this.liveSpeakerEl.dataset.speaker = segment.speaker ?? '?';
      this.liveTextEl.textContent = segment.text;
      this.liveElement.classList.add('active');
    } else {
      this.liveSpeakerEl.textContent = '';
      this.liveTextEl.textContent = '';
      this.liveElement.classList.remove('active');
    }
  }

  /**
   * Update current playback time
   * @param {number} time - Current time in seconds
   */
  updateTime(time) {
    this.currentTime = time;
    this.updateLiveDisplay();
    this.highlightCurrentSegment(time);
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

    if (this.transcriptElement) {
      this.transcriptElement.innerHTML = '';
    }

    this.updateLiveDisplay();
    this.emit('clear');
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
   */
  rerender() {
    if (!this.transcriptElement) return;

    this.transcriptElement.innerHTML = '';
    for (const segment of this.segments) {
      this.appendToTranscript(segment);
    }
  }
}
