/**
 * WebRTC client for ultra-low-latency audio streaming
 *
 * Features:
 * - Automatic jitter buffer (handled by browser)
 * - Opus codec (optimized for speech)
 * - ~100-300ms latency
 * - Native audio playback via <audio> element
 * - Dynamic configuration from server (/api/config)
 */
import { createEventEmitter } from './utils.js';

export class WebRTCClient {
  /**
   * @param {string} wsUrl - WebSocket URL for signaling
   * @param {Object} options - Configuration options
   * @param {Array} options.iceServers - ICE servers for WebRTC (from server config)
   */
  constructor(wsUrl, options = {}) {
    this.wsUrl = wsUrl;

    // Default ICE servers (fallback if not provided by server)
    const defaultIceServers = [
      { urls: 'stun:stun.l.google.com:19302' }
    ];

    this.options = {
      iceServers: options.iceServers || defaultIceServers,
      // Use 'all' to allow direct connections when server provides public IP candidates
      iceTransportPolicy: 'all',
      ...options,
    };

    /** @type {WebSocket|null} */
    this.ws = null;

    /** @type {RTCPeerConnection|null} */
    this.pc = null;

    /** @type {HTMLAudioElement|null} */
    this.audioElement = null;

    /** @type {MediaStream|null} */
    this.remoteStream = null;

    this.connected = false;
    this.clientId = null;

    // Event emitter
    const emitter = createEventEmitter();
    this.on = emitter.on.bind(emitter);
    this.off = emitter.off.bind(emitter);
    this.emit = emitter.emit.bind(emitter);
  }

  /**
   * Connect to WebRTC server
   * @param {HTMLAudioElement} audioElement - Audio element for playback
   */
  async connect(audioElement) {
    this.audioElement = audioElement;

    return new Promise((resolve, reject) => {
      // Create WebSocket for signaling
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('[WebRTC] Signaling connected');
        this.setupPeerConnection();
      };

      this.ws.onclose = (event) => {
        console.log('[WebRTC] Signaling disconnected:', event.code, event.reason);
        this.connected = false;
        this.emit('disconnect', { code: event.code, reason: event.reason });
      };

      this.ws.onerror = (error) => {
        console.error('[WebRTC] Signaling error:', error);
        this.emit('error', error);
        reject(error);
      };

      this.ws.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data);
          await this.handleSignalingMessage(msg, resolve);
        } catch (e) {
          console.error('[WebRTC] Error handling message:', e);
        }
      };
    });
  }

  /**
   * Set up WebRTC peer connection
   */
  setupPeerConnection() {
    // Create peer connection
    this.pc = new RTCPeerConnection({
      iceServers: this.options.iceServers,
    });

    // Handle incoming audio track
    this.pc.ontrack = (event) => {
      console.log('[WebRTC] Received audio track');
      this.remoteStream = event.streams[0];

      if (this.audioElement) {
        this.audioElement.srcObject = this.remoteStream;
        this.audioElement.play().catch(e => {
          console.warn('[WebRTC] Autoplay blocked, user interaction required:', e);
          this.emit('autoplayBlocked');
        });
      }

      this.emit('trackReceived', event.streams[0]);
    };

    // Handle ICE candidates
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('[WebRTC] Sending ICE candidate to server:', event.candidate.candidate);
        this.ws.send(JSON.stringify({
          type: 'ice-candidate',
          candidate: event.candidate.toJSON(),
        }));
      } else {
        console.log('[WebRTC] ICE gathering complete');
      }
    };

    // Handle connection state changes
    this.pc.onconnectionstatechange = () => {
      console.log('[WebRTC] Connection state:', this.pc.connectionState);

      switch (this.pc.connectionState) {
        case 'connected':
          this.connected = true;
          this.emit('connected');
          break;
        case 'disconnected':
        case 'failed':
          this.connected = false;
          this.emit('connectionFailed');
          break;
        case 'closed':
          this.connected = false;
          this.emit('closed');
          break;
      }
    };

    // Handle ICE connection state
    this.pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE state:', this.pc.iceConnectionState);
      if (this.pc.iceConnectionState === 'failed') {
        console.error('[WebRTC] ICE connection failed');
        this.emit('connectionFailed');
      }
    };

    // Handle ICE gathering state
    this.pc.onicegatheringstatechange = () => {
      console.log('[WebRTC] ICE gathering state:', this.pc.iceGatheringState);
    };

    // Signal ready to receive offer
    this.ws.send(JSON.stringify({ type: 'ready' }));
  }

  /**
   * Handle signaling messages from server
   * @param {Object} msg - Signaling message
   * @param {Function} resolve - Promise resolve function
   */
  async handleSignalingMessage(msg, resolve) {
    switch (msg.type) {
      case 'welcome':
        console.log('[WebRTC] Welcome:', msg.message);
        this.clientId = msg.client_id;
        this.emit('welcome', msg);
        break;

      case 'offer':
        console.log('[WebRTC] Received offer');
        try {
          await this.pc.setRemoteDescription({
            type: 'offer',
            sdp: msg.sdp,
          });

          const answer = await this.pc.createAnswer();
          await this.pc.setLocalDescription(answer);

          this.ws.send(JSON.stringify({
            type: 'answer',
            sdp: answer.sdp,
          }));

          console.log('[WebRTC] Sent answer');
          resolve();
        } catch (e) {
          console.error('[WebRTC] Error handling offer:', e);
          this.emit('error', e);
        }
        break;

      case 'ice-candidate':
        if (msg.candidate) {
          console.log('[WebRTC] Received ICE candidate from server:', msg.candidate);
          try {
            await this.pc.addIceCandidate(msg.candidate);
            console.log('[WebRTC] Added ICE candidate successfully');
          } catch (e) {
            console.error('[WebRTC] Error adding ICE candidate:', e);
          }
        }
        break;

      case 'subtitle':
        this.emit('subtitle', {
          text: msg.text,
          speaker: msg.speaker,
          start: msg.start,
          end: msg.end,
          is_final: msg.is_final,
        });
        break;

      case 'status':
        this.emit('status', {
          bufferTime: msg.buffer_time,
          totalDuration: msg.total_duration,
        });
        break;

      case 'end':
        this.emit('end', {
          totalDuration: msg.total_duration,
        });
        break;

      case 'error':
        console.error('[WebRTC] Server error:', msg.message);
        this.emit('serverError', { message: msg.message });
        break;
    }
  }

  /**
   * Disconnect from server
   */
  disconnect() {
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.audioElement) {
      this.audioElement.srcObject = null;
    }

    this.remoteStream = null;
    this.connected = false;

    this.emit('disconnect', { code: 1000, reason: 'User disconnect' });
  }

  /**
   * Get current playback time
   * @returns {number} Current time in seconds
   */
  get currentTime() {
    return this.audioElement?.currentTime || 0;
  }

  /**
   * Get total duration
   * @returns {number} Duration in seconds
   */
  get duration() {
    return this.audioElement?.duration || 0;
  }

  /**
   * Check if playing
   * @returns {boolean}
   */
  get playing() {
    return this.audioElement && !this.audioElement.paused;
  }

  /**
   * Play audio
   */
  async play() {
    if (this.audioElement) {
      try {
        await this.audioElement.play();
      } catch (e) {
        console.error('[WebRTC] Play error:', e);
        this.emit('error', e);
      }
    }
  }

  /**
   * Pause audio
   */
  pause() {
    if (this.audioElement) {
      this.audioElement.pause();
    }
  }

  /**
   * Set volume
   * @param {number} volume - Volume level (0.0 to 1.0)
   */
  setVolume(volume) {
    if (this.audioElement) {
      this.audioElement.volume = Math.max(0, Math.min(1, volume));
    }
  }

  /**
   * Get connection stats
   * @returns {Promise<Object>} Connection statistics
   */
  async getStats() {
    if (!this.pc) {
      return null;
    }

    const stats = await this.pc.getStats();
    const result = {
      bytesReceived: 0,
      packetsReceived: 0,
      packetsLost: 0,
      jitter: 0,
      roundTripTime: 0,
    };

    stats.forEach(report => {
      if (report.type === 'inbound-rtp' && report.kind === 'audio') {
        result.bytesReceived = report.bytesReceived || 0;
        result.packetsReceived = report.packetsReceived || 0;
        result.packetsLost = report.packetsLost || 0;
        result.jitter = report.jitter || 0;
      }
      if (report.type === 'candidate-pair' && report.state === 'succeeded') {
        result.roundTripTime = report.currentRoundTripTime || 0;
      }
    });

    return result;
  }
}
