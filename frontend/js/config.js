/**
 * Configuration for the transcription frontend
 */
export const config = {
  // WebSocket server URL (axum serves at /ws endpoint)
  wsUrl: 'ws://localhost:8080/ws',

  // Audio settings
  audio: {
    sampleRate: 16000,
    channels: 1,
    bufferSize: 4096, // Web Audio buffer size
  },

  // Subtitle display settings
  subtitles: {
    // Maximum segments to keep in transcript view
    maxSegments: 1000,
    // Auto-scroll transcript
    autoScroll: true,
    // Show timestamps
    showTimestamps: true,
  },

  // Speaker colors (up to 8 speakers)
  speakerColors: [
    '#4A90D9', // Blue
    '#50C878', // Green
    '#E9967A', // Salmon
    '#DDA0DD', // Plum
    '#F0E68C', // Khaki
    '#87CEEB', // Sky Blue
    '#FFB6C1', // Light Pink
    '#98FB98', // Pale Green
  ],

  // Reconnection settings
  reconnect: {
    enabled: true,
    delay: 2000,      // Initial delay in ms
    maxDelay: 30000,  // Max delay in ms
    maxAttempts: 10,  // 0 = infinite
  },
};
