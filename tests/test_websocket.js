#!/usr/bin/env node
/**
 * WebSocket signaling test for WebRTC transcriber
 * Tests the WebSocket connection and signaling flow
 */

const WebSocket = require('ws');

const WS_URL = process.argv[2] || 'ws://localhost:8080/ws';
const TIMEOUT_MS = 10000;

console.log(`[Test] Connecting to ${WS_URL}`);

let receivedWelcome = false;
let receivedOffer = false;
let receivedSubtitle = false;
let testPassed = false;

const ws = new WebSocket(WS_URL);

const timeout = setTimeout(() => {
  console.log('[Test] TIMEOUT - Test took too long');
  console.log(`[Test] Results: welcome=${receivedWelcome}, offer=${receivedOffer}, subtitle=${receivedSubtitle}`);
  ws.close();
  process.exit(receivedWelcome && receivedOffer ? 0 : 1);
}, TIMEOUT_MS);

ws.on('open', () => {
  console.log('[Test] WebSocket connected');
  // Send ready message to initiate WebRTC signaling
  ws.send(JSON.stringify({ type: 'ready' }));
  console.log('[Test] Sent "ready" message');
});

ws.on('message', (data) => {
  try {
    const msg = JSON.parse(data.toString());
    console.log(`[Test] Received: ${msg.type}`);

    switch (msg.type) {
      case 'welcome':
        console.log(`[Test] Welcome: ${msg.message}, client_id: ${msg.client_id}`);
        receivedWelcome = true;
        break;

      case 'offer':
        console.log('[Test] Received SDP offer (WebRTC signaling works!)');
        console.log(`[Test] SDP preview: ${msg.sdp.substring(0, 100)}...`);
        receivedOffer = true;

        // Send a mock answer (won't establish real WebRTC, but tests signaling)
        const mockAnswer = {
          type: 'answer',
          sdp: 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\nc=IN IP4 0.0.0.0\r\na=rtpmap:111 opus/48000/2\r\n'
        };
        ws.send(JSON.stringify(mockAnswer));
        console.log('[Test] Sent mock answer');
        break;

      case 'subtitle':
        console.log(`[Test] Subtitle: "${msg.text}" [${msg.start}s-${msg.end}s] Speaker: ${msg.speaker}`);
        receivedSubtitle = true;
        break;

      case 'status':
        console.log(`[Test] Status: buffer=${msg.buffer_time}s, total=${msg.total_duration}s`);
        break;

      case 'ice-candidate':
        console.log('[Test] Received ICE candidate');
        break;

      case 'error':
        console.log(`[Test] Server error: ${msg.message}`);
        break;

      default:
        console.log(`[Test] Unknown message type: ${msg.type}`);
    }

    // Consider test passed if we got welcome and offer
    if (receivedWelcome && receivedOffer && !testPassed) {
      testPassed = true;
      console.log('[Test] PASSED - WebSocket signaling works correctly!');

      // Wait a bit to see if we get subtitles
      setTimeout(() => {
        clearTimeout(timeout);
        console.log(`[Test] Final results: welcome=${receivedWelcome}, offer=${receivedOffer}, subtitle=${receivedSubtitle}`);
        ws.close();
        process.exit(0);
      }, 5000);
    }
  } catch (e) {
    console.error('[Test] Error parsing message:', e);
  }
});

ws.on('error', (error) => {
  console.error('[Test] WebSocket error:', error.message);
  clearTimeout(timeout);
  process.exit(1);
});

ws.on('close', (code, reason) => {
  console.log(`[Test] WebSocket closed: code=${code}, reason=${reason}`);
  clearTimeout(timeout);
});
