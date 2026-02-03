/**
 * Diagnostic: captures raw WebSocket subtitle messages and simulates
 * what the "Current" vs "Transcript" panels would display.
 *
 * Run: npx ts-node tests/e2e/debug-current-panel.ts
 */

import WebSocket from 'ws';

const SESSION_ID = process.argv[2] || 'b6f563a4';
const WS_URL = `ws://localhost:8080/ws/${SESSION_ID}`;
const CAPTURE_SECS = 30;

interface SubtitleMsg {
  type: string;
  text: string;
  raw_text?: string;
  growing_text: string;
  full_transcript?: string;
  delta: string;
  tail_changed: boolean;
  speaker: number | null;
  start: number;
  end: number;
  is_final: boolean;
  inference_time_ms?: number;
}

let messageCount = 0;
let finalCount = 0;
let partialCount = 0;
let currentSegment: SubtitleMsg | null = null;
let transcriptSegments: SubtitleMsg[] = [];

console.log(`Connecting to ${WS_URL}...`);
console.log(`Will capture for ${CAPTURE_SECS}s\n`);

const ws = new WebSocket(WS_URL);

ws.on('open', () => {
  console.log('Connected!\n');
  console.log('=== Raw Messages ===');
  console.log('(showing first 40 subtitle messages)\n');
});

ws.on('message', (data: WebSocket.Data) => {
  try {
    const msg = JSON.parse(data.toString());
    if (msg.type !== 'subtitle') return;

    messageCount++;
    const sub = msg as SubtitleMsg;

    if (sub.is_final) {
      finalCount++;
    } else {
      partialCount++;
    }

    // Show first 40 messages in detail
    if (messageCount <= 40) {
      const marker = sub.is_final ? 'FINAL' : 'partial';
      const ft = (sub as any).full_transcript || '';
      console.log(`[${messageCount}] ${marker} | text="${(sub.text || '').substring(0, 60)}" | growing_text="${(sub.growing_text || '').substring(0, 80)}" | full_buf_len=${ft.length} | delta="${(sub.delta || '').substring(0, 40)}"`);
    }

    // Simulate frontend logic (_processSegment)
    if (sub.is_final) {
      transcriptSegments.push(sub);
      // Clear currentSegment if start matches
      if (currentSegment && currentSegment.start === sub.start) {
        currentSegment = null;
      }
    } else {
      currentSegment = sub;
    }

    // Every 5 seconds, log the "Current" panel state
    if (messageCount % 10 === 0) {
      const displaySegment = currentSegment || (transcriptSegments.length > 0 ? transcriptSegments[transcriptSegments.length - 1] : null);
      const displayText = displaySegment ? (displaySegment.growing_text || displaySegment.text) : '(empty)';
      const source = currentSegment ? 'PARTIAL' : (transcriptSegments.length > 0 ? 'FINAL-fallback' : 'none');

      console.log(`\n--- Panel State (after ${messageCount} msgs) ---`);
      console.log(`  Current panel source: ${source}`);
      console.log(`  Current panel text: "${displayText.substring(0, 100)}"`);
      console.log(`  Transcript segments: ${transcriptSegments.length}`);
      console.log(`  Stats: ${finalCount} final, ${partialCount} partial\n`);
    }
  } catch (e) {
    // ignore non-JSON
  }
});

ws.on('error', (err) => {
  console.error('WebSocket error:', err.message);
});

ws.on('close', () => {
  console.log('\nConnection closed');
  printSummary();
});

setTimeout(() => {
  console.log(`\n\n=== Capture complete (${CAPTURE_SECS}s) ===`);
  printSummary();
  ws.close();
  process.exit(0);
}, CAPTURE_SECS * 1000);

function printSummary() {
  console.log('\n=== SUMMARY ===');
  console.log(`Total subtitle messages: ${messageCount}`);
  console.log(`  FINAL:   ${finalCount} (${messageCount > 0 ? ((finalCount / messageCount) * 100).toFixed(1) : 0}%)`);
  console.log(`  partial: ${partialCount} (${messageCount > 0 ? ((partialCount / messageCount) * 100).toFixed(1) : 0}%)`);
  console.log(`\nTranscript panel segments: ${transcriptSegments.length}`);
  console.log(`Current panel has content: ${currentSegment !== null}`);

  if (currentSegment) {
    console.log(`Current panel text: "${(currentSegment.growing_text || currentSegment.text).substring(0, 100)}"`);
  } else {
    console.log(`Current panel: EMPTY (no partial segments received)`);
  }

  if (partialCount === 0) {
    console.log('\n*** ROOT CAUSE: All messages are is_final=true!');
    console.log('    The "Current" panel only displays partial (is_final=false) segments.');
    console.log('    SentenceBuffer or RealtimeCanary is marking everything as final.');
  }

  // Show last 5 transcript segment texts
  console.log('\nLast 5 transcript segments:');
  transcriptSegments.slice(-5).forEach((s, i) => {
    console.log(`  [${i}] "${(s.growing_text || s.text).substring(0, 80)}"`);
  });
}
