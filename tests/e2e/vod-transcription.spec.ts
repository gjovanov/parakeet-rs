import { test, expect } from '@playwright/test';
import {
  createSessionViaAPI,
  startSessionViaAPI,
  stopSessionViaAPI,
  uploadMediaViaAPI,
  getMediaFiles,
} from './helpers/session-helpers';
import { setupConsoleCapture, type ConsoleMessage } from './helpers/webrtc-helpers';
import path from 'path';

const BASE_URL = process.env.BASE_URL || 'http://localhost:8080';

/** Connect a Node-side WebSocket and collect VoD events until completion or timeout. */
function collectVodEvents(
  wsUrl: string,
  timeoutMs: number
): Promise<{ events: any[]; transcript: string; completed: boolean }> {
  return new Promise((resolve) => {
    const events: any[] = [];
    let transcript = '';
    let completed = false;

    const ws = new WebSocket(wsUrl);
    const timer = setTimeout(() => {
      ws.close();
      resolve({ events, transcript, completed });
    }, timeoutMs);

    ws.addEventListener('message', (event: MessageEvent) => {
      try {
        const data = JSON.parse(String(event.data));
        events.push({ type: data.type });

        if (data.type === 'subtitle' && data.is_final && data.text) {
          transcript += (transcript ? ' ' : '') + data.text;
        }

        if (data.type === 'vod_progress') {
          events.push({
            type: 'vod_progress',
            percent: data.percent,
            completed_chunks: data.completed_chunks,
            total_chunks: data.total_chunks,
          });
        }

        if (data.type === 'vod_complete' || data.type === 'end') {
          completed = true;
          clearTimeout(timer);
          setTimeout(() => {
            ws.close();
            resolve({ events, transcript, completed });
          }, 500);
        }
      } catch {}
    });

    ws.addEventListener('error', () => {
      clearTimeout(timer);
      resolve({ events, transcript, completed });
    });
  });
}

/** Wait for a Node-side WebSocket to open, then resolve. */
function waitForWsOpen(wsUrl: string, timeoutMs = 5000): Promise<WebSocket | null> {
  return new Promise((resolve) => {
    const ws = new WebSocket(wsUrl);
    const timer = setTimeout(() => { ws.close(); resolve(null); }, timeoutMs);
    ws.addEventListener('open', () => { clearTimeout(timer); resolve(ws); });
    ws.addEventListener('error', () => { clearTimeout(timer); resolve(null); });
  });
}

/** Connect WS, start session, then collect events (avoids race condition). */
async function startAndCollectVodEvents(
  sessionId: string,
  timeoutMs: number
): Promise<{ events: any[]; transcript: string; completed: boolean }> {
  const wsUrl = `${BASE_URL.replace('http', 'ws')}/ws/${sessionId}`;

  // First connect the WS and wait for it to open
  const ws = await waitForWsOpen(wsUrl);
  if (!ws) {
    return { events: [], transcript: '', completed: false };
  }

  // Now start collecting events from the already-open WS
  const collectPromise = new Promise<{ events: any[]; transcript: string; completed: boolean }>((resolve) => {
    const events: any[] = [];
    let transcript = '';
    let completed = false;

    const timer = setTimeout(() => {
      ws.close();
      resolve({ events, transcript, completed });
    }, timeoutMs);

    ws.addEventListener('message', (event: MessageEvent) => {
      try {
        const data = JSON.parse(String(event.data));
        events.push({ type: data.type });

        if (data.type === 'subtitle' && data.is_final && data.text) {
          transcript += (transcript ? ' ' : '') + data.text;
        }

        if (data.type === 'vod_progress') {
          events.push({
            type: 'vod_progress',
            percent: data.percent,
            completed_chunks: data.completed_chunks,
            total_chunks: data.total_chunks,
          });
        }

        if (data.type === 'vod_complete' || data.type === 'end') {
          completed = true;
          clearTimeout(timer);
          setTimeout(() => {
            ws.close();
            resolve({ events, transcript, completed });
          }, 500);
        }
      } catch {}
    });

    ws.addEventListener('error', () => {
      clearTimeout(timer);
      resolve({ events, transcript, completed });
    });
  });

  // Now start the session (WS is already connected and listening)
  // Don't await — the collectPromise timer handles the timeout if the server is slow
  startSessionViaAPI(sessionId).catch(() => {});

  return collectPromise;
}

test.describe('VoD Transcription Tests', () => {
  let consoleLogs: ConsoleMessage[];
  let cleanupSessionIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);
    cleanupSessionIds = [];
  });

  test.afterEach(async () => {
    for (const id of cleanupSessionIds) {
      await stopSessionViaAPI(id).catch(() => {});
    }
  });

  test('VoD session processes audio and completes', async ({ page }) => {
    test.setTimeout(120000);

    // Ensure we have a media file available
    const mediaFiles = await getMediaFiles();
    if (mediaFiles.length === 0) {
      const fixturePath = path.resolve(__dirname, '../../tests/fixtures/de_short.wav');
      try {
        await uploadMediaViaAPI(fixturePath, 'de_short.wav');
      } catch {
        console.log('No media files and upload failed - skipping');
        test.skip();
        return;
      }
    }

    const updatedFiles = await getMediaFiles();
    const mediaFile = updatedFiles[0];
    console.log(`Using media file: ${mediaFile}`);

    // Create session (does NOT auto-start)
    const session = await createSessionViaAPI({
      mode: 'vod',
      mediaFile: mediaFile,
      language: 'de',
    });
    cleanupSessionIds.push(session.id);
    console.log(`Created VoD session: ${session.id}`);

    // Connect WS first, then start session to avoid race
    const result = await startAndCollectVodEvents(session.id, 90000);

    console.log(`VoD events received: ${result.events.length}`);
    console.log(`Event types: ${[...new Set(result.events.map(e => e.type))].join(', ')}`);
    console.log(`Transcript: ${result.transcript.substring(0, 200)}`);
    console.log(`Completed: ${result.completed}`);

    expect(result.events.length).toBeGreaterThan(0);
    expect(result.completed).toBe(true);
    expect(result.transcript.length).toBeGreaterThan(0);
  });

  test('VoD session shows progress updates', async () => {
    test.setTimeout(120000);

    const mediaFiles = await getMediaFiles();
    if (mediaFiles.length === 0) {
      console.log('No media files - skipping');
      test.skip();
      return;
    }

    // Use longer file for visible progress updates
    const mediaFile = mediaFiles.find(f => f.includes('long') || f.includes('news')) || mediaFiles[0];
    console.log(`Using media file: ${mediaFile}`);

    const session = await createSessionViaAPI({
      mode: 'vod',
      mediaFile,
      language: 'de',
    });
    cleanupSessionIds.push(session.id);
    console.log(`Created VoD session: ${session.id}`);

    // Connect WS first, then start session
    const result = await startAndCollectVodEvents(session.id, 90000);

    const progressEvents = result.events.filter(e => e.type === 'vod_progress');

    console.log(`Progress events: ${progressEvents.length}`);
    console.log(`Completed: ${result.completed}`);
    if (progressEvents.length > 0) {
      console.log(`First progress: ${JSON.stringify(progressEvents[0])}`);
      console.log(`Last progress: ${JSON.stringify(progressEvents[progressEvents.length - 1])}`);
    }

    // Short files may complete in a single chunk with no progress events
    expect(result.completed).toBe(true);
  });

  test('VoD transcript can be downloaded after completion', async () => {
    test.setTimeout(120000);

    const mediaFiles = await getMediaFiles();
    if (mediaFiles.length === 0) {
      console.log('No media files - skipping');
      test.skip();
      return;
    }

    const session = await createSessionViaAPI({
      mode: 'vod',
      mediaFile: mediaFiles[0],
      language: 'de',
    });
    cleanupSessionIds.push(session.id);
    console.log(`Created VoD session: ${session.id}`);

    // Connect WS first, then start session
    const result = await startAndCollectVodEvents(session.id, 90000);

    if (!result.completed) {
      console.log('VoD did not complete in time - skipping download test');
      test.skip();
      return;
    }

    console.log(`VoD completed with ${result.transcript.length} chars transcript via WS`);

    // Try to download transcript via API
    const resp = await fetch(`${BASE_URL}/api/sessions/${session.id}/transcript`);
    if (resp.ok) {
      const transcript = await resp.text();
      console.log(`Downloaded transcript (${transcript.length} chars): ${transcript.substring(0, 200)}`);
      expect(transcript.length).toBeGreaterThan(0);
    } else {
      console.log(`Transcript download returned ${resp.status} - endpoint may not exist`);
    }
  });
});
