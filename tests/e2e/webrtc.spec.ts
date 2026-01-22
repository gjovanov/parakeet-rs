import { test, expect } from '@playwright/test';
import {
  getWebRTCState,
  waitForConnectionState,
  waitForICEState,
  waitForPackets,
  getDetailedDiagnostics,
  setupConsoleCapture,
  formatStateForLog,
  type ConsoleMessage,
} from './helpers/webrtc-helpers';

test.describe('WebRTC Connection Tests', () => {
  let consoleLogs: ConsoleMessage[];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);

    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== 'passed') {
      console.log('\n=== Console Logs ===');
      consoleLogs
        .filter((log) => log.text.includes('[WebRTC]'))
        .forEach((log) => {
          console.log(`[${log.type}] ${log.text}`);
        });

      const diagnostics = await getDetailedDiagnostics(page);
      console.log('\n=== WebRTC Diagnostics ===');
      console.log(JSON.stringify(diagnostics, null, 2));

      const state = await getWebRTCState(page);
      console.log('\n=== Final State ===');
      console.log(formatStateForLog(state));
    }
  });

  test('should load the page and display sessions', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Multi-Session Transcription');

    const sessionPanel = page.locator('#session-panel');
    await expect(sessionPanel).toBeVisible();
  });

  test('should establish WebRTC connection when joining a session', async ({ page }) => {
    test.setTimeout(60000);

    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping connection test');
      test.skip();
      return;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    console.log('Waiting for WebRTC client to be available...');
    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    console.log('Waiting for ICE connection...');
    const iceState = await waitForICEState(page, ['connected', 'completed'], 30000);
    console.log(`ICE state reached: ${iceState}`);

    console.log('Waiting for peer connection...');
    await waitForConnectionState(page, 'connected', 30000);

    const finalState = await getWebRTCState(page);
    console.log('Final state:', formatStateForLog(finalState));

    expect(finalState?.connectionState).toBe('connected');
  });

  test('should receive audio packets after connection', async ({ page }) => {
    test.setTimeout(90000);

    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping packet test');
      test.skip();
      return;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    console.log('Waiting for connection...');
    await waitForConnectionState(page, 'connected', 30000);

    console.log('Waiting for packets (5 seconds)...');
    await page.waitForTimeout(5000);

    const packets = await waitForPackets(page, 1, 10000);
    console.log(`Received ${packets} packets`);

    expect(packets).toBeGreaterThan(0);
  });

  test('diagnostics: capture full WebRTC state', async ({ page }) => {
    test.setTimeout(90000);

    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping diagnostics test');
      test.skip();
      return;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    console.log('=== Initial State ===');
    let state = await getWebRTCState(page);
    console.log(formatStateForLog(state));

    for (let i = 0; i < 12; i++) {
      await page.waitForTimeout(5000);
      state = await getWebRTCState(page);
      console.log(`\n=== State at ${(i + 1) * 5}s ===`);
      console.log(formatStateForLog(state));

      if (state?.connectionState === 'connected' && state?.packetsReceived > 0) {
        console.log('\nConnection established and packets flowing!');
        break;
      }

      if (state?.connectionState === 'failed' || state?.iceConnectionState === 'failed') {
        console.log('\nConnection FAILED - capturing diagnostics...');
        const diagnostics = await getDetailedDiagnostics(page);
        console.log(JSON.stringify(diagnostics, null, 2));
        break;
      }
    }

    const finalDiagnostics = await getDetailedDiagnostics(page);
    console.log('\n=== Final Diagnostics ===');
    console.log(JSON.stringify(finalDiagnostics, null, 2));
  });

  test('should detect ICE failures with detailed diagnostics', async ({ page }) => {
    test.setTimeout(60000);

    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping ICE failure test');
      test.skip();
      return;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    const iceCandidates: any[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('ICE candidate')) {
        iceCandidates.push({
          timestamp: Date.now(),
          message: text,
        });
      }
    });

    for (let i = 0; i < 6; i++) {
      await page.waitForTimeout(5000);
      const state = await getWebRTCState(page);
      console.log(`\n[${(i + 1) * 5}s] ICE: ${state?.iceConnectionState}, Conn: ${state?.connectionState}`);

      if (state?.iceConnectionState === 'failed') {
        console.log('\n=== ICE FAILURE DETECTED ===');
        console.log('ICE candidates collected:', iceCandidates.length);
        iceCandidates.forEach((c) => console.log(`  ${c.message}`));

        const diagnostics = await getDetailedDiagnostics(page);
        console.log('\nFull diagnostics:', JSON.stringify(diagnostics, null, 2));

        expect(state.iceConnectionState).not.toBe('failed');
        return;
      }

      if (state?.iceConnectionState === 'connected' || state?.iceConnectionState === 'completed') {
        console.log('\nICE connected successfully!');
        expect(['connected', 'completed']).toContain(state.iceConnectionState);
        return;
      }
    }

    const finalState = await getWebRTCState(page);
    console.log('\nTest completed - final ICE state:', finalState?.iceConnectionState);
  });

  test('should verify audio element receives stream', async ({ page }) => {
    test.setTimeout(60000);

    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping audio element test');
      test.skip();
      return;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    await waitForConnectionState(page, 'connected', 30000);

    const audioState = await page.evaluate(() => {
      const client = (window as any)._webrtcClient;
      const audio = client?.audioElement;

      return {
        hasSrcObject: !!audio?.srcObject,
        paused: audio?.paused,
        muted: audio?.muted,
        volume: audio?.volume,
        readyState: audio?.readyState,
        networkState: audio?.networkState,
        streamActive: client?.remoteStream?.active,
        trackCount: client?.remoteStream?.getTracks()?.length ?? 0,
        trackDetails: client?.remoteStream?.getTracks()?.map((t: any) => ({
          kind: t.kind,
          enabled: t.enabled,
          muted: t.muted,
          readyState: t.readyState,
        })) ?? [],
      };
    });

    console.log('Audio element state:', JSON.stringify(audioState, null, 2));

    expect(audioState.hasSrcObject).toBe(true);
    expect(audioState.streamActive).toBe(true);
    expect(audioState.trackCount).toBeGreaterThan(0);
  });
});
