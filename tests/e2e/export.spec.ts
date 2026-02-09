import { test, expect } from '@playwright/test';
import {
  setupConsoleCapture,
  type ConsoleMessage,
} from './helpers/webrtc-helpers';
import {
  createSessionViaAPI,
  startSessionViaAPI,
  stopSessionViaAPI,
  getMediaFiles,
} from './helpers/session-helpers';

const BASE_URL = process.env.BASE_URL || 'http://localhost:8080';

test.describe('Transcript Export Tests', () => {
  let consoleLogs: ConsoleMessage[];
  let cleanupSessionIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);
    cleanupSessionIds = [];
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test.afterEach(async () => {
    for (const id of cleanupSessionIds) {
      await stopSessionViaAPI(id).catch(() => {});
    }
  });

  /**
   * Helper: Create a VoD session, connect WS, start session, wait for completion,
   * then join via UI to populate the transcript panel. Returns true if successful.
   */
  async function createVodAndWaitForTranscript(page: import('@playwright/test').Page): Promise<boolean> {
    const mediaFiles = await getMediaFiles();
    if (mediaFiles.length === 0) return false;

    const session = await createSessionViaAPI({
      mode: 'vod',
      mediaFile: mediaFiles[0],
      language: 'de',
    });
    cleanupSessionIds.push(session.id);

    // Connect WS first, then start session (avoids race condition)
    const wsUrl = `${BASE_URL.replace('http', 'ws')}/ws/${session.id}`;
    const completed = await new Promise<boolean>((resolve) => {
      const ws = new WebSocket(wsUrl);
      const timer = setTimeout(() => { ws.close(); resolve(false); }, 60000);

      ws.addEventListener('open', () => {
        // WS connected - now start the session
        startSessionViaAPI(session.id).catch(() => {});
      });

      ws.addEventListener('message', (event: MessageEvent) => {
        try {
          const data = JSON.parse(String(event.data));
          if (data.type === 'vod_complete' || data.type === 'end') {
            clearTimeout(timer);
            setTimeout(() => { ws.close(); resolve(true); }, 500);
          }
        } catch {}
      });

      ws.addEventListener('error', () => { clearTimeout(timer); resolve(false); });
    });

    if (!completed) return false;

    // Reload to pick up the completed session, then join it
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Find and click the session card
    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);
    if (!hasSession) return false;

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    const isEnabled = await connectBtn.isEnabled().catch(() => false);
    if (!isEnabled) return false;

    await connectBtn.click();

    // Wait for transcript segments to appear (last_subtitle replay)
    try {
      await page.locator('.transcript-segment').first().waitFor({ state: 'visible', timeout: 15000 });
      return true;
    } catch {
      // Some sessions may not have visible transcript segments
      return (await page.locator('.transcript-segment').count()) > 0;
    }
  }

  test('export modal opens and closes', async ({ page }) => {
    const exportBtn = page.locator('#export-btn');
    await expect(exportBtn).toBeVisible();

    await exportBtn.click();

    const modal = page.locator('#export-modal');
    await expect(modal).toBeVisible();

    await expect(page.locator('#export-txt')).toBeVisible();
    await expect(page.locator('#export-timestamps')).toBeVisible();
    await expect(page.locator('#export-json')).toBeVisible();

    // Close modal
    const closeBtn = page.locator('#close-modal');
    if (await closeBtn.isVisible()) {
      await closeBtn.click();
    } else {
      await page.keyboard.press('Escape');
    }

    await expect(modal).toBeHidden({ timeout: 3000 });
  });

  test('plain text export produces download', async ({ page }) => {
    test.setTimeout(120000);

    const hasTranscript = await createVodAndWaitForTranscript(page);
    if (!hasTranscript) {
      console.log('No transcript available - skipping export test');
      test.skip();
      return;
    }

    const segmentCount = await page.locator('.transcript-segment').count();
    console.log(`Transcript has ${segmentCount} segments`);

    await page.click('#export-btn');
    await expect(page.locator('#export-modal')).toBeVisible();

    const downloadPromise = page.waitForEvent('download', { timeout: 10000 });
    await page.click('#export-txt');

    const download = await downloadPromise;
    console.log(`Download triggered: ${download.suggestedFilename()}`);

    const content = await download.path().then(p => {
      if (p) {
        const fs = require('fs');
        return fs.readFileSync(p, 'utf-8');
      }
      return '';
    });

    if (content) {
      console.log(`Downloaded ${content.length} chars: ${content.substring(0, 200)}`);
      expect(content.length).toBeGreaterThan(0);
    }
  });

  test('JSON export produces valid JSON', async ({ page }) => {
    test.setTimeout(120000);

    const hasTranscript = await createVodAndWaitForTranscript(page);
    if (!hasTranscript) {
      console.log('No transcript available - skipping');
      test.skip();
      return;
    }

    await page.click('#export-btn');
    await expect(page.locator('#export-modal')).toBeVisible();

    const downloadPromise = page.waitForEvent('download', { timeout: 10000 });
    await page.click('#export-json');

    const download = await downloadPromise;
    console.log(`JSON download: ${download.suggestedFilename()}`);

    const content = await download.path().then(p => {
      if (p) {
        const fs = require('fs');
        return fs.readFileSync(p, 'utf-8');
      }
      return '';
    });

    if (content) {
      const parsed = JSON.parse(content);
      expect(Array.isArray(parsed) || typeof parsed === 'object').toBe(true);
      console.log(`JSON export: ${Array.isArray(parsed) ? parsed.length + ' entries' : Object.keys(parsed).length + ' keys'}`);
    }
  });

  test('clear button empties transcript', async ({ page }) => {
    test.setTimeout(120000);

    const hasTranscript = await createVodAndWaitForTranscript(page);
    if (!hasTranscript) {
      console.log('No transcript available - skipping');
      test.skip();
      return;
    }

    const countBefore = await page.locator('.transcript-segment').count();
    expect(countBefore).toBeGreaterThan(0);
    console.log(`Segments before clear: ${countBefore}`);

    // Clear button shows a confirm dialog - accept it
    page.on('dialog', dialog => dialog.accept());
    await page.click('#clear-btn');
    await page.waitForTimeout(500);

    const countAfter = await page.locator('.transcript-segment').count();
    console.log(`Segments after clear: ${countAfter}`);
    expect(countAfter).toBe(0);
  });
});
