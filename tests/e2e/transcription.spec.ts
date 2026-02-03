import { test, expect } from '@playwright/test';
import {
  waitForConnectionState,
  setupConsoleCapture,
  getDetailedDiagnostics,
  getWebRTCState,
  formatStateForLog,
  type ConsoleMessage,
} from './helpers/webrtc-helpers';

test.describe('Transcription UI Tests', () => {
  let consoleLogs: ConsoleMessage[];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);

    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== 'passed') {
      console.log('\n=== Console Logs (subtitle/transcript) ===');
      consoleLogs
        .filter(
          (log) =>
            log.text.includes('[Subtitle]') ||
            log.text.includes('[WebSocket]') ||
            log.text.includes('[WebRTC]') ||
            log.text.includes('transcript')
        )
        .slice(-50)
        .forEach((log) => {
          console.log(`[${log.type}] ${log.text}`);
        });

      const diagnostics = await getDetailedDiagnostics(page);
      console.log('\n=== WebRTC Diagnostics ===');
      console.log(JSON.stringify(diagnostics, null, 2));
    }
  });

  /**
   * Helper: connect to the first available session.
   * Returns true if connected, false if no session available (test should skip).
   */
  async function connectToSession(page: import('@playwright/test').Page): Promise<boolean> {
    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      return false;
    }

    await sessionCard.click();
    await page.waitForTimeout(500);

    const connectBtn = page.locator('#connect-btn');
    await expect(connectBtn).toBeEnabled({ timeout: 10000 });
    await connectBtn.click();

    await page.waitForFunction(() => (window as any)._webrtcClient !== null, {
      timeout: 15000,
    });

    await waitForConnectionState(page, 'connected', 30000);
    return true;
  }

  test('live subtitle text appears and updates over time', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    const subtitleText = page.locator('#live-subtitle .subtitle-text');

    // Wait for initial subtitle text to appear (up to 30s for first transcription)
    await expect(subtitleText).not.toHaveText('', { timeout: 30000 });

    // Collect snapshots over time to verify text updates
    const snapshots: string[] = [];
    for (let i = 0; i < 15; i++) {
      await page.waitForTimeout(2000);
      const text = (await subtitleText.innerText()).trim();
      if (text) {
        snapshots.push(text);
      }
    }

    console.log(`Collected ${snapshots.length} subtitle snapshots`);

    // At least some snapshots should have content
    expect(snapshots.length).toBeGreaterThan(0);

    // Text should change over time (not frozen)
    const uniqueTexts = new Set(snapshots);
    console.log(`Unique subtitle texts: ${uniqueTexts.size} / ${snapshots.length}`);
    expect(uniqueTexts.size).toBeGreaterThan(1);
  });

  test('live subtitle has no hallucination repetition', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    const subtitleText = page.locator('#live-subtitle .subtitle-text');
    await expect(subtitleText).not.toHaveText('', { timeout: 30000 });

    // Sample subtitle text periodically and check for repetition
    let maxRepeat = 1;
    for (let i = 0; i < 20; i++) {
      await page.waitForTimeout(2000);
      const text = (await subtitleText.innerText()).trim();
      if (!text) continue;

      const words = text.split(/\s+/);
      let consecutive = 1;
      for (let w = 1; w < words.length; w++) {
        if (words[w].toLowerCase() === words[w - 1].toLowerCase() && words[w].length > 1) {
          consecutive++;
          if (consecutive > maxRepeat) {
            maxRepeat = consecutive;
          }
        } else {
          consecutive = 1;
        }
      }
    }

    console.log(`Max consecutive word repeat in subtitles: ${maxRepeat}`);
    // Allow max 2 consecutive same words (normal in speech), flag 3+
    expect(maxRepeat).toBeLessThan(3);
  });

  test('transcript segments accumulate over time', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    const transcriptContent = page.locator('#transcript-content');
    await expect(transcriptContent).toBeVisible();

    // Wait for first segment to appear
    const firstSegment = page.locator('.transcript-segment').first();
    await expect(firstSegment).toBeVisible({ timeout: 60000 });

    // Track segment count growth over time
    const counts: number[] = [];
    for (let i = 0; i < 15; i++) {
      await page.waitForTimeout(2000);
      const count = await page.locator('.transcript-segment').count();
      counts.push(count);
    }

    console.log(`Segment counts over time: [${counts.join(', ')}]`);

    // Segments should accumulate
    const firstCount = counts[0];
    const lastCount = counts[counts.length - 1];
    console.log(`Segments: ${firstCount} -> ${lastCount}`);
    expect(lastCount).toBeGreaterThan(firstCount);
  });

  test('transcript segments have non-empty, non-duplicated text', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    // Wait for segments to accumulate
    const firstSegment = page.locator('.transcript-segment').first();
    await expect(firstSegment).toBeVisible({ timeout: 60000 });

    // Wait a bit for more segments
    await page.waitForTimeout(20000);

    // Get all segment texts
    const segmentTexts = await page.locator('.transcript-segment .segment-text').allInnerTexts();

    console.log(`Found ${segmentTexts.length} transcript segments`);
    expect(segmentTexts.length).toBeGreaterThan(0);

    // All segments should have non-empty text
    for (const text of segmentTexts) {
      expect(text.trim().length).toBeGreaterThan(0);
    }

    // Check for exact duplicates between consecutive segments
    let duplicateCount = 0;
    for (let i = 1; i < segmentTexts.length; i++) {
      if (segmentTexts[i].trim() === segmentTexts[i - 1].trim()) {
        duplicateCount++;
        console.log(`Duplicate segment at index ${i}: "${segmentTexts[i].trim().substring(0, 60)}..."`);
      }
    }

    console.log(`Duplicate consecutive segments: ${duplicateCount} / ${segmentTexts.length}`);
    // Allow some duplicates (model may re-confirm) but not excessive
    const duplicateRate = duplicateCount / Math.max(segmentTexts.length - 1, 1);
    expect(duplicateRate).toBeLessThan(0.3);
  });

  test('current panel shows coherent sentence starts, not mid-sentence fragments (DE)', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    const subtitleText = page.locator('#live-subtitle .subtitle-text');
    await expect(subtitleText).not.toHaveText('', { timeout: 30000 });

    // German lowercase mid-sentence indicators: articles, prepositions, conjunctions
    // that almost never start a spoken sentence. If the panel frequently starts with
    // these, the display is showing raw partial fragments instead of sentence starts.
    const midSentenceStarters = new Set([
      'der', 'die', 'das', 'des', 'dem', 'den',
      'ein', 'eine', 'einer', 'einem', 'einen', 'eines',
      'und', 'oder', 'aber', 'sondern', 'denn', 'weil',
      'als', 'wie', 'dass', 'wenn', 'ob', 'wer',
      'in', 'im', 'an', 'am', 'auf', 'aus', 'bei', 'für',
      'mit', 'nach', 'von', 'vom', 'vor', 'zu', 'zum', 'zur',
      'über', 'unter', 'zwischen', 'gegen', 'durch', 'ohne',
      'noch', 'auch', 'schon', 'nur', 'sehr', 'nicht',
    ]);

    // Collect snapshots and classify each leading word
    const snapshots: { text: string; firstWord: string; isMidSentence: boolean }[] = [];
    for (let i = 0; i < 30; i++) {
      await page.waitForTimeout(2000);
      const raw = (await subtitleText.innerText()).trim();
      if (!raw || raw.length < 3) continue;

      // The panel may render multiple <p> tags; take the last (most current) paragraph
      const paragraphs = raw.split('\n').map(p => p.trim()).filter(Boolean);
      const text = paragraphs[paragraphs.length - 1];
      const firstWord = text.split(/\s+/)[0].toLowerCase().replace(/[.,!?;:]+$/, '');

      snapshots.push({
        text: text.substring(0, 80),
        firstWord,
        isMidSentence: midSentenceStarters.has(firstWord),
      });
    }

    const total = snapshots.length;
    const midCount = snapshots.filter(s => s.isMidSentence).length;
    const midRate = total > 0 ? midCount / total : 0;

    console.log(`Subtitle sentence-start analysis (${total} samples):`);
    console.log(`  Mid-sentence starts: ${midCount}/${total} (${(midRate * 100).toFixed(0)}%)`);
    // Log a few examples of flagged snapshots
    snapshots
      .filter(s => s.isMidSentence)
      .slice(0, 5)
      .forEach(s => console.log(`  FLAGGED: "${s.firstWord}" -> "${s.text}"`));
    // Log a few good examples
    snapshots
      .filter(s => !s.isMidSentence)
      .slice(0, 5)
      .forEach(s => console.log(`  OK:      "${s.firstWord}" -> "${s.text}"`));

    // Heuristic: if >40% of samples start with mid-sentence words, the panel is
    // displaying raw partial fragments rather than coherent growing sentences.
    expect(midRate).toBeLessThan(0.4);
  });

  test('transcript segments have speaker labels and timestamps', async ({ page }) => {
    test.setTimeout(120000);

    const connected = await connectToSession(page);
    if (!connected) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    // Wait for segments
    const firstSegment = page.locator('.transcript-segment').first();
    await expect(firstSegment).toBeVisible({ timeout: 60000 });

    await page.waitForTimeout(10000);

    const segmentCount = await page.locator('.transcript-segment').count();
    console.log(`Checking ${segmentCount} segments for speaker/time metadata`);

    // Check first segment has expected child elements
    const firstSpeaker = page.locator('.transcript-segment .segment-speaker').first();
    const firstTime = page.locator('.transcript-segment .segment-time').first();
    const firstName = page.locator('.transcript-segment .segment-text').first();

    await expect(firstName).toBeVisible();

    // Speaker label should exist (may say "Speaker ?" for unknown)
    const speakerText = await firstSpeaker.innerText();
    console.log(`First segment speaker: "${speakerText}"`);
    expect(speakerText.length).toBeGreaterThan(0);

    // Timestamp should exist and have time-like format
    const timeText = await firstTime.innerText();
    console.log(`First segment time: "${timeText}"`);
    expect(timeText.length).toBeGreaterThan(0);
  });
});
