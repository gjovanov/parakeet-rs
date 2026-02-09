import { test, expect } from '@playwright/test';
import {
  navigateToCreateTab,
  navigateToSessionsTab,
  navigateToMediaTab,
  waitForFormPopulated,
  stopSessionViaAPI,
  getSessions,
} from './helpers/session-helpers';
import { setupConsoleCapture, type ConsoleMessage } from './helpers/webrtc-helpers';

test.describe('Session Lifecycle Tests', () => {
  let consoleLogs: ConsoleMessage[];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('page loads with correct title and session panel', async ({ page }) => {
    await expect(page).toHaveTitle(/Multi-Session Transcription/);
    await expect(page.locator('#session-panel')).toBeVisible();
    await expect(page.locator('h1')).toContainText('Multi-Session Transcription');
  });

  test('session tabs navigate correctly', async ({ page }) => {
    // Sessions tab should be active by default
    const sessionsTab = page.locator('.session-tab[data-tab="sessions"]');
    await expect(sessionsTab).toHaveClass(/active/);

    // Switch to Create tab
    await navigateToCreateTab(page);
    await expect(page.locator('#create-content')).toBeVisible();

    // Switch to Media tab
    await navigateToMediaTab(page);
    await expect(page.locator('#media-content')).toBeVisible();

    // Switch back to Sessions
    await navigateToSessionsTab(page);
  });

  test('create form populates dropdowns from server', async ({ page }) => {
    await navigateToCreateTab(page);
    await waitForFormPopulated(page);

    // Model select should have options
    const modelOptions = await page.locator('#model-select option').count();
    expect(modelOptions).toBeGreaterThan(1);
    console.log(`Models available: ${modelOptions - 1}`);

    // Mode select should have options
    const modeOptions = await page.locator('#mode-select option').count();
    expect(modeOptions).toBeGreaterThan(0);
    console.log(`Modes available: ${modeOptions}`);

    // Language select should have German pre-selected
    const selectedLang = await page.locator('#language-select').inputValue();
    expect(selectedLang).toBe('de');
  });

  test('create session via UI with media file', async ({ page }) => {
    test.setTimeout(30000);

    await navigateToCreateTab(page);
    await waitForFormPopulated(page);

    // Select model (first available)
    const firstModel = await page.locator('#model-select option:not([value=""])').first().getAttribute('value');
    if (!firstModel) {
      console.log('No models available - skipping');
      test.skip();
      return;
    }
    await page.selectOption('#model-select', firstModel);

    // Check if media files are available
    const mediaOptions = await page.locator('#media-select option:not([value=""])').count();
    if (mediaOptions === 0) {
      console.log('No media files available - skipping');
      test.skip();
      return;
    }

    // Select first media file
    const firstMedia = await page.locator('#media-select option:not([value=""])').first().getAttribute('value');
    await page.selectOption('#media-select', firstMedia!);

    // Select mode
    await page.selectOption('#mode-select', 'speedy');

    // Click Create Session
    const countBefore = await getSessions().then(s => s.length);
    await page.click('#create-session-btn');

    // Wait for session to appear in the sessions tab
    await page.waitForTimeout(2000);

    // Switch to sessions tab to verify
    await navigateToSessionsTab(page);

    // Should have at least one session card
    const sessionCards = page.locator('.session-card');
    await expect(sessionCards.first()).toBeVisible({ timeout: 10000 });

    const countAfter = await sessionCards.count();
    console.log(`Sessions: ${countBefore} -> ${countAfter}`);
    expect(countAfter).toBeGreaterThanOrEqual(countBefore);

    // Clean up: stop the created session
    const sessions = await getSessions();
    if (sessions.length > 0) {
      const newest = sessions[sessions.length - 1];
      await stopSessionViaAPI(newest.id);
    }
  });

  test('session card shows model and mode info', async ({ page }) => {
    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    // Session card should have title and meta info
    const title = await sessionCard.locator('.session-title').innerText();
    expect(title.length).toBeGreaterThan(0);
    console.log(`Session title: ${title}`);

    const meta = await sessionCard.locator('.session-meta').innerText();
    expect(meta.length).toBeGreaterThan(0);
    console.log(`Session meta: ${meta}`);

    // Status badge should exist
    const status = await sessionCard.locator('.session-status').innerText();
    expect(status.length).toBeGreaterThan(0);
    console.log(`Session status: ${status}`);
  });

  test('selecting a session enables the join button', async ({ page }) => {
    const sessionCard = page.locator('.session-card').first();
    const hasSession = await sessionCard.isVisible().catch(() => false);

    if (!hasSession) {
      console.log('No sessions available - skipping');
      test.skip();
      return;
    }

    const connectBtn = page.locator('#connect-btn');

    // Before clicking, button should be disabled
    await expect(connectBtn).toBeDisabled();

    // Click a session card
    await sessionCard.click();
    await page.waitForTimeout(500);

    // Now the session card should be selected
    await expect(sessionCard).toHaveClass(/selected/);

    // Connect button should become enabled
    await expect(connectBtn).toBeEnabled({ timeout: 5000 });
  });

  test('mode selection shows/hides advanced configs', async ({ page }) => {
    await navigateToCreateTab(page);
    await waitForFormPopulated(page);

    // Parallel config should be hidden by default (speedy mode)
    await expect(page.locator('#parallel-config')).toBeHidden();
    await expect(page.locator('#pause-config')).toBeHidden();

    // Switch to a parallel mode if available
    const hasParallel = await page.locator('#mode-select option[value="parallel"]').count();
    if (hasParallel > 0) {
      await page.selectOption('#mode-select', 'parallel');
      await page.waitForTimeout(300);
      await expect(page.locator('#parallel-config')).toBeVisible();
      console.log('Parallel config shown for parallel mode');
    }

    // Switch to pause_based mode if available
    const hasPauseBased = await page.locator('#mode-select option[value="pause_based"]').count();
    if (hasPauseBased > 0) {
      await page.selectOption('#mode-select', 'pause_based');
      await page.waitForTimeout(300);
      await expect(page.locator('#pause-config')).toBeVisible();
      console.log('Pause config shown for pause_based mode');
    }

    // Switch back to speedy - parallel config should hide
    // Note: speedy is in PAUSE_MODES so pause config stays visible
    await page.selectOption('#mode-select', 'speedy');
    await page.waitForTimeout(300);
    await expect(page.locator('#parallel-config')).toBeHidden();

    // Switch to a non-pause mode (e.g. low_latency or ultra_low_latency) to hide pause config
    const hasLowLatency = await page.locator('#mode-select option[value="low_latency"]').count();
    if (hasLowLatency > 0) {
      await page.selectOption('#mode-select', 'low_latency');
      await page.waitForTimeout(300);
      await expect(page.locator('#pause-config')).toBeHidden();
      console.log('Pause config hidden for low_latency mode');
    }
  });

  test('FAB controls visibility toggle', async ({ page }) => {
    await navigateToCreateTab(page);
    await waitForFormPopulated(page);

    // FAB URL group and send type group should be hidden by default
    await expect(page.locator('#fab-url-group')).toBeHidden();
    await expect(page.locator('#fab-send-type-group')).toBeHidden();

    // Select FAB "enabled" → both groups should become visible
    await page.selectOption('#fab-enabled-select', 'enabled');
    await page.waitForTimeout(300);
    await expect(page.locator('#fab-url-group')).toBeVisible();
    await expect(page.locator('#fab-send-type-group')).toBeVisible();

    // Verify send type dropdown has 3 options
    const sendTypeOptions = await page.locator('#fab-send-type-select option').count();
    expect(sendTypeOptions).toBe(3);

    // Select FAB "disabled" → both groups should hide
    await page.selectOption('#fab-enabled-select', 'disabled');
    await page.waitForTimeout(300);
    await expect(page.locator('#fab-url-group')).toBeHidden();
    await expect(page.locator('#fab-send-type-group')).toBeHidden();

    // Select FAB "default" → both groups should stay hidden
    await page.selectOption('#fab-enabled-select', 'default');
    await page.waitForTimeout(300);
    await expect(page.locator('#fab-url-group')).toBeHidden();
    await expect(page.locator('#fab-send-type-group')).toBeHidden();
  });

  test('config API returns fabSendType', async ({ page }) => {
    const BASE_URL = process.env.BASE_URL || 'http://localhost:8080';
    const resp = await fetch(`${BASE_URL}/api/config`);
    expect(resp.ok).toBeTruthy();
    const config = await resp.json();
    expect(config.fabSendType).toBeDefined();
    expect(['growing', 'confirmed']).toContain(config.fabSendType);
    console.log(`FAB send type from config: ${config.fabSendType}`);
  });

  test('FAB send type default label from server config', async ({ page }) => {
    await navigateToCreateTab(page);
    await waitForFormPopulated(page);

    // The first option of fab-send-type-select should reflect server config
    const firstOptionText = await page.locator('#fab-send-type-select option').first().innerText();
    // Should contain either "growing" or "confirmed" from server config
    const hasConfigValue = firstOptionText.toLowerCase().includes('growing') || firstOptionText.toLowerCase().includes('confirmed');
    expect(hasConfigValue).toBeTruthy();
    console.log(`FAB send type default label: ${firstOptionText}`);
  });

  test('source tabs switch between media and SRT', async ({ page }) => {
    await navigateToCreateTab(page);

    // Media source should be visible by default
    await expect(page.locator('#media-source-content')).toBeVisible();
    await expect(page.locator('#srt-source-content')).toBeHidden();

    // Click SRT tab
    await page.click('.source-tab[data-source="srt"]');
    await page.waitForTimeout(300);

    await expect(page.locator('#media-source-content')).toBeHidden();
    await expect(page.locator('#srt-source-content')).toBeVisible();

    // Click back to media
    await page.click('.source-tab[data-source="media"]');
    await page.waitForTimeout(300);

    await expect(page.locator('#media-source-content')).toBeVisible();
    await expect(page.locator('#srt-source-content')).toBeHidden();
  });
});
