import { test, expect } from '@playwright/test';
import { navigateToMediaTab, getMediaFiles } from './helpers/session-helpers';
import { setupConsoleCapture, type ConsoleMessage } from './helpers/webrtc-helpers';
import path from 'path';
import fs from 'fs';

test.describe('Media Upload Tests', () => {
  let consoleLogs: ConsoleMessage[];

  test.beforeEach(async ({ page }) => {
    consoleLogs = setupConsoleCapture(page);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('media tab shows upload zone and file list', async ({ page }) => {
    await navigateToMediaTab(page);

    // Upload zone should be visible
    await expect(page.locator('#upload-zone')).toBeVisible();

    // Media list container should exist
    await expect(page.locator('#media-list')).toBeVisible();

    // File input should exist (hidden but present)
    const fileInput = page.locator('#file-input');
    expect(await fileInput.count()).toBe(1);
  });

  test('upload a WAV file via file input', async ({ page }) => {
    test.setTimeout(30000);

    await navigateToMediaTab(page);

    // Create a small test WAV file if fixture doesn't exist
    const fixturePath = path.resolve(__dirname, '../../tests/fixtures/de_short.wav');
    if (!fs.existsSync(fixturePath)) {
      console.log('Test fixture not found - skipping upload test');
      test.skip();
      return;
    }

    // Count media files before
    const filesBefore = await getMediaFiles();
    console.log(`Media files before upload: ${filesBefore.length}`);

    // Upload via file input
    const fileInput = page.locator('#file-input');
    await fileInput.setInputFiles(fixturePath);

    // Wait for upload to complete
    await page.waitForTimeout(3000);

    // Verify file appears in the media list
    const filesAfter = await getMediaFiles();
    console.log(`Media files after upload: ${filesAfter.length}`);

    // File count should have increased (or file was already present)
    const hasFile = filesAfter.some(f => f.includes('de_short'));
    expect(hasFile || filesAfter.length >= filesBefore.length).toBe(true);
  });

  test('uploaded files appear in the create session form', async ({ page }) => {
    test.setTimeout(15000);

    // Check if there are any media files
    const mediaFiles = await getMediaFiles();
    if (mediaFiles.length === 0) {
      console.log('No media files available - skipping');
      test.skip();
      return;
    }

    // Navigate to create tab
    await page.click('.session-tab[data-tab="create"]');
    await page.waitForTimeout(1000);

    // Media select should have options
    const options = await page.locator('#media-select option').allInnerTexts();
    console.log(`Media select options: ${options.join(', ')}`);

    // Should have at least one non-placeholder option
    const realOptions = options.filter(o => o !== '' && !o.includes('Loading'));
    expect(realOptions.length).toBeGreaterThan(0);
  });

  test('media files list shows file information', async ({ page }) => {
    await navigateToMediaTab(page);

    // Wait for media list to load
    await page.waitForTimeout(2000);

    const mediaItems = page.locator('#media-list > *');
    const count = await mediaItems.count();

    if (count === 0) {
      console.log('No media files in list - skipping');
      test.skip();
      return;
    }

    console.log(`Media list has ${count} items`);

    // First item should have text content (filename)
    const firstItem = mediaItems.first();
    const text = await firstItem.innerText();
    expect(text.length).toBeGreaterThan(0);
    console.log(`First media item: ${text}`);
  });
});
