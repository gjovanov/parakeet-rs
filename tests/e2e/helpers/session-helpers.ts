import { Page, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:8080';

export interface SessionInfo {
  id: string;
  model: string;
  mode: string;
  status: string;
}

/**
 * Create a session via the REST API (faster than UI for setup).
 */
export async function createSessionViaAPI(options: {
  model?: string;
  mode?: string;
  language?: string;
  mediaFile?: string;
  srtStreamId?: string;
  sentenceCompletion?: string;
  fabEnabled?: string;
  fabSendType?: string;
}): Promise<SessionInfo> {
  const body: Record<string, any> = {
    model_id: options.model || 'canary-1b',
    mode: options.mode || 'speedy',
    language: options.language || 'de',
    sentence_completion: options.sentenceCompletion || 'off',
  };

  if (options.mediaFile) {
    // media_id is the filename without extension
    body.media_id = options.mediaFile.replace(/\.\w+$/, '');
  }
  if (options.srtStreamId) {
    body.srt_channel_id = options.srtStreamId;
  }
  if (options.fabEnabled) {
    body.fab_enabled = options.fabEnabled;
  }
  if (options.fabSendType) {
    body.fab_send_type = options.fabSendType;
  }

  const resp = await fetch(`${BASE_URL}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    throw new Error(`Create session failed: ${resp.status} ${await resp.text()}`);
  }

  const json = await resp.json();
  const data = json.data || json;
  return {
    id: data.id,
    model: data.model_id || body.model_id,
    mode: data.mode || body.mode,
    status: data.state || data.status || 'created',
  };
}

/**
 * Start a session via the REST API.
 */
export async function startSessionViaAPI(sessionId: string): Promise<void> {
  const resp = await fetch(`${BASE_URL}/api/sessions/${sessionId}/start`, {
    method: 'POST',
  });
  if (!resp.ok) {
    throw new Error(`Start session failed: ${resp.status} ${await resp.text()}`);
  }
}

/**
 * Stop a session via the REST API.
 */
export async function stopSessionViaAPI(sessionId: string): Promise<void> {
  const resp = await fetch(`${BASE_URL}/api/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  // DELETE may return 404 if already stopped - that's OK
}

/**
 * Upload a media file via the REST API.
 */
export async function uploadMediaViaAPI(filePath: string, fileName: string): Promise<string> {
  const fs = await import('fs');
  const fileBytes = fs.readFileSync(filePath);
  const blob = new Blob([fileBytes], { type: 'audio/wav' });
  const formData = new FormData();
  formData.append('file', blob, fileName);

  const resp = await fetch(`${BASE_URL}/api/media/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    throw new Error(`Upload failed: ${resp.status} ${await resp.text()}`);
  }

  const data = await resp.json();
  return data.filename || fileName;
}

/**
 * Get the list of available models from the server.
 */
export async function getModels(): Promise<string[]> {
  const resp = await fetch(`${BASE_URL}/api/models`);
  const json = await resp.json();
  const models = json.data || json.models || [];
  return models.map((m: any) => m.id || m);
}

/**
 * Get the list of available media files.
 */
export async function getMediaFiles(): Promise<string[]> {
  const resp = await fetch(`${BASE_URL}/api/media`);
  const json = await resp.json();
  const files = json.data || json.files || [];
  return files.map((f: any) => f.filename || f.name || f);
}

/**
 * Get the list of active sessions.
 */
export async function getSessions(): Promise<any[]> {
  const resp = await fetch(`${BASE_URL}/api/sessions`);
  const json = await resp.json();
  return json.data || json.sessions || [];
}

/**
 * Wait for the session list in the UI to refresh and show a specific session.
 */
export async function waitForSessionInUI(page: Page, sessionId: string, timeoutMs = 15000): Promise<void> {
  await page.waitForFunction(
    (id) => {
      const cards = document.querySelectorAll('.session-card');
      for (const card of cards) {
        if (card.getAttribute('data-session-id') === id || card.textContent?.includes(id.substring(0, 8))) {
          return true;
        }
      }
      return false;
    },
    sessionId,
    { timeout: timeoutMs }
  );
}

/**
 * Navigate to the "Create New" tab in the session panel.
 */
export async function navigateToCreateTab(page: Page): Promise<void> {
  const createTab = page.locator('.session-tab[data-tab="create"]');
  await createTab.click();
  await expect(page.locator('#create-content')).toBeVisible();
}

/**
 * Navigate to the "Sessions" tab.
 */
export async function navigateToSessionsTab(page: Page): Promise<void> {
  const sessionsTab = page.locator('.session-tab[data-tab="sessions"]');
  await sessionsTab.click();
  await expect(page.locator('#sessions-content')).toBeVisible();
}

/**
 * Navigate to the "Media Files" tab.
 */
export async function navigateToMediaTab(page: Page): Promise<void> {
  const mediaTab = page.locator('.session-tab[data-tab="media"]');
  await mediaTab.click();
  await expect(page.locator('#media-content')).toBeVisible();
}

/**
 * Wait for dropdowns to be populated (models, modes, etc.)
 */
export async function waitForFormPopulated(page: Page, timeoutMs = 10000): Promise<void> {
  // Wait for model select to have more than the placeholder option
  await page.waitForFunction(
    () => {
      const select = document.querySelector('#model-select') as HTMLSelectElement;
      return select && select.options.length > 1;
    },
    { timeout: timeoutMs }
  );
}
