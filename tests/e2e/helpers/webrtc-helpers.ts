import { Page, expect } from '@playwright/test';

export interface WebRTCState {
  connectionState: RTCPeerConnectionState | null;
  iceConnectionState: RTCIceConnectionState | null;
  iceGatheringState: RTCIceGatheringState | null;
  signalingState: RTCSignalingState | null;
  packetsReceived: number;
  bytesReceived: number;
  packetsLost: number;
  jitter: number;
  roundTripTime: number;
  audioElementReady: boolean;
  remoteStreamActive: boolean;
  trackCount: number;
}

export interface ConsoleMessage {
  type: string;
  text: string;
  timestamp: number;
}

export async function getWebRTCState(page: Page): Promise<WebRTCState | null> {
  return page.evaluate(async () => {
    const client = (window as any)._webrtcClient;
    if (!client) {
      return null;
    }

    const stats = await client.getStats?.();

    return {
      connectionState: client.pc?.connectionState ?? null,
      iceConnectionState: client.pc?.iceConnectionState ?? null,
      iceGatheringState: client.pc?.iceGatheringState ?? null,
      signalingState: client.pc?.signalingState ?? null,
      packetsReceived: stats?.packetsReceived ?? 0,
      bytesReceived: stats?.bytesReceived ?? 0,
      packetsLost: stats?.packetsLost ?? 0,
      jitter: stats?.jitter ?? 0,
      roundTripTime: stats?.roundTripTime ?? 0,
      audioElementReady: !!client.audioElement?.srcObject,
      remoteStreamActive: client.remoteStream?.active ?? false,
      trackCount: client.remoteStream?.getTracks()?.length ?? 0,
    };
  });
}

export async function waitForConnectionState(
  page: Page,
  targetState: RTCPeerConnectionState,
  timeoutMs: number = 30000
): Promise<void> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    const state = await getWebRTCState(page);
    if (state?.connectionState === targetState) {
      return;
    }
    await page.waitForTimeout(500);
  }

  const finalState = await getWebRTCState(page);
  throw new Error(
    `Timeout waiting for connectionState "${targetState}". ` +
    `Current state: ${finalState?.connectionState ?? 'no client'}`
  );
}

export async function waitForICEState(
  page: Page,
  targetStates: RTCIceConnectionState[],
  timeoutMs: number = 30000
): Promise<RTCIceConnectionState> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    const state = await getWebRTCState(page);
    if (state?.iceConnectionState && targetStates.includes(state.iceConnectionState)) {
      return state.iceConnectionState;
    }
    await page.waitForTimeout(500);
  }

  const finalState = await getWebRTCState(page);
  throw new Error(
    `Timeout waiting for ICE state in [${targetStates.join(', ')}]. ` +
    `Current state: ${finalState?.iceConnectionState ?? 'no client'}`
  );
}

export async function waitForPackets(
  page: Page,
  minPackets: number = 1,
  timeoutMs: number = 10000
): Promise<number> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    const state = await getWebRTCState(page);
    if (state && state.packetsReceived >= minPackets) {
      return state.packetsReceived;
    }
    await page.waitForTimeout(500);
  }

  const finalState = await getWebRTCState(page);
  throw new Error(
    `Timeout waiting for ${minPackets} packets. ` +
    `Received: ${finalState?.packetsReceived ?? 0}`
  );
}

export async function getDetailedDiagnostics(page: Page): Promise<object> {
  return page.evaluate(async () => {
    const client = (window as any)._webrtcClient;
    if (!client) {
      return { error: 'No WebRTC client found' };
    }

    const debugStatus = await client.debugStatus?.();
    const rawStats: any[] = [];

    if (client.pc) {
      const stats = await client.pc.getStats();
      stats.forEach((report: any) => {
        rawStats.push({
          type: report.type,
          id: report.id,
          ...Object.fromEntries(
            Object.entries(report).filter(([key]) => key !== 'type' && key !== 'id')
          ),
        });
      });
    }

    return {
      debugStatus,
      rawStats,
      config: {
        iceServers: client.options?.iceServers,
        iceTransportPolicy: client.options?.iceTransportPolicy,
        wsUrl: client.wsUrl,
      },
    };
  });
}

export async function captureConsoleLogs(page: Page): Promise<ConsoleMessage[]> {
  const logs: ConsoleMessage[] = [];

  page.on('console', (msg) => {
    logs.push({
      type: msg.type(),
      text: msg.text(),
      timestamp: Date.now(),
    });
  });

  return logs;
}

export function setupConsoleCapture(page: Page): ConsoleMessage[] {
  const logs: ConsoleMessage[] = [];

  page.on('console', (msg) => {
    logs.push({
      type: msg.type(),
      text: msg.text(),
      timestamp: Date.now(),
    });
  });

  return logs;
}

export async function clickPlayButton(page: Page): Promise<void> {
  const playButton = page.locator('#play-button, [data-action="play"], button:has-text("Play")');
  await playButton.waitFor({ state: 'visible', timeout: 10000 });
  await playButton.click();
}

export async function selectSession(page: Page, sessionId?: string): Promise<void> {
  if (sessionId) {
    const sessionCard = page.locator(`[data-session-id="${sessionId}"]`);
    await sessionCard.click();
  } else {
    const firstSession = page.locator('.session-card, [data-session-id]').first();
    await firstSession.waitFor({ state: 'visible', timeout: 10000 });
    await firstSession.click();
  }
}

export function formatStateForLog(state: WebRTCState | null): string {
  if (!state) {
    return 'No WebRTC client available';
  }

  return [
    `Connection: ${state.connectionState}`,
    `ICE: ${state.iceConnectionState}`,
    `ICE Gathering: ${state.iceGatheringState}`,
    `Signaling: ${state.signalingState}`,
    `Packets: ${state.packetsReceived}`,
    `Bytes: ${state.bytesReceived}`,
    `Lost: ${state.packetsLost}`,
    `Jitter: ${state.jitter?.toFixed(4) ?? 'N/A'}`,
    `RTT: ${state.roundTripTime?.toFixed(4) ?? 'N/A'}`,
    `Audio Ready: ${state.audioElementReady}`,
    `Stream Active: ${state.remoteStreamActive}`,
    `Tracks: ${state.trackCount}`,
  ].join(' | ');
}
