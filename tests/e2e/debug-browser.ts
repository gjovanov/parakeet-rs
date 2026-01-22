import { chromium } from 'playwright';

async function main() {
  console.log('Launching browser...');

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--autoplay-policy=no-user-gesture-required',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
    ],
  });

  console.log('Browser launched!');

  const context = await browser.newContext({
    permissions: ['microphone'],
  });

  console.log('Context created!');

  const page = await context.newPage();
  console.log('Page created!');

  page.on('console', msg => {
    console.log(`[Browser Console] ${msg.type()}: ${msg.text()}`);
  });

  console.log('Navigating to http://172.31.244.165:8080...');
  await page.goto('http://172.31.244.165:8080', { timeout: 30000 });
  console.log('Page loaded!');

  const title = await page.title();
  console.log(`Page title: ${title}`);

  // Check for session cards
  const sessionCards = await page.locator('.session-card').count();
  console.log(`Found ${sessionCards} session cards`);

  if (sessionCards > 0) {
    console.log('Clicking first session...');
    await page.locator('.session-card').first().click();
    await page.waitForTimeout(1000);

    const connectBtn = page.locator('#connect-btn');
    const isEnabled = await connectBtn.isEnabled();
    console.log(`Connect button enabled: ${isEnabled}`);

    if (isEnabled) {
      console.log('Clicking connect button...');
      await connectBtn.click();

      console.log('Waiting for WebRTC client...');
      await page.waitForFunction(
        () => (window as any)._webrtcClient !== null,
        { timeout: 15000 }
      ).catch(e => console.log('WebRTC client not found:', e.message));

      // Wait and check WebRTC state
      for (let i = 0; i < 6; i++) {
        await page.waitForTimeout(5000);

        const state = await page.evaluate(async () => {
          const client = (window as any)._webrtcClient;
          if (!client) return null;

          const stats = await client.getStats?.();
          return {
            connectionState: client.pc?.connectionState,
            iceConnectionState: client.pc?.iceConnectionState,
            packetsReceived: stats?.packetsReceived || 0,
          };
        });

        console.log(`[${(i+1)*5}s] State:`, JSON.stringify(state));

        if (state?.connectionState === 'connected' && state?.packetsReceived > 0) {
          console.log('SUCCESS: WebRTC connected and receiving packets!');
          break;
        }

        if (state?.connectionState === 'failed' || state?.iceConnectionState === 'failed') {
          console.log('FAILED: WebRTC connection failed');
          break;
        }
      }
    }
  }

  await browser.close();
  console.log('Browser closed');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
