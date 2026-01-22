import { firefox } from 'playwright';

async function main() {
  console.log('Launching Firefox...');

  const browser = await firefox.launch({
    headless: true,
    firefoxUserPrefs: {
      // Disable mDNS for ICE candidates
      'media.peerconnection.ice.obfuscate_host_addresses': false,
      // Allow autoplay
      'media.autoplay.default': 0,
      'media.autoplay.enabled.user-gestures-needed': false,
    },
  });

  const context = await browser.newContext({
    permissions: ['microphone'],
  });

  const page = await context.newPage();

  // Capture and log all console messages
  page.on('console', msg => {
    console.log(`[Firefox] ${msg.type()}: ${msg.text()}`);
  });

  console.log('Navigating to http://172.31.244.165:8080...');
  await page.goto('http://172.31.244.165:8080', { timeout: 30000 });

  const sessionCards = await page.locator('.session-card').count();
  if (sessionCards === 0) {
    console.log('No sessions available');
    await browser.close();
    return;
  }

  await page.locator('.session-card').first().click();
  await page.waitForTimeout(1000);

  const connectBtn = page.locator('#connect-btn');
  if (await connectBtn.isEnabled()) {
    await connectBtn.click();

    // Wait for connection
    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    // Wait longer to see connection result
    await page.waitForTimeout(5000);

    const status = await page.evaluate(() => {
      const client = (window as any)._webrtcClient;
      if (!client?.pc) return null;

      return {
        iceConnectionState: client.pc.iceConnectionState,
        connectionState: client.pc.connectionState,
        signalingState: client.pc.signalingState,
        iceGatheringState: client.pc.iceGatheringState,
      };
    });

    console.log('\n=== Firefox WebRTC Status ===');
    console.log(JSON.stringify(status, null, 2));
  }

  await browser.close();
  console.log('Firefox closed');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
