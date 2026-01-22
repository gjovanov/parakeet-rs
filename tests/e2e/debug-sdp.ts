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

  const context = await browser.newContext({
    permissions: ['microphone'],
  });

  const page = await context.newPage();

  // Capture and log all console messages
  page.on('console', msg => {
    console.log(`[Browser] ${msg.type()}: ${msg.text()}`);
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

    // Wait for connection and get SDP details
    await page.waitForFunction(
      () => (window as any)._webrtcClient !== null,
      { timeout: 15000 }
    );

    await page.waitForTimeout(2000);

    const sdpDetails = await page.evaluate(() => {
      const client = (window as any)._webrtcClient;
      if (!client?.pc) return null;

      const localDesc = client.pc.localDescription;
      const remoteDesc = client.pc.remoteDescription;

      // Extract ice-ufrag from SDPs
      const extractUfrag = (sdp: string) => {
        const match = sdp.match(/a=ice-ufrag:(\S+)/);
        return match ? match[1] : null;
      };

      // Get media sections
      const extractMediaSections = (sdp: string) => {
        const sections: any[] = [];
        const lines = sdp.split('\r\n');
        let currentSection: any = null;

        for (const line of lines) {
          if (line.startsWith('m=')) {
            if (currentSection) sections.push(currentSection);
            currentSection = { type: line, mid: null, ufrag: null };
          } else if (currentSection) {
            if (line.startsWith('a=mid:')) {
              currentSection.mid = line.substring(6);
            } else if (line.startsWith('a=ice-ufrag:')) {
              currentSection.ufrag = line.substring(12);
            }
          }
        }
        if (currentSection) sections.push(currentSection);
        return sections;
      };

      return {
        localDescription: {
          type: localDesc?.type,
          ufrag: localDesc?.sdp ? extractUfrag(localDesc.sdp) : null,
          mediaSections: localDesc?.sdp ? extractMediaSections(localDesc.sdp) : [],
          fullSdp: localDesc?.sdp?.substring(0, 1500),
        },
        remoteDescription: {
          type: remoteDesc?.type,
          ufrag: remoteDesc?.sdp ? extractUfrag(remoteDesc.sdp) : null,
          mediaSections: remoteDesc?.sdp ? extractMediaSections(remoteDesc.sdp) : [],
          fullSdp: remoteDesc?.sdp?.substring(0, 1500),
        },
      };
    });

    console.log('\n=== SDP Details ===');
    console.log(JSON.stringify(sdpDetails, null, 2));
  }

  await browser.close();
  console.log('Browser closed');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
