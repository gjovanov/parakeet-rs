import { chromium } from 'playwright';

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required', '--no-sandbox']
  });
  const page = await browser.newPage();

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('Connection State') || text.includes('ICE') || text.includes('packets')) {
      console.log(`[Browser] ${text}`);
    }
  });

  await page.goto('http://172.31.244.165:8080', { timeout: 30000 });
  await page.waitForTimeout(1000);
  await page.locator('.session-card').first().click();
  await page.waitForTimeout(500);
  await page.locator('#connect-btn').click();

  // Wait longer for TURN relay connection
  await page.waitForTimeout(10000);

  const status = await page.evaluate(() => {
    const c = (window as any)._webrtcClient;
    if (!c?.pc) return null;
    return {
      ice: c.pc.iceConnectionState,
      conn: c.pc.connectionState,
      signaling: c.pc.signalingState
    };
  });

  console.log('\n=== Final Status ===');
  console.log(JSON.stringify(status, null, 2));

  await browser.close();
}

main().catch(e => console.error(e));
