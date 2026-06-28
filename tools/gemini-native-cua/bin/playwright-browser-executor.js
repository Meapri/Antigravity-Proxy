#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => data += chunk);
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function normalizeUrl(value) {
  const text = String(value || '').trim();
  if (!text) return 'about:blank';
  if (/^[a-z][a-z0-9+.-]*:/i.test(text)) return text;
  if (/^(?:[a-z0-9-]+\.)+[a-z]{2,}(?:\/.*)?$/i.test(text)) return `https://${text}`;
  return `https://www.google.com/search?q=${encodeURIComponent(text)}`;
}

function keyName(key) {
  const raw = String(key || '').trim();
  const lower = raw.toLowerCase();
  const map = {
    return: 'Enter', enter: 'Enter', escape: 'Escape', esc: 'Escape', tab: 'Tab',
    backspace: 'Backspace', delete: 'Delete', space: ' ', arrowup: 'ArrowUp',
    arrowdown: 'ArrowDown', arrowleft: 'ArrowLeft', arrowright: 'ArrowRight',
  };
  return map[lower] || raw;
}

function playwrightShortcut(key) {
  return String(key || '')
    .split('+')
    .map(part => part.trim())
    .filter(Boolean)
    .map(part => {
      const lower = part.toLowerCase();
      if (lower === 'ctrl' || lower === 'control') return 'Control';
      if (lower === 'cmd' || lower === 'meta') return 'Meta';
      if (lower === 'alt' || lower === 'option') return 'Alt';
      if (lower === 'shift') return 'Shift';
      return keyName(part);
    })
    .join('+');
}

async function main() {
  const input = JSON.parse(await readStdin() || '{}');
  const stateDir = input.stateDir || path.join(process.env.HOME || '/tmp', '.local/share/gemini-native-cua/browser');
  const userDataDir = input.userDataDir || path.join(stateDir, 'profile');
  const selectorPath = path.join(stateDir, 'selectors.json');
  const sessionPath = path.join(stateDir, 'session.json');
  ensureDir(stateDir);
  ensureDir(userDataDir);

  const headless = input.headless === false || process.env.GEMINI_NATIVE_CUA_HEADLESS === '0' ? false : true;
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless,
    viewport: { width: Number(input.width || 1280), height: Number(input.height || 900) },
    args: ['--disable-dev-shm-usage', '--no-first-run', '--no-default-browser-check'],
  });
  try {
    const action = String(input.action || 'capture');
    const args = input.args || {};
    let savedSession = {};
    try { savedSession = JSON.parse(fs.readFileSync(sessionPath, 'utf8')); } catch {}
    let pages = context.pages();
    let page = pages.find(p => p.url() && p.url() !== 'about:blank' && (!savedSession.url || p.url() === savedSession.url))
      || pages.find(p => p.url() && p.url() !== 'about:blank')
      || pages[0]
      || await context.newPage();
    if (!['open', 'navigate'].includes(action) && page.url() === 'about:blank' && savedSession.url && savedSession.url !== 'about:blank') {
      await page.goto(savedSession.url, { waitUntil: 'domcontentloaded', timeout: Number(input.navigationTimeoutMs || 30000) }).catch(() => {});
    }
    page.setDefaultTimeout(Number(input.timeoutMs || 12000));

    if (action === 'open') {
      if (page.url() === 'about:blank') await page.goto('about:blank');
    } else if (action === 'navigate') {
      await page.goto(normalizeUrl(args.url), { waitUntil: 'domcontentloaded', timeout: Number(input.navigationTimeoutMs || 30000) });
      await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {});
    } else if (action === 'click') {
      if (args.element_index !== undefined && args.element_index !== null) {
        let selectors = {};
        try { selectors = JSON.parse(fs.readFileSync(selectorPath, 'utf8')); } catch {}
        const selector = selectors[String(args.element_index)];
        if (!selector) throw new Error(`no selector cached for element_index=${args.element_index}`);
        await page.locator(selector).first().click({ timeout: 12000 });
      } else if (args.x !== undefined && args.y !== undefined) {
        await page.mouse.click(Number(args.x), Number(args.y), { clickCount: action === 'double_click' ? 2 : 1 });
      } else {
        throw new Error('click requires element_index or x/y');
      }
    } else if (action === 'double_click') {
      if (args.element_index !== undefined && args.element_index !== null) {
        let selectors = {};
        try { selectors = JSON.parse(fs.readFileSync(selectorPath, 'utf8')); } catch {}
        const selector = selectors[String(args.element_index)];
        if (!selector) throw new Error(`no selector cached for element_index=${args.element_index}`);
        await page.locator(selector).first().dblclick({ timeout: 12000 });
      } else if (args.x !== undefined && args.y !== undefined) await page.mouse.dblclick(Number(args.x), Number(args.y));
      else throw new Error('double_click requires element_index or x/y');
    } else if (action === 'type') {
      await page.keyboard.type(String(args.text || ''), { delay: 1 });
    } else if (action === 'press') {
      const key = playwrightShortcut(args.key || args.keys || '');
      if (!key) throw new Error('press requires key');
      await page.keyboard.press(key);
    } else if (action === 'scroll') {
      const direction = String(args.direction || 'down').toLowerCase();
      const amount = Math.max(1, Math.min(Number(args.amount || 3), 50));
      const delta = amount * 350 * (direction === 'up' ? -1 : 1);
      await page.mouse.wheel(0, delta);
    } else if (action === 'wait') {
      await page.waitForTimeout(Math.max(0, Number(args.seconds || args.duration || 1) * 1000));
    } else if (action !== 'capture') {
      throw new Error(`unsupported browser action: ${action}`);
    }

    await page.waitForTimeout(Number(input.settleMs || 300));
    const title = await page.title().catch(() => '');
    const url = page.url();
    const elements = await page.evaluate(() => {
      function cssPath(el) {
        if (el.id) return `#${CSS.escape(el.id)}`;
        const parts = [];
        let node = el;
        while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.documentElement) {
          const tag = node.tagName.toLowerCase();
          const sameTagSiblings = Array.from(node.parentElement ? node.parentElement.children : [])
            .filter(child => child.tagName === node.tagName);
          const nth = sameTagSiblings.indexOf(node) + 1;
          parts.unshift(`${tag}:nth-of-type(${nth})`);
          node = node.parentElement;
        }
        return parts.length ? parts.join(' > ') : el.tagName.toLowerCase();
      }
      const candidates = Array.from(document.querySelectorAll('a,button,input,textarea,select,[role="button"],[tabindex]:not([tabindex="-1"])'));
      const out = [];
      let idx = 1;
      for (const el of candidates.slice(0, 180)) {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        if (rect.width < 2 || rect.height < 2 || style.visibility === 'hidden' || style.display === 'none') continue;
        const tag = el.tagName.toLowerCase();
        const label = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('value') || el.href || '').replace(/\s+/g, ' ').trim();
        const selector = cssPath(el);
        out.push({
          element_index: idx,
          role: el.getAttribute('role') || tag,
          label: label.slice(0, 180),
          frame: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
          selector,
          href: el.href || undefined,
        });
        idx += 1;
      }
      return out;
    }).catch(() => []);
    const selectors = {};
    for (const el of elements) selectors[String(el.element_index)] = el.selector;
    fs.writeFileSync(selectorPath, JSON.stringify(selectors, null, 2));
    const text = await page.locator('body').innerText({ timeout: 3000 }).catch(() => '');
    const screenshotPath = path.join(stateDir, `screenshot-${Date.now()}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: false }).catch(() => {});
    const viewport = page.viewportSize() || { width: 1280, height: 900 };
    fs.writeFileSync(sessionPath, JSON.stringify({ url, title, updatedAt: new Date().toISOString() }, null, 2));
    const result = {
      ok: true,
      action,
      url,
      title,
      viewport,
      elements: elements.map(({ selector, ...rest }) => rest),
      text: String(text || '').slice(0, 6000),
      screenshot_path: fs.existsSync(screenshotPath) ? screenshotPath : undefined,
      screenshot_mime: 'image/png',
    };
    console.log(JSON.stringify(result));
  } finally {
    await context.close();
  }
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
