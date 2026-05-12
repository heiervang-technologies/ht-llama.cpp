import { expect, test, type Page } from '@playwright/test';

const CONFIG_KEY = 'LlamaCppWebui.config';

/**
 * Bootstraps localStorage with a clean config (empty aiCommands so built-in
 * defaults are used) and mocks POST /v1/chat/completions with a fixed set of
 * streamed content chunks formatted as an SSE response. Playwright fulfills
 * the route with a complete body — the client still parses each `data:` line
 * and yields each chunk as a delta.
 */
async function primeClient(
	page: Page,
	chunks: string[],
	extraConfig: Record<string, unknown> = {}
) {
	await page.addInitScript(
		([key, cfg]) => {
			localStorage.setItem(key as string, JSON.stringify(cfg));
		},
		[
			CONFIG_KEY,
			{
				alwaysShowSidebarOnDesktop: true,
				backendBaseUrl: '',
				apiKey: '',
				aiCommands: '',
				inlineCompletionEnabled: false,
				...extraConfig
			}
		]
	);

	await page.route('**/v1/chat/completions', async (route) => {
		if (route.request().method() !== 'POST') {
			return route.continue();
		}
		const sse =
			chunks
				.map((c) => `data: ${JSON.stringify({ choices: [{ delta: { content: c } }] })}\n\n`)
				.join('') + 'data: [DONE]\n\n';
		await route.fulfill({
			status: 200,
			contentType: 'text/event-stream',
			body: sse
		});
	});
}

async function openNewDoc(page: Page) {
	await page.goto('/');
	const newDocButton = page.getByRole('button', { name: 'New doc' });
	await newDocButton.waitFor({ state: 'visible', timeout: 10000 });
	await newDocButton.click();
	await page.waitForURL(/#\/doc\//, { timeout: 10000 });
	await page.locator('.cm-content').first().waitFor({ state: 'visible', timeout: 15000 });
}

async function openCommandsMenu(page: Page) {
	// The trigger button carries the visible text "Commands" on desktop
	// viewports and a Wand2 icon. title attribute adds a stable tooltip.
	const trigger = page.getByRole('button', { name: /Commands/i }).first();
	await trigger.waitFor({ state: 'visible', timeout: 5000 });
	await trigger.click();
}

test.describe('AI commands menu', () => {
	test('append mode: Summarize streams output after a separator', async ({ page }) => {
		await primeClient(page, ['First bullet. ', 'Second bullet.']);
		await openNewDoc(page);

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('Existing document content.');
		// Let the autosave debounce fire.
		await page.waitForTimeout(700);

		const chatReq = page.waitForRequest(
			(req) => /\/v1\/chat\/completions$/.test(req.url()) && req.method() === 'POST',
			{ timeout: 10000 }
		);

		await openCommandsMenu(page);
		await page.getByRole('menuitem', { name: /^Summarize/ }).click();

		await chatReq;

		await expect(editor).toContainText('Existing document content.', { timeout: 5000 });
		await expect(editor).toContainText('First bullet.', { timeout: 5000 });
		await expect(editor).toContainText('Second bullet.', { timeout: 5000 });
		// Separator between original document and appended output.
		await expect(editor).toContainText('---', { timeout: 5000 });
	});

	test('replace mode: Fix grammar replaces selected text with streamed output', async ({
		page
	}) => {
		await primeClient(page, ['corrected']);
		await openNewDoc(page);

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('This is the bad grammer.');
		await page.waitForTimeout(700);

		// Select the word "grammer" via keyboard: cursor sits after the period,
		// so ArrowLeft moves before it, and Ctrl+Shift+ArrowLeft selects the
		// previous word deterministically (dblclick on CM6 nodes was unreliable).
		await page.keyboard.press('ArrowLeft');
		await page.keyboard.press('Control+Shift+ArrowLeft');

		const chatReq = page.waitForRequest(
			(req) => /\/v1\/chat\/completions$/.test(req.url()) && req.method() === 'POST',
			{ timeout: 10000 }
		);

		await openCommandsMenu(page);
		await page.getByRole('menuitem', { name: /Fix grammar/i }).click();

		await chatReq;

		await expect(editor).toContainText('This is the bad corrected.', { timeout: 5000 });
		await expect(editor).not.toContainText('grammer');
	});

	test('replace mode without selection shows a warning toast', async ({ page }) => {
		await primeClient(page, ['should not be called']);
		await openNewDoc(page);

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('Hello world.');
		await page.waitForTimeout(500);

		let chatCalled = false;
		page.on('request', (req) => {
			if (/\/v1\/chat\/completions$/.test(req.url()) && req.method() === 'POST') {
				chatCalled = true;
			}
		});

		await openCommandsMenu(page);
		await page.getByRole('menuitem', { name: /Rewrite selection/i }).click();

		// Toast library is svelte-sonner; its items render with role="status".
		await expect(page.getByText(/requires a text selection/i)).toBeVisible({ timeout: 3000 });
		expect(chatCalled).toBe(false);
	});

	test('Ctrl+Shift+K opens the commands menu', async ({ page }) => {
		await primeClient(page, ['unused']);
		await openNewDoc(page);

		// Menu should not be visible yet.
		await expect(page.getByRole('menuitem', { name: /^Summarize/ })).toHaveCount(0);

		// Pressing the shortcut anywhere in the doc screen opens the dropdown.
		// Focus the editor first so the keydown fires at the page level, not
		// swallowed by an input with native shortcut handling.
		await page.locator('.cm-content').first().click();
		await page.keyboard.press('Control+Shift+K');

		await expect(page.getByRole('menuitem', { name: /^Summarize/ })).toBeVisible({
			timeout: 2000
		});
	});

	test('Stop button cancels a running command mid-stream', async ({ page }) => {
		// Install a custom route handler that streams slowly so we have time
		// to click Stop before completion. We cannot fulfil + delay in one
		// call — instead write the body chunks with explicit timing.
		await page.addInitScript(
			([key, cfg]) => {
				localStorage.setItem(key as string, JSON.stringify(cfg));
			},
			[
				CONFIG_KEY,
				{
					alwaysShowSidebarOnDesktop: true,
					backendBaseUrl: '',
					apiKey: '',
					aiCommands: '',
					inlineCompletionEnabled: false
				}
			]
		);
		await page.route('**/v1/chat/completions', async (route) => {
			if (route.request().method() !== 'POST') {
				return route.continue();
			}
			// Never-ending SSE stream: the client will keep waiting until abort.
			// Playwright's route.fulfill cannot stream, so emulate by responding
			// with a body that's valid SSE but intentionally lacks [DONE] and
			// add a small delay so the test has a window to hit Stop.
			await new Promise((r) => setTimeout(r, 3000));
			await route.fulfill({
				status: 200,
				contentType: 'text/event-stream',
				body:
					`data: ${JSON.stringify({ choices: [{ delta: { content: 'partial' } }] })}\n\n` +
					'data: [DONE]\n\n'
			});
		});

		await openNewDoc(page);
		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('Seed content.');
		await page.waitForTimeout(500);

		await openCommandsMenu(page);
		await page.getByRole('menuitem', { name: /^Summarize/ }).click();

		// While the route handler is still waiting, the trigger button should
		// have flipped into the "Stop" state. Click it.
		const stopBtn = page.getByRole('button', { name: /Stop/i }).first();
		await stopBtn.waitFor({ state: 'visible', timeout: 2000 });
		await stopBtn.click();

		// The button should flip back to "Commands" once the abort is applied.
		await expect(page.getByRole('button', { name: /Commands/i }).first()).toBeVisible({
			timeout: 3000
		});
	});
});
