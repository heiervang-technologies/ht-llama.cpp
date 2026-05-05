import { expect, test } from '@playwright/test';

const CONFIG_KEY = 'LlamaCppWebui.config';

/**
 * Bootstraps localStorage so the app starts with inline completions enabled
 * and a short delay, then installs a route mock that replies with a fixed
 * string on POST /completion. Reload the page after calling so both the
 * settings store and the route handler are in place before the doc mounts.
 */
async function primeClient(
	page: import('@playwright/test').Page,
	completionText: string,
	extraConfig: Record<string, unknown> = {}
) {
	await page.addInitScript(
		([key, cfg]) => {
			localStorage.setItem(key as string, JSON.stringify(cfg));
		},
		[
			CONFIG_KEY,
			{
				inlineCompletionEnabled: true,
				inlineCompletionDelay: 200,
				inlineCompletionMaxTokens: 16,
				alwaysShowSidebarOnDesktop: true,
				backendBaseUrl: '',
				apiKey: '',
				...extraConfig
			}
		]
	);

	await page.route('**/completion', async (route) => {
		if (route.request().method() !== 'POST') {
			return route.continue();
		}
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ content: completionText })
		});
	});
}

async function openNewDoc(page: import('@playwright/test').Page) {
	await page.goto('/');

	// The sidebar is open by default on desktop viewports. The sidebar exposes a
	// "New doc" button that calls docsStore.createDoc and navigates to the new id.
	const newDocButton = page.getByRole('button', { name: 'New doc' });
	await newDocButton.waitFor({ state: 'visible', timeout: 10000 });
	await newDocButton.click();

	await page.waitForURL(/#\/doc\//, { timeout: 10000 });
	await page.locator('.cm-content').first().waitFor({ state: 'visible', timeout: 15000 });
}

test.describe('inline AI completions', () => {
	test('ghost text appears and Tab accepts it', async ({ page }) => {
		await primeClient(page, ' continues the sentence.');
		await openNewDoc(page);

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('The story');

		// Wait for the mocked completion request to round-trip.
		await page.waitForRequest((req) => /\/completion$/.test(req.url()) && req.method() === 'POST');

		// Ghost widget should be rendered as a span with the cm-ghost-text class.
		const ghost = page.locator('.cm-ghost-text');
		await expect(ghost).toHaveText(' continues the sentence.', { timeout: 5000 });

		// Accept and verify the text is now part of the document.
		await page.keyboard.press('Tab');
		await expect(editor).toContainText('The story continues the sentence.');
		await expect(ghost).toHaveCount(0);
	});

	test('Escape dismisses ghost text without inserting', async ({ page }) => {
		await primeClient(page, ' was never written.');
		await openNewDoc(page);

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('The book');

		await page.waitForRequest((req) => /\/completion$/.test(req.url()) && req.method() === 'POST');
		await expect(page.locator('.cm-ghost-text')).toBeVisible();

		await page.keyboard.press('Escape');
		await expect(page.locator('.cm-ghost-text')).toHaveCount(0);
		await expect(editor).toContainText('The book');
		await expect(editor).not.toContainText('was never written');
	});

	test('disabled setting does not fire requests', async ({ page }) => {
		await primeClient(page, ' never called.', { inlineCompletionEnabled: false });
		await openNewDoc(page);

		let firedRequest = false;
		page.on('request', (req) => {
			if (/\/completion$/.test(req.url()) && req.method() === 'POST') {
				firedRequest = true;
			}
		});

		const editor = page.locator('.cm-content').first();
		await editor.click();
		await page.keyboard.type('Nothing should happen here.');
		await page.waitForTimeout(1500);

		expect(firedRequest).toBe(false);
		await expect(page.locator('.cm-ghost-text')).toHaveCount(0);
	});
});
