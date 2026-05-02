import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import { defineConfig, searchForWorkspaceRoot } from 'vite';
import devtoolsJson from 'vite-plugin-devtools-json';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { llamaCppBuildPlugin } from './scripts/vite-plugin-llama-cpp-build';
import { visualizer } from 'rollup-plugin-visualizer';

const ANALYZE_BUNDLE = process.env.ANALYZE === '1';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
	resolve: {
		alias: {
			'katex-fonts': resolve('node_modules/katex/dist/fonts')
		}
	},

	build: {
		assetsInlineLimit: 32000,
		chunkSizeWarningLimit: 3072,
		minify: true
	},

	esbuild: {
		lineLimit: 500,
		minifyIdentifiers: false
	},

	css: {
		preprocessorOptions: {
			scss: {
				additionalData: `
					$use-woff2: true;
					$use-woff: false;
					$use-ttf: false;
				`,
				// Silence the Sass deprecation spam from katex upstream
				// — `katex/src/styles/katex.scss` still uses the legacy
				// @import + global `nth()` / `length()` builtins. We're
				// tracking katex, not maintaining it, so silencing the
				// channel keeps the build log readable without masking
				// deprecations in our own stylesheets.
				silenceDeprecations: ['import', 'global-builtin'],
				quietDeps: true
			}
		}
	},

	plugins: [
		tailwindcss(),
		sveltekit(),
		devtoolsJson(),
		llamaCppBuildPlugin(),
		// Bundle analyzer — opt-in via `ANALYZE=1 npm run build`. Writes
		// dist/stats.html with a treemap of every module's contribution
		// to the final bundle. Lets us answer "what is in those 8 MB"
		// without guessing.
		ANALYZE_BUNDLE &&
			visualizer({
				filename: 'dist/stats.html',
				template: 'treemap',
				gzipSize: true,
				brotliSize: true,
				open: false
			})
	].filter(Boolean),

	test: {
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					environment: 'browser',
					browser: {
						enabled: true,
						provider: 'playwright',
						instances: [{ browser: 'chromium' }]
					},
					include: ['tests/client/**/*.svelte.{test,spec}.{js,ts}'],
					setupFiles: ['./vitest-setup-client.ts']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'unit',
					environment: 'node',
					include: ['tests/unit/**/*.{test,spec}.{js,ts}']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'ui',
					environment: 'browser',
					browser: {
						enabled: true,
						provider: 'playwright',
						instances: [{ browser: 'chromium', headless: true }]
					},
					include: ['tests/stories/**/*.stories.{js,ts,svelte}'],
					setupFiles: ['./.storybook/vitest.setup.ts']
				},
				plugins: [
					storybookTest({
						storybookScript: 'pnpm run storybook --no-open'
					})
				]
			}
		]
	},

	server: {
		proxy: {
			'/v1': 'http://localhost:8080',
			'/props': 'http://localhost:8080',
			'/models': 'http://localhost:8080',
			'/cors-proxy': 'http://localhost:8080',
			'/lora-adapters': 'http://localhost:8080'
		},
		headers: {
			'Cross-Origin-Embedder-Policy': 'require-corp',
			'Cross-Origin-Opener-Policy': 'same-origin'
		},
		fs: {
			allow: [searchForWorkspaceRoot(process.cwd()), resolve(__dirname, 'tests')]
		}
	}
});
