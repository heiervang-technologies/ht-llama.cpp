import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],

	kit: {
		paths: {
			relative: true
		},
		router: { type: 'hash' },
		adapter: adapter({
			pages: '../public',
			assets: '../public',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		// NOTE: `bundleStrategy: 'single'` pins the whole app into one
		// `bundle.js`. Switching to `'split'` would let Rollup emit
		// per-route chunks + pull dynamic-import() targets into their
		// own files — the pattern we want for pdfjs / xterm / codemirror
		// / hljs. But the current post-build scripts (scripts/post-
		// build.sh + scripts/vite-plugin-llama-cpp-build.ts) are both
		// hard-wired to the single-bundle layout: they wipe
		// `public/_app/` and rewrite every URL to `./bundle.js`.
		// Making split work means teaching them to preserve the chunk
		// tree, keep relative paths, AND verifying Tauri's file:// /
		// android asset loader can still resolve them. Bigger diff than
		// one sitting. Leaving single as a known follow-up.
		output: {
			bundleStrategy: 'single'
		},
		alias: {
			$styles: 'src/styles'
		},
		version: {
			name: 'llama-server-webui'
		}
	},

	extensions: ['.svelte', '.svx']
};

export default config;
