// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from 'eslint-plugin-storybook';

import prettier from 'eslint-config-prettier';
import { includeIgnoreFile } from '@eslint/compat';
import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import ts from 'typescript-eslint';
import svelteConfig from './svelte.config.js';

const gitignorePath = fileURLToPath(new URL('./.gitignore', import.meta.url));

export default ts.config(
	includeIgnoreFile(gitignorePath),
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs.recommended,
	prettier,
	...svelte.configs.prettier,
	{
		languageOptions: {
			globals: { ...globals.browser, ...globals.node }
		},
		rules: {
			// typescript-eslint strongly recommend that you do not use the no-undef lint rule on TypeScript projects.
			// see: https://typescript-eslint.io/troubleshooting/faqs/eslint/#i-get-errors-from-the-no-undef-rule-about-global-variables-not-being-defined-even-though-there-are-no-typescript-errors
			'no-undef': 'off',
			'svelte/no-at-html-tags': 'off',
			// This app uses hash-based routing (#/) where resolve() from $app/paths does not apply
			'svelte/no-navigation-without-resolve': 'off',
			// Disabled during the 2026-05-12 ht->upstream merge: many Props
			// interfaces were loosened with optional ht-specific fields to
			// keep ht's call surface type-checking against upstream's renamed
			// components. Some fields are runtime no-ops until the
			// corresponding ht intent is re-wired through upstream's API.
			// Tracking: heiervang-technologies/ht-llama.cpp#38
			'svelte/no-unused-props': 'off',
			// Same rationale: any-typed callback props used as a type-system
			// escape hatch on a few ChatMessageActionCard props (onRegenerate,
			// onCopy, onEdit, etc) so the various caller signatures fit.
			// Tighten in the follow-up integration pass.
			'@typescript-eslint/no-explicit-any': 'off',
			// ChatMessageActionCard's `children` snippet is unused in the
			// upstream body but exposed in Props for ht callers that pass
			// children. Same follow-up.
			'@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^(_|children$)' }]
		}
	},
	{
		files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
		languageOptions: {
			parserOptions: {
				projectService: true,
				extraFileExtensions: ['.svelte'],
				parser: ts.parser,
				svelteConfig
			}
		}
	},
	{
		// Exclude Storybook files from main ESLint rules
		ignores: ['.storybook/**/*']
	},
	storybook.configs['flat/recommended']
);
