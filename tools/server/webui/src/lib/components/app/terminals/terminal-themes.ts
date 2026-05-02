/**
 * Terminal themes + overlay effects.
 *
 * Each theme carries an xterm palette (consumed by `new Terminal({
 * theme })`) plus an optional `overlay` class list that the
 * `TerminalView` component applies to the wrapping div. Overlays
 * give us CRT scanlines / vignette / phosphor bloom without having
 * to touch the canvas renderer — they're pure CSS layered on top.
 *
 * Keep the palettes small and hand-picked; exotic colour matches
 * should ship as new entries, not parameters.
 */

import type { ITheme } from '@xterm/xterm';

export interface TerminalTheme {
	id: string;
	label: string;
	description: string;
	/** xterm palette — the 16 ANSI slots + background/foreground. */
	palette: ITheme;
	/** Optional CSS class applied to the terminal wrapper for
	 *  scanline / vignette / curvature effects. Classes are defined
	 *  in the same component's `<style>` block. */
	overlay?: 'crt' | 'amber-crt' | 'matrix' | 'paper' | 'blueprint';
	/** Override fontFamily per theme (e.g. a pixel font for CRT). */
	fontFamily?: string;
}

const JETBRAINS =
	'"JetBrainsMono Nerd Font Mono", "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, "SF Mono", "Fira Code", "Cascadia Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace';

// VT323 and Press Start 2P ship on most Omarchy systems via Google
// Fonts preinstall; if missing the stack falls back to the Nerd Font.
const PIXEL = `"VT323", "Press Start 2P", ${JETBRAINS}`;

export const TERMINAL_THEMES: TerminalTheme[] = [
	{
		id: 'default',
		label: 'Default',
		description: 'Clean dark theme matching the app chrome.',
		palette: {
			background: '#0b0b10',
			foreground: '#e6e6e6',
			cursor: '#e6e6e6',
			cursorAccent: '#0b0b10',
			selectionBackground: '#394267',
			black: '#1a1b26',
			red: '#f7768e',
			green: '#9ece6a',
			yellow: '#e0af68',
			blue: '#7aa2f7',
			magenta: '#bb9af7',
			cyan: '#7dcfff',
			white: '#c0caf5',
			brightBlack: '#414868',
			brightRed: '#ff7a93',
			brightGreen: '#b9f27c',
			brightYellow: '#ff9e64',
			brightBlue: '#7da6ff',
			brightMagenta: '#bb9af7',
			brightCyan: '#0db9d7',
			brightWhite: '#c0caf5'
		}
	},
	{
		id: 'crt',
		label: 'CRT',
		description: 'Phosphor green scanlines + curvature. Looks like a 1980s VT220.',
		overlay: 'crt',
		fontFamily: PIXEL,
		palette: {
			background: '#0a140a',
			foreground: '#33ff66',
			cursor: '#66ff99',
			cursorAccent: '#0a140a',
			selectionBackground: '#1a552a',
			black: '#0a140a',
			red: '#66ff66',
			green: '#33ff66',
			yellow: '#aaff66',
			blue: '#33ff99',
			magenta: '#66ffaa',
			cyan: '#66ffcc',
			white: '#aaffaa',
			brightBlack: '#228833',
			brightRed: '#99ff66',
			brightGreen: '#66ff99',
			brightYellow: '#ccff66',
			brightBlue: '#66ffbb',
			brightMagenta: '#99ffbb',
			brightCyan: '#99ffdd',
			brightWhite: '#ccffcc'
		}
	},
	{
		id: 'amber',
		label: 'Amber CRT',
		description: 'Amber phosphor (DEC VT220 amber option).',
		overlay: 'amber-crt',
		fontFamily: PIXEL,
		palette: {
			background: '#110700',
			foreground: '#ffbb33',
			cursor: '#ffd166',
			cursorAccent: '#110700',
			selectionBackground: '#552a00',
			black: '#110700',
			red: '#ffaa33',
			green: '#ffc066',
			yellow: '#ffd166',
			blue: '#ff9933',
			magenta: '#ffaa66',
			cyan: '#ffcc99',
			white: '#ffe0b3',
			brightBlack: '#663300',
			brightRed: '#ffcc66',
			brightGreen: '#ffd699',
			brightYellow: '#ffdd77',
			brightBlue: '#ffbb55',
			brightMagenta: '#ffccaa',
			brightCyan: '#ffe0cc',
			brightWhite: '#fff0d9'
		}
	},
	{
		id: 'matrix',
		label: 'Matrix',
		description: 'Dense green-on-black, no curvature. The other kind of matrix.',
		overlay: 'matrix',
		palette: {
			background: '#000000',
			foreground: '#00ff41',
			cursor: '#00ff41',
			cursorAccent: '#000000',
			selectionBackground: '#003b1a',
			black: '#000000',
			red: '#00cc33',
			green: '#00ff41',
			yellow: '#00ff66',
			blue: '#00aa33',
			magenta: '#00ff77',
			cyan: '#00ffbb',
			white: '#008822',
			brightBlack: '#003311',
			brightRed: '#66ff77',
			brightGreen: '#88ffaa',
			brightYellow: '#aaffbb',
			brightBlue: '#33ff77',
			brightMagenta: '#77ffaa',
			brightCyan: '#99ffdd',
			brightWhite: '#ccffcc'
		}
	},
	{
		id: 'solarized',
		label: 'Solarized Dark',
		description: 'The classic Ethan Schoonover palette.',
		palette: {
			background: '#002b36',
			foreground: '#839496',
			cursor: '#93a1a1',
			cursorAccent: '#002b36',
			selectionBackground: '#073642',
			black: '#073642',
			red: '#dc322f',
			green: '#859900',
			yellow: '#b58900',
			blue: '#268bd2',
			magenta: '#d33682',
			cyan: '#2aa198',
			white: '#eee8d5',
			brightBlack: '#586e75',
			brightRed: '#cb4b16',
			brightGreen: '#586e75',
			brightYellow: '#657b83',
			brightBlue: '#839496',
			brightMagenta: '#6c71c4',
			brightCyan: '#93a1a1',
			brightWhite: '#fdf6e3'
		}
	},
	{
		id: 'paper',
		label: 'Paper',
		description: 'Light theme for screenshots / docs.',
		overlay: 'paper',
		palette: {
			background: '#f6f4ee',
			foreground: '#1c1c1c',
			cursor: '#1c1c1c',
			cursorAccent: '#f6f4ee',
			selectionBackground: '#c6d4dd',
			black: '#1c1c1c',
			red: '#c3432a',
			green: '#587a3a',
			yellow: '#8a5f00',
			blue: '#2a5aa0',
			magenta: '#8e3f7f',
			cyan: '#277886',
			white: '#3d3d3d',
			brightBlack: '#6e6e6e',
			brightRed: '#d65a40',
			brightGreen: '#6d9247',
			brightYellow: '#a47200',
			brightBlue: '#406eb7',
			brightMagenta: '#a45590',
			brightCyan: '#3894a0',
			brightWhite: '#101010'
		}
	},
	{
		id: 'blueprint',
		label: 'Blueprint',
		description: 'Cyan-on-navy, grid-paper overlay.',
		overlay: 'blueprint',
		palette: {
			background: '#0a2540',
			foreground: '#9ae6ff',
			cursor: '#9ae6ff',
			cursorAccent: '#0a2540',
			selectionBackground: '#1a4060',
			black: '#0a2540',
			red: '#ff8fb3',
			green: '#9aeabf',
			yellow: '#ffe29a',
			blue: '#7ecbff',
			magenta: '#c2a6ff',
			cyan: '#9ae6ff',
			white: '#e0f2ff',
			brightBlack: '#335577',
			brightRed: '#ffb3c9',
			brightGreen: '#bcf2d4',
			brightYellow: '#ffebb0',
			brightBlue: '#a8d8ff',
			brightMagenta: '#d4c0ff',
			brightCyan: '#bff0ff',
			brightWhite: '#ffffff'
		}
	}
];

export const DEFAULT_TERMINAL_THEME_ID = 'default';

export function resolveTheme(id: string | null | undefined): TerminalTheme {
	return (
		TERMINAL_THEMES.find((t) => t.id === id) ??
		TERMINAL_THEMES.find((t) => t.id === DEFAULT_TERMINAL_THEME_ID) ??
		TERMINAL_THEMES[0]
	);
}

export const MONO_FONT_STACK = JETBRAINS;
