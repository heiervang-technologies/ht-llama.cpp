import type { Root as HastRoot, Element as HastElement } from 'hast';
import { visit } from 'unist-util-visit';

const WRAPPER_CLASS = 'ht-image-artifact';

/**
 * Wraps each `<img>` in a figure-like artifact container with a caption derived
 * from alt text or the src filename, so AI-rendered images visually separate
 * from surrounding prose. The toggle button is wired up after render by the
 * companion `setupImageArtifactToggles` helper.
 */
export function rehypeWrapImagesAsArtifacts() {
	return (tree: HastRoot) => {
		visit(tree, 'element', (node, index, parent) => {
			if (node.tagName !== 'img' || !parent || index === undefined) return;

			// Skip images already wrapped (e.g. nested <a><img></a> in a wrapper)
			if (
				parent.type === 'element' &&
				Array.isArray(parent.properties?.className) &&
				parent.properties.className.includes(WRAPPER_CLASS)
			) {
				return;
			}

			const alt = typeof node.properties?.alt === 'string' ? node.properties.alt : '';
			const src = typeof node.properties?.src === 'string' ? node.properties.src : '';
			const caption = alt.trim() || deriveCaptionFromSrc(src) || 'Image';

			const wrapper: HastElement = {
				type: 'element',
				tagName: 'figure',
				properties: { className: [WRAPPER_CLASS], 'data-collapsed': 'false' },
				children: [
					{
						type: 'element',
						tagName: 'button',
						properties: {
							type: 'button',
							className: [`${WRAPPER_CLASS}-toggle`],
							'aria-label': 'Toggle image'
						},
						children: [
							{
								type: 'element',
								tagName: 'span',
								properties: { className: [`${WRAPPER_CLASS}-chevron`] },
								children: []
							},
							{
								type: 'element',
								tagName: 'span',
								properties: { className: [`${WRAPPER_CLASS}-caption`] },
								children: [{ type: 'text', value: caption }]
							}
						]
					},
					{
						type: 'element',
						tagName: 'div',
						properties: { className: [`${WRAPPER_CLASS}-body`] },
						children: [node]
					}
				]
			};

			parent.children.splice(index, 1, wrapper);
		});
	};
}

function deriveCaptionFromSrc(src: string): string {
	if (!src) return '';
	if (src.startsWith('data:')) return 'Inline image';
	try {
		const url = new URL(src, 'http://_');
		const last = url.pathname.split('/').filter(Boolean).pop();
		return last ? decodeURIComponent(last) : '';
	} catch {
		return '';
	}
}

/**
 * Wire up collapse toggles after markdown render. Safe to call repeatedly —
 * uses a dataset flag to avoid double-binding.
 */
export function setupImageArtifactToggles(container: HTMLElement | null | undefined) {
	if (!container) return;
	const toggles = container.querySelectorAll<HTMLButtonElement>(`.${WRAPPER_CLASS}-toggle`);
	for (const toggle of toggles) {
		if (toggle.dataset.htBound === 'true') continue;
		toggle.dataset.htBound = 'true';
		toggle.addEventListener('click', () => {
			const figure = toggle.closest<HTMLElement>(`.${WRAPPER_CLASS}`);
			if (!figure) return;
			const collapsed = figure.dataset.collapsed === 'true';
			figure.dataset.collapsed = collapsed ? 'false' : 'true';
		});
	}
}
