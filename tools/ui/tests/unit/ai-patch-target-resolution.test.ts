/**
 * Commit 5 — target resolution heuristic unit tests.
 *
 * `resolveTargetFromAssistantContext` decides which buffer a streamed
 * SEARCH/REPLACE block should edit. Four paths, in priority order:
 *
 *   1. Filename line above fence (regex-gated) → doc via Dexie.
 *   2. Explicit currentArtifactId on the turn → artifact target.
 *   3. Autocapture slot with a seed → inline target.
 *   4. None of the above → null (caller emits E_NO_TARGET).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface Doc {
	id: string;
	name: string;
	content: string;
}

const fixtures = {
	docs: new Map<string, Doc>(),
	findDocCalls: [] as string[]
};

function reset() {
	fixtures.docs.clear();
	fixtures.findDocCalls = [];
}

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {
		findDocByName: async (name: string) => {
			fixtures.findDocCalls.push(name);
			for (const doc of fixtures.docs.values()) {
				if (doc.name.toLowerCase() === name.toLowerCase()) return doc;
			}
			return undefined;
		}
	}
}));

import {
	isResolvableFilename,
	resolveTargetFromAssistantContext
} from '$lib/editor/ai-patch/target-resolution';

beforeEach(() => {
	reset();
});

describe('isResolvableFilename', () => {
	it('accepts common doc/code extensions', () => {
		for (const name of ['foo.md', 'README.markdown', 'index.html', 'icon.svg', 'app.ts', 'x.tsx']) {
			expect(isResolvableFilename(name)).toBe(true);
		}
	});

	it('rejects plain identifiers and unsupported kinds', () => {
		for (const name of ['MyClass', 'foo.rs', 'README', 'no extension at all']) {
			expect(isResolvableFilename(name)).toBe(false);
		}
	});

	it('tolerates trailing whitespace', () => {
		expect(isResolvableFilename('notes.md   ')).toBe(true);
	});
});

describe('resolveTargetFromAssistantContext', () => {
	it('resolves to a doc target when the filename matches a known doc', async () => {
		fixtures.docs.set('doc-1', { id: 'doc-1', name: 'notes.md', content: '# hi' });
		const target = await resolveTargetFromAssistantContext(
			{ search: 'hi', replace: 'hello', filename: 'notes.md' },
			{}
		);
		expect(target).toEqual({ kind: 'doc', docId: 'doc-1' });
		expect(fixtures.findDocCalls).toEqual(['notes.md']);
	});

	it('matches doc names case-insensitively', async () => {
		fixtures.docs.set('doc-1', { id: 'doc-1', name: 'Notes.MD', content: '' });
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '', filename: 'notes.md' },
			{}
		);
		expect(target).toEqual({ kind: 'doc', docId: 'doc-1' });
	});

	it('returns null (E_NO_TARGET) when the filename is present but no doc matches — does NOT fall through to artifact', async () => {
		fixtures.docs.set('doc-1', { id: 'doc-1', name: 'other.md', content: '' });
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '', filename: 'missing.md' },
			{ currentArtifactId: 'art-1' }
		);
		expect(target).toBeNull();
	});

	it('resolves to the current artifact when the fence is naked and currentArtifactId is set', async () => {
		const target = await resolveTargetFromAssistantContext(
			{ search: 'foo', replace: 'bar' },
			{ currentArtifactId: 'art-42' }
		);
		expect(target).toEqual({ kind: 'artifact', artifactId: 'art-42' });
		// No Dexie round-trip when no filename is provided.
		expect(fixtures.findDocCalls).toEqual([]);
	});

	it('falls back to an inline target when only the autocapture slot is available', async () => {
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '' },
			{
				conversationId: 'conv-1',
				parentMessageId: 'user-msg-1',
				inlineSeed: {
					kind: 'code',
					title: 'scratch',
					mimeType: 'text/plain',
					baseText: 'seed body'
				}
			}
		);
		expect(target).toEqual({
			kind: 'inline',
			conversationId: 'conv-1',
			parentMessageId: 'user-msg-1',
			artifactIndex: 0,
			seed: {
				kind: 'code',
				title: 'scratch',
				mimeType: 'text/plain',
				baseText: 'seed body'
			}
		});
	});

	it('prefers currentArtifactId over the autocapture slot when both are provided', async () => {
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '' },
			{
				conversationId: 'conv-1',
				parentMessageId: 'user-msg-1',
				currentArtifactId: 'art-explicit',
				inlineSeed: {
					kind: 'code',
					title: 't',
					mimeType: 'text/plain',
					baseText: ''
				}
			}
		);
		expect(target).toEqual({ kind: 'artifact', artifactId: 'art-explicit' });
	});

	it('returns null when no filename, no artifact, no seed — naked fence with no context', async () => {
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '' },
			{ conversationId: 'conv-1', parentMessageId: 'user-msg-1' }
		);
		expect(target).toBeNull();
	});

	it('returns null for a filename that fails the extension regex even if a doc with that name exists', async () => {
		// The resolver guards at the regex level first: the model emitting
		// `class Foo` above the fence is not a filename, even if Dexie
		// happened to have a row named "class Foo".
		fixtures.docs.set('doc-1', { id: 'doc-1', name: 'class Foo', content: '' });
		const target = await resolveTargetFromAssistantContext(
			{ search: '', replace: '', filename: 'class Foo' },
			{}
		);
		expect(target).toBeNull();
		// Critically: we did not even probe Dexie for an unsupported name.
		expect(fixtures.findDocCalls).toEqual([]);
	});
});
