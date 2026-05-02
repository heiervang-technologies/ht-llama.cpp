/**
 * AI patch module — streaming SEARCH/REPLACE parser, fuzz-match anchor
 * finder, and lazy-elision detector.
 *
 * See `./README.md` for scope and `../../../../docs/research/diff-edit-tool-design.md`
 * for the design brief.
 */

export * from './types';
export { StreamingPatchParser, type StreamingPatchParserOptions } from './parser';
export { findAnchor, type AnchorResult, type AnchorVia } from './fuzz-match';
export { detectElision, type ElisionHit } from './elision';
export { LimitedPatchStream, DEFAULT_BYTE_BUDGET, type LimitedPatchStreamOptions } from './limiter';
export { ShadowDoc, type ShadowBlockSummary } from './shadow-doc';
export {
	PatchSession,
	resolveTarget,
	summarizePatchErrors,
	type PatchSessionCtorOptions,
	type BlockOutcome,
	type SummarizedErrors
} from './dispatcher';
export {
	RepairLoop,
	formatRepairMessage,
	MAX_REFLECTIONS,
	MAX_SUGGESTIONS,
	REPAIRABLE_CODES,
	type DispatcherFailure,
	type RepairEvent
} from './repair-loop';
export { injectRepairTurn, type RepairInjectionMeta } from './repair-injector';
export {
	createPatchStreamHook,
	runPatchRepairLoop,
	type PatchStreamHook,
	type PatchSessionAttempt,
	type RunPatchRepairLoopOptions,
	type RunStreamContext
} from './stream-hook';
export {
	getPatchSession,
	registerPatchSession,
	stopPatchRepairLoop,
	unregisterPatchSession,
	type SessionHandle
} from './session-registry';
export {
	consumeCompletedPatchSession,
	getReflectionCount,
	handleCompletedPatchSession,
	recordCompletedPatchSession,
	type HandleCompletedPatchSessionContext,
	type RunAssistantTurn
} from './chat-integration';
export {
	createChatPatchBootstrap,
	type ChatPatchBootstrap,
	type ChatPatchBootstrapOptions
} from './chat-bootstrap';
export {
	isResolvableFilename,
	resolveTargetFromAssistantContext,
	type InlineSeed,
	type TargetResolutionContext
} from './target-resolution';
export { validateSyntax, type SyntaxResult } from './syntax-gate';
export {
	attachAbortListener,
	attachPatchView,
	buildCommitChange,
	clearInflight,
	handleUserTransaction,
	patchStateField,
	setInflight,
	type CM6Attachment,
	type CM6AttachmentOptions,
	type InflightAnchor,
	type PatchAbortTarget
} from './cm6-bridge';
