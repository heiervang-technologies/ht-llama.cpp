import { ColorMode } from '$lib/enums/ui';
import { Monitor, Moon, Sun } from '@lucide/svelte';

export const SETTING_CONFIG_DEFAULT: Record<string, string | number | boolean | undefined> = {
	// Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value.
	// Do not use nested objects, keep it single level. Prefix the key if you need to group them.
	apiKey: '',
	backendBaseUrl: '',
	systemMessage: '',
	showSystemMessage: true,
	theme: ColorMode.SYSTEM,
	themePrimaryHue: 295,
	themeSecondaryHue: 190,
	showThoughtInProgress: false,
	disableReasoningParsing: false,
	excludeReasoningFromContext: false,
	showRawOutputSwitch: false,
	keepStatsVisible: false,
	showMessageStats: true,
	askForTitleConfirmation: false,
	titleGenerationUseFirstLine: false,
	pasteLongTextToFileLen: 2500,
	copyTextAttachmentsAsPlainText: false,
	pdfAsImage: false,
	disableAutoScroll: false,
	renderUserContentAsMarkdown: true,
	alwaysShowSidebarOnDesktop: false,
	autoShowSidebarOnNewChat: true,
	sendOnEnter: true,
	fullHeightCodeBlocks: false,
	showRawModelNames: false,
	mcpServers: '[]',
	mcpServerUsageStats: '{}', // JSON object: { [serverId]: usageCount }
	agenticMaxTurns: 10,
	agenticMaxToolPreviewLines: 25,
	showToolCallInProgress: false,
	alwaysShowAgenticTurns: false,
	// sampling params: empty means "use server default"
	// the server / preset is the source of truth
	// empty values are shown as placeholders from /props in the UI
	// and are NOT sent in API requests, letting the server decide
	samplers: '',
	backend_sampling: false,
	temperature: undefined,
	dynatemp_range: undefined,
	dynatemp_exponent: undefined,
	top_k: undefined,
	top_p: undefined,
	min_p: undefined,
	xtc_probability: undefined,
	xtc_threshold: undefined,
	typ_p: undefined,
	repeat_last_n: undefined,
	repeat_penalty: undefined,
	presence_penalty: undefined,
	frequency_penalty: undefined,
	dry_multiplier: undefined,
	dry_base: undefined,
	dry_allowed_length: undefined,
	dry_penalty_last_n: undefined,
	max_tokens: undefined,
	custom: '', // custom json-stringified object
	preEncodeConversation: false,
	// experimental features
	pyInterpreterEnabled: false,
	enableContinueGeneration: false,
	// tts
	ttsEnabled: false,
	ttsAutoplay: true,
	ttsBaseUrl: '',
	ttsApiKey: '',
	ttsModel: '',
	ttsVoice: '',
	ttsFormat: 'wav',
	ttsRefAudio: '',
	ttsRefAudioName: '',
	showVoicePicker: true,
	// stt
	sttEnabled: false,
	sttAutoTranscribe: true,
	sttAutoSend: false,
	sttBaseUrl: '',
	sttApiKey: '',
	sttModel: '',
	sttLanguage: '',
	// inline AI completions (ghost text in the doc editor)
	inlineCompletionEnabled: false,
	inlineCompletionDelay: 800,
	inlineCompletionMaxTokens: 48,
	// user-defined AI commands invoked from the doc editor header
	// (stored as JSON string; empty string = use built-in defaults)
	aiCommands: '',
	// model name (not id) to pre-select when opening a new chat / empty
	// conversation. Empty means "no preference — let the first available
	// model win." When set, the routes bootstrap selectModelByName on
	// mount if the model is currently available; a missing default is
	// silently ignored.
	defaultModel: '',
	// When true, `role: 'tool'` messages render as their own standalone
	// cards in the chat log instead of being folded into the preceding
	// assistant turn's agentic section. Combines well with
	// `showSystemMessage` + `alwaysShowAgenticTurns` for a full
	// prompt-transparency view.
	showToolMessagesAsStandalone: false,
	// Override for the `ht-termd` sidecar URL. Empty = Tauri auto-spawn
	// or llama-server `/props.terminals.url` take precedence. Set for
	// web deployments that point at a remote termd instance (rare).
	terminalsBaseUrl: '',
	// Bearer token paired with `terminalsBaseUrl`. Required when the
	// termd daemon is launched with `--token`; ignored otherwise.
	// Sent as `Authorization: Bearer <token>` on HTTP and `?token=<t>`
	// on the WS upgrade (browsers can't set WS headers).
	terminalsToken: '',
	// Base URL of the OpenAI-compatible image-generation proxy (e.g.
	// `http://images.ht.local`). Used by the `generate_image` built-in
	// tool. Empty string = feature disabled; the tool returns a clean
	// "not configured" error instead of trying a default URL.
	imagesBaseUrl: '',
	// Optional API key forwarded as `Authorization: Bearer <key>` to
	// the images proxy. Typical deployments on a trusted LAN leave
	// this empty.
	imagesApiKey: '',
	// Media generation toggles — default OFF so the model never picks
	// a ~60 s image-gen tool unless the user explicitly opts in. When
	// true, the corresponding builtin tool (`generate_image` /
	// `generate_video`) is advertised in the model's tool list and
	// dispatchable; when false it is hidden and a stale tool_call in
	// conversation history returns a "disabled in Settings" error to
	// the model. See `builtin-tools.ts`.
	imageGenEnabled: false,
	videoGenEnabled: false
};

export const SETTING_CONFIG_INFO: Record<string, string> = {
	apiKey: 'Set the API Key if you are using <code>--api-key</code> option for the server.',
	backendBaseUrl:
		'Base URL of the llama-server backend (e.g. <code>http://192.168.8.158:30184</code>). Leave empty to use the same origin as the webui. When set, a pill in the header shows the active hostname.',
	systemMessage: 'The starting message that defines how model should behave.',
	showSystemMessage: 'Display the system message at the top of each conversation.',
	theme:
		'Choose the color theme for the interface. You can choose between System (follows your device settings), Light, or Dark.',
	pasteLongTextToFileLen:
		'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
	copyTextAttachmentsAsPlainText:
		'When copying a message with text attachments, combine them into a single plain text string instead of a special format that can be pasted back as attachments.',
	samplers:
		'The order at which samplers are applied, in simplified way. Default is "top_k;typ_p;top_p;min_p;temperature": top_k->typ_p->top_p->min_p->temperature',
	backend_sampling:
		'Enable backend-based samplers. When enabled, supported samplers run on the accelerator backend for faster sampling.',
	temperature:
		'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
	dynatemp_range:
		'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
	dynatemp_exponent:
		'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
	top_k: 'Keeps only k top tokens.',
	top_p: 'Limits tokens to those that together have a cumulative probability of at least p',
	min_p:
		'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
	xtc_probability:
		'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
	xtc_threshold:
		'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
	typ_p: 'Sorts and limits tokens based on the difference between log-probability and entropy.',
	repeat_last_n: 'Last n tokens to consider for penalizing repetition',
	repeat_penalty: 'Controls the repetition of token sequences in the generated text',
	presence_penalty: 'Limits tokens based on whether they appear in the output or not.',
	frequency_penalty: 'Limits tokens based on how often they appear in the output.',
	dry_multiplier:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
	dry_base:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
	dry_allowed_length:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
	dry_penalty_last_n:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
	max_tokens: 'The maximum number of token per output. Use -1 for infinite (no limit).',
	custom: 'Custom JSON parameters to send to the API. Must be valid JSON format.',
	showThoughtInProgress: 'Expand thought process by default when generating messages.',
	disableReasoningParsing:
		'Send reasoning_format=none so the server returns thinking tokens inline instead of extracting them into a separate field.',
	excludeReasoningFromContext:
		'Strip thinking from previous messages before sending. When off, thinking is sent back via the reasoning_content field so the model sees its own chain-of-thought across turns.',
	showRawOutputSwitch:
		'Show toggle button to display messages as plain text instead of Markdown-formatted content',
	keepStatsVisible: 'Keep processing statistics visible after generation finishes.',
	showMessageStats:
		'Display generation statistics (tokens/second, token count, duration) below each assistant message.',
	askForTitleConfirmation:
		'Ask for confirmation before automatically changing conversation title when editing the first message.',
	titleGenerationUseFirstLine:
		'Use only the first non-empty line of the prompt to generate the conversation title.',
	pdfAsImage:
		'Parse PDF as image instead of text. Automatically falls back to text processing for non-vision models.',
	disableAutoScroll:
		'Disable automatic scrolling while messages stream so you can control the viewport position manually.',
	renderUserContentAsMarkdown: 'Render user messages using markdown formatting in the chat.',
	alwaysShowSidebarOnDesktop:
		'Always keep the sidebar visible on desktop instead of auto-hiding it.',
	autoShowSidebarOnNewChat:
		'Automatically show sidebar when starting a new chat. Disable to keep the sidebar hidden until you click on it.',
	sendOnEnter:
		'Use Enter to send messages and Shift + Enter for new lines. When disabled, use Ctrl/Cmd + Enter.',
	fullHeightCodeBlocks:
		'Always display code blocks at their full natural height, overriding any height limits.',
	showRawModelNames:
		'Display full raw model identifiers (e.g. "ggml-org/GLM-4.7-Flash-GGUF:Q8_0") instead of parsed names with badges.',
	mcpServers:
		'Configure MCP servers as a JSON list. Use the form in the MCP Client settings section to edit.',
	mcpServerUsageStats:
		'Usage statistics for MCP servers. Tracks how many times tools from each server have been used.',
	agenticMaxTurns:
		'Maximum number of tool execution cycles before stopping (prevents infinite loops).',
	agenticMaxToolPreviewLines:
		'Number of lines shown in tool output previews (last N lines). Only these previews and the final LLM response persist after the agentic loop completes.',
	showToolCallInProgress:
		'Automatically expand tool call details while executing and keep them expanded after completion.',
	pyInterpreterEnabled:
		'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
	preEncodeConversation:
		'After each response, re-submit the conversation to pre-fill the server KV cache. Makes the next turn faster since the prompt is already encoded while you read the response.',
	enableContinueGeneration:
		'Enable "Continue" button for assistant messages. Currently works only with non-reasoning models.',
	ttsEnabled: 'Enable text-to-speech for assistant messages.',
	ttsAutoplay: 'Automatically speak assistant messages when generation completes.',
	ttsBaseUrl:
		'Base URL of an OpenAI-compatible TTS server (e.g. <code>http://192.168.8.123:30384</code>). Must implement <code>POST /v1/audio/speech</code>.',
	ttsApiKey:
		'Optional API key sent as <code>Authorization: Bearer &lt;key&gt;</code> to the TTS server.',
	ttsModel:
		'TTS model identifier (e.g. <code>qwen3-tts</code>, <code>tts-1</code>, <code>kokoro</code>).',
	ttsVoice:
		'Voice name passed to the TTS server (e.g. <code>Chelsie</code>, <code>alloy</code>, <code>af_bella</code>).',
	ttsFormat:
		'Audio format requested from the TTS server (<code>wav</code>, <code>mp3</code>, <code>opus</code>, <code>flac</code>).',
	ttsRefAudio:
		'Reference audio clip for voice cloning (Qwen3-TTS). Stored as a <code>data:</code> URI. Upload a short sample of the target voice; the server will use x-vector extraction to match it.',
	ttsRefAudioName: 'Original filename of the uploaded reference audio (for display only).',
	showVoicePicker:
		'Show the voice picker as a block in the chat composer chain. Lets you swap voices without opening settings. Disable to keep the chain to LoRA + model only.',
	sttEnabled: 'Enable speech-to-text for mic recordings.',
	sttAutoTranscribe:
		'Automatically transcribe recordings into the textarea instead of attaching them as audio files.',
	sttAutoSend:
		'After transcription finishes, automatically submit the message. Enables a heads-down voice-only flow.',
	sttBaseUrl:
		'Base URL of an OpenAI-compatible STT server (e.g. <code>http://192.168.8.123:30189</code>). Must implement <code>POST /v1/audio/transcriptions</code>.',
	sttApiKey:
		'Optional API key sent as <code>Authorization: Bearer &lt;key&gt;</code> to the STT server.',
	sttModel: 'STT model identifier (e.g. <code>Qwen/Qwen3-ASR-1.7B</code>, <code>whisper-1</code>).',
	sttLanguage:
		'Optional ISO 639-1 language hint (e.g. <code>en</code>, <code>no</code>). Leave blank for auto-detection.',
	inlineCompletionEnabled:
		'Show AI ghost-text suggestions in the doc editor while you type. Tab accepts, Esc dismisses.',
	inlineCompletionDelay:
		'Milliseconds of idle time before an inline completion is requested (min 200).',
	inlineCompletionMaxTokens:
		'Max tokens requested per inline completion. Smaller values feel snappier.',
	defaultModel:
		'Preferred model name pre-selected when opening a new chat (e.g. <code>gemma-4-e4b</code>). Leave empty to let the first available model win. Ignored silently when the named model is unavailable at load time.',
	showToolMessagesAsStandalone:
		'Render each tool call + tool result as its own card in the chat log, before the next user message. Off by default (tool exchanges stay folded inside the assistant turn).',
	terminalsBaseUrl:
		'Base URL of the <code>ht-termd</code> sidecar (e.g. <code>http://127.0.0.1:43127</code>). Leave empty in the Tauri app (the sidecar is auto-spawned). Only set this for web deployments pointing at a remote daemon.',
	terminalsToken:
		"Bearer token for the <code>ht-termd</code> sidecar. Only needed when the daemon was started with <code>--token</code>; required for Tailscale-reachable deployments so random peers can't spawn shells.",
	imagesBaseUrl:
		'OpenAI-compatible image-generation proxy URL (e.g. <code>http://images.ht.local</code> or <code>http://192.168.8.170:30385</code>). Enables the <code>generate_image</code> tool so the model can create images; each returned image lands in the artifact gallery automatically.',
	imagesApiKey:
		"Optional bearer token for the images proxy. Leave empty for trusted-LAN deployments that don't require auth.",
	imageGenEnabled:
		'Let the model invoke the <code>generate_image</code> tool when replying. Requires <em>Images base URL</em> to be set. Off by default — enable when you want a turn to be able to produce an image. Expected wait: ~60 s with <code>z-image-turbo</code>.',
	videoGenEnabled:
		'Let the model invoke the <code>generate_video</code> tool when replying. Backend is async (202 + poll) and currently experimental (wan22-i2v latent-channel fixes still landing). Off by default.'
};

export const SETTINGS_COLOR_MODES_CONFIG = [
	{ value: ColorMode.SYSTEM, label: 'System', icon: Monitor },
	{ value: ColorMode.LIGHT, label: 'Light', icon: Sun },
	{ value: ColorMode.DARK, label: 'Dark', icon: Moon }
];
