/**
 * Settings key constants for ChatSettings configuration.
 *
 * These keys correspond to properties in SettingsConfigType and are used
 * in settings field configurations to ensure consistency.
 */
export const SETTINGS_KEYS = {
	// General
	THEME: 'theme',
	THEME_MODE: 'themeMode',
	THEME_PRIMARY_HUE: 'themePrimaryHue',
	THEME_SECONDARY_HUE: 'themeSecondaryHue',
	THEME_CHROMA_SCALE: 'themeChromaScale',
	API_KEY: 'apiKey',
	BACKEND_BASE_URL: 'backendBaseUrl',
	SYSTEM_MESSAGE: 'systemMessage',
	PASTE_LONG_TEXT_TO_FILE_LEN: 'pasteLongTextToFileLen',
	COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT: 'copyTextAttachmentsAsPlainText',
	SEND_ON_ENTER: 'sendOnEnter',
	ENABLE_CONTINUE_GENERATION: 'enableContinueGeneration',
	PDF_AS_IMAGE: 'pdfAsImage',
	ASK_FOR_TITLE_CONFIRMATION: 'askForTitleConfirmation',
	TITLE_GENERATION_USE_FIRST_LINE: 'titleGenerationUseFirstLine',
	// Display
	SHOW_MESSAGE_STATS: 'showMessageStats',
	SHOW_THOUGHT_IN_PROGRESS: 'showThoughtInProgress',
	KEEP_STATS_VISIBLE: 'keepStatsVisible',
	// TTS
	TTS_ENABLED: 'ttsEnabled',
	TTS_AUTOPLAY: 'ttsAutoplay',
	TTS_BASE_URL: 'ttsBaseUrl',
	TTS_API_KEY: 'ttsApiKey',
	TTS_MODEL: 'ttsModel',
	TTS_VOICE: 'ttsVoice',
	TTS_FORMAT: 'ttsFormat',
	TTS_REF_AUDIO: 'ttsRefAudio',
	TTS_REF_AUDIO_NAME: 'ttsRefAudioName',
	SHOW_VOICE_PICKER: 'showVoicePicker',
	// STT
	STT_ENABLED: 'sttEnabled',
	STT_AUTO_TRANSCRIBE: 'sttAutoTranscribe',
	STT_AUTO_SEND: 'sttAutoSend',
	STT_BASE_URL: 'sttBaseUrl',
	STT_API_KEY: 'sttApiKey',
	STT_MODEL: 'sttModel',
	STT_LANGUAGE: 'sttLanguage',
	// Inline AI completions
	INLINE_COMPLETION_ENABLED: 'inlineCompletionEnabled',
	INLINE_COMPLETION_DELAY: 'inlineCompletionDelay',
	INLINE_COMPLETION_MAX_TOKENS: 'inlineCompletionMaxTokens',
	// AI commands (doc editor)
	AI_COMMANDS: 'aiCommands',
	RENDER_USER_CONTENT_AS_MARKDOWN: 'renderUserContentAsMarkdown',
	DISABLE_AUTO_SCROLL: 'disableAutoScroll',
	ALWAYS_SHOW_SIDEBAR_ON_DESKTOP: 'alwaysShowSidebarOnDesktop',
	AUTO_SHOW_SIDEBAR_ON_NEW_CHAT: 'autoShowSidebarOnNewChat',
	FULL_HEIGHT_CODE_BLOCKS: 'fullHeightCodeBlocks',
	SHOW_RAW_MODEL_NAMES: 'showRawModelNames',
	// Sampling
	TEMPERATURE: 'temperature',
	DYNATEMP_RANGE: 'dynatemp_range',
	DYNATEMP_EXPONENT: 'dynatemp_exponent',
	TOP_K: 'top_k',
	TOP_P: 'top_p',
	MIN_P: 'min_p',
	XTC_PROBABILITY: 'xtc_probability',
	XTC_THRESHOLD: 'xtc_threshold',
	TYP_P: 'typ_p',
	MAX_TOKENS: 'max_tokens',
	SAMPLERS: 'samplers',
	BACKEND_SAMPLING: 'backend_sampling',
	// Penalties
	REPEAT_LAST_N: 'repeat_last_n',
	REPEAT_PENALTY: 'repeat_penalty',
	PRESENCE_PENALTY: 'presence_penalty',
	FREQUENCY_PENALTY: 'frequency_penalty',
	DRY_MULTIPLIER: 'dry_multiplier',
	DRY_BASE: 'dry_base',
	DRY_ALLOWED_LENGTH: 'dry_allowed_length',
	DRY_PENALTY_LAST_N: 'dry_penalty_last_n',
	// Images / video
	IMAGES_BASE_URL: 'imagesBaseUrl',
	IMAGES_API_KEY: 'imagesApiKey',
	IMAGE_GEN_ENABLED: 'imageGenEnabled',
	VIDEO_GEN_ENABLED: 'videoGenEnabled',
	// MCP
	AGENTIC_MAX_TURNS: 'agenticMaxTurns',
	ALWAYS_SHOW_AGENTIC_TURNS: 'alwaysShowAgenticTurns',
	AGENTIC_MAX_TOOL_PREVIEW_LINES: 'agenticMaxToolPreviewLines',
	SHOW_TOOL_CALL_IN_PROGRESS: 'showToolCallInProgress',
	// Connections — Nextcloud (WebDAV)
	NEXTCLOUD_URL: 'nextcloudUrl',
	NEXTCLOUD_USERNAME: 'nextcloudUsername',
	NEXTCLOUD_REMOTE_ROOT: 'nextcloudRemoteRoot',
	NEXTCLOUD_AUTO_UPLOAD: 'nextcloudAutoUpload',
	NEXTCLOUD_MIRROR_DELETES: 'nextcloudMirrorDeletes',
	// Performance
	PRE_ENCODE_CONVERSATION: 'preEncodeConversation',
	// Developer
	DISABLE_REASONING_PARSING: 'disableReasoningParsing',
	EXCLUDE_REASONING_FROM_CONTEXT: 'excludeReasoningFromContext',
	SHOW_RAW_OUTPUT_SWITCH: 'showRawOutputSwitch',
	CUSTOM: 'custom'
} as const;
