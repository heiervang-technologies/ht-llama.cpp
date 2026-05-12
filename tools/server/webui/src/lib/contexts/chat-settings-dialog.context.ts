import { getContext, setContext } from 'svelte';
import type { SettingsSectionTitle } from '$lib/types';

export interface ChatSettingsDialogContext {
	open: (initialSection?: SettingsSectionTitle) => void;
}

const CHAT_SETTINGS_DIALOG_KEY = Symbol.for('chat-settings-dialog');

export function setChatSettingsDialogContext(
	ctx: ChatSettingsDialogContext
): ChatSettingsDialogContext {
	return setContext(CHAT_SETTINGS_DIALOG_KEY, ctx);
}

export function getChatSettingsDialogContext(): ChatSettingsDialogContext {
	return getContext(CHAT_SETTINGS_DIALOG_KEY);
}
