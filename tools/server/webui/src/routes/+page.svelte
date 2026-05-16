<script lang="ts">
	import { ChatScreen, DialogModelNotAvailable } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, isConversationsInitialized } from '$lib/stores/conversations.svelte';
	import { modelsStore, modelOptions } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';

	let qParam = $derived(page.url.searchParams.get('q'));
	let modelParam = $derived(page.url.searchParams.get('model'));
	let newChatParam = $derived(page.url.searchParams.get('new_chat'));

	// Dialog state for model not available error
	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelOptions().map((m) => m.model));

	/**
	 * Clear URL params after message is sent to prevent re-sending on refresh
	 */
	function clearUrlParams() {
		const url = new URL(page.url);

		url.searchParams.delete('q');
		url.searchParams.delete('model');
		url.searchParams.delete('new_chat');

		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		await modelsStore.fetch();

		if (modelParam) {
			const model = modelsStore.findModelByName(modelParam);

			if (model) {
				try {
					await modelsStore.selectModelById(model.id);
				} catch (error) {
					console.error('Failed to select model:', error);
					requestedModelName = modelParam;
					showModelNotAvailable = true;

					return;
				}
			} else {
				requestedModelName = modelParam;
				showModelNotAvailable = true;

				return;
			}
		}

		// Handle ?q= parameter - create new conversation and send message
		if (qParam !== null) {
			await conversationsStore.createConversation();
			clearUrlParams();
		} else if (modelParam || newChatParam === 'true') {
			clearUrlParams();
		}
	}

	onMount(async () => {
		if (!isConversationsInitialized()) {
			await conversationsStore.initialize();
		}

		conversationsStore.clearActiveConversation();
		chatStore.clearUIState();

		if (
			isRouterMode() &&
			modelsStore.selectedModelName &&
			!modelsStore.isModelLoaded(modelsStore.selectedModelName)
		) {
			modelsStore.clearSelection();

			const first = modelOptions().find((m) => modelsStore.loadedModelIds.includes(m.model));
			if (first) {
				await modelsStore.selectModelById(first.id);
			}
		}

		// Apply the configured default model when nothing's selected yet and the
		// named model is currently available. URL param takes precedence (handled
		// below). Silent when the default doesn't match any available model — a
		// stale config shouldn't block the user from chatting.
		const configured = (config().defaultModel ?? '').toString().trim();
		if (configured && !modelsStore.selectedModelName && !modelParam) {
			if (modelsStore.models.length === 0) await modelsStore.fetch();
			const match = modelsStore.findModelByName(configured);
			if (match) {
				try {
					await modelsStore.selectModelById(match.id);
				} catch (err) {
					console.warn('[default-model] select failed', configured, err);
				}
			}
		}

		// Handle URL params only if we have ?q= or ?model= or ?new_chat=true
		if (qParam !== null || modelParam !== null || newChatParam === 'true') {
			await handleUrlParams();
		}
	});
</script>

<svelte:head>
	<title>heierchat</title>
</svelte:head>

<ChatScreen showCenteredEmpty />

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	modelName={requestedModelName}
	availableModels={availableModelNames}
/>
