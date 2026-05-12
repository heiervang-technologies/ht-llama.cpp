<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { untrack } from 'svelte';
	import { ChatSettings } from '$lib/components/app';
	import type { SettingsSectionTitle } from '$lib/constants';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
		initialSection?: SettingsSectionTitle;
	}

	let { onOpenChange, open = false, initialSection }: Props = $props();

	let chatSettingsRef: ChatSettings | undefined = $state();

	function handleClose() {
		onOpenChange?.(false);
	}

	function handleSave() {
		onOpenChange?.(false);
	}

	// Reset localConfig to current store config ONCE when the dialog transitions to open.
	// untrack() prevents reactive reads inside reset() (config.* properties) from becoming
	// deps of this effect — otherwise any config mutation would refire the effect and
	// clobber in-progress edits (e.g. unsaved randomized theme hues) on every click.
	$effect(() => {
		if (open && chatSettingsRef) {
			const ref = chatSettingsRef;
			untrack(() => ref.reset());
		}
	});
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] max-w-4xl! flex-col gap-0 rounded-none
			p-0 md:h-[64vh] md:max-h-[64vh] md:min-h-0 md:rounded-lg"
	>
		<ChatSettings bind:this={chatSettingsRef} onSave={handleSave} {initialSection} />
	</Dialog.Content>
</Dialog.Root>
