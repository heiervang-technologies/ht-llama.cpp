<script lang="ts">
	import { page } from '$app/state';
	import { DocScreen } from '$lib/components/app';
	import { activeDoc } from '$lib/stores/docs.svelte';

	let docId = $derived(page.params.id ?? '');
</script>

<svelte:head>
	<title>{activeDoc()?.name || 'Document'} - heierchat</title>
</svelte:head>

{#if docId}
	<DocScreen {docId} />
{/if}
