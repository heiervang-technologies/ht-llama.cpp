<script lang="ts">
	import { page } from '$app/state';
	import { ArtifactDetail } from '$lib/components/app';

	let artifactId = $derived(page.params.id ?? '');
</script>

<svelte:head>
	<title>Artifact — ht-llama.cpp</title>
</svelte:head>

{#if artifactId}
	<ArtifactDetail {artifactId} />
{/if}
