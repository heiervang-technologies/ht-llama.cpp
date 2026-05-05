<script lang="ts">
	import { Server } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { getBackendBaseUrl, getBackendHostLabel } from '$lib/utils/backend-url';

	const label = $derived(getBackendHostLabel());
	const baseUrl = $derived(getBackendBaseUrl());
</script>

{#if label}
	<Badge
		variant="tertiary"
		class="h-9 gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
		title={baseUrl}
	>
		<Server class="h-3.5 w-3.5 opacity-80" />
		<span class="font-mono">{label}</span>
	</Badge>
{/if}
