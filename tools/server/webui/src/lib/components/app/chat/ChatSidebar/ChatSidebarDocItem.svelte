<!-- HT integration shim: ChatSidebar moved doc-item to navigation/SidebarNavigation/. -->
<!-- Re-export the actual component so old import paths in ChatSidebar still work. -->
<script lang="ts">
	import RealChatSidebarDocItem from '../../navigation/SidebarNavigation/ChatSidebarDocItem.svelte';
	import type { ComponentProps } from 'svelte';
	let props: ComponentProps<typeof RealChatSidebarDocItem> = $props();
</script>

<RealChatSidebarDocItem {...props} />
