<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { onMount, untrack } from 'svelte';
	import {
		ChatSidebar,
		DialogConversationTitleUpdate,
		DialogChatSettings
	} from '$lib/components/app';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeMessages } from '$lib/stores/conversations.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { goto } from '$app/navigation';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import type { SettingsSectionTitle } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { setChatSettingsDialogContext } from '$lib/contexts';
	import { isTauri, restoreZoom, zoomIn, zoomOut, zoomReset } from '$lib/utils/tauri-window';

	let { children } = $props();

	// Tag the boot splash (defined inline in app.html) with `.hide` once
	// the root layout is mounted and interactive. The CSS transition
	// fades it out over 250 ms, then we remove it from the DOM so it
	// can't intercept clicks. This closes the "green-black blank
	// screen" window cold-boot used to spend on bundle parse.
	onMount(() => {
		// Reapply persisted webview zoom on cold boot — Tauri's webkit2gtk
		// resets to 1.0× across launches, so we re-call setZoom with the
		// last value the user picked. No-op in browser tabs.
		restoreZoom();

		if (typeof document === 'undefined') return;
		const splash = document.getElementById('boot-splash');
		if (!splash) return;
		splash.classList.add('hide');
		splash.addEventListener('transitionend', () => splash.remove(), { once: true });
		// Safety net in case `transitionend` never fires (e.g. user has
		// `prefers-reduced-motion`; some browsers skip transitions in
		// that case so the event is suppressed).
		setTimeout(() => splash.remove(), 600);
	});

	let isChatRoute = $derived(page.route.id === '/(chat)/chat/[id]');
	let isDocRoute = $derived(page.route.id === '/doc/[id]');
	let isHomeRoute = $derived(page.route.id === '/');
	let isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	let showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());
	let alwaysShowSidebarOnDesktop = $derived(config().alwaysShowSidebarOnDesktop);
	let autoShowSidebarOnNewChat = $derived(config().autoShowSidebarOnNewChat);
	let isMobile = new IsMobile();
	let isDesktop = $derived(!isMobile.current);
	let sidebarOpen = $state(false);
	let innerHeight = $state<number | undefined>();
	let chatSidebar:
		| { activateSearchMode?: () => void; editActiveConversation?: () => void }
		| undefined = $state();

	// Conversation title update dialog state
	let titleUpdateDialogOpen = $state(false);
	let titleUpdateCurrentTitle = $state('');
	let titleUpdateNewTitle = $state('');
	let titleUpdateResolve: ((value: boolean) => void) | null = null;

	let chatSettingsDialogOpen = $state(false);
	let chatSettingsDialogInitialSection = $state<SettingsSectionTitle | undefined>(undefined);

	setChatSettingsDialogContext({
		open: (initialSection?: SettingsSectionTitle) => {
			chatSettingsDialogInitialSection = initialSection;
			chatSettingsDialogOpen = true;
		}
	});

	// Global keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.key === KeyboardKey.K_LOWER) {
			event.preventDefault();
			if (chatSidebar?.activateSearchMode) {
				chatSidebar.activateSearchMode();
				sidebarOpen = true;
			}
		}

		if (isCtrlOrCmd && event.shiftKey && event.key === KeyboardKey.O_UPPER) {
			event.preventDefault();
			goto('?new_chat=true#/');
		}

		if (event.shiftKey && isCtrlOrCmd && event.key === KeyboardKey.E_UPPER) {
			event.preventDefault();

			if (chatSidebar?.editActiveConversation) {
				chatSidebar.editActiveConversation();
			}
		}

		// Webview zoom — webkit2gtk inside Tauri doesn't bind Ctrl+= /
		// Ctrl+- / Ctrl+0 by default, so the desktop shell silently
		// ignored them. Hook them up to the shared zoom helper, which
		// persists in localStorage and reapplies after reload. In a
		// browser the chrome handles its own zoom so we let the event
		// bubble and only intervene in Tauri.
		if (isCtrlOrCmd && (event.key === '+' || event.key === '=')) {
			if (isTauri()) {
				event.preventDefault();
				zoomIn();
			}
		} else if (isCtrlOrCmd && event.key === '-') {
			if (isTauri()) {
				event.preventDefault();
				zoomOut();
			}
		} else if (isCtrlOrCmd && event.key === '0') {
			if (isTauri()) {
				event.preventDefault();
				zoomReset();
			}
		}
	}

	function handleTitleUpdateCancel() {
		titleUpdateDialogOpen = false;
		if (titleUpdateResolve) {
			titleUpdateResolve(false);
			titleUpdateResolve = null;
		}
	}

	function handleTitleUpdateConfirm() {
		titleUpdateDialogOpen = false;
		if (titleUpdateResolve) {
			titleUpdateResolve(true);
			titleUpdateResolve = null;
		}
	}

	$effect(() => {
		if (alwaysShowSidebarOnDesktop && isDesktop) {
			sidebarOpen = true;
			return;
		}

		if (isHomeRoute && !isNewChatMode) {
			// Auto-collapse sidebar when navigating to home route (but not in new chat mode)
			sidebarOpen = false;
		} else if (isHomeRoute && isNewChatMode) {
			// Keep sidebar open in new chat mode
			sidebarOpen = true;
		} else if (isChatRoute || isDocRoute) {
			// On chat/doc routes, only auto-show sidebar if setting is enabled
			if (autoShowSidebarOnNewChat) {
				sidebarOpen = true;
			}
			// If setting is disabled, don't change sidebar state - let user control it manually
		} else {
			// Other routes follow default behavior
			sidebarOpen = showSidebarByDefault;
		}
	});

	// Initialize server properties on app load (run once)
	$effect(() => {
		// Only fetch if we don't already have props
		if (!serverStore.props) {
			untrack(() => {
				serverStore.fetch();
			});
		}
	});

	// Sync settings when server props are loaded
	$effect(() => {
		const serverProps = serverStore.props;

		if (serverProps) {
			settingsStore.syncWithServerDefaults();
		}
	});

	// Fetch router models when in router mode (for status and modalities)
	// Wait for models to be loaded first, run only once
	let routerModelsFetched = false;

	$effect(() => {
		const isRouter = isRouterMode();
		const modelsCount = modelsStore.models.length;

		// Only fetch router models once when we have models loaded and in router mode
		if (isRouter && modelsCount > 0 && !routerModelsFetched) {
			routerModelsFetched = true;
			untrack(() => {
				modelsStore.fetchRouterModels();
			});
		}
	});

	// Background MCP server health checks on app load
	// Fetch enabled servers from settings and run health checks in background
	$effect(() => {
		if (!browser) return;

		const mcpServers = mcpStore.getServers();

		// Only run health checks if we have enabled servers with URLs
		const enabledServers = mcpServers.filter((s) => s.enabled && s.url.trim());

		if (enabledServers.length > 0) {
			untrack(() => {
				// Run health checks in background (don't await)
				mcpStore.runHealthChecksForServers(enabledServers, false).catch((error) => {
					console.warn('[layout] MCP health checks failed:', error);
				});
			});
		}
	});

	// Monitor API key changes and redirect to error page if removed or changed when required
	$effect(() => {
		const apiKey = config().apiKey;

		if (
			(page.route.id === '/' ||
				page.route.id === '/(chat)' ||
				page.route.id === '/(chat)/chat/[id]') &&
			page.status !== 401 &&
			page.status !== 403
		) {
			const headers: Record<string, string> = {
				'Content-Type': 'application/json'
			};

			if (apiKey && apiKey.trim() !== '') {
				headers.Authorization = `Bearer ${apiKey.trim()}`;
			}

			fetch(`${base}/props`, { headers })
				.then((response) => {
					if (response.status === 401 || response.status === 403) {
						window.location.reload();
					}
				})
				.catch((e) => {
					console.error('Error checking API key:', e);
				});
		}
	});

	// Push theme hue / chroma / mode onto the document root so app.css picks them up.
	$effect(() => {
		if (!browser) return;
		const primary = Number(config().themePrimaryHue);
		const secondary = Number(config().themeSecondaryHue);
		const chroma = Number(config().themeChromaScale);
		const mode = String(config().themeMode ?? 'colorful');
		const root = document.documentElement;
		if (Number.isFinite(primary)) root.style.setProperty('--hue-primary', String(primary));
		if (Number.isFinite(secondary)) root.style.setProperty('--hue-secondary', String(secondary));
		if (Number.isFinite(chroma))
			root.style.setProperty('--theme-chroma-scale', String(Math.max(0, Math.min(1, chroma))));
		root.classList.toggle('theme-pure-black', mode === 'pure-black');
		root.classList.toggle('theme-pure-white', mode === 'pure-white');
	});

	// Set up title update confirmation callback
	$effect(() => {
		conversationsStore.setTitleUpdateConfirmationCallback(
			async (currentTitle: string, newTitle: string) => {
				return new Promise<boolean>((resolve) => {
					titleUpdateCurrentTitle = currentTitle;
					titleUpdateNewTitle = newTitle;
					titleUpdateResolve = resolve;
					titleUpdateDialogOpen = true;
				});
			}
		);
	});
</script>

<Tooltip.Provider delayDuration={TOOLTIP_DELAY_DURATION}>
	<ModeWatcher />

	<Toaster richColors />

	<DialogChatSettings
		open={chatSettingsDialogOpen}
		onOpenChange={(open: boolean) => (chatSettingsDialogOpen = open)}
		initialSection={chatSettingsDialogInitialSection}
	/>

	<DialogConversationTitleUpdate
		bind:open={titleUpdateDialogOpen}
		currentTitle={titleUpdateCurrentTitle}
		newTitle={titleUpdateNewTitle}
		onConfirm={handleTitleUpdateConfirm}
		onCancel={handleTitleUpdateCancel}
	/>

	<Sidebar.Provider bind:open={sidebarOpen}>
		<div class="flex h-screen w-full" style:height="{innerHeight}px">
			<Sidebar.Root class="h-full">
				<ChatSidebar bind:this={chatSidebar} />
			</Sidebar.Root>

			{#if !(alwaysShowSidebarOnDesktop && isDesktop)}
				<Sidebar.Trigger
					class="transition-left absolute left-0 z-[900] duration-200 ease-linear {sidebarOpen
						? 'md:left-[var(--sidebar-width)]'
						: 'md:left-0!'}"
					style="translate: 1rem 1rem;"
				/>
			{/if}

			<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
				{@render children?.()}
			</Sidebar.Inset>
		</div>
	</Sidebar.Provider>
</Tooltip.Provider>

<svelte:window onkeydown={handleKeydown} bind:innerHeight />
