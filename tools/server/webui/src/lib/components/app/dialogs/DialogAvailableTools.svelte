<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Wrench, ChevronRight, Server, FlaskConical, Sparkles, Copy } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { config } from '$lib/stores/settings.svelte';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	// Everything derives off the live stores so the dialog mirrors exactly
	// what the chat pipeline would send on the next request.
	let mcpToolDefs = $derived(mcpStore.getToolDefinitionsForLLM());
	let connections = $derived([...mcpStore.getConnections().values()]);

	// Built-in feature flags that shape what the model sees or can do — not
	// currently sent as tools (they live at prompt / pipeline layer), but
	// they belong in a "what's active" surface.
	let builtins = $derived([
		{
			name: 'Reasoning parsing',
			on: !config().disableReasoningParsing,
			detail: config().disableReasoningParsing
				? 'Disabled — reasoning tokens treated as plain content.'
				: 'Enabled — <think> blocks get their own channel.'
		},
		{
			name: 'Python interpreter',
			on: Boolean(config().pyInterpreterEnabled),
			detail: config().pyInterpreterEnabled
				? 'Enabled (experimental) — model outputs in ```python blocks are runnable.'
				: 'Disabled — python blocks render as static code.'
		},
		{
			name: 'Continue generation',
			on: Boolean(config().enableContinueGeneration),
			detail: config().enableContinueGeneration
				? 'Enabled (experimental) — explicit "continue" resumes prior turn.'
				: 'Disabled.'
		},
		{
			name: 'Inline AI completions',
			on: Boolean(config().inlineCompletionEnabled),
			detail: config().inlineCompletionEnabled
				? 'On in the doc editor — ghost text via /v1/completions on idle.'
				: 'Off.'
		}
	]);

	// Doc-editor AI commands surface — not sent as OpenAI tools, but part of
	// the capability story the user should see in one place.
	let userCommandsRaw = $derived((config().aiCommands as string) ?? '');
	let userCommands = $derived.by(() => {
		const raw = userCommandsRaw.trim();
		if (!raw) return [] as Array<{ id: string; title: string; description?: string }>;
		try {
			const parsed = JSON.parse(raw);
			if (Array.isArray(parsed)) {
				return parsed.filter((c) => c && typeof c === 'object' && 'id' in c && 'title' in c);
			}
		} catch {
			/* fall through */
		}
		return [];
	});

	let expanded = $state<Record<string, boolean>>({});
	function toggle(key: string) {
		expanded[key] = !expanded[key];
	}

	function copyJson(def: unknown, label: string) {
		try {
			void navigator.clipboard.writeText(JSON.stringify(def, null, 2));
			toast.success(`Copied ${label} schema`);
		} catch (err) {
			toast.error(`Copy failed: ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	function copyAll() {
		try {
			void navigator.clipboard.writeText(JSON.stringify(mcpToolDefs, null, 2));
			toast.success(`Copied ${mcpToolDefs.length} tool definitions`);
		} catch (err) {
			toast.error(`Copy failed: ${err instanceof Error ? err.message : String(err)}`);
		}
	}
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="max-h-[85vh] max-w-3xl overflow-hidden">
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Wrench class="h-5 w-5" />
				Available tools
			</Dialog.Title>
			<Dialog.Description>
				Exactly what the model will see on the next chat turn. Includes MCP tools currently sent in
				the request, plus built-in features and user-defined AI commands that shape behaviour at
				other layers.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex max-h-[65vh] flex-col gap-6 overflow-y-auto pr-1">
			<!-- MCP tools — the set actually sent in the tools[] array. -->
			<section class="flex flex-col gap-2">
				<header class="flex items-center justify-between">
					<h3 class="flex items-center gap-2 text-sm font-semibold">
						<Server class="h-4 w-4 text-primary" />
						MCP tools
						<span class="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] uppercase text-primary">
							sent to model
						</span>
						<span class="text-xs font-normal text-muted-foreground">
							{mcpToolDefs.length} total
						</span>
					</h3>
					{#if mcpToolDefs.length > 0}
						<Button size="sm" variant="ghost" onclick={copyAll}>
							<Copy class="h-3.5 w-3.5" />
							Copy all JSON
						</Button>
					{/if}
				</header>

				{#if connections.length === 0}
					<p class="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
						No MCP servers configured. Open Settings → MCP to add one.
					</p>
				{:else}
					{#each connections as conn (conn.serverName)}
						{@const server = conn.serverName}
						{@const serverKey = `srv:${server}`}
						{@const online = conn.tools.length > 0}
						<div class="rounded-md border">
							<button
								type="button"
								class="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-muted/40"
								onclick={() => toggle(serverKey)}
								aria-expanded={expanded[serverKey] ?? false}
							>
								<ChevronRight
									class="h-4 w-4 shrink-0 transition-transform {expanded[serverKey]
										? 'rotate-90'
										: ''}"
								/>
								<span class="font-medium">{server}</span>
								<span
									class="rounded-full px-2 py-0.5 text-[10px] uppercase {online
										? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
										: 'bg-muted text-muted-foreground'}"
								>
									{online ? `${conn.tools.length} tools` : 'offline / no tools'}
								</span>
							</button>
							{#if expanded[serverKey]}
								<div class="divide-y border-t">
									{#if conn.tools.length === 0}
										<p class="px-3 py-2 text-xs text-muted-foreground">
											Server is connected but exposes no tools.
										</p>
									{:else}
										{#each conn.tools as tool (tool.name)}
											{@const toolKey = `${serverKey}::${tool.name}`}
											{@const def = mcpToolDefs.find((t) => t.function.name === tool.name)}
											<div class="px-3 py-2">
												<button
													type="button"
													class="flex w-full items-center justify-between gap-2 text-left"
													onclick={() => toggle(toolKey)}
													aria-expanded={expanded[toolKey] ?? false}
												>
													<div class="flex min-w-0 flex-col">
														<span class="font-mono text-xs font-medium">{tool.name}</span>
														{#if tool.description}
															<span class="truncate text-xs text-muted-foreground">
																{tool.description}
															</span>
														{/if}
													</div>
													<ChevronRight
														class="h-4 w-4 shrink-0 transition-transform {expanded[toolKey]
															? 'rotate-90'
															: ''}"
													/>
												</button>
												{#if expanded[toolKey] && def}
													<div class="mt-2 flex flex-col gap-1">
														<div class="flex items-center justify-end">
															<Button
																size="sm"
																variant="ghost"
																class="h-6 gap-1 text-xs"
																onclick={() => copyJson(def, tool.name)}
															>
																<Copy class="h-3 w-3" />
																Copy JSON
															</Button>
														</div>
														<pre
															class="overflow-x-auto rounded-md bg-muted/60 p-2 text-[11px] leading-snug"><code
																>{JSON.stringify(def, null, 2)}</code
															></pre>
													</div>
												{/if}
											</div>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			</section>

			<!-- Built-in behaviour flags — not tool-calls, but model-visible. -->
			<section class="flex flex-col gap-2">
				<h3 class="flex items-center gap-2 text-sm font-semibold">
					<FlaskConical class="h-4 w-4 text-primary" />
					Built-in features
					<span
						class="rounded-full bg-muted px-2 py-0.5 text-[10px] uppercase text-muted-foreground"
					>
						prompt / pipeline layer
					</span>
				</h3>
				<ul class="flex flex-col gap-1 rounded-md border p-2">
					{#each builtins as b (b.name)}
						<li class="flex items-start gap-2 px-2 py-1 text-sm">
							<span
								class="mt-0.5 inline-flex h-4 w-10 flex-shrink-0 items-center justify-center rounded-full text-[10px] font-medium uppercase {b.on
									? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
									: 'bg-muted text-muted-foreground'}"
							>
								{b.on ? 'On' : 'Off'}
							</span>
							<div class="flex flex-col">
								<span class="font-medium">{b.name}</span>
								<span class="text-xs text-muted-foreground">{b.detail}</span>
							</div>
						</li>
					{/each}
				</ul>
			</section>

			<!-- Doc-editor AI commands — local, not sent to the model as tools. -->
			<section class="flex flex-col gap-2">
				<h3 class="flex items-center gap-2 text-sm font-semibold">
					<Sparkles class="h-4 w-4 text-primary" />
					AI commands
					<span
						class="rounded-full bg-muted px-2 py-0.5 text-[10px] uppercase text-muted-foreground"
					>
						doc editor, local
					</span>
					<span class="text-xs font-normal text-muted-foreground">
						{userCommands.length > 0 ? `${userCommands.length} custom` : 'built-in defaults'}
					</span>
				</h3>
				{#if userCommands.length === 0}
					<p class="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
						Using the built-in defaults (shorten / expand / fix grammar / summarize / etc.). Define
						your own in Settings → AI Commands.
					</p>
				{:else}
					<ul class="flex flex-col gap-1 rounded-md border p-2">
						{#each userCommands as cmd (cmd.id)}
							<li class="flex flex-col px-2 py-1 text-sm">
								<span class="font-mono text-xs font-medium">{cmd.title}</span>
								{#if cmd.description}
									<span class="text-xs text-muted-foreground">{cmd.description}</span>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</section>
		</div>

		<Dialog.Footer>
			<Button variant="secondary" onclick={() => (open = false)}>Close</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
