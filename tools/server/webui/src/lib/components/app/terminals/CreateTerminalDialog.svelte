<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import { Info } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { selectedModelName } from '$lib/stores/models.svelte';
	import {
		DEFAULT_RECIPE_ID,
		TERMINAL_RECIPES,
		resolveRecipe,
		type RecipeBuild
	} from './terminal-recipes';
	import type { CreateTerminalBody } from '$lib/services/termd.service';

	interface Props {
		open: boolean;
		onOpenChange?: (next: boolean) => void;
		/** Called when the user clicks Create with the assembled body. */
		onSubmit: (body: CreateTerminalBody) => void;
	}

	let { open = $bindable(), onOpenChange, onSubmit }: Props = $props();

	let name = $state('');
	let recipeId = $state(DEFAULT_RECIPE_ID);
	let customBootstrap = $state('');

	let activeRecipe = $derived(resolveRecipe(recipeId));
	let ctx = $derived({
		backendBaseUrl: String(config().backendBaseUrl ?? '').trim(),
		modelName: selectedModelName() ?? '',
		apiKey: String(config().apiKey ?? '').trim()
	});
	let recipeBuild = $derived<RecipeBuild>(activeRecipe.build(ctx));

	let hasBackend = $derived(ctx.backendBaseUrl.length > 0);

	function previewEnv(build: RecipeBuild): string[] {
		return Object.entries(build.env ?? {}).map(([k]) => k);
	}
	function previewFiles(build: RecipeBuild): string[] {
		return (build.files ?? []).map((f) => f.path);
	}

	function reset() {
		name = '';
		recipeId = DEFAULT_RECIPE_ID;
		customBootstrap = '';
	}

	function handleSubmit() {
		const build = activeRecipe.build(ctx);
		const body: CreateTerminalBody = {
			name: name.trim() || undefined,
			env: build.env,
			files: build.files
		};
		// Custom bootstrap appends after the recipe's — that way
		// "Codex → local model + my extra steps" is a two-line
		// workflow instead of a fork.
		const merged = [build.bootstrap?.trim(), customBootstrap.trim()].filter(Boolean).join('\n\n');
		if (merged) body.bootstrap = merged;
		onSubmit(body);
		reset();
	}

	function handleOpenChange(next: boolean) {
		open = next;
		onOpenChange?.(next);
		if (!next) reset();
	}
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange}>
	<Dialog.Content class="max-w-2xl">
		<Dialog.Header>
			<Dialog.Title>New sandbox terminal</Dialog.Title>
			<Dialog.Description>
				Pick a recipe to auto-configure agent CLIs inside the sandbox, or start a plain shell.
				Bootstrap runs once as root after the container starts; output shows up in the bootstrap log
				tab.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex flex-col gap-4">
			<div class="flex flex-col gap-1.5">
				<Label for="terminal-name">Name (optional)</Label>
				<Input id="terminal-name" bind:value={name} placeholder="e.g. codex-playground" />
			</div>

			<div class="flex flex-col gap-2">
				<Label>Recipe</Label>
				<div class="flex flex-col gap-2">
					{#each TERMINAL_RECIPES as r (r.id)}
						{@const disabled = r.requiresBackend && !hasBackend}
						<button
							type="button"
							{disabled}
							onclick={() => (recipeId = r.id)}
							class="flex items-start gap-3 rounded-md border p-3 text-left text-sm transition hover:border-primary/60
							{recipeId === r.id ? 'border-primary bg-primary/5' : 'border-border'}
							{disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}"
						>
							<span
								class="mt-1 inline-flex h-3.5 w-3.5 flex-shrink-0 items-center justify-center rounded-full border
								{recipeId === r.id ? 'border-primary' : 'border-muted-foreground/40'}"
							>
								{#if recipeId === r.id}
									<span class="h-1.5 w-1.5 rounded-full bg-primary"></span>
								{/if}
							</span>
							<div class="flex flex-1 flex-col gap-0.5">
								<span class="font-medium">{r.label}</span>
								<span class="text-xs text-muted-foreground">{r.description}</span>
								{#if disabled}
									<span class="text-[11px] text-amber-500">
										Needs a backend URL in Settings → Server.
									</span>
								{/if}
							</div>
						</button>
					{/each}
				</div>
			</div>

			{#if recipeBuild.env || recipeBuild.files}
				<div class="flex flex-col gap-1 rounded-md border border-dashed p-3 text-xs">
					<div class="flex items-center gap-1.5 text-muted-foreground">
						<Info class="h-3 w-3" />
						Recipe preview
					</div>
					{#if previewEnv(recipeBuild).length > 0}
						<div>
							<span class="font-medium">Env:</span>
							<code class="font-mono text-[11px]">{previewEnv(recipeBuild).join(', ')}</code>
						</div>
					{/if}
					{#if previewFiles(recipeBuild).length > 0}
						<div>
							<span class="font-medium">Files:</span>
							<code class="font-mono text-[11px]">{previewFiles(recipeBuild).join(', ')}</code>
						</div>
					{/if}
				</div>
			{/if}

			<div class="flex flex-col gap-1.5">
				<Label for="terminal-bootstrap">Extra bootstrap (appended)</Label>
				<Textarea
					id="terminal-bootstrap"
					bind:value={customBootstrap}
					placeholder="# e.g. install extras&#10;apt-get update -qq && apt-get install -y ripgrep fzf"
					class="min-h-[120px] font-mono text-xs"
				/>
				<p class="text-[11px] text-muted-foreground">
					Runs as root inside /workspace. Output captured to
					<code>/var/log/ht-termd-bootstrap.log</code>.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="ghost" onclick={() => handleOpenChange(false)}>Cancel</Button>
			<Button onclick={handleSubmit}>Spawn</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
