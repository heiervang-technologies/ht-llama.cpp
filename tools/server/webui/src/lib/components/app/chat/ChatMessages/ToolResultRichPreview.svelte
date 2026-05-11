<script lang="ts">
	import { Globe, Newspaper, FileText, ImageIcon, ExternalLink, ChevronDown } from '@lucide/svelte';
	import MarkdownContent from '$lib/components/app/content/MarkdownContent.svelte';

	// Rich card-style renderer for the tool_call JSON results from the
	// web tool family (web_search, search_images, search_news,
	// fetch_url, fetch_image). The default rendering in
	// ChatMessageAgenticContent dumps the raw JSON into a code block —
	// that's fine for unknown tools but loses a lot of UX for these
	// since the structure is well-known and frequently consumed.
	//
	// Returns null (renders nothing) when the toolName is unrecognised
	// or the result JSON doesn't parse — caller falls back to the
	// existing code-block renderer.

	interface Props {
		toolName: string | undefined;
		toolResult: string | undefined;
	}

	let { toolName, toolResult }: Props = $props();

	type WebResult = {
		title: string;
		url: string;
		content: string;
		engine: string;
		score: number | null;
		published_date: string | null;
	};
	type ImageResult = {
		title: string;
		url: string;
		img_src: string;
		thumbnail: string;
		resolution: string;
		source: string;
		engine: string;
	};
	type NewsResult = {
		title: string;
		url: string;
		content: string;
		source: string;
		published_date: string | null;
		engine: string;
	};
	type FetchUrlResult = {
		url: string;
		title: string;
		byline: string | null;
		excerpt: string;
		site_name: string;
		length: number;
		markdown: string;
		truncated: boolean;
	};

	function parse(): unknown {
		if (!toolResult) return null;
		try {
			return JSON.parse(toolResult);
		} catch {
			return null;
		}
	}

	let parsed = $derived(parse());
	let kind = $derived.by<
		'web_search' | 'search_images' | 'search_news' | 'fetch_url' | 'fetch_image' | 'unknown'
	>(() => {
		if (!toolName) return 'unknown';
		switch (toolName) {
			case 'web_search':
				return 'web_search';
			case 'search_images':
				return 'search_images';
			case 'search_news':
				return 'search_news';
			case 'fetch_url':
				return 'fetch_url';
			case 'fetch_image':
				return 'fetch_image';
		}
		return 'unknown';
	});

	function hostOf(u: string): string {
		try {
			return new URL(u).host.replace(/^www\./, '');
		} catch {
			return u;
		}
	}

	function fmtDate(s: string | null | undefined): string {
		if (!s) return '';
		const d = new Date(s);
		if (isNaN(d.getTime())) return s;
		const now = Date.now();
		const diff = now - d.getTime();
		const day = 86_400_000;
		if (diff < day) return `${Math.max(1, Math.round(diff / 3_600_000))}h ago`;
		if (diff < 7 * day) return `${Math.round(diff / day)}d ago`;
		return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
	}

	function isWebSearch(p: unknown): p is {
		query: string;
		count: number;
		results: WebResult[];
		suggestions?: string[];
		answers?: unknown[];
		infoboxes?: unknown[];
	} {
		return Boolean(
			p &&
				typeof p === 'object' &&
				'results' in (p as Record<string, unknown>) &&
				Array.isArray((p as { results?: unknown }).results)
		);
	}

	function isFetchUrl(p: unknown): p is FetchUrlResult {
		return Boolean(
			p &&
				typeof p === 'object' &&
				'markdown' in (p as Record<string, unknown>) &&
				'url' in (p as Record<string, unknown>)
		);
	}

	const VISIBLE_BY_DEFAULT = 6;
	let showAll = $state(false);
	let markdownOpen = $state(false);
</script>

{#if kind === 'web_search' && isWebSearch(parsed)}
	{@const results = (parsed.results as WebResult[]) ?? []}
	{@const visible = showAll ? results : results.slice(0, VISIBLE_BY_DEFAULT)}
	<div class="my-2 overflow-hidden rounded-lg border border-border/40 bg-card">
		<header class="flex items-center gap-2 border-b border-border/30 bg-muted/30 px-3 py-2 text-xs">
			<Globe class="h-3.5 w-3.5 text-primary" aria-hidden="true" />
			<span class="font-medium">Web search</span>
			<span class="text-muted-foreground">·</span>
			<span class="text-muted-foreground">
				{parsed.count} result{parsed.count === 1 ? '' : 's'} for
				<span class="text-foreground">«{parsed.query}»</span>
			</span>
		</header>
		<ul class="divide-y divide-border/30">
			{#each visible as r (r.url + r.title)}
				<li class="group px-3 py-2.5 transition-colors hover:bg-muted/20">
					<a
						href={r.url}
						target="_blank"
						rel="noopener noreferrer"
						class="block min-w-0"
						title={r.url}
					>
						<div class="flex items-center gap-2 text-[11px] text-muted-foreground">
							<span class="truncate">{hostOf(r.url)}</span>
							{#if r.engine}
								<span
									class="rounded-sm bg-muted/60 px-1.5 py-0.5 text-[10px] tracking-wide uppercase"
								>
									{r.engine}
								</span>
							{/if}
							{#if r.published_date}
								<span>·</span>
								<span>{fmtDate(r.published_date)}</span>
							{/if}
							<ExternalLink
								class="ml-auto h-3 w-3 opacity-0 transition-opacity group-hover:opacity-60"
								aria-hidden="true"
							/>
						</div>
						<div
							class="mt-0.5 line-clamp-2 text-sm font-medium text-foreground group-hover:text-primary"
						>
							{r.title}
						</div>
						{#if r.content}
							<div class="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
								{r.content}
							</div>
						{/if}
					</a>
				</li>
			{/each}
		</ul>
		{#if results.length > VISIBLE_BY_DEFAULT}
			<button
				type="button"
				onclick={() => (showAll = !showAll)}
				class="flex w-full items-center justify-center gap-1 border-t border-border/30 bg-muted/20 px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted/40"
			>
				{showAll ? 'Show fewer' : `Show all ${results.length}`}
				<ChevronDown
					class="h-3 w-3 transition-transform {showAll ? 'rotate-180' : ''}"
					aria-hidden="true"
				/>
			</button>
		{/if}
		{#if Array.isArray(parsed.suggestions) && parsed.suggestions.length > 0}
			<footer class="flex flex-wrap gap-1.5 border-t border-border/30 bg-muted/20 px-3 py-2">
				<span class="text-[10px] tracking-wide text-muted-foreground uppercase">Related</span>
				{#each parsed.suggestions.slice(0, 6) as s, i (i)}
					<span class="rounded-full bg-muted/60 px-2 py-0.5 text-[11px] text-muted-foreground">
						{s}
					</span>
				{/each}
			</footer>
		{/if}
	</div>
{:else if kind === 'search_news' && isWebSearch(parsed)}
	{@const results = (parsed.results as unknown as NewsResult[]) ?? []}
	{@const visible = showAll ? results : results.slice(0, VISIBLE_BY_DEFAULT)}
	<div class="my-2 overflow-hidden rounded-lg border border-border/40 bg-card">
		<header class="flex items-center gap-2 border-b border-border/30 bg-muted/30 px-3 py-2 text-xs">
			<Newspaper class="h-3.5 w-3.5 text-primary" aria-hidden="true" />
			<span class="font-medium">News</span>
			<span class="text-muted-foreground">·</span>
			<span class="text-muted-foreground">
				{parsed.count} for
				<span class="text-foreground">«{parsed.query}»</span>
			</span>
		</header>
		<ul class="divide-y divide-border/30">
			{#each visible as r (r.url + r.title)}
				<li class="group px-3 py-2.5 transition-colors hover:bg-muted/20">
					<a
						href={r.url}
						target="_blank"
						rel="noopener noreferrer"
						class="block min-w-0"
						title={r.url}
					>
						<div class="flex items-center gap-2 text-[11px] text-muted-foreground">
							<span class="truncate font-medium text-foreground/80">
								{r.source || hostOf(r.url)}
							</span>
							{#if r.published_date}
								<span>·</span>
								<span>{fmtDate(r.published_date)}</span>
							{/if}
							<ExternalLink
								class="ml-auto h-3 w-3 opacity-0 transition-opacity group-hover:opacity-60"
								aria-hidden="true"
							/>
						</div>
						<div
							class="mt-0.5 line-clamp-2 text-sm font-medium text-foreground group-hover:text-primary"
						>
							{r.title}
						</div>
						{#if r.content}
							<div class="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
								{r.content}
							</div>
						{/if}
					</a>
				</li>
			{/each}
		</ul>
		{#if results.length > VISIBLE_BY_DEFAULT}
			<button
				type="button"
				onclick={() => (showAll = !showAll)}
				class="flex w-full items-center justify-center gap-1 border-t border-border/30 bg-muted/20 px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted/40"
			>
				{showAll ? 'Show fewer' : `Show all ${results.length}`}
				<ChevronDown
					class="h-3 w-3 transition-transform {showAll ? 'rotate-180' : ''}"
					aria-hidden="true"
				/>
			</button>
		{/if}
	</div>
{:else if kind === 'search_images' && isWebSearch(parsed)}
	{@const results = (parsed.results as unknown as ImageResult[]) ?? []}
	{@const visible = showAll ? results : results.slice(0, 12)}
	<div class="my-2 overflow-hidden rounded-lg border border-border/40 bg-card">
		<header class="flex items-center gap-2 border-b border-border/30 bg-muted/30 px-3 py-2 text-xs">
			<ImageIcon class="h-3.5 w-3.5 text-primary" aria-hidden="true" />
			<span class="font-medium">Image search</span>
			<span class="text-muted-foreground">·</span>
			<span class="text-muted-foreground">
				{parsed.count} for
				<span class="text-foreground">«{parsed.query}»</span>
			</span>
		</header>
		<div class="grid grid-cols-2 gap-1.5 p-2 sm:grid-cols-3 md:grid-cols-4">
			{#each visible as r (r.img_src + r.title)}
				<a
					href={r.url || r.img_src}
					target="_blank"
					rel="noopener noreferrer"
					class="group relative aspect-square overflow-hidden rounded-md bg-muted/30 transition-shadow hover:shadow-md"
					title={r.title}
				>
					<img
						src={r.thumbnail || r.img_src}
						alt={r.title}
						loading="lazy"
						class="h-full w-full object-cover transition-transform group-hover:scale-105"
					/>
					<div
						class="absolute inset-x-0 bottom-0 translate-y-full bg-gradient-to-t from-black/80 to-transparent p-1.5 text-[10px] text-white transition-transform group-hover:translate-y-0"
					>
						<div class="line-clamp-2 leading-tight">{r.title}</div>
						{#if r.resolution || r.source}
							<div class="mt-0.5 flex items-center gap-1.5 text-[9px] opacity-80">
								{#if r.resolution}<span>{r.resolution}</span>{/if}
								{#if r.source}<span class="truncate">{r.source}</span>{/if}
							</div>
						{/if}
					</div>
				</a>
			{/each}
		</div>
		{#if results.length > 12}
			<button
				type="button"
				onclick={() => (showAll = !showAll)}
				class="flex w-full items-center justify-center gap-1 border-t border-border/30 bg-muted/20 px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted/40"
			>
				{showAll ? 'Show fewer' : `Show all ${results.length}`}
				<ChevronDown
					class="h-3 w-3 transition-transform {showAll ? 'rotate-180' : ''}"
					aria-hidden="true"
				/>
			</button>
		{/if}
	</div>
{:else if kind === 'fetch_url' && isFetchUrl(parsed)}
	<div class="my-2 overflow-hidden rounded-lg border border-border/40 bg-card">
		<header class="flex items-center gap-2 border-b border-border/30 bg-muted/30 px-3 py-2 text-xs">
			<FileText class="h-3.5 w-3.5 text-primary" aria-hidden="true" />
			<span class="font-medium">Fetched</span>
			<span class="text-muted-foreground">·</span>
			<a
				href={parsed.url}
				target="_blank"
				rel="noopener noreferrer"
				class="truncate text-muted-foreground hover:text-primary"
				title={parsed.url}
			>
				{hostOf(parsed.url)}
			</a>
			<span class="ml-auto text-[10px] text-muted-foreground">
				{parsed.length.toLocaleString()} chars{parsed.truncated ? ' · truncated' : ''}
			</span>
		</header>
		<div class="space-y-1 px-3 py-2.5">
			{#if parsed.title}
				<div class="text-sm font-semibold text-foreground">{parsed.title}</div>
			{/if}
			{#if parsed.byline || parsed.site_name}
				<div class="text-[11px] text-muted-foreground">
					{#if parsed.byline}{parsed.byline}{/if}
					{#if parsed.byline && parsed.site_name}
						·
					{/if}
					{#if parsed.site_name}{parsed.site_name}{/if}
				</div>
			{/if}
			{#if parsed.excerpt}
				<p class="line-clamp-3 text-xs leading-relaxed text-muted-foreground">{parsed.excerpt}</p>
			{/if}
		</div>
		<button
			type="button"
			onclick={() => (markdownOpen = !markdownOpen)}
			class="flex w-full items-center justify-center gap-1 border-t border-border/30 bg-muted/20 px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted/40"
		>
			{markdownOpen ? 'Hide article' : 'Read article'}
			<ChevronDown
				class="h-3 w-3 transition-transform {markdownOpen ? 'rotate-180' : ''}"
				aria-hidden="true"
			/>
		</button>
		{#if markdownOpen}
			<div class="max-h-[28rem] overflow-y-auto border-t border-border/30 px-3 py-3">
				<MarkdownContent content={parsed.markdown} />
			</div>
		{/if}
	</div>
{/if}
