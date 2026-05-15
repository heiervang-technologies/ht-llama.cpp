/**
 * Per-turn primers for the front-end "Deep research" composer toggle.
 *
 * The mode does not introduce a new endpoint, tool, or backend code path —
 * it primes the existing agentic loop with a research-shaped system prompt
 * and bumps the max-turn ceiling so the loop has room to plan, search, and
 * synthesise instead of bailing after the default ~10 turns. All tool
 * invocations (web_search, fetch_url, fetch_image, MCP-provided tools)
 * remain the model's choice from the same registry the normal chat uses.
 */

/** Default agenticConfig.maxTurns is 10 — research sessions routinely need
 *  more (plan + 5-10 searches + read + revise + synthesise + cite). 30 is
 *  the per-turn ceiling for deep research; the loop still terminates the
 *  moment the model emits a final answer with no tool calls. */
export const DEEP_RESEARCH_MAX_TURNS = 30;

/** One-shot system message prepended to the next assistant turn when
 *  deep-research mode is on. Not persisted to the conversation. */
export const DEEP_RESEARCH_SYSTEM_PROMPT = `You are operating in deep-research mode. The user has asked for a thorough, evidence-backed answer rather than a quick reply.

Methodology — follow this order:

1. **Plan.** Open with a brief plan: list the 3-6 sub-questions you intend to investigate. The plan stays visible to the user.
2. **Search.** Use the available web search and fetch tools to gather primary sources. Prefer original documents, papers, official docs, and reputable reporting over aggregators. Run multiple searches with refined queries when the first results are thin.
3. **Read deeply.** When a source looks promising, fetch the full page rather than relying on the snippet. Extract the specific claims relevant to each sub-question.
4. **Triangulate.** Cross-check important claims across at least two independent sources. Flag any disagreement explicitly rather than papering over it.
5. **Synthesise.** Once you have enough material, produce a structured markdown report. Use headings per sub-question, inline citations as numeric footnotes \`[1]\`, \`[2]\`, ..., and a final \`## Sources\` list pairing each footnote with the URL.
6. **Acknowledge gaps.** End with a short "Limitations" section listing what you couldn't verify, what your sources disagreed on, and what would need a human follow-up.

Discipline:

- Tool calls are the primary work. Don't pad with prose between calls — the user sees the agentic timeline and prefers terse running commentary.
- Don't stop after one or two searches if the question genuinely needs more. The turn budget is high in this mode; spend it.
- Do not fabricate citations. Every \`[n]\` must correspond to a URL you actually fetched.
- If a tool call fails, retry with a different query before giving up. Don't silently drop a sub-question.
- The final synthesis is your assistant content; it should be the last thing emitted, after all tool work is done.`;
