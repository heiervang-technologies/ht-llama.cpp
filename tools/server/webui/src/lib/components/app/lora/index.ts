/**
 *
 * LORA
 *
 * Components for LoRA adapter management. Displays available adapters
 * with toggle switches and scale sliders. Only renders when the server
 * has LoRA adapters loaded.
 *
 */

/**
 * **LoraAdapters** - Collapsible LoRA adapter management panel
 *
 * Shows loaded LoRA adapters with per-adapter enable/disable toggle
 * and scale slider (0.0 - 2.0). Includes an "Apply" button when
 * changes are pending.
 *
 * **Architecture:**
 * - Fetches adapter list from server via loraStore on mount
 * - Conditionally renders only when adapters are available
 * - Uses loraStore for state management
 *
 * @example
 * ```svelte
 * <LoraAdapters />
 * ```
 */
export { default as LoraAdapters } from './LoraAdapters.svelte';
