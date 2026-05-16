/**
 * Standalone image-generation playground. Three-column desktop
 * layout (input rail · canvas · history rail), collapsible
 * accordion stack on mobile. Shares the imageGenEnabled gate, the
 * /v1/images proxy, and the gallery sink with the chat-side tools
 * and the /image slash command — only metadata.source differs
 * (`playground` here so the four entry points stay distinguishable).
 */
export { default as ImagesScreen } from './ImagesScreen.svelte';
export { default as ImageJobsPanel } from './ImageJobsPanel.svelte';
