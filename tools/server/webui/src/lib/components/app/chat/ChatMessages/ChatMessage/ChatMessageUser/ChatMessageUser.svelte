<script lang="ts">
	import {
		ChatMessageActions,
		ChatMessageEditForm,
		ChatMessageUser
	} from '$lib/components/app/chat';
	import { getMessageEditContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';

	interface Props {
		class?: string;
		message?: DatabaseMessage;
		siblingInfo?: ChatMessageSiblingInfo | null;
		deletionInfo?: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		showDeleteDialog?: boolean;
		onEdit?: () => void;
		onDelete?: () => void;
		onConfirmDelete?: () => void;
		onForkConversation?: (options: { name: string; includeAttachments: boolean }) => void;
		onShowDeleteDialogChange?: (show: boolean) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onCopy?: () => void;
		content?: string;
		attachments?: DatabaseMessageExtra[];
		renderMarkdown?: boolean;
		textColorClass?: string;
		cardBgClass?: string;
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		deletionInfo,
		showDeleteDialog,
		onEdit,
		onDelete,
		onConfirmDelete,
		onForkConversation,
		onShowDeleteDialogChange,
		onNavigateToSibling,
		onCopy
	}: Props = $props();

	// Get contexts
	const editCtx = getMessageEditContext();
</script>

<div
	aria-label="User message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<ChatMessageUser content={message.content} attachments={message.extra} renderMarkdown={true} />

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActions
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onForkConversation}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
