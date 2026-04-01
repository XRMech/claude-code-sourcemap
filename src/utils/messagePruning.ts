import { Message, UserMessage } from '../query.js'

// Tools whose results are read-only and can be safely deduplicated/pruned
const PRUNABLE_READ_TOOLS = new Set([
  'View',         // FileReadTool
  'Search',       // GrepTool
  'GlobTool',     // GlobTool
  'LS',           // LSTool
  'ReadNotebook', // NotebookReadTool
])

// Tools where a later result for the same input supersedes an earlier one
const SUPERSEDABLE_TOOLS = new Set([
  'View',  // same file read twice → keep latest
])

// Read-only bash commands whose output can be pruned
const READ_ONLY_BASH_PREFIXES = [
  'cat ', 'head ', 'tail ', 'less ', 'more ',
  'ls ', 'dir ',
  'find ', 'locate ',
  'grep ', 'rg ', 'ag ', 'ack ',
  'git status', 'git log', 'git diff', 'git show', 'git branch',
  'npm test', 'npm run test', 'npx jest', 'npx vitest', 'pytest',
  'npm run lint', 'npm run typecheck', 'npx tsc', 'npx eslint',
  'npm run build', 'cargo build', 'cargo test', 'go test',
  'echo ', 'pwd', 'whoami', 'date', 'env',
  'wc ', 'sort ', 'uniq ', 'cut ', 'tr ',
]

/**
 * Determines if a message contains a prunable (read-only) tool result.
 */
export function isPrunableMessage(msg: UserMessage): boolean {
  if (!msg._toolName) return false
  if (PRUNABLE_READ_TOOLS.has(msg._toolName)) return true
  if (msg._toolName === 'Bash' && isBashReadOnly(msg)) return true
  return false
}

function isBashReadOnly(msg: UserMessage): boolean {
  const command = msg._toolInput?.command as string | undefined
  if (!command) return false
  const trimmed = command.trim()
  return READ_ONLY_BASH_PREFIXES.some(prefix => trimmed.startsWith(prefix))
}

/**
 * Extracts the file path from a tool result message, if applicable.
 * Used for deduplicating file reads.
 */
function extractFilePath(msg: UserMessage): string | null {
  if (!msg._toolInput) return null
  return (msg._toolInput.file_path as string)
    ?? (msg._toolInput.path as string)
    ?? null
}

/**
 * Replaces the tool_result content in a UserMessage with new text.
 * Preserves the message structure and tool_use_id linkage.
 */
function replaceToolResultContent(msg: UserMessage, newContent: string): UserMessage {
  const content = msg.message.content
  if (!Array.isArray(content)) return msg

  const newMessageContent = content.map(block => {
    if (
      typeof block === 'object' &&
      'type' in block &&
      block.type === 'tool_result'
    ) {
      return { ...block, content: newContent }
    }
    return block
  })

  return {
    ...msg,
    message: { ...msg.message, content: newMessageContent },
  }
}

/**
 * Phase 1: Deduplicate file reads.
 *
 * If the same file was read multiple times, only the latest read keeps
 * its full content. Earlier reads are replaced with a pointer message.
 *
 * This is free (no API cost) and zero-risk — the model can always
 * re-read the file if it needs the content.
 */
export function deduplicateMessages(messages: Message[]): Message[] {
  // Track latest read index for each file path (supersedable tools only)
  const latestReadByPath = new Map<string, number>()

  // First pass (reverse): find the latest read index for each file
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.type !== 'user') continue
    if (!msg._toolName || !SUPERSEDABLE_TOOLS.has(msg._toolName)) continue
    if (msg._pruneState === 'deduped') continue

    const filePath = extractFilePath(msg)
    if (!filePath) continue

    if (!latestReadByPath.has(filePath)) {
      latestReadByPath.set(filePath, i)
    }
  }

  // Second pass: replace older reads with a pointer
  return messages.map((msg, i) => {
    if (msg.type !== 'user') return msg
    if (!msg._toolName || !SUPERSEDABLE_TOOLS.has(msg._toolName)) return msg
    if (msg._pruneState === 'deduped') return msg

    const filePath = extractFilePath(msg)
    if (!filePath) return msg

    const latestIndex = latestReadByPath.get(filePath)
    if (latestIndex !== undefined && i < latestIndex) {
      const replaced = replaceToolResultContent(msg,
        `[File ${filePath} was read again later in the conversation — see that result instead. Re-read with View tool if needed.]`
      )
      replaced._pruneState = 'deduped'
      return replaced
    }
    return msg
  })
}

/**
 * Main pruning pipeline.
 *
 * Phase 1 (current): Deduplication only.
 * Phase 2 (future): Add truncation for warm-tier messages.
 * Phase 3 (future): Add Haiku summarization for cold-tier messages.
 */
export function pruneMessages(
  messages: Message[],
  _currentTurn: number,
): Message[] {
  // Phase 1: Deduplicate file reads (free, zero-risk)
  return deduplicateMessages(messages)
}
