import { Message, UserMessage } from '../query.js'
import { queryHaiku } from '../services/claude.js'

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

// --- Phase 2: Warm-tier truncation ---

const HOT_WINDOW = parseInt(process.env.PRUNE_HOT_WINDOW ?? '10')
const HEAD_LINES = 50
const TAIL_LINES = 50
const TRUNCATION_THRESHOLD_CHARS = 5000

/**
 * Extracts the text content from a tool_result block in a UserMessage.
 */
function getToolResultText(msg: UserMessage): string | null {
  const content = msg.message.content
  if (!Array.isArray(content)) return null

  for (const block of content) {
    if (
      typeof block === 'object' &&
      'type' in block &&
      block.type === 'tool_result'
    ) {
      if (typeof block.content === 'string') return block.content
      // Content can be an array of text/image blocks
      if (Array.isArray(block.content)) {
        return block.content
          .filter((b: { type: string }) => b.type === 'text')
          .map((b: { type: string; text: string }) => b.text)
          .join('\n')
      }
    }
  }
  return null
}

/**
 * Phase 2: Truncate warm-tier tool results.
 *
 * Messages older than HOT_WINDOW turns that contain large read-only
 * tool results are truncated to head/tail snippets. The model can
 * always re-read the file or re-run the command for full content.
 *
 * This is free (no API cost) and low-risk.
 */
export function truncateWarmMessages(
  messages: Message[],
  currentTurn: number,
): Message[] {
  return messages.map(msg => {
    if (msg.type !== 'user') return msg
    if (!isPrunableMessage(msg)) return msg
    // Don't re-truncate already pruned messages
    if (msg._pruneState === 'deduped' || msg._pruneState === 'truncated' || msg._pruneState === 'summarized') return msg

    const age = currentTurn - (msg._turnCreated ?? 0)
    if (age <= HOT_WINDOW) return msg

    const text = getToolResultText(msg)
    if (!text || text.length < TRUNCATION_THRESHOLD_CHARS) return msg

    const lines = text.split('\n')
    if (lines.length <= HEAD_LINES + TAIL_LINES + 5) return msg

    const truncatedLineCount = lines.length - HEAD_LINES - TAIL_LINES
    const toolLabel = msg._toolName ?? 'tool'
    const filePath = extractFilePath(msg)
    const locationHint = filePath ? ` for ${filePath}` : ''

    const truncated = [
      ...lines.slice(0, HEAD_LINES),
      '',
      `[... ${truncatedLineCount} lines truncated from ${toolLabel} result${locationHint} — re-read file or re-run command for full content ...]`,
      '',
      ...lines.slice(-TAIL_LINES),
    ].join('\n')

    const replaced = replaceToolResultContent(msg, truncated)
    replaced._pruneState = 'truncated'
    return replaced
  })
}

// --- Phase 3: Cold-tier Haiku summarization ---

const COLD_WINDOW = parseInt(process.env.PRUNE_COLD_WINDOW ?? '20')
const SUMMARY_THRESHOLD_CHARS = 2000
const SUMMARY_MAX_CONCURRENCY = 5
const ENABLE_SUMMARIZATION = process.env.PRUNE_SUMMARIZE !== 'false'

const SUMMARY_SYSTEM_PROMPT = [
  `You are a tool result summarizer for a coding assistant. Given a tool result, produce a concise summary that preserves:
- File paths and key line numbers referenced
- Structure: exports, classes, functions, key types found
- Important values: error messages, test pass/fail counts, config values
- Any TODOs, FIXMEs, or warnings
Keep the summary under 200 tokens. Be factual, not interpretive.`,
]

/**
 * Phase 3: Summarize cold-tier tool results using Haiku.
 *
 * Messages older than COLD_WINDOW turns that contain large read-only
 * tool results are replaced with a Haiku-generated summary.
 * The model retains awareness of what was seen (key facts, line numbers,
 * structure) but at ~95% fewer tokens.
 *
 * Cost: ~700 Haiku tokens per summary (~$0.0005 each).
 * Fallback: if Haiku fails, keeps the truncated version from Phase 2.
 */
export async function summarizeColdMessages(
  messages: Message[],
  currentTurn: number,
  signal?: AbortSignal,
): Promise<Message[]> {
  // Collect indices of messages that need summarization
  const candidates: { index: number; text: string }[] = []

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.type !== 'user') continue
    if (!isPrunableMessage(msg)) continue
    if (msg._pruneState === 'summarized' || msg._pruneState === 'deduped') continue

    const age = currentTurn - (msg._turnCreated ?? 0)
    if (age <= COLD_WINDOW) continue

    const text = getToolResultText(msg)
    if (!text || text.length < SUMMARY_THRESHOLD_CHARS) continue

    candidates.push({ index: i, text })
  }

  if (candidates.length === 0) return messages

  const results = [...messages]

  // Process in batches to limit concurrency
  for (let batchStart = 0; batchStart < candidates.length; batchStart += SUMMARY_MAX_CONCURRENCY) {
    const batch = candidates.slice(batchStart, batchStart + SUMMARY_MAX_CONCURRENCY)

    const summaries = await Promise.all(
      batch.map(async ({ index, text }) => {
        const msg = messages[index] as UserMessage
        const toolName = msg._toolName ?? 'unknown'
        const filePath = extractFilePath(msg)
        const command = msg._toolInput?.command as string | undefined

        // Build context header for the summarizer
        let contextHeader = `Tool: ${toolName}`
        if (filePath) contextHeader += `\nFile: ${filePath}`
        if (command) contextHeader += `\nCommand: ${command}`

        try {
          const response = await queryHaiku({
            systemPrompt: SUMMARY_SYSTEM_PROMPT,
            userPrompt: `${contextHeader}\n\nResult (first 4000 chars):\n${text.slice(0, 4000)}`,
            signal,
          })
          const summary = response.message.content[0]?.type === 'text'
            ? response.message.content[0].text
            : null
          return { index, summary }
        } catch {
          // On failure, keep whatever state the message is in (likely truncated from Phase 2)
          return { index, summary: null }
        }
      }),
    )

    for (const { index, summary } of summaries) {
      if (!summary) continue // keep existing content on failure

      const msg = results[index] as UserMessage
      const toolName = msg._toolName ?? 'tool'
      const filePath = extractFilePath(msg)
      const command = (msg._toolInput?.command as string | undefined)

      let label = `${toolName} result`
      if (filePath) label += ` for ${filePath}`
      else if (command) label += `: ${command.slice(0, 80)}`

      const replaced = replaceToolResultContent(msg,
        `[Summarized ${label}]\n${summary}\n[Re-read file or re-run command for full content]`,
      )
      replaced._pruneState = 'summarized'
      results[index] = replaced
    }
  }

  return results
}

/**
 * Main pruning pipeline.
 *
 * Phase 1: Deduplication — replace older duplicate file reads with pointers.
 * Phase 2: Truncation — trim large warm-tier results to head/tail snippets.
 * Phase 3: Summarization — replace cold-tier results with Haiku summaries.
 *
 * Operates on a copy of messages for the API payload. The original
 * messages array (stored in REPL state and logs) is never modified.
 */
export async function pruneMessages(
  messages: Message[],
  currentTurn: number,
  signal?: AbortSignal,
): Promise<Message[]> {
  // Phase 1: Deduplicate file reads (free, zero-risk)
  let pruned = deduplicateMessages(messages)

  // Phase 2: Truncate warm-tier results (free, low-risk)
  pruned = truncateWarmMessages(pruned, currentTurn)

  // Phase 3: Summarize cold-tier results (low Haiku cost, medium risk)
  if (ENABLE_SUMMARIZATION) {
    pruned = await summarizeColdMessages(pruned, currentTurn, signal)
  }

  return pruned
}
