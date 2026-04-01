import { useEffect, useRef } from 'react'
import { type Message } from '../query.js'
import { overwriteLog, getMessagesPath } from '../utils/log.js'

const LOG_DEBOUNCE_MS = 500

export function useLogMessages(
  messages: Message[],
  messageLogName: string,
  forkNumber: number,
): void {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const latestRef = useRef({ messages, messageLogName, forkNumber })
  latestRef.current = { messages, messageLogName, forkNumber }

  useEffect(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
    }
    timerRef.current = setTimeout(() => {
      const { messages: msgs, messageLogName: name, forkNumber: fork } = latestRef.current
      overwriteLog(
        getMessagesPath(name, fork, 0),
        msgs.filter(_ => _.type !== 'progress'),
      )
      timerRef.current = null
    }, LOG_DEBOUNCE_MS)

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [messages, messageLogName, forkNumber])

  // Flush on unmount to ensure final state is saved
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        const { messages: msgs, messageLogName: name, forkNumber: fork } = latestRef.current
        overwriteLog(
          getMessagesPath(name, fork, 0),
          msgs.filter(_ => _.type !== 'progress'),
        )
      }
    }
  }, [])
}
