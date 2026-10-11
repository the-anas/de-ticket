import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import type { Journey } from '../lib/types'
import { useApp, type ChatMessage } from '../state/store'
import { JourneySummary, LineBadges } from './JourneyCard'

const SUGGESTIONS = [
  'Plan a day trip for Saturday',
  'Can I get to Salzburg with the ticket?',
  'Which routes get crowded at weekends?',
]

function ChatJourneys({ journeys }: { journeys: Journey[] }) {
  const showJourneys = useApp((s) => s.showJourneys)
  return (
    <div className="chat-journeys">
      {journeys.map((journey) => (
        <button key={journey.id} type="button" className="chat-journey" onClick={() => showJourneys([journey])}>
          <JourneySummary journey={journey} />
          <LineBadges journey={journey} />
        </button>
      ))}
    </div>
  )
}

function Message({ message }: { message: ChatMessage }) {
  if (message.role === 'user') return <div className="msg msg--user">{message.text}</div>
  return (
    <div className="msg msg--assistant">
      {message.text && <Markdown>{message.text}</Markdown>}
      {message.activity && (
        <p className="activity">
          <span className="spinner" aria-hidden="true" /> {message.activity}
        </p>
      )}
      {message.journeys && message.journeys.length > 0 && <ChatJourneys journeys={message.journeys} />}
      {message.error && <p className="note note--bad">{message.error}</p>}
      {message.sources && message.sources.length > 0 && !message.activity && (
        <p className="sources">
          Notes consulted: {[...new Set(message.sources.slice(0, 4).map((s) => `${s.document} – ${s.section}`))].join('; ')}
        </p>
      )}
    </div>
  )
}

export default function ChatBar() {
  const messages = useApp((s) => s.messages)
  const busy = useApp((s) => s.chatBusy)
  const open = useApp((s) => s.chatOpen)
  const setOpen = useApp((s) => s.setChatOpen)
  const sendChat = useApp((s) => s.sendChat)
  const clearChat = useApp((s) => s.clearChat)
  const [draft, setDraft] = useState('')
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [messages, open])

  const submit = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || busy) return
    setDraft('')
    void sendChat(trimmed)
  }

  return (
    <section className="chat" aria-label="Trip assistant">
      {open && messages.length > 0 && (
        <div className="panel chat__log-wrap">
          <div className="chat__log-head">
            <span className="eyebrow">Trip assistant</span>
            <span>
              <button type="button" className="text-button" onClick={clearChat}>
                New chat
              </button>
              <button type="button" className="icon-button" aria-label="Hide conversation" onClick={() => setOpen(false)}>
                ×
              </button>
            </span>
          </div>
          <div className="chat__log" ref={logRef} aria-live="polite">
            {messages.map((message, i) => (
              <Message key={i} message={message} />
            ))}
          </div>
        </div>
      )}

      {messages.length === 0 && (
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => submit(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        className="panel chat__bar"
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
      >
        {!open && messages.length > 0 && (
          <button type="button" className="text-button" onClick={() => setOpen(true)}>
            Show chat
          </button>
        )}
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask for a trip plan, a destination idea, or what the ticket covers…"
          aria-label="Message the trip assistant"
        />
        <button type="submit" className="send" disabled={busy || !draft.trim()}>
          {busy ? 'Working…' : 'Ask'}
        </button>
      </form>
    </section>
  )
}
