import type { ChatEvent, Destination, Journey, PopularRoute, Station } from './types'

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { signal })
  if (!res.ok) throw new Error(`${path} failed with ${res.status}`)
  return res.json() as Promise<T>
}

type Point = { lat: number; lon: number }

export const searchStations = (q: string, signal?: AbortSignal) =>
  get<Station[]>(`/api/stations?q=${encodeURIComponent(q)}`, signal)

export const fetchDestinations = () => get<Destination[]>('/api/destinations')

export const fetchPopular = (from: Point, signal?: AbortSignal) =>
  get<PopularRoute[]>(`/api/popular?lat=${from.lat}&lon=${from.lon}`, signal)

export const fetchJourneys = (from: Point, to: Point, signal?: AbortSignal) =>
  get<Journey[]>(`/api/journeys?from=${from.lat},${from.lon}&to=${to.lat},${to.lon}`, signal)

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

/** POSTs the conversation and calls onEvent for each server-sent event. */
export async function streamChat(
  messages: ChatTurn[],
  origin: Station | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      origin: origin && { name: origin.name, lat: origin.lat, lon: origin.lon },
    }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`Chat request failed with ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let end
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as ChatEvent)
      }
    }
  }
}
