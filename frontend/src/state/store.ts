import { create } from 'zustand'
import { fetchDestinations, fetchJourneys, fetchPopular, streamChat } from '../lib/api'
import type { Destination, DocSource, Journey, PopularRoute, Station } from '../lib/types'

export type RouteState =
  | { status: 'loading' }
  | { status: 'ok'; journey: Journey }
  | { status: 'none' }
  | { status: 'error' }

export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  // assistant only
  activity?: string | null
  journeys?: Journey[]
  sources?: DocSource[]
  error?: string
}

interface Drawn {
  journeys: Journey[]
  places: Station[]
}

interface AppState {
  destinations: Destination[]
  origin: Station | null
  popular: PopularRoute[]
  popularStatus: 'idle' | 'loading' | 'ready' | 'error'
  /** journey per destination name, for the current origin */
  routes: Record<string, RouteState>
  selected: string | null
  drawn: Drawn

  messages: ChatMessage[]
  chatBusy: boolean
  chatOpen: boolean

  loadDestinations: () => Promise<void>
  setOrigin: (station: Station | null) => Promise<void>
  selectDestination: (dest: Destination | null) => void
  showJourneys: (journeys: Journey[], places?: Station[]) => void
  sendChat: (text: string) => Promise<void>
  setChatOpen: (open: boolean) => void
  clearChat: () => void
}

// Bumped whenever the origin changes, so slow responses for an old origin are dropped.
let originGeneration = 0
let chatAbort: AbortController | null = null

const ACTIVITY: Record<string, string> = {
  search_stations: 'Looking up stations…',
  find_journeys: 'Checking the timetable…',
  search_docs: 'Reading the travel notes…',
  show_on_map: 'Drawing on the map…',
}

export const useApp = create<AppState>((set, get) => {
  const loadRoute = async (dest: Destination, generation: number) => {
    const origin = get().origin
    if (!origin) return
    const existing = get().routes[dest.name]
    if (existing && existing.status !== 'error') return
    set((s) => ({ routes: { ...s.routes, [dest.name]: { status: 'loading' } } }))
    let result: RouteState
    try {
      const journeys = await fetchJourneys(origin, dest)
      const best = journeys.find((j) => j.dticket_valid)
      result = best ? { status: 'ok', journey: best } : { status: 'none' }
    } catch {
      result = { status: 'error' }
    }
    if (generation !== originGeneration) return
    set((s) => ({ routes: { ...s.routes, [dest.name]: result } }))
    if (get().selected === dest.name && result.status === 'ok') {
      set({ drawn: { journeys: [result.journey], places: [] } })
    }
  }

  const patchLast = (patch: (m: ChatMessage) => ChatMessage) =>
    set((s) => {
      const last = s.messages[s.messages.length - 1]
      // the chat may have been cleared while a reply was still streaming
      if (!last || last.role !== 'assistant') return s
      return { messages: [...s.messages.slice(0, -1), patch(last)] }
    })

  return {
    destinations: [],
    origin: null,
    popular: [],
    popularStatus: 'idle',
    routes: {},
    selected: null,
    drawn: { journeys: [], places: [] },
    messages: [],
    chatBusy: false,
    chatOpen: false,

    loadDestinations: async () => {
      try {
        set({ destinations: await fetchDestinations() })
      } catch {
        // the map still works without the curated markers
      }
    },

    setOrigin: async (station) => {
      const generation = ++originGeneration
      set({
        origin: station,
        popular: [],
        routes: {},
        selected: null,
        drawn: { journeys: [], places: [] },
        popularStatus: station ? 'loading' : 'idle',
      })
      if (!station) return
      let popular: PopularRoute[]
      try {
        popular = await fetchPopular(station)
      } catch {
        if (generation === originGeneration) set({ popularStatus: 'error' })
        return
      }
      if (generation !== originGeneration) return
      set({ popular, popularStatus: 'ready' })
      // The transit API allows only a slow sustained rate, so journey times
      // are fetched one after another and appear as they arrive.
      for (const route of popular) {
        if (generation !== originGeneration) return
        await loadRoute(route.destination, generation)
      }
    },

    selectDestination: (dest) => {
      if (!dest) {
        set({ selected: null, drawn: { journeys: [], places: [] } })
        return
      }
      set({ selected: dest.name })
      const route = get().routes[dest.name]
      if (route?.status === 'ok') {
        set({ drawn: { journeys: [route.journey], places: [] } })
      } else {
        set({ drawn: { journeys: [], places: [] } })
        if (get().origin) void loadRoute(dest, originGeneration)
      }
    },

    showJourneys: (journeys, places = []) => set({ selected: null, drawn: { journeys, places } }),

    sendChat: async (text) => {
      if (get().chatBusy) return
      const history = [
        ...get()
          .messages.filter((m) => m.text.trim())
          .map((m) => ({ role: m.role, content: m.text })),
        { role: 'user' as const, content: text },
      ]
      set((s) => ({
        chatBusy: true,
        chatOpen: true,
        messages: [
          ...s.messages,
          { role: 'user', text },
          { role: 'assistant', text: '', activity: 'Thinking…' },
        ],
      }))
      chatAbort = new AbortController()
      try {
        await streamChat(
          history,
          get().origin,
          (event) => {
            switch (event.type) {
              case 'text':
                patchLast((m) => ({ ...m, text: m.text + event.text, activity: null }))
                break
              case 'tool':
                patchLast((m) => ({ ...m, activity: ACTIVITY[event.name] ?? 'Working…' }))
                break
              case 'map':
                patchLast((m) => ({ ...m, journeys: event.journeys }))
                set({ selected: null, drawn: { journeys: event.journeys, places: event.places } })
                break
              case 'sources':
                patchLast((m) => {
                  const seen = new Set((m.sources ?? []).map((s) => s.source + s.section))
                  const fresh = event.sources.filter((s) => !seen.has(s.source + s.section))
                  return { ...m, sources: [...(m.sources ?? []), ...fresh] }
                })
                break
              case 'error':
                patchLast((m) => ({ ...m, error: event.message, activity: null }))
                break
              case 'done':
                patchLast((m) => ({ ...m, activity: null }))
                break
            }
          },
          chatAbort.signal,
        )
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          patchLast((m) => ({ ...m, error: 'Could not reach the assistant. Is the backend running?' }))
        }
      } finally {
        patchLast((m) => ({ ...m, activity: null }))
        set({ chatBusy: false })
      }
    },

    setChatOpen: (open) => set({ chatOpen: open }),

    clearChat: () => {
      chatAbort?.abort()
      set({ messages: [], chatOpen: false })
    },
  }
})
