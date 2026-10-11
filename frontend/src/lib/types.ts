export interface Station {
  id: string
  name: string
  lat: number
  lon: number
  country: string | null
  region?: string | null
}

export interface Stop {
  name: string
  lat: number
  lon: number
  time: string | null
  track: string | null
}

export interface Leg {
  mode: string
  line: string | null
  operator: string | null
  headsign: string | null
  origin: Stop
  destination: Stop
  departure: string
  arrival: string
  duration_min: number
  geometry: [number, number][]
}

export interface Journey {
  id: string
  departure: string
  arrival: string
  duration_min: number
  transfers: number
  legs: Leg[]
  dticket_valid: boolean
  dticket_notes: string[]
}

export interface Destination {
  name: string
  station: string
  lat: number
  lon: number
  region: string
  tags: string[]
  blurb: string
}

export interface PopularRoute {
  destination: Destination
  distance_km: number
  /** counted from the search time, so it includes the wait for the first train */
  approx_min: number
  transfers: number
}

export interface DocSource {
  source: string
  document: string
  section: string
}

export type ChatEvent =
  | { type: 'text'; text: string }
  | { type: 'tool'; name: string; input: Record<string, unknown> }
  | { type: 'map'; journeys: Journey[]; places: Station[] }
  | { type: 'sources'; sources: DocSource[] }
  | { type: 'error'; message: string }
  | { type: 'done' }
