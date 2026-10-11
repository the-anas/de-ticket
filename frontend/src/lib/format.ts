import type { Leg } from './types'

const clock = new Intl.DateTimeFormat('de-DE', {
  hour: '2-digit',
  minute: '2-digit',
  timeZone: 'Europe/Berlin',
})
const day = new Intl.DateTimeFormat('en-GB', {
  weekday: 'short',
  day: 'numeric',
  month: 'short',
  timeZone: 'Europe/Berlin',
})

export const formatTime = (iso: string) => clock.format(new Date(iso))
export const formatDay = (iso: string) => day.format(new Date(iso))

export function formatDuration(minutes: number): string {
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  if (h === 0) return `${m} min`
  return m === 0 ? `${h} h` : `${h} h ${m} min`
}

export const MODE_COLORS: Record<string, string> = {
  REGIONAL_RAIL: '#ff5d5d',
  REGIONAL_FAST_RAIL: '#ff5d5d',
  SUBURBAN: '#3ddc84',
  SUBWAY: '#4da3ff',
  TRAM: '#ff9f43',
  BUS: '#b79cff',
  FERRY: '#36cfc9',
  WALK: '#8b97ad',
}

export const modeColor = (mode: string) => MODE_COLORS[mode] ?? '#ffc53d'

const MODE_LABELS: Record<string, string> = {
  REGIONAL_RAIL: 'Regional train',
  REGIONAL_FAST_RAIL: 'Regional express',
  SUBURBAN: 'S-Bahn',
  SUBWAY: 'U-Bahn',
  TRAM: 'Tram',
  BUS: 'Bus',
  FERRY: 'Ferry',
  WALK: 'Walk',
}

export const legLabel = (leg: Leg) => leg.line ?? MODE_LABELS[leg.mode] ?? leg.mode

export const rideLegs = (legs: Leg[]) => legs.filter((l) => l.mode !== 'WALK')
