import { useEffect, useRef, useState } from 'react'
import { searchStations } from '../lib/api'
import { formatDuration } from '../lib/format'
import type { Station } from '../lib/types'
import { useApp } from '../state/store'
import { JourneyLegs, JourneySummary, LineBadges } from './JourneyCard'

function StartCityBox() {
  const origin = useApp((s) => s.origin)
  const setOrigin = useApp((s) => s.setOrigin)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Station[]>([])
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [failed, setFailed] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const q = query.trim()
    if (q.length < 2 || q === origin?.name) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      searchStations(q, controller.signal)
        .then((stations) => {
          setResults(stations)
          setActive(0)
          setFailed(false)
          setOpen(true)
        })
        .catch((e) => {
          if (e.name !== 'AbortError') setFailed(true)
        })
    }, 250)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [query, origin])

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  // results of an earlier, longer query must not linger under a short one
  const options = query.trim().length >= 2 ? results : []

  const choose = (station: Station) => {
    setQuery(station.name)
    setOpen(false)
    setResults([])
    void setOrigin(station)
  }

  return (
    <div className="start-box" ref={boxRef}>
      <label htmlFor="start-city" className="eyebrow">
        Start city
      </label>
      <div className="start-box__field">
        <input
          id="start-city"
          type="text"
          autoComplete="off"
          placeholder="Where are you starting from?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => options.length > 0 && setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              setActive((a) => Math.min(a + 1, options.length - 1))
            } else if (e.key === 'ArrowUp') {
              e.preventDefault()
              setActive((a) => Math.max(a - 1, 0))
            } else if (e.key === 'Enter' && open && options[active]) {
              choose(options[active])
            } else if (e.key === 'Escape') {
              setOpen(false)
            }
          }}
          role="combobox"
          aria-expanded={open}
          aria-controls="start-city-options"
        />
        {origin && (
          <button
            type="button"
            className="icon-button"
            aria-label="Clear start city"
            onClick={() => {
              setQuery('')
              void setOrigin(null)
            }}
          >
            ×
          </button>
        )}
      </div>
      {failed && <p className="note note--bad">Station search is unavailable right now.</p>}
      {open && options.length > 0 && (
        <ul className="options" id="start-city-options" role="listbox">
          {options.map((station, i) => (
            <li
              key={station.id}
              role="option"
              aria-selected={i === active}
              className={i === active ? 'is-active' : ''}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault()
                choose(station)
              }}
            >
              {station.name}
              {station.country && station.country !== 'de' && (
                <span className="muted"> · {station.country.toUpperCase()}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function PopularList() {
  const popular = useApp((s) => s.popular)
  const status = useApp((s) => s.popularStatus)
  const routes = useApp((s) => s.routes)
  const selected = useApp((s) => s.selected)
  const destinations = useApp((s) => s.destinations)
  const selectDestination = useApp((s) => s.selectDestination)

  if (status === 'loading') return <p className="muted pad">Finding destinations…</p>
  if (status === 'error') return <p className="note note--bad pad">Could not load destinations. Is the backend running?</p>
  if (status !== 'ready') return null

  // a destination picked on the map that is not among the nearest ones
  const extra =
    selected && !popular.some((p) => p.destination.name === selected)
      ? destinations.find((d) => d.name === selected)
      : undefined
  const items = [...(extra ? [{ destination: extra, distance_km: null }] : []), ...popular]

  return (
    <div className="popular">
      <div className="eyebrow pad">Popular from here, regional trains only</div>
      <ul>
        {items.map(({ destination, distance_km }) => {
          const route = routes[destination.name]
          const isSelected = selected === destination.name
          return (
            <li key={destination.name} className={isSelected ? 'card is-selected' : 'card'}>
              <button
                type="button"
                className="card__button"
                aria-expanded={isSelected}
                onClick={() => selectDestination(isSelected ? null : destination)}
              >
                <div className="card__top">
                  <span className="card__name">{destination.name}</span>
                  <span className="card__time">
                    {route?.status === 'ok' && formatDuration(route.journey.duration_min)}
                    {route?.status === 'loading' && <span className="spinner" aria-label="Loading" />}
                    {route?.status === 'none' && <span className="muted">no route</span>}
                    {route?.status === 'error' && <span className="muted">retry</span>}
                  </span>
                </div>
                <div className="muted card__meta">
                  {destination.region}
                  {distance_km != null && ` · ${distance_km} km away`}
                </div>
                {route?.status === 'ok' && <LineBadges journey={route.journey} />}
              </button>
              {isSelected && (
                <div className="card__detail">
                  <p className="card__blurb">{destination.blurb}</p>
                  {route?.status === 'ok' && (
                    <>
                      <JourneySummary journey={route.journey} />
                      <JourneyLegs journey={route.journey} />
                    </>
                  )}
                  {route?.status === 'loading' && <p className="muted">Checking the timetable…</p>}
                  {route?.status === 'none' && (
                    <p className="note">No connection covered by the ticket was found for the next departure window.</p>
                  )}
                  {route?.status === 'error' && <p className="note note--bad">The timetable lookup failed. Select again to retry.</p>}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default function StartPanel() {
  const origin = useApp((s) => s.origin)
  return (
    <aside className="panel start-panel">
      <header className="brand">
        <span className="brand__mark" aria-hidden="true" />
        <div>
          <h1>D-Ticket Trip Planner</h1>
          <p className="muted">Routes that stay inside the Deutschlandticket</p>
        </div>
      </header>
      <StartCityBox />
      {origin ? (
        <PopularList />
      ) : (
        <p className="muted pad hint">
          Pick a start city to see where regional trains can take you, or ask the assistant below.
        </p>
      )}
    </aside>
  )
}
