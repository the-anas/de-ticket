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

  // The origin can also be set by clicking a city on the map; mirror it here.
  const [shownOrigin, setShownOrigin] = useState(origin)
  if (origin !== shownOrigin) {
    setShownOrigin(origin)
    setQuery(origin?.name ?? '')
    setOpen(false)
  }

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

  // Escape anywhere on the page clears the selected start city.
  useEffect(() => {
    if (!origin) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') void setOrigin(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [origin, setOrigin])

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
    setResults([])
    void setOrigin(station)
  }

  return (
    <div className="start-box" ref={boxRef}>
      <label htmlFor="start-city" className="eyebrow">
        Pick your starting location
      </label>
      <div className="start-box__field">
        <input
          id="start-city"
          type="text"
          autoComplete="off"
          placeholder="Where are you starting from?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={(e) => {
            // typing then replaces the current city instead of appending to it
            e.target.select()
            if (options.length > 0) setOpen(true)
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              setActive((a) => Math.min(a + 1, options.length - 1))
            } else if (e.key === 'ArrowUp') {
              e.preventDefault()
              setActive((a) => Math.max(a - 1, 0))
            } else if (e.key === 'Enter' && open && options[active]) {
              choose(options[active])
            } else if (e.key === 'Escape' && open && options.length > 0) {
              // first Escape only closes the list; otherwise it falls through
              // to the page-wide handler that clears the start city
              setOpen(false)
              e.stopPropagation()
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
            onClick={() => void setOrigin(null)}
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
              <span className="muted">
                {station.region && ` · ${station.region}`}
                {station.country && station.country !== 'de' && ` · ${station.country.toUpperCase()}`}
              </span>
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

  // Destinations the ticket cannot reach are left out rather than listed as dead ends.
  const reachable = popular.filter((p) => routes[p.destination.name]?.status !== 'none')
  // A city clicked on the map that is not among the suggestions gets its own card on top.
  const picked =
    selected && !popular.some((p) => p.destination.name === selected)
      ? destinations.find((d) => d.name === selected)
      : undefined
  const pickedUnreachable = picked && routes[picked.name]?.status === 'none'
  const items = [
    ...(picked && !pickedUnreachable ? [{ destination: picked, distance_km: null, approx_min: null }] : []),
    ...reachable,
  ]
  if (items.length === 0) {
    return (
      <p className="note pad-box">
        {pickedUnreachable
          ? `There is no possible route with the Deutschlandticket to ${picked.name}.`
          : 'There are no possible routes with the Deutschlandticket from here.'}
      </p>
    )
  }

  return (
    <div className="popular">
      <div className="eyebrow pad">Popular from here, regional trains only</div>
      {pickedUnreachable && (
        <p className="note pad-box">There is no possible route with the Deutschlandticket to {picked.name}.</p>
      )}
      <ul>
        {items.map(({ destination, distance_km, approx_min }) => {
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
                    {!route && approx_min != null && (
                      <span title="From now, including the wait for the next departure">
                        ≈ {formatDuration(approx_min)}
                      </span>
                    )}
                    {route?.status === 'loading' && <span className="spinner" aria-label="Loading" />}
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
        <img className="brand__logo" src="/D-Ticket_Logo.svg" alt="" />
        <h1>D-Ticket Trip Planner</h1>
      </header>
      <StartCityBox />
      {origin && <PopularList />}
    </aside>
  )
}
