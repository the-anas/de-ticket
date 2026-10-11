import { formatDay, formatDuration, formatTime, legLabel, modeColor, rideLegs } from '../lib/format'
import type { Journey } from '../lib/types'

/** Line badges in travel order, e.g. RE5 › S3. */
export function LineBadges({ journey }: { journey: Journey }) {
  return (
    <span className="badges">
      {rideLegs(journey.legs).map((leg, i) => (
        <span key={i} className="badge" style={{ borderColor: modeColor(leg.mode), color: modeColor(leg.mode) }}>
          {legLabel(leg)}
        </span>
      ))}
    </span>
  )
}

export function JourneySummary({ journey }: { journey: Journey }) {
  return (
    <div className="journey-summary">
      <span className="journey-summary__times">
        {formatTime(journey.departure)} – {formatTime(journey.arrival)}
      </span>
      <span className="muted">
        {formatDuration(journey.duration_min)} · {journey.transfers === 0 ? 'direct' : `${journey.transfers} change${journey.transfers > 1 ? 's' : ''}`}
      </span>
    </div>
  )
}

export function JourneyLegs({ journey }: { journey: Journey }) {
  return (
    <div className="legs">
      <div className="muted legs__day">{formatDay(journey.departure)}</div>
      {rideLegs(journey.legs).map((leg, i) => (
        <div key={i} className="leg" style={{ borderLeftColor: modeColor(leg.mode) }}>
          <div className="leg__head">
            <span className="badge" style={{ borderColor: modeColor(leg.mode), color: modeColor(leg.mode) }}>
              {legLabel(leg)}
            </span>
            {leg.headsign && <span className="muted">towards {leg.headsign}</span>}
          </div>
          <div className="leg__row">
            <span className="leg__time">{formatTime(leg.departure)}</span>
            <span>{leg.origin.name}</span>
            {leg.origin.track && <span className="muted">pl. {leg.origin.track}</span>}
          </div>
          <div className="leg__row">
            <span className="leg__time">{formatTime(leg.arrival)}</span>
            <span>{leg.destination.name}</span>
            {leg.destination.track && <span className="muted">pl. {leg.destination.track}</span>}
          </div>
        </div>
      ))}
      {journey.dticket_notes.map((note) => (
        <p key={note} className={journey.dticket_valid ? 'note' : 'note note--bad'}>
          {note}
        </p>
      ))}
    </div>
  )
}
