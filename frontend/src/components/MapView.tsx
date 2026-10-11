import type { FeatureCollection, MultiPolygon, Feature } from 'geojson'
import { setWorkerUrl, type Map as MapLibreMap } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useMemo, useRef, useState } from 'react'
import Map, { AttributionControl, Layer, Marker, NavigationControl, Source, type MapRef } from 'react-map-gl/maplibre'
import { modeColor } from '../lib/format'
import { useApp } from '../state/store'

// MapLibre looks for its worker next to its own module, which no longer holds
// once Vite has bundled the library, so hand it the bundled worker explicitly.
setWorkerUrl(workerUrl)

const STYLE_URL = 'https://tiles.openfreemap.org/styles/dark'
const GERMANY_VIEW: [number, number, number, number] = [5.6, 47.1, 15.3, 55.2]
const MAX_BOUNDS: [number, number, number, number] = [-2, 43.5, 23, 58.5]
const RAIL_COLOR = '#f2b84b'

/** The base style is a road map; turn it into one where railways lead. */
function emphasiseRail(map: MapLibreMap) {
  for (const layer of map.getStyle().layers) {
    if (layer.type !== 'line' && layer.type !== 'symbol') continue
    if (layer.id.startsWith('railway')) {
      if (layer.id.includes('dashline')) {
        map.setLayoutProperty(layer.id, 'visibility', 'none')
      } else {
        const transit = layer.id.includes('transit') || layer.id.includes('minor')
        // the base style only shows railways from zoom 13; bring them in as
        // early as the tiles carry them
        map.setLayerZoomRange(layer.id, transit ? 11 : 6, 24)
        map.setPaintProperty(layer.id, 'line-color', RAIL_COLOR)
        map.setPaintProperty(layer.id, 'line-opacity', transit ? 0.35 : 0.7)
        map.setPaintProperty(layer.id, 'line-width', [
          'interpolate', ['linear'], ['zoom'], 6, 0.6, 10, transit ? 1 : 1.6, 14, transit ? 1.6 : 2.6,
        ])
      }
    } else if (layer.id.startsWith('highway_name') || layer.id.startsWith('road_oneway')) {
      map.setLayoutProperty(layer.id, 'visibility', 'none')
    } else if (layer.id.startsWith('highway') && layer.type === 'line') {
      map.setPaintProperty(layer.id, 'line-opacity', 0.3)
    }
  }
}

/** Leaves room for the start panel (wide screens) and the prompt bar. */
function overviewPadding() {
  const wide = window.innerWidth > 900
  return { top: wide ? 30 : 150, bottom: 130, left: wide ? 400 : 20, right: wide ? 60 : 20 }
}

/** Everything except Germany, used to dim the neighbouring countries. */
function outsideMask(germany: Feature<MultiPolygon>): Feature {
  const world = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]
  const holes = germany.geometry.coordinates.map((polygon) => polygon[0])
  return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [world, ...holes] } }
}

export default function MapView() {
  const mapRef = useRef<MapRef>(null)
  const [germany, setGermany] = useState<Feature<MultiPolygon> | null>(null)
  const destinations = useApp((s) => s.destinations)
  const origin = useApp((s) => s.origin)
  const popular = useApp((s) => s.popular)
  const selected = useApp((s) => s.selected)
  const drawn = useApp((s) => s.drawn)
  const selectDestination = useApp((s) => s.selectDestination)

  useEffect(() => {
    fetch('/germany.geojson')
      .then((r) => r.json())
      .then(setGermany)
      .catch(() => undefined)
  }, [])

  const legs = useMemo<FeatureCollection>(
    () => ({
      type: 'FeatureCollection',
      features: drawn.journeys.flatMap((journey) =>
        journey.legs
          .filter((leg) => leg.geometry.length > 1)
          .map((leg) => ({
            type: 'Feature' as const,
            properties: { color: modeColor(leg.mode), walk: leg.mode === 'WALK' },
            geometry: { type: 'LineString' as const, coordinates: leg.geometry },
          })),
      ),
    }),
    [drawn.journeys],
  )

  const stops = useMemo<FeatureCollection>(() => {
    const seen = new Set<string>()
    const features: FeatureCollection['features'] = []
    const add = (name: string, lon: number, lat: number) => {
      if (!name || seen.has(name)) return
      seen.add(name)
      features.push({ type: 'Feature', properties: { name }, geometry: { type: 'Point', coordinates: [lon, lat] } })
    }
    for (const journey of drawn.journeys) {
      for (const leg of journey.legs) {
        if (leg.mode === 'WALK') continue
        add(leg.origin.name, leg.origin.lon, leg.origin.lat)
        add(leg.destination.name, leg.destination.lon, leg.destination.lat)
      }
    }
    for (const place of drawn.places) add(place.name, place.lon, place.lat)
    return { type: 'FeatureCollection', features }
  }, [drawn])

  // Frame whatever was just drawn, keeping clear of the side panel and prompt bar.
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const points: [number, number][] = [
      ...drawn.journeys.flatMap((j) => j.legs.flatMap((l) => l.geometry)),
      ...drawn.places.map((p) => [p.lon, p.lat] as [number, number]),
    ]
    if (points.length === 0) return
    const lons = points.map((p) => p[0])
    const lats = points.map((p) => p[1])
    const wide = window.innerWidth > 900
    map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { padding: { top: 90, bottom: 170, left: wide ? 430 : 40, right: wide ? 80 : 40 }, maxZoom: 11, duration: 900 },
    )
  }, [drawn])

  useEffect(() => {
    if (origin && drawn.journeys.length === 0) {
      mapRef.current?.flyTo({
        center: [origin.lon, origin.lat],
        zoom: 7.2,
        duration: 900,
        padding: overviewPadding(),
      })
    }
    // only when the origin changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [origin])

  const nearby = useMemo(() => new Set(popular.map((p) => p.destination.name)), [popular])

  return (
    <Map
      ref={mapRef}
      mapStyle={STYLE_URL}
      initialViewState={{ bounds: GERMANY_VIEW, fitBoundsOptions: { padding: overviewPadding() } }}
      maxBounds={MAX_BOUNDS}
      minZoom={4.2}
      maxZoom={16}
      dragRotate={false}
      pitchWithRotate={false}
      attributionControl={false}
      onLoad={(e) => emphasiseRail(e.target)}
      style={{ position: 'absolute', inset: 0 }}
    >
      <NavigationControl position="top-right" showCompass={false} />
      {/* bottom-left is the one corner the panels leave free */}
      <AttributionControl position="bottom-left" compact />

      {germany && (
        <>
          <Source id="mask" type="geojson" data={outsideMask(germany)}>
            <Layer id="mask-fill" type="fill" paint={{ 'fill-color': '#05080f', 'fill-opacity': 0.55 }} />
          </Source>
          <Source id="germany" type="geojson" data={germany}>
            <Layer
              id="germany-glow"
              type="line"
              paint={{ 'line-color': RAIL_COLOR, 'line-width': 6, 'line-blur': 8, 'line-opacity': 0.35 }}
            />
            <Layer id="germany-line" type="line" paint={{ 'line-color': RAIL_COLOR, 'line-width': 1, 'line-opacity': 0.7 }} />
          </Source>
        </>
      )}

      <Source id="legs" type="geojson" data={legs}>
        <Layer
          id="legs-casing"
          type="line"
          filter={['!', ['get', 'walk']]}
          layout={{ 'line-cap': 'round', 'line-join': 'round' }}
          paint={{ 'line-color': '#05080f', 'line-width': 8, 'line-opacity': 0.9 }}
        />
        <Layer
          id="legs-line"
          type="line"
          filter={['!', ['get', 'walk']]}
          layout={{ 'line-cap': 'round', 'line-join': 'round' }}
          paint={{ 'line-color': ['get', 'color'], 'line-width': 4.5 }}
        />
        <Layer
          id="legs-walk"
          type="line"
          filter={['get', 'walk']}
          paint={{ 'line-color': ['get', 'color'], 'line-width': 2.5, 'line-dasharray': [1, 2] }}
        />
      </Source>

      <Source id="stops" type="geojson" data={stops}>
        <Layer
          id="stops-dot"
          type="circle"
          paint={{ 'circle-radius': 5, 'circle-color': '#ffffff', 'circle-stroke-color': '#05080f', 'circle-stroke-width': 2 }}
        />
        <Layer
          id="stops-label"
          type="symbol"
          layout={{
            'text-field': ['get', 'name'],
            'text-font': ['Noto Sans Regular'],
            'text-size': 12,
            'text-offset': [0, 1.1],
            'text-anchor': 'top',
            'text-optional': true,
          }}
          paint={{ 'text-color': '#ffffff', 'text-halo-color': '#05080f', 'text-halo-width': 1.6 }}
        />
      </Source>

      {destinations.map((dest) => (
        <Marker key={dest.name} longitude={dest.lon} latitude={dest.lat} anchor="center">
          <button
            type="button"
            className={[
              'dest-marker',
              nearby.has(dest.name) ? 'is-nearby' : '',
              selected === dest.name ? 'is-selected' : '',
            ].join(' ')}
            title={origin ? `${dest.name}: show the route` : `${dest.name}: ${dest.blurb}`}
            aria-label={dest.name}
            onClick={() => selectDestination(dest)}
          >
            <span className="dest-marker__label">{dest.name}</span>
          </button>
        </Marker>
      ))}

      {origin && (
        <Marker longitude={origin.lon} latitude={origin.lat} anchor="center">
          <div className="origin-marker" title={origin.name} />
        </Marker>
      )}
    </Map>
  )
}
