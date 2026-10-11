import { useEffect } from 'react'
import ChatBar from './components/ChatBar'
import MapView from './components/MapView'
import StartPanel from './components/StartPanel'
import { useApp } from './state/store'

export default function App() {
  const loadDestinations = useApp((s) => s.loadDestinations)
  useEffect(() => {
    void loadDestinations()
  }, [loadDestinations])

  return (
    <div className="app">
      <MapView />
      <StartPanel />
      <ChatBar />
    </div>
  )
}
