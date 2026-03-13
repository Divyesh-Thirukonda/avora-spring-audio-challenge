import { useEffect, useState, useMemo, useCallback } from 'react'
import { useAudio } from './audio/useAudio'
import { Visualizer } from './visualizers/Visualizer'
import './App.css'

/**
 * Background color for the visualizer page.
 */
const BACKGROUND_COLOR = '#000'

function App() {
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  })

  const handleResize = useCallback(() => {
    setDimensions({
      width: window.innerWidth,
      height: window.innerHeight,
    })
  }, [])

  useEffect(() => {
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [handleResize])

  const analyserOptions = useMemo<AnalyserOptions>(() => ({
    fftSize: 2048,
    smoothingTimeConstant: 0.5,
    minDecibels: -100,
    maxDecibels: -30,
  }), [])

  const audioOptions = useMemo<MediaTrackConstraints>(() => ({
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: true,
  }), [])

  const { frequencyData, timeDomainData, isActive, start } = useAudio({
    analyser: analyserOptions,
    audio: audioOptions,
  })

  useEffect(() => {
    start()
  }, [start])

  return (
    <div className="app" style={{ backgroundColor: BACKGROUND_COLOR }}>
      <div className="visualizer-container" style={{ width: dimensions.width, height: dimensions.height }}>
        <Visualizer
          frequencyData={frequencyData}
          timeDomainData={timeDomainData}
          isActive={isActive}
          width={dimensions.width}
          height={dimensions.height}
        />
      </div>
    </div>
  )
}

export default App
