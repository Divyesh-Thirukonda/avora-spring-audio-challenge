import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import {
  CALIBRATION,
  adaptiveAlpha,
} from './calibration'

export interface VisualizerProps {
  frequencyData: React.RefObject<Uint8Array>
  timeDomainData: React.RefObject<Uint8Array>
  isActive: boolean
  width: number
  height: number
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeData {
  index: number
  gridX: number
  gridY: number
  gridZ: number
  home: THREE.Vector3
  position: THREE.Vector3
  activation: number
  traffic: number
  pulse: number
  baseSize: number
  freqBand: number        // which frequency group (0-9) this node listens to
  harmonicTrait: number   // 0 = harmonic/tonal, 1 = inharmonic/noisy   (Y-axis)
  fluxTrait: number       // 0 = sustained, 1 = transient               (Z-axis)
  neighbors: number[]
  edgeIds: number[]
}

interface EdgeData {
  a: number
  b: number
  traffic: number
}

interface PacketData {
  edgeIndex: number
  fromNode: number
  toNode: number
  progress: number
  speed: number
  energy: number
  band: number
}

interface SceneProps {
  frequencyData: React.RefObject<Uint8Array>
  timeDomainData: React.RefObject<Uint8Array>
}

// ---------------------------------------------------------------------------
// StreamStats — self-calibrating online statistics for a single feature
//
// Instead of comparing values against hardcoded thresholds, every feature
// tracks its own running mean (μ) and variance (σ²) via exponential moving
// average. Events are detected as z-scores: how many σ above the mean.
//
// This means a quiet song's drum hits still trigger packets (because they
// stand out from THAT song's baseline), and a loud song doesn't constantly
// trigger everything.
// ---------------------------------------------------------------------------

class StreamStats {
  private mean: number = 0
  private varAcc: number = 0   // running variance accumulator (EMA of (x - mean)²)
  private count: number = 0
  private alpha: number         // EMA smoothing factor (0 < α < 1)

  /**
   * @param windowFrames How many frames of "memory". Larger = more stable
   *   baseline, smaller = adapts faster. Uses EMA so this is approximate.
   */
  constructor(windowFrames: number) {
    // α = 2/(N+1) gives an EMA with ~N-frame effective window
    this.alpha = 2 / (windowFrames + 1)
  }

  /** Feed a new raw value. Updates mean and variance. */
  push(value: number) {
    if (this.count === 0) {
      this.mean = value
      this.varAcc = 0
      this.count = 1
      return
    }
    this.count++
    const delta = value - this.mean
    this.mean += this.alpha * delta
    this.varAcc += this.alpha * (delta * delta - this.varAcc)
  }

  /** Standard deviation of the running stream. */
  get std(): number {
    return Math.sqrt(Math.max(0, this.varAcc))
  }

  /** How many σ above the running mean this value is. Negative = below mean. */
  zScore(value: number): number {
    const s = this.std
    if (s < 1e-8) return value > this.mean ? 1 : 0
    return (value - this.mean) / s
  }

  /**
   * Normalize a value to [0, 1] relative to the stream's own range.
   * Uses [mean - σ, mean + 2σ] as the effective range, so p50 ≈ 0.33
   * and the top ~2% of values hit 1.0.
   */
  normalize(value: number): number {
    const s = this.std
    if (s < 1e-8) return 0.5
    const lo = this.mean - s
    const hi = this.mean + 2 * s
    if (hi <= lo) return 0.5
    return Math.max(0, Math.min(1, (value - lo) / (hi - lo)))
  }

  /** Whether the stream has seen enough data to be meaningful. */
  get warmedUp(): boolean {
    return this.count > 10
  }
}

// ---------------------------------------------------------------------------
// StreamAnalyzer — wraps all per-feature StreamStats + per-band stats
// ---------------------------------------------------------------------------

class StreamAnalyzer {
  // Per-band energy trackers (one per frequency group)
  bandStats: StreamStats[]
  // Per-band onset trackers — tracks the *change* in each band
  bandOnsetStats: StreamStats[]

  // Global feature trackers
  energy: StreamStats
  centroid: StreamStats
  harmonicRatio: StreamStats
  spectralSpread: StreamStats
  flux: StreamStats
  crest: StreamStats
  rms: StreamStats

  // State for frame-to-frame deltas
  prevBandEnergies: Float32Array
  prevSpectrum: Float32Array
  rmsRing: Float32Array       // Ring buffer for attack slope detection
  rmsRingIdx: number

  constructor() {
    // ~90 frames ≈ 3 seconds at 30fps — adapts within a few seconds
    const windowFrames = 90

    this.bandStats = Array.from({ length: NUM_BANDS }, () => new StreamStats(windowFrames))
    this.bandOnsetStats = Array.from({ length: NUM_BANDS }, () => new StreamStats(windowFrames))

    this.energy = new StreamStats(windowFrames)
    this.centroid = new StreamStats(windowFrames)
    this.harmonicRatio = new StreamStats(windowFrames)
    this.spectralSpread = new StreamStats(windowFrames)
    this.flux = new StreamStats(windowFrames)
    this.crest = new StreamStats(windowFrames)
    this.rms = new StreamStats(windowFrames)

    this.prevBandEnergies = new Float32Array(NUM_BANDS)
    this.prevSpectrum = new Float32Array(1024)
    this.rmsRing = new Float32Array(6) // 6 frames ≈ 200ms lookback for attack slope
    this.rmsRingIdx = 0
  }
}

/** All features extracted + normalized relative to the running stream. */
interface AudioFeatures {
  bandEnergies: Float32Array        // raw normalised band energies [0, 1]
  bandOnsets: Float32Array          // raw onset deltas per band
  bandOnsetZScores: Float32Array    // how unusual each band's onset is (z-scores)
  dominantOnsetBand: number         // band with strongest onset z-score
  energy: number                    // stream-normalized [0, 1]
  centroid: number                  // stream-normalized [0, 1]
  harmonicRatio: number             // stream-normalized [0, 1]
  spectralSpread: number            // stream-normalized [0, 1]
  flux: number                      // stream-normalized [0, 1]
  crest: number                     // stream-normalized [0, 1]
  attackSlope: number               // stream-normalized [0, 1]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRID_SIZE = 10
const NODE_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE
const EDGE_COUNT = 3 * GRID_SIZE * GRID_SIZE * (GRID_SIZE - 1)
const MAX_PACKETS = 220
const NODE_SPACING = 1.25
const CUBE_SPAN = (GRID_SIZE - 1) * NODE_SPACING
const AXIS_LENGTH = CUBE_SPAN * 0.85 + 3.5
const NUM_BANDS = CALIBRATION.frequencyGroups.length // 10

// Persistence windows from calibration for frame-rate-independent decay
const TRAFFIC_DECAY_MS = CALIBRATION.persistenceWindowsMs[0]    // 33ms
const ACTIVATION_DECAY_MS = CALIBRATION.persistenceWindowsMs[2] // 117ms
const PULSE_DECAY_MS = CALIBRATION.persistenceWindowsMs[1]      // 62ms
const EDGE_DECAY_MS = CALIBRATION.persistenceWindowsMs[3]       // 221ms

const _tmpColor = new THREE.Color()
const _obj = new THREE.Object3D()
const _packetPos = new THREE.Vector3()

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function hash01(seed: number) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123
  return x - Math.floor(x)
}

function centeredCoord(value: number) {
  return (value - (GRID_SIZE - 1) / 2) * NODE_SPACING
}

function nodeIndex(x: number, y: number, z: number) {
  return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

function createGraph() {
  const nodes: NodeData[] = []
  const edges: EdgeData[] = []

  for (let z = 0; z < GRID_SIZE; z++) {
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const index = nodeIndex(x, y, z)
        const px = centeredCoord(x)
        const py = centeredCoord(y)
        const pz = centeredCoord(z)

        const freqBand = x
        const harmonicTrait = y / (GRID_SIZE - 1)
        const fluxTrait = z / (GRID_SIZE - 1)

        const home = new THREE.Vector3(px, py, pz)

        nodes.push({
          index,
          gridX: x,
          gridY: y,
          gridZ: z,
          home,
          position: home.clone(),
          activation: 0,
          traffic: 0,
          pulse: 0,
          baseSize: 0.05 + hash01(index * 11 + 7) * 0.02,
          freqBand,
          harmonicTrait,
          fluxTrait,
          neighbors: [],
          edgeIds: [],
        })
      }
    }
  }

  const addEdge = (a: number, b: number) => {
    const edgeIndex = edges.length
    edges.push({ a, b, traffic: 0 })
    nodes[a].neighbors.push(b)
    nodes[a].edgeIds.push(edgeIndex)
    nodes[b].neighbors.push(a)
    nodes[b].edgeIds.push(edgeIndex)
  }

  for (let z = 0; z < GRID_SIZE; z++) {
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const current = nodeIndex(x, y, z)
        if (x + 1 < GRID_SIZE) addEdge(current, nodeIndex(x + 1, y, z))
        if (y + 1 < GRID_SIZE) addEdge(current, nodeIndex(x, y + 1, z))
        if (z + 1 < GRID_SIZE) addEdge(current, nodeIndex(x, y, z + 1))
      }
    }
  }

  return { nodes, edges }
}

// ---------------------------------------------------------------------------
// Feature extraction — all normalized relative to the running stream
// ---------------------------------------------------------------------------

function extractFeatures(
  frequencyData: Uint8Array,
  timeDomainData: Uint8Array,
  analyzer: StreamAnalyzer,
  bandEnergies: Float32Array,
  bandOnsets: Float32Array,
  bandOnsetZScores: Float32Array,
): AudioFeatures {
  bandEnergies.fill(0)
  bandOnsets.fill(0)
  bandOnsetZScores.fill(0)

  const empty: AudioFeatures = {
    bandEnergies, bandOnsets, bandOnsetZScores, dominantOnsetBand: 0,
    energy: 0, centroid: 0, harmonicRatio: 0, spectralSpread: 0,
    flux: 0, crest: 0, attackSlope: 0,
  }

  if (!frequencyData.length || !timeDomainData.length) return empty

  const fLen = frequencyData.length
  const tLen = timeDomainData.length

  // ------- Frequency domain -------

  let totalFreqSum = 0
  let totalFreqSqSum = 0
  for (let i = 0; i < fLen; i++) {
    const v = frequencyData[i] / 255
    totalFreqSum += v
    totalFreqSqSum += v * v
  }

  // === Band energies + onsets ===
  let maxOnsetZ = -Infinity
  let dominantOnsetBand = 0

  for (let b = 0; b < NUM_BANDS; b++) {
    const [start, end] = CALIBRATION.frequencyGroups[b]
    const binStart = Math.min(start, fLen)
    const binEnd = Math.min(end, fLen)
    if (binEnd <= binStart) continue
    let sum = 0
    for (let i = binStart; i < binEnd; i++) {
      sum += frequencyData[i] / 255
    }
    const rawBandEnergy = sum / (binEnd - binStart)

    // Feed into the stream's running stats, then normalize RELATIVE to stream
    analyzer.bandStats[b].push(rawBandEnergy)
    bandEnergies[b] = analyzer.bandStats[b].normalize(rawBandEnergy)

    // Onset: positive-only delta (energy rising)
    const rawOnset = Math.max(0, rawBandEnergy - analyzer.prevBandEnergies[b])
    analyzer.prevBandEnergies[b] = rawBandEnergy

    analyzer.bandOnsetStats[b].push(rawOnset)
    bandOnsets[b] = rawOnset

    // z-score of onset: "how unusual is this onset for THIS band?"
    const z = analyzer.bandOnsetStats[b].zScore(rawOnset)
    bandOnsetZScores[b] = z

    if (z > maxOnsetZ) {
      maxOnsetZ = z
      dominantOnsetBand = b
    }
  }

  // === Energy ===
  const rawEnergy = fLen > 0 ? Math.sqrt(totalFreqSqSum / fLen) : 0
  analyzer.energy.push(rawEnergy)
  const energy = analyzer.energy.normalize(rawEnergy)

  // === Spectral centroid ===
  let weightedSum = 0
  for (let i = 0; i < fLen; i++) {
    weightedSum += i * (frequencyData[i] / 255)
  }
  const rawCentroid = totalFreqSum > 0 ? (weightedSum / totalFreqSum) / fLen : 0
  analyzer.centroid.push(rawCentroid)
  const centroid = analyzer.centroid.normalize(rawCentroid)

  // === Spectral spread ===
  const centroidBin = totalFreqSum > 0 ? weightedSum / totalFreqSum : 0
  let spreadSum = 0
  for (let i = 0; i < fLen; i++) {
    const v = frequencyData[i] / 255
    spreadSum += ((i - centroidBin) ** 2) * v
  }
  const rawSpread = totalFreqSum > 0 ? Math.sqrt(spreadSum / totalFreqSum) / fLen : 0
  analyzer.spectralSpread.push(rawSpread)
  const spectralSpread = analyzer.spectralSpread.normalize(rawSpread)

  // === Harmonic ratio ===
  let rawHarmonicRatio = 0
  const bassEnd = Math.min(Math.floor(fLen / 4), fLen)
  let maxBassVal = 0
  let fundamentalBin = 0
  for (let i = 1; i < bassEnd; i++) {
    const v = frequencyData[i] / 255
    if (v > maxBassVal) {
      maxBassVal = v
      fundamentalBin = i
    }
  }
  if (fundamentalBin > 0 && totalFreqSqSum > 0.001) {
    let harmonicEnergy = 0
    for (let h = 1; h <= 5; h++) {
      const hBin = fundamentalBin * h
      if (hBin >= fLen) break
      const lo = Math.max(0, hBin - 2)
      const hi = Math.min(fLen, hBin + 3)
      for (let i = lo; i < hi; i++) {
        const v = frequencyData[i] / 255
        harmonicEnergy += v * v
      }
    }
    rawHarmonicRatio = clamp(harmonicEnergy / totalFreqSqSum, 0, 1)
  }
  analyzer.harmonicRatio.push(rawHarmonicRatio)
  const harmonicRatio = analyzer.harmonicRatio.normalize(rawHarmonicRatio)

  // === Spectral flux ===
  let fluxSum = 0
  for (let i = 0; i < fLen; i++) {
    const curr = frequencyData[i] / 255
    const diff = curr - analyzer.prevSpectrum[i]
    fluxSum += diff * diff
    analyzer.prevSpectrum[i] = curr
  }
  const rawFlux = fluxSum / fLen
  analyzer.flux.push(rawFlux)
  const flux = analyzer.flux.normalize(rawFlux)

  // ------- Time domain -------

  let sumSq = 0
  let peak = 0
  for (let i = 0; i < tLen; i++) {
    const sample = (timeDomainData[i] - 128) / 128
    sumSq += sample * sample
    const absSample = Math.abs(sample)
    if (absSample > peak) peak = absSample
  }

  const rawRms = Math.sqrt(sumSq / tLen)
  analyzer.rms.push(rawRms)

  const rawCrest = rawRms > 0.001 ? peak / rawRms : 1
  analyzer.crest.push(rawCrest)
  const crest = analyzer.crest.normalize(rawCrest)

  // === Attack slope (stream-adaptive) ===
  // Compare current RMS to the RMS from ~6 frames ago
  const ringLen = analyzer.rmsRing.length
  const oldIdx = analyzer.rmsRingIdx % ringLen
  const oldRms = analyzer.rmsRing[oldIdx]
  analyzer.rmsRing[analyzer.rmsRingIdx % ringLen] = rawRms
  analyzer.rmsRingIdx++

  // Attack = positive RMS slope, normalized by the stream's RMS σ
  let attackSlope = 0
  if (analyzer.rmsRingIdx > ringLen && analyzer.rms.std > 1e-6) {
    const slope = Math.max(0, rawRms - oldRms)
    // Normalize by stream's own RMS σ: slope of 1σ ≈ 0.5
    attackSlope = clamp(slope / (analyzer.rms.std * 2), 0, 1)
  }

  return {
    bandEnergies, bandOnsets, bandOnsetZScores, dominantOnsetBand,
    energy, centroid, harmonicRatio, spectralSpread,
    flux, crest, attackSlope,
  }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

function GraphScene({ frequencyData, timeDomainData }: SceneProps) {
  const graph = useMemo(() => createGraph(), [])
  const nodesRef = useRef<NodeData[]>(graph.nodes)
  const edgesRef = useRef<EdgeData[]>(graph.edges)
  const packetsRef = useRef<PacketData[]>([])

  const nodeMeshRef = useRef<THREE.InstancedMesh>(null)
  const glowMeshRef = useRef<THREE.InstancedMesh>(null)
  const packetMeshRef = useRef<THREE.InstancedMesh>(null)
  const lineGeoRef = useRef<THREE.BufferGeometry>(null)

  // Stream analyzer — self-calibrates to whichever song/voice is playing
  const analyzerRef = useRef(new StreamAnalyzer())

  const bandEnergiesRef = useRef(new Float32Array(NUM_BANDS))
  const bandOnsetsRef = useRef(new Float32Array(NUM_BANDS))
  const bandOnsetZScoresRef = useRef(new Float32Array(NUM_BANDS))
  const linePositionsRef = useRef(new Float32Array(EDGE_COUNT * 2 * 3))
  const lineColorsRef = useRef(new Float32Array(EDGE_COUNT * 2 * 3))
  const axesHelper = useMemo(() => new THREE.AxesHelper(AXIS_LENGTH), [])
  const cubeFrameGeometry = useMemo(
    () => new THREE.EdgesGeometry(new THREE.BoxGeometry(CUBE_SPAN, CUBE_SPAN, CUBE_SPAN)),
    [],
  )

  useEffect(() => {
    const nodes = nodesRef.current
    const nodeMesh = nodeMeshRef.current
    const glowMesh = glowMeshRef.current
    const packetMesh = packetMeshRef.current
    const lineGeo = lineGeoRef.current
    const linePositions = linePositionsRef.current
    const lineColors = lineColorsRef.current

    if (lineGeo) {
      const positionAttr = new THREE.BufferAttribute(linePositions, 3)
      const colorAttr = new THREE.BufferAttribute(lineColors, 3)
      positionAttr.setUsage(THREE.DynamicDrawUsage)
      colorAttr.setUsage(THREE.DynamicDrawUsage)
      lineGeo.setAttribute('position', positionAttr)
      lineGeo.setAttribute('color', colorAttr)
      lineGeo.setDrawRange(0, 0)
    }

    if (nodeMesh) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        _tmpColor.setHSL(0, 0, 0.65)
        nodeMesh.setColorAt(i, _tmpColor)
        _obj.position.copy(node.position)
        _obj.scale.setScalar(node.baseSize)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)
      }
      nodeMesh.instanceMatrix.needsUpdate = true
      if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true
    }

    if (glowMesh) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        _tmpColor.setHSL(0, 0, 0.4)
        _tmpColor.multiplyScalar(0.1)
        glowMesh.setColorAt(i, _tmpColor)
        _obj.position.copy(node.position)
        _obj.scale.setScalar(node.baseSize * 2.2)
        _obj.updateMatrix()
        glowMesh.setMatrixAt(i, _obj.matrix)
      }
      glowMesh.instanceMatrix.needsUpdate = true
      if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true
    }

    if (packetMesh) {
      for (let i = 0; i < MAX_PACKETS; i++) {
        _obj.position.set(0, 0, 0)
        _obj.scale.setScalar(0)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(i, _obj.matrix)
      }
      packetMesh.instanceMatrix.needsUpdate = true
      if (packetMesh.instanceColor) packetMesh.instanceColor.needsUpdate = true
    }
  }, [])

  useFrame((_, delta) => {
    const nodes = nodesRef.current
    const edges = edgesRef.current
    const packets = packetsRef.current
    const nodeMesh = nodeMeshRef.current
    const glowMesh = glowMeshRef.current
    const packetMesh = packetMeshRef.current
    const lineGeo = lineGeoRef.current
    const linePositions = linePositionsRef.current
    const lineColors = lineColorsRef.current

    if (!nodeMesh || !glowMesh || !packetMesh || !lineGeo) return

    const frameMs = delta * 1000

    // Frame-rate-independent decay from calibration persistence windows
    const trafficAlpha = adaptiveAlpha(TRAFFIC_DECAY_MS, frameMs)
    const activationAlpha = adaptiveAlpha(ACTIVATION_DECAY_MS, frameMs)
    const pulseAlpha = adaptiveAlpha(PULSE_DECAY_MS, frameMs)
    const edgeDecayAlpha = adaptiveAlpha(EDGE_DECAY_MS, frameMs)

    // ===== 1. EXTRACT FEATURES (self-calibrating) =====
    const features = extractFeatures(
      frequencyData.current ?? new Uint8Array(0),
      timeDomainData.current ?? new Uint8Array(0),
      analyzerRef.current,
      bandEnergiesRef.current,
      bandOnsetsRef.current,
      bandOnsetZScoresRef.current,
    )

    const {
      bandEnergies, bandOnsetZScores, dominantOnsetBand,
      energy, centroid, harmonicRatio, spectralSpread,
      flux, crest, attackSlope,
    } = features

    // ===== 2. UPDATE NODE ACTIVATIONS =====
    //
    // X: band onset z-scores — nodes glow when their band has an *unusual* energy spike
    // Y: harmonic ratio vs spectral spread (stream-normalized)
    // Z: flux + crest + attack slope (stream-normalized)

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]

      // X-axis: onset z-score — "is this band having an unusual spike?"
      // z > 0 = above average onset, z > 1 = notable, z > 2 = strong hit
      // We convert z-score to activation: z=0→0, z=1→0.5, z=2→1.0
      const onsetZ = bandOnsetZScores[node.freqBand]
      const xActivation = clamp(onsetZ * 0.5, 0, 1.2)
      // Also add a warmth term from the band's normalized energy level
      const bandWarmth = bandEnergies[node.freqBand] * 0.35

      // Y-axis: harmonic density matching (stream-normalized)
      const t_y = node.harmonicTrait
      const harmonicMatch = harmonicRatio * (1 - t_y)
      const inharmonicMatch = spectralSpread * t_y
      const yActivation = harmonicMatch + inharmonicMatch

      // Z-axis: temporal dynamics (stream-normalized)
      const t_z = node.fluxTrait
      const sustainedMatch = (1 - flux) * (1 - t_z)
      const transientMatch = ((flux + crest + attackSlope) / 3) * t_z
      const zActivation = sustainedMatch + transientMatch

      // Combined: onset drives it, y/z shape it, energy gates it
      const targetActivation = clamp(
        (xActivation + bandWarmth) *
        (0.6 + 0.4 * yActivation) *
        (0.6 + 0.4 * zActivation) *
        (0.5 + energy * 0.8),
        0, 1.4,
      )

      const blended = targetActivation + node.traffic * 0.25 + node.pulse * 0.3
      node.activation += (blended - node.activation) * activationAlpha
      node.traffic *= (1 - trafficAlpha)
      node.pulse *= (1 - pulseAlpha)
      node.position.copy(node.home)
    }

    // ===== 3. DECAY EDGES =====
    for (let i = 0; i < edges.length; i++) {
      edges[i].traffic *= (1 - edgeDecayAlpha)
    }

    // ===== 4. SPAWN PACKETS =====
    //
    // No hardcoded thresholds! A packet spawns when a band's onset z-score
    // exceeds 1.0 (i.e., it's more than 1σ above that band's average onset).
    // This self-calibrates:
    //   - Quiet song → small onsets are still 1σ above the quiet baseline
    //   - Loud song → only genuine hits exceed the loud baseline
    //
    // Speed = centroid (stream-normalized): bright → fast, bass → slow
    // Direction = toward dominant onset band (highest z-score)
    // Energy = onset z-score × node activation

    const frameScale = Math.min(delta * 60, 1.5)

    for (let band = 0; band < NUM_BANDS && packets.length < MAX_PACKETS; band++) {
      const onsetZ = bandOnsetZScores[band]

      // Stream-adaptive threshold: only spawn if this onset is
      // at least 1σ above the band's average onset magnitude.
      // No magic numbers — this adapts to any audio source.
      if (onsetZ < 1.0) continue

      // Spawn count scales with how extreme the onset is (z-score driven)
      // z=1 → ~1 candidate, z=3 → ~3 candidates
      const numCandidates = Math.ceil(onsetZ)

      // Spawn probability per candidate scales with energy
      const spawnChance = clamp(energy * 0.6 + 0.15, 0, 1) * (frameMs / CALIBRATION.sampleIntervalMs)

      for (let c = 0; c < numCandidates && packets.length < MAX_PACKETS; c++) {
        if (Math.random() >= spawnChance) continue

        const gy = Math.floor(Math.random() * GRID_SIZE)
        const gz = Math.floor(Math.random() * GRID_SIZE)
        const nodeIdx = nodeIndex(band, gy, gz)
        const node = nodes[nodeIdx]

        // Direction: toward dominant onset band + activation pull
        let totalWeight = 0
        const weights = new Array(node.neighbors.length)
        for (let j = 0; j < node.neighbors.length; j++) {
          const neighborIdx = node.neighbors[j]
          const neighbor = nodes[neighborIdx]
          const edge = edges[node.edgeIds[j]]

          // Lateral pull toward dominant onset band
          const myDist = Math.abs(node.gridX - dominantOnsetBand)
          const neighborDist = Math.abs(neighbor.gridX - dominantOnsetBand)
          const lateralPull = myDist > neighborDist ? 1.5 : 0.3

          const activationPull = neighbor.activation * neighbor.activation
          const weight = lateralPull + activationPull * 2 + edge.traffic * 0.8 + 0.05
          weights[j] = weight
          totalWeight += weight
        }

        if (totalWeight <= 0) continue

        let selection = Math.random() * totalWeight
        let targetSlot = 0
        for (let j = 0; j < weights.length; j++) {
          selection -= weights[j]
          if (selection <= 0) {
            targetSlot = j
            break
          }
        }

        const edgeIndex = node.edgeIds[targetSlot]
        const toNode = node.neighbors[targetSlot]
        const edge = edges[edgeIndex]

        // Speed: driven by pitch (centroid), rate of freq change (flux), and transients (attackSlope)
        // This makes packets zip extremely fast during complex, rapidly changing audio
        const packetSpeed = clamp(0.024 + centroid * 0.04 + flux * 0.15 + attackSlope * 0.12, 0.024, 0.3)

        // Energy: incorporates overall "super loud" peaks (crest) and transients (attackSlope/onsetZ)
        const packetEnergy = clamp(
          (onsetZ / 3) * node.activation * (0.6 + crest * 0.6) + (attackSlope * 0.4),
          0.05, 1.8,
        )

        edge.traffic = clamp(edge.traffic + packetEnergy * 0.12, 0, 1.8)
        node.traffic = clamp(node.traffic + packetEnergy * 0.08, 0, 1.4)
        node.pulse = clamp(node.pulse + (onsetZ / 3) * 0.5, 0, 1.4)

        packets.push({
          edgeIndex,
          fromNode: node.index,
          toNode,
          progress: 0,
          speed: packetSpeed,
          energy: packetEnergy,
          band,
        })
      }
    }

    // ===== 5. RENDER NODES =====
    if (nodeMesh && glowMesh) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        const activity = clamp(node.activation + node.traffic * 0.4 + node.pulse * 0.35, 0, 1.5)
        const size = node.baseSize * (1 + activity * 0.8)

        _obj.position.copy(node.position)
        _obj.scale.setScalar(size)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)

        const brightness = 0.3 + activity * 0.5
        _tmpColor.setHSL(0, 0, brightness)
        _tmpColor.multiplyScalar(0.9 + activity * 0.55)
        nodeMesh.setColorAt(i, _tmpColor)

        // Glow scale explodes when the audio is "super loud" (energy) or rapidly changing (flux)
        const dynamicGlow = 1.3 + activity * (0.2 + energy * 1.5 + flux * 1.0)
        _obj.scale.setScalar(size * dynamicGlow)
        _obj.updateMatrix()
        glowMesh.setMatrixAt(i, _obj.matrix)

        // Color blasts to pure brilliant white during sharp attacks and loud peaks
        const flash = activity * (1.0 + attackSlope * 1.5 + crest * 0.5)
        _tmpColor.setHSL(0, 0, clamp(0.25 + flash * 0.6, 0, 1))
        _tmpColor.multiplyScalar(0.06 + flash * 0.8)
        glowMesh.setColorAt(i, _tmpColor)
      }

      nodeMesh.instanceMatrix.needsUpdate = true
      glowMesh.instanceMatrix.needsUpdate = true
      if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true
      if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true
    }

    // ===== 6. RENDER EDGES =====
    if (lineGeo) {
      let vertexIndex = 0
      for (let i = 0; i < edges.length; i++) {
        const edge = edges[i]
        const a = nodes[edge.a]
        const b = nodes[edge.b]

        const score = edge.traffic * 0.7 + (a.activation + b.activation) * 0.12
        const brightness = clamp(0.04 + score * 0.65, 0.04, 0.95)
        _tmpColor.setHSL(0, 0, 0.28 + brightness * 0.28)
        _tmpColor.multiplyScalar(0.32 + brightness * 0.48)

        linePositions[vertexIndex * 3] = a.position.x
        linePositions[vertexIndex * 3 + 1] = a.position.y
        linePositions[vertexIndex * 3 + 2] = a.position.z
        lineColors[vertexIndex * 3] = _tmpColor.r
        lineColors[vertexIndex * 3 + 1] = _tmpColor.g
        lineColors[vertexIndex * 3 + 2] = _tmpColor.b
        vertexIndex++

        linePositions[vertexIndex * 3] = b.position.x
        linePositions[vertexIndex * 3 + 1] = b.position.y
        linePositions[vertexIndex * 3 + 2] = b.position.z
        lineColors[vertexIndex * 3] = _tmpColor.r
        lineColors[vertexIndex * 3 + 1] = _tmpColor.g
        lineColors[vertexIndex * 3 + 2] = _tmpColor.b
        vertexIndex++
      }

      for (let i = vertexIndex * 3; i < linePositions.length; i++) {
        linePositions[i] = 0
        lineColors[i] = 0
      }

      const positionAttr = lineGeo.getAttribute('position') as THREE.BufferAttribute
      const colorAttr = lineGeo.getAttribute('color') as THREE.BufferAttribute
      positionAttr.needsUpdate = true
      colorAttr.needsUpdate = true
      lineGeo.setDrawRange(0, vertexIndex)
    }

    // ===== 7. UPDATE PACKETS =====
    if (packetMesh) {
      let activePackets = 0

      for (let i = packets.length - 1; i >= 0; i--) {
        const packet = packets[i]
        const edge = edges[packet.edgeIndex]
        packet.progress += packet.speed * frameScale
        edge.traffic = clamp(edge.traffic + packet.energy * 0.004 * frameScale, 0, 1.8)

        if (packet.progress >= 1) {
          const destination = nodes[packet.toNode]
          destination.activation = clamp(destination.activation + packet.energy * 0.45, 0, 1.4)
          destination.traffic = clamp(destination.traffic + packet.energy * 0.35, 0, 1.5)
          destination.pulse = clamp(destination.pulse + packet.energy * crest * 0.5, 0, 1.5)
          packets.splice(i, 1)
          continue
        }

        const from = nodes[packet.fromNode]
        const to = nodes[packet.toNode]
        _packetPos.lerpVectors(from.position, to.position, packet.progress)
        _obj.position.copy(_packetPos)
        _obj.scale.setScalar(0.04 + packet.energy * 0.08 + Math.sin(packet.progress * Math.PI) * 0.025)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(activePackets, _obj.matrix)

        _tmpColor.setHSL(0, 0, 0.65 + packet.energy * 0.25)
        _tmpColor.multiplyScalar(1.1 + packet.energy * 0.7)
        packetMesh.setColorAt(activePackets, _tmpColor)
        activePackets++
      }

      for (let i = activePackets; i < MAX_PACKETS; i++) {
        _obj.position.set(0, 0, 0)
        _obj.scale.setScalar(0)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(i, _obj.matrix)
      }

      packetMesh.count = Math.max(activePackets, 1)
      packetMesh.instanceMatrix.needsUpdate = true
      if (packetMesh.instanceColor) packetMesh.instanceColor.needsUpdate = true
    }
  })

  return (
    <>
      <primitive object={axesHelper} />

      <lineSegments geometry={cubeFrameGeometry}>
        <lineBasicMaterial
          color="#cbd5e1"
          transparent
          opacity={0.3}
          depthWrite={false}
          toneMapped={false}
        />
      </lineSegments>

      <lineSegments frustumCulled={false}>
        <bufferGeometry ref={lineGeoRef} />
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={0.42}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </lineSegments>

      <instancedMesh
        ref={nodeMeshRef}
        args={[undefined, undefined, NODE_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 10, 8]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      <instancedMesh
        ref={glowMeshRef}
        args={[undefined, undefined, NODE_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 8, 6]} />
        <meshBasicMaterial
          transparent
          opacity={0.12}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </instancedMesh>

      <instancedMesh
        ref={packetMeshRef}
        args={[undefined, undefined, MAX_PACKETS]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 8, 6]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
    </>
  )
}

export function Visualizer({
  frequencyData,
  timeDomainData,
  isActive,
  width,
  height,
}: VisualizerProps) {
  return (
    <div style={{ width, height, position: 'relative', background: '#05070c' }}>
      <Canvas
        camera={{ position: [15, 13, 18], fov: 42, near: 0.1, far: 120 }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.NoToneMapping
          gl.outputColorSpace = THREE.SRGBColorSpace
        }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#05070c']} />

        <ambientLight intensity={0.2} />
        <pointLight position={[14, 16, 14]} intensity={0.36} color="#ffffff" />
        <pointLight position={[-14, -8, 10]} intensity={0.14} color="#a3a3a3" />

        <OrbitControls
          enableDamping
          dampingFactor={0.06}
          autoRotate
          autoRotateSpeed={0.12}
          minDistance={14}
          maxDistance={38}
          enablePan={false}
        />

        {isActive && (
          <GraphScene
            frequencyData={frequencyData}
            timeDomainData={timeDomainData}
          />
        )}

        <EffectComposer>
          <Bloom
            intensity={0.42}
            luminanceThreshold={0.22}
            luminanceSmoothing={0.7}
            mipmapBlur
          />
        </EffectComposer>
      </Canvas>

      {!isActive && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
          }}
        >
          <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#94a3b8' }}>
            awaiting microphone input
          </span>
        </div>
      )}
    </div>
  )
}
