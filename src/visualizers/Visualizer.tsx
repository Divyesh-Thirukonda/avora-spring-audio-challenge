import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'
import {
  CALIBRATION,
  normalizeByQuantiles,
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
  band: number            // which frequency band spawned this packet
}

interface SceneProps {
  frequencyData: React.RefObject<Uint8Array>
  timeDomainData: React.RefObject<Uint8Array>
}

/** All features extracted from a single audio frame, normalized to [0, 1]. */
interface AudioFeatures {
  /** Per-band energy, normalized by groupEnergyQuantiles. Length = 10. */
  bandEnergies: Float32Array
  /** Per-band onset: how much each band ROSE since last frame [0, 1]. Length = 10. */
  bandOnsets: Float32Array
  /** Index of the band with the strongest onset this frame. */
  dominantOnsetBand: number
  /** Overall onset strength [0, 1]: how much the spectrum changed (positive-only flux). */
  onsetStrength: number
  /** Global energy level [0, 1] */
  energy: number
  /** Spectral centroid [0, 1]: low = bass-heavy, high = treble-heavy */
  centroid: number
  /** Harmonic ratio [0, 1]: how much energy sits on harmonic peaks of the fundamental */
  harmonicRatio: number
  /** Spectral spread [0, 1]: how wide the spectrum is (bandwidth) */
  spectralSpread: number
  /** Spectral flux [0, 1]: low = steady, high = changing rapidly */
  flux: number
  /** Crest factor [0, 1]: low = flat waveform, high = sharp peaks/transients */
  crest: number
  /** Attack slope [0, 1]: how fast amplitude is increasing (> 0 = getting louder) */
  attackSlope: number
}

// ---------------------------------------------------------------------------
// Constants — derived from calibration data, not arbitrary
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
const TRAFFIC_DECAY_MS = CALIBRATION.persistenceWindowsMs[0]    // 33ms  — fast
const ACTIVATION_DECAY_MS = CALIBRATION.persistenceWindowsMs[2] // 117ms — medium
const PULSE_DECAY_MS = CALIBRATION.persistenceWindowsMs[1]      // 62ms  — quick flash
const EDGE_DECAY_MS = CALIBRATION.persistenceWindowsMs[3]       // 221ms — visible trails

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

        // X → frequency band (0-9, maps to CALIBRATION.frequencyGroups)
        //     Nodes light up when their band has an *onset* (energy spike)
        const freqBand = x

        // Y → harmonic density: 0 = harmonic/tonal (pure notes), 1 = inharmonic/noisy
        //     Data shows harmonic_ratio anti-correlates with spectral_spread at r=-0.96
        const harmonicTrait = y / (GRID_SIZE - 1)

        // Z → temporal dynamics: 0 = sustained/steady, 1 = transient/attacking
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
// Audio feature extraction — all normalised via calibration quantiles
// ---------------------------------------------------------------------------

function extractFeatures(
  frequencyData: Uint8Array,
  timeDomainData: Uint8Array,
  bandEnergies: Float32Array,
  bandOnsets: Float32Array,
  prevBandEnergies: Float32Array,
  prevSpectrum: Float32Array,
  rmsHistory: Float32Array,
  rmsHistoryIndex: { value: number },
): AudioFeatures {
  bandEnergies.fill(0)
  bandOnsets.fill(0)

  const fq = CALIBRATION.featureQuantiles
  const empty: AudioFeatures = {
    bandEnergies, bandOnsets, dominantOnsetBand: 0, onsetStrength: 0,
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

  // === Band energies + per-band onsets ===
  // Each band's onset = how much it ROSE above its previous level
  let maxOnset = 0
  let dominantOnsetBand = 0
  let totalOnset = 0

  for (let b = 0; b < NUM_BANDS; b++) {
    const [start, end] = CALIBRATION.frequencyGroups[b]
    const binStart = Math.min(start, fLen)
    const binEnd = Math.min(end, fLen)
    if (binEnd <= binStart) continue
    let sum = 0
    for (let i = binStart; i < binEnd; i++) {
      sum += frequencyData[i] / 255
    }
    const raw = sum / (binEnd - binStart)
    bandEnergies[b] = normalizeByQuantiles(raw, CALIBRATION.groupEnergyQuantiles[b])

    // Onset: positive difference only (energy rising = onset, falling = ignored)
    const onset = Math.max(0, bandEnergies[b] - prevBandEnergies[b])
    bandOnsets[b] = onset
    totalOnset += onset

    if (onset > maxOnset) {
      maxOnset = onset
      dominantOnsetBand = b
    }

    prevBandEnergies[b] = bandEnergies[b]
  }

  // Normalize onset strength: total onset across all bands, clamped
  const onsetStrength = clamp(totalOnset / NUM_BANDS * 5, 0, 1)

  // === Energy ===
  const rawEnergy = fLen > 0 ? Math.sqrt(totalFreqSqSum / fLen) : 0
  const energy = normalizeByQuantiles(rawEnergy, fq.energy)

  // === Spectral centroid ===
  let weightedSum = 0
  for (let i = 0; i < fLen; i++) {
    weightedSum += i * (frequencyData[i] / 255)
  }
  const rawCentroid = totalFreqSum > 0 ? (weightedSum / totalFreqSum) / fLen : 0
  const centroid = normalizeByQuantiles(rawCentroid, fq.centroid)

  // === Spectral spread (bandwidth) ===
  const centroidBin = totalFreqSum > 0 ? weightedSum / totalFreqSum : 0
  let spreadSum = 0
  for (let i = 0; i < fLen; i++) {
    const v = frequencyData[i] / 255
    spreadSum += ((i - centroidBin) ** 2) * v
  }
  const rawSpread = totalFreqSum > 0 ? Math.sqrt(spreadSum / totalFreqSum) / fLen : 0
  // Normalize: typical spread range from analysis is ~0.01-0.25
  const spectralSpread = clamp(rawSpread * 5, 0, 1)

  // === Harmonic ratio ===
  // Find the fundamental (strongest bin in bass) and check how much energy
  // sits on its harmonic series (2f, 3f, 4f, 5f)
  let harmonicRatio = 0
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
    harmonicRatio = clamp(harmonicEnergy / totalFreqSqSum, 0, 1)
  }

  // === Spectral flux ===
  let fluxSum = 0
  for (let i = 0; i < fLen; i++) {
    const curr = frequencyData[i] / 255
    const diff = curr - prevSpectrum[i]
    fluxSum += diff * diff
    prevSpectrum[i] = curr
  }
  const rawFlux = fluxSum / fLen
  const flux = normalizeByQuantiles(rawFlux, fq.flux)

  // ------- Time domain -------

  let sumSq = 0
  let peak = 0

  for (let i = 0; i < tLen; i++) {
    const sample = (timeDomainData[i] - 128) / 128
    sumSq += sample * sample
    const absSample = Math.abs(sample)
    if (absSample > peak) peak = absSample
  }

  const rms = Math.sqrt(sumSq / tLen)

  const rawCrest = rms > 0.001 ? peak / rms : 1
  const crest = normalizeByQuantiles(rawCrest, fq.crest)

  // === Attack slope (amplitude rising?) ===
  // Track RMS over a short window (4 frames) to detect volume increases
  const histLen = rmsHistory.length
  const idx = rmsHistoryIndex.value % histLen
  rmsHistory[idx] = rms
  rmsHistoryIndex.value++

  let attackSlope = 0
  if (rmsHistoryIndex.value >= histLen) {
    const oldIdx = (rmsHistoryIndex.value - histLen + histLen) % histLen
    const oldRms = rmsHistory[oldIdx]
    const slope = rms - oldRms
    attackSlope = clamp(slope * 50, 0, 1) // Positive only, scaled up
  }

  return {
    bandEnergies, bandOnsets, dominantOnsetBand, onsetStrength,
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

  const bandEnergiesRef = useRef(new Float32Array(NUM_BANDS))
  const bandOnsetsRef = useRef(new Float32Array(NUM_BANDS))
  const prevBandEnergiesRef = useRef(new Float32Array(NUM_BANDS))
  const prevSpectrumRef = useRef(new Float32Array(1024))
  const rmsHistoryRef = useRef(new Float32Array(4))
  const rmsHistoryIndexRef = useRef({ value: 0 })
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

    // ===== 1. EXTRACT AUDIO FEATURES =====
    const features = extractFeatures(
      frequencyData.current ?? new Uint8Array(0),
      timeDomainData.current ?? new Uint8Array(0),
      bandEnergiesRef.current,
      bandOnsetsRef.current,
      prevBandEnergiesRef.current,
      prevSpectrumRef.current,
      rmsHistoryRef.current,
      rmsHistoryIndexRef.current,
    )

    const {
      bandEnergies, bandOnsets, dominantOnsetBand, onsetStrength: _os,
      energy, centroid, harmonicRatio, spectralSpread,
      flux, crest, attackSlope,
    } = features

    // ===== 2. UPDATE NODE ACTIVATIONS =====
    //
    // Each node responds to 3 data-driven questions:
    //
    // X (gridX = freqBand): "Is my frequency band currently having an onset?"
    //   → bandOnsets[band] gives per-band onset detection: nodes light up
    //     when their band's energy RISES, not just when it's loud
    //
    // Y (harmonicTrait): "Does the current sound match my harmonic character?"
    //   → Low-y nodes glow for harmonic/tonal sounds (high harmonicRatio)
    //   → High-y nodes glow for inharmonic/noisy sounds (high spectralSpread)
    //   → Data confirms: harmonicRatio and spectralSpread anti-correlate at r=-0.96
    //
    // Z (fluxTrait): "Does the current sound match my temporal character?"
    //   → Low-z nodes glow for sustained sounds (low flux, low attackSlope)
    //   → High-z nodes glow for transients (high flux, high crest)

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]

      // X-axis: onset-reactive — nodes light on energy *rises*, not just level
      // Blend: mostly onset (responsiveness) + some sustained energy (warmth)
      const bandOnset = bandOnsets[node.freqBand] ?? 0
      const bandLevel = bandEnergies[node.freqBand] ?? 0
      const xActivation = bandOnset * 3 + bandLevel * 0.4

      // Y-axis: harmonic density matching
      // → harmonicRatio is high when energy sits on harmonic peaks
      // → spectralSpread is high when energy is scattered (noise-like)
      const t_y = node.harmonicTrait
      const harmonicMatch = harmonicRatio * (1 - t_y)     // strong at y=0
      const inharmonicMatch = spectralSpread * t_y         // strong at y=9
      const yActivation = harmonicMatch + inharmonicMatch

      // Z-axis: temporal dynamics matching
      const t_z = node.fluxTrait
      const sustainedMatch = (1 - flux) * (1 - t_z)       // strong at z=0
      const transientMatch = ((flux + crest + attackSlope) / 3) * t_z  // strong at z=9
      const zActivation = sustainedMatch + transientMatch

      // Combined: multiplicative so silence on any axis dims the node
      const targetActivation = clamp(
        xActivation * (0.25 + 0.75 * yActivation) * (0.25 + 0.75 * zActivation) * (0.3 + energy * 0.7),
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
    // WHEN:  A band has an onset (its energy is rising) — per-band onset detection
    // WHERE: At the node in that band's x-column with the highest activation
    // SPEED: Driven by centroid — bright/high-pitched sounds make fast packets,
    //        dark/bass sounds make slow packets (data: centroid range 0.01-0.37)
    // DIRECTION: Toward the dominant onset band — packets cascade laterally
    //            across the x-axis toward where the biggest onset is.
    //            Within y/z, attracted to active neighbors.

    const frameScale = Math.min(delta * 60, 1.5)

    for (let band = 0; band < NUM_BANDS && packets.length < MAX_PACKETS; band++) {
      const onset = bandOnsets[band]
      if (onset < 0.08) continue // Only spawn on meaningful onsets

      // Spawn probability scales with onset magnitude and overall energy
      const spawnChance = onset * energy * 0.15 * (frameMs / CALIBRATION.sampleIntervalMs)

      // Find the best node in this band's column to spawn from (highest activation)
      // We sample a few random y,z positions rather than scanning all 100
      const numCandidates = Math.ceil(spawnChance * 8) + 1

      for (let c = 0; c < numCandidates && packets.length < MAX_PACKETS; c++) {
        if (Math.random() >= spawnChance) continue

        // Pick a random y,z in this band's column
        const gy = Math.floor(Math.random() * GRID_SIZE)
        const gz = Math.floor(Math.random() * GRID_SIZE)
        const nodeIdx = nodeIndex(band, gy, gz)
        const node = nodes[nodeIdx]

        if (node.activation < 0.1) continue

        // === DIRECTION ===
        // Primary drive: toward the dominant onset band (lateral cascade)
        // Secondary drive: toward neighbors with high activation
        let totalWeight = 0
        const weights = new Array(node.neighbors.length)
        for (let j = 0; j < node.neighbors.length; j++) {
          const neighborIdx = node.neighbors[j]
          const neighbor = nodes[neighborIdx]
          const edge = edges[node.edgeIds[j]]

          // Lateral pull: prefer neighbors closer to the dominant onset band
          const myDistToDominant = Math.abs(node.gridX - dominantOnsetBand)
          const neighborDistToDominant = Math.abs(neighbor.gridX - dominantOnsetBand)
          const lateralPull = myDistToDominant > neighborDistToDominant ? 1.5 : 0.3

          // Activation pull: flow toward bright nodes
          const activationPull = neighbor.activation * neighbor.activation

          // Edge affinity: prefer edges that already have traffic (convoy)
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

        // === SPEED ===
        // Driven by spectral centroid: bright sounds = fast, bass = slow
        // centroid ranges [0, 1] after normalization
        // Base speed ensures packets always move; centroid adds 4x variation
        const packetSpeed = 0.008 + centroid * 0.032

        // === ENERGY ===
        // Packet brightness = onset magnitude × node activation × crest boost
        // Crest makes transient packets flash brighter on arrival
        const packetEnergy = clamp(onset * node.activation * (0.6 + crest * 0.4), 0.05, 1.2)

        // Spawn effects on source node and edge
        edge.traffic = clamp(edge.traffic + packetEnergy * 0.12, 0, 1.8)
        node.traffic = clamp(node.traffic + packetEnergy * 0.08, 0, 1.4)
        node.pulse = clamp(node.pulse + onset * 0.5, 0, 1.4)

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

        // Size: proportional to activation
        const size = node.baseSize * (1 + activity * 0.8)

        _obj.position.copy(node.position)
        _obj.scale.setScalar(size)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)

        // Brightness: linear ramp from dim to bright
        const brightness = 0.3 + activity * 0.5
        _tmpColor.setHSL(0, 0, brightness)
        _tmpColor.multiplyScalar(0.9 + activity * 0.55)
        nodeMesh.setColorAt(i, _tmpColor)

        // Glow
        _obj.scale.setScalar(size * (1.3 + activity * 0.2))
        _obj.updateMatrix()
        glowMesh.setMatrixAt(i, _obj.matrix)

        _tmpColor.setHSL(0, 0, 0.25 + activity * 0.15)
        _tmpColor.multiplyScalar(0.06 + activity * 0.14)
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
          // === ARRIVAL EFFECT ===
          // Destination node gets:
          // - activation boost proportional to packet energy
          // - pulse flash proportional to crest (sharp transients hit harder)
          // - traffic proportional to energy (sustains cascade)
          const destination = nodes[packet.toNode]
          destination.activation = clamp(destination.activation + packet.energy * 0.18, 0, 1.4)
          destination.traffic = clamp(destination.traffic + packet.energy * 0.22, 0, 1.5)
          destination.pulse = clamp(destination.pulse + packet.energy * crest * 0.4, 0, 1.5)
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
