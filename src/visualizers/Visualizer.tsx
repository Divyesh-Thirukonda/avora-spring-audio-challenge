import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

export interface VisualizerProps {
  frequencyData: React.RefObject<Uint8Array>
  timeDomainData: React.RefObject<Uint8Array>
  isActive: boolean
  width: number
  height: number
}

interface NodeData {
  index: number
  gridX: number
  home: THREE.Vector3
  position: THREE.Vector3
  activation: number
  traffic: number
  pulse: number
  baseSize: number
  hue: number
  freqBand: number
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
  hue: number
}

interface SceneProps {
  frequencyData: React.RefObject<Uint8Array>
  timeDomainData: React.RefObject<Uint8Array>
}

const GRID_SIZE = 10
const NODE_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE
const EDGE_COUNT = 3 * GRID_SIZE * GRID_SIZE * (GRID_SIZE - 1)
const MAX_PACKETS = 220
const NODE_SPACING = 1.25
const CUBE_SPAN = (GRID_SIZE - 1) * NODE_SPACING
const AXIS_LENGTH = CUBE_SPAN * 0.85 + 3.5

const _tmpColor = new THREE.Color()
const _obj = new THREE.Object3D()
const _packetPos = new THREE.Vector3()

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
        const freqBand = Math.round((x / (GRID_SIZE - 1)) * 15)
        const freqMix = x / (GRID_SIZE - 1)
        const home = new THREE.Vector3(px, py, pz)

        nodes.push({
          index,
          gridX: x,
          home,
          position: home.clone(),
          activation: 0,
          traffic: 0,
          pulse: 0,
          baseSize: 0.05 + hash01(index * 11 + 7) * 0.025,
          hue: 0.56 + freqMix * 0.14,
          freqBand,
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

function sampleAudio(
  frequencyData: Uint8Array,
  timeDomainData: Uint8Array,
  bandEnergies: Float32Array,
) {
  bandEnergies.fill(0)

  if (!frequencyData.length || !timeDomainData.length) {
    return { waveAmp: 0, activeBandAverage: 0 }
  }

  let waveAmp = 0
  for (let i = 0; i < timeDomainData.length; i++) {
    waveAmp += Math.abs(timeDomainData[i] - 128)
  }
  waveAmp /= timeDomainData.length * 128

  const bandCount = bandEnergies.length
  const bandSize = Math.max(1, Math.floor(frequencyData.length / bandCount))
  let activeBandAverage = 0

  for (let band = 0; band < bandCount; band++) {
    let sum = 0
    for (let i = 0; i < bandSize; i++) {
      const sampleIndex = band * bandSize + i
      if (sampleIndex >= frequencyData.length) break
      sum += frequencyData[sampleIndex]
    }
    const weight = 1 + Math.pow(band / 3, 1.5)
    const energy = clamp((sum / (bandSize * 255)) * weight, 0, 1)
    bandEnergies[band] = energy
    activeBandAverage += energy
  }

  return { waveAmp, activeBandAverage: activeBandAverage / bandEnergies.length }
}

function GraphScene({ frequencyData, timeDomainData }: SceneProps) {
  const graph = useMemo(() => createGraph(), [])
  const nodesRef = useRef<NodeData[]>(graph.nodes)
  const edgesRef = useRef<EdgeData[]>(graph.edges)
  const packetsRef = useRef<PacketData[]>([])

  const nodeMeshRef = useRef<THREE.InstancedMesh>(null)
  const glowMeshRef = useRef<THREE.InstancedMesh>(null)
  const packetMeshRef = useRef<THREE.InstancedMesh>(null)
  const lineGeoRef = useRef<THREE.BufferGeometry>(null)

  const bandEnergiesRef = useRef(new Float32Array(16))
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
    const bandEnergies = bandEnergiesRef.current
    const linePositions = linePositionsRef.current
    const lineColors = lineColorsRef.current

    if (!nodeMesh || !glowMesh || !packetMesh || !lineGeo) return

    const frameScale = Math.min(delta * 60, 1.5)
    const decayNodeTraffic = Math.pow(0.965, frameScale)
    const decayEdgeTraffic = Math.pow(0.972, frameScale)
    const decayPulse = Math.pow(0.9, frameScale)
    const { waveAmp, activeBandAverage } = sampleAudio(
      frequencyData.current ?? new Uint8Array(0),
      timeDomainData.current ?? new Uint8Array(0),
      bandEnergies,
    )

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      const bandEnergy = bandEnergies[node.freqBand] ?? 0
      const targetActivation = clamp(
        bandEnergy * 0.9 + node.traffic * 0.45 + waveAmp * 0.15,
        0,
        1.2,
      )
      node.activation += (targetActivation - node.activation) * (0.12 * frameScale)
      node.traffic *= decayNodeTraffic
      node.pulse *= decayPulse
      node.position.copy(node.home)
    }

    for (let i = 0; i < edges.length; i++) {
      edges[i].traffic *= decayEdgeTraffic
    }

    const spawnChanceBase = (0.0015 + activeBandAverage * 0.012 + waveAmp * 0.004) * frameScale
    for (let i = 0; i < nodes.length && packets.length < MAX_PACKETS; i++) {
      const node = nodes[i]
      if (node.activation < 0.2) continue

      const spawnChance = spawnChanceBase + node.activation * 0.014 + node.traffic * 0.005
      if (Math.random() >= spawnChance) continue

      let totalWeight = 0
      const weights = new Array(node.neighbors.length)
      for (let j = 0; j < node.neighbors.length; j++) {
        const neighborIndex = node.neighbors[j]
        const edge = edges[node.edgeIds[j]]
        const neighbor = nodes[neighborIndex]
        const weight =
          0.25 +
          neighbor.activation * 1.2 +
          neighbor.traffic * 0.8 +
          edge.traffic * 1.5 +
          (neighbor.gridX > node.gridX ? 0.08 : 0)
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
      const energy = clamp(node.activation * 0.8 + waveAmp * 0.5, 0.1, 1.15)

      edge.traffic = clamp(edge.traffic + energy * 0.12, 0, 1.8)
      node.traffic = clamp(node.traffic + energy * 0.08, 0, 1.4)
      node.pulse = clamp(node.pulse + 0.25, 0, 1.4)

      packets.push({
        edgeIndex,
        fromNode: node.index,
        toNode,
        progress: 0,
        speed: 0.018 + energy * 0.028,
        energy,
        hue: node.hue,
      })
    }

    if (nodeMesh && glowMesh) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        const activity = clamp(node.activation * 0.75 + node.traffic * 0.55 + node.pulse * 0.45, 0, 1.5)
        const size = node.baseSize * (1 + waveAmp * 0.18 + activity * 0.72)

        _obj.position.copy(node.position)
        _obj.scale.setScalar(size)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)

        _tmpColor.setHSL(0, 0, 0.5 + activity * 0.2)
        _tmpColor.multiplyScalar(0.9 + activity * 0.55)
        nodeMesh.setColorAt(i, _tmpColor)

        _obj.scale.setScalar(size * (1.35 + activity * 0.18))
        _obj.updateMatrix()
        glowMesh.setMatrixAt(i, _obj.matrix)

        _tmpColor.setHSL(0, 0, 0.3 + activity * 0.1)
        _tmpColor.multiplyScalar(0.07 + activity * 0.12)
        glowMesh.setColorAt(i, _tmpColor)
      }

      nodeMesh.instanceMatrix.needsUpdate = true
      glowMesh.instanceMatrix.needsUpdate = true
      if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true
      if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true
    }

    if (lineGeo) {
      let vertexIndex = 0
      for (let i = 0; i < edges.length; i++) {
        const edge = edges[i]
        const a = nodes[edge.a]
        const b = nodes[edge.b]
        const score =
          edge.traffic * 0.85 +
          (a.activation + b.activation) * 0.22 +
          (a.traffic + b.traffic) * 0.42
        const brightness = clamp(0.08 + score * 0.58, 0.08, 0.95)
        _tmpColor.setHSL(0, 0, 0.35 + brightness * 0.2)
        _tmpColor.multiplyScalar(0.38 + brightness * 0.45)

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

    if (packetMesh) {
      let activePackets = 0

      for (let i = packets.length - 1; i >= 0; i--) {
        const packet = packets[i]
        const edge = edges[packet.edgeIndex]
        packet.progress += packet.speed * frameScale
        edge.traffic = clamp(edge.traffic + packet.energy * 0.006 * frameScale, 0, 1.8)

        if (packet.progress >= 1) {
          const destination = nodes[packet.toNode]
          destination.activation = clamp(destination.activation + 0.12 + packet.energy * 0.18, 0, 1.4)
          destination.traffic = clamp(destination.traffic + 0.16 + packet.energy * 0.2, 0, 1.5)
          destination.pulse = clamp(destination.pulse + 0.35, 0, 1.5)
          packets.splice(i, 1)
          continue
        }

        const from = nodes[packet.fromNode]
        const to = nodes[packet.toNode]
        _packetPos.lerpVectors(from.position, to.position, packet.progress)
        _obj.position.copy(_packetPos)
        _obj.scale.setScalar(0.045 + packet.energy * 0.085 + Math.sin(packet.progress * Math.PI) * 0.03)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(activePackets, _obj.matrix)

        _tmpColor.setHSL(0, 0, 0.7 + packet.energy * 0.2)
        _tmpColor.multiplyScalar(1.2 + packet.energy * 0.7)
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
