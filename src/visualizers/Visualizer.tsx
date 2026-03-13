import { useRef, useEffect, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

export interface VisualizerProps {
  frequencyData: React.RefObject<Uint8Array<ArrayBuffer>>
  timeDomainData: React.RefObject<Uint8Array<ArrayBuffer>>
  isActive: boolean
  width: number
  height: number
}

/* ─── Data structures ─── */

interface NodeData {
  position: THREE.Vector3
  velocity: THREE.Vector3
  origin: THREE.Vector3
  freqBand: number
  activation: number
  baseSize: number
  hue: number
  connections: number[]
  pulseTime: number
}

interface PacketData {
  fromNode: number
  toNode: number
  progress: number
  speed: number
  hue: number
}

/* ─── Constants ─── */

const NODE_COUNT = 80
const MAX_PACKETS = 300
// Extended bounds for the horizontal layout
const BOUNDS_X = 18
const BOUNDS_YZ = 8

/* ─── Initialization ─── */

function createNodes(count: number): NodeData[] {
  const nodes: NodeData[] = []

  for (let i = 0; i < count; i++) {
    // Distribute freqbands evenly 0-15
    const freqBand = Math.floor((i / count) * 16)
    
    // Map frequency to position: Bass on the left, Treble on the right
    const normalizedFreq = freqBand / 15 // 0.0 to 1.0
    const x = (normalizedFreq - 0.5) * (BOUNDS_X * 2 * 0.8) + (Math.random() - 0.5) * 4
    const y = (Math.random() - 0.5) * BOUNDS_YZ * 2
    const z = (Math.random() - 0.5) * BOUNDS_YZ * 2

    // Map frequency to hue: Bass = Red/Orange, Mids = Green, Treble = Blue/Purple
    const hue = normalizedFreq * 300 + (Math.random() * 40 - 20)

    nodes.push({
      position: new THREE.Vector3(x, y, z),
      velocity: new THREE.Vector3(
        (Math.random() - 0.5) * 0.02,
        (Math.random() - 0.5) * 0.02,
        (Math.random() - 0.5) * 0.02,
      ),
      origin: new THREE.Vector3(x, y, z),
      freqBand,
      activation: 0,
      baseSize: 0.15 + Math.random() * 0.2, // slightly larger
      hue: (hue + 360) % 360,
      connections: [],
      pulseTime: 0,
    })
  }

  // Wire connections by proximity but favor connections within similar frequency bands
  for (let i = 0; i < count; i++) {
    const sorted: { idx: number; dist: number }[] = []
    for (let j = 0; j < count; j++) {
      if (i === j) continue
      const dist = nodes[i].position.distanceTo(nodes[j].position)
      // Heavily penalize connecting to very distant frequency bands
      const freqPenalty = Math.abs(nodes[i].freqBand - nodes[j].freqBand) * 1.5
      sorted.push({ idx: j, dist: dist + freqPenalty })
    }
    sorted.sort((a, b) => a.dist - b.dist)
    nodes[i].connections = sorted
      .slice(0, 3 + Math.floor(Math.random() * 3)) // 3-5 connections
      .map((s) => s.idx)
  }

  return nodes
}

/* ─── Reusable temp objects ─── */

const _tmpColor = new THREE.Color()
const _obj = new THREE.Object3D()

/* ─── NeuralScene ─── */

interface SceneProps {
  frequencyData: React.RefObject<Uint8Array<ArrayBuffer>>
  timeDomainData: React.RefObject<Uint8Array<ArrayBuffer>>
}

function NeuralScene({ frequencyData, timeDomainData }: SceneProps) {
  const nodesRef = useRef<NodeData[]>([])
  const packetsRef = useRef<PacketData[]>([])

  // Instanced mesh refs
  const nodeMeshRef = useRef<THREE.InstancedMesh>(null)
  const glowMeshRef = useRef<THREE.InstancedMesh>(null)
  const packetMeshRef = useRef<THREE.InstancedMesh>(null)

  // Line geometry ref
  const lineGeoRef = useRef<THREE.BufferGeometry>(null)

  // Pre-allocate line buffers
  const maxLineVerts = NODE_COUNT * 6 * 2
  const linePositions = useMemo(() => new Float32Array(maxLineVerts * 3), [maxLineVerts])
  const lineColors = useMemo(() => new Float32Array(maxLineVerts * 3), [maxLineVerts])

  // Initialize nodes + instance colors on mount
  useEffect(() => {
    nodesRef.current = createNodes(NODE_COUNT)
    packetsRef.current = []

    // Force-initialize instanceColor buffers with bright base colors
    const nodeMesh = nodeMeshRef.current
    const glowMesh = glowMeshRef.current
    const packetMesh = packetMeshRef.current
    const nodes = nodesRef.current

    if (nodeMesh) {
      for (let i = 0; i < nodes.length; i++) {
        _tmpColor.setHSL(nodes[i].hue / 360, 0.8, 0.6)
        nodeMesh.setColorAt(i, _tmpColor)

        _obj.position.copy(nodes[i].position)
        _obj.scale.setScalar(nodes[i].baseSize)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)
      }
      nodeMesh.instanceMatrix.needsUpdate = true
      if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true
    }

    if (glowMesh) {
      for (let i = 0; i < nodes.length; i++) {
        _tmpColor.setHSL(nodes[i].hue / 360, 0.9, 0.5)
        _tmpColor.multiplyScalar(0.2)
        glowMesh.setColorAt(i, _tmpColor)

        _obj.position.copy(nodes[i].position)
        _obj.scale.setScalar(nodes[i].baseSize * 4)
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
        _tmpColor.set(0, 0, 0)
        packetMesh.setColorAt(i, _tmpColor)
      }
      packetMesh.instanceMatrix.needsUpdate = true
      if (packetMesh.instanceColor) packetMesh.instanceColor.needsUpdate = true
    }
  }, [])

  useFrame(({ clock }) => {
    const nodes = nodesRef.current
    const packets = packetsRef.current
    if (!nodes.length) return

    const freqData = frequencyData.current
    const timeData = timeDomainData.current
    const t = clock.elapsedTime

    // ── Audio analysis ──
    let waveAmp = 0
    for (let i = 0; i < timeData.length; i++) {
      waveAmp += Math.abs(timeData[i] - 128)
    }
    waveAmp /= timeData.length * 128

    const bandCount = 16
    const bandEnergies = new Float32Array(bandCount)
    const bandSize = Math.floor(freqData.length / bandCount)
    for (let b = 0; b < bandCount; b++) {
      let sum = 0
      for (let i = 0; i < bandSize; i++) {
        sum += freqData[b * bandSize + i]
      }
      // Apply exponential frequency weighting: high frequencies naturally have
      // exponentially less amplitude than bass, so we need a massive boost for treble
      // b=0 (bass) multiplier is 1.0; b=15 (highest treble) multiplier is ~57.0
      const energyMultiplier = 1 + Math.pow(b / 2, 2)
      bandEnergies[b] = Math.min(1.0, (sum / (bandSize * 255)) * energyMultiplier)
    }

    const breathing = 1 + waveAmp * 0.5

    // ── Update node physics ──
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      const bandEnergy = bandEnergies[node.freqBand]
      node.activation += (bandEnergy - node.activation) * 0.15

      // Deterministic pseudo-random trigger based on time, node index, and audio
      // This ensures consistent packet spawning that still looks organic
      const timeHash = (Math.floor(t * 10) * 13.7 + i * 29.3) % 1

      // Spawn packets on strong activation
      if (node.activation > 0.3 && timeHash < node.activation * 0.3) {
        node.pulseTime = 1
        for (let ci = 0; ci < node.connections.length; ci++) {
          const connIdx = node.connections[ci]
          // Deterministic check per connection
          const connHash = (timeHash * 43.1 + ci * 17.9) % 1
          if (packets.length < MAX_PACKETS && connHash < 0.35) {
            // Speed derived from band energy — higher energy = faster packets
            const speedVal = bandEnergy * 0.025 + (1 - bandEnergy) * 0.015
            packets.push({
              fromNode: i,
              toNode: connIdx,
              progress: 0,
              speed: speedVal,
              hue: node.hue,
            })
          }
        }
      }

      node.pulseTime *= 0.92

      // Spring physics to hover around origin
      const dx = node.origin.x - node.position.x
      const dy = node.origin.y - node.position.y
      const dz = node.origin.z - node.position.z

      // Pull back to origin, stronger when far away
      node.velocity.x += dx * 0.002
      node.velocity.y += dy * 0.002
      node.velocity.z += dz * 0.002

      // Add a slight organic swirling force
      node.velocity.x += Math.sin(t * 0.5 + i) * 0.001
      node.velocity.y += Math.cos(t * 0.6 + i) * 0.001
      node.velocity.z += Math.sin(t * 0.7 - i) * 0.001

      // Dampen velocity to prevent infinite acceleration
      node.velocity.multiplyScalar(0.95)

      // Apply velocity
      node.position.add(node.velocity)
    }

    // ── Update instanced node meshes ──
    const nodeMesh = nodeMeshRef.current
    const glowMesh = glowMeshRef.current
    if (nodeMesh && glowMesh) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        const size = node.baseSize * breathing * (1 + node.activation * 1.5 + node.pulseTime * 1.2)

        // Core sphere
        _obj.position.copy(node.position)
        _obj.scale.setScalar(size)
        _obj.updateMatrix()
        nodeMesh.setMatrixAt(i, _obj.matrix)

        // Color: vibrant and bright via HDR
        const L = 0.5 + node.activation * 0.35
        _tmpColor.setHSL(node.hue / 360, 0.85, L)
        _tmpColor.r *= 1.0 + node.activation * 1.0
        _tmpColor.g *= 1.0 + node.activation * 1.0
        _tmpColor.b *= 1.0 + node.activation * 1.0
        nodeMesh.setColorAt(i, _tmpColor)

        // Glow halo — barely extends past the core
        const glowSize = size * (1.15 + node.activation * 0.15)
        _obj.scale.setScalar(glowSize)
        _obj.updateMatrix()
        glowMesh.setMatrixAt(i, _obj.matrix)

        const gb = 0.04 + node.activation * 0.12
        _tmpColor.setHSL(node.hue / 360, 0.7, 0.4 + node.activation * 0.2)
        _tmpColor.r *= gb
        _tmpColor.g *= gb
        _tmpColor.b *= gb
        glowMesh.setColorAt(i, _tmpColor)
      }
      nodeMesh.instanceMatrix.needsUpdate = true
      nodeMesh.instanceColor!.needsUpdate = true
      glowMesh.instanceMatrix.needsUpdate = true
      glowMesh.instanceColor!.needsUpdate = true
    }

    // ── Update connection lines ──
    const lineGeo = lineGeoRef.current
    if (lineGeo) {
      let vi = 0
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        for (const connIdx of node.connections) {
          if (connIdx <= i) continue
          const other = nodes[connIdx]
          const dist = node.position.distanceTo(other.position)
          if (dist > BOUNDS_X * 0.9) continue

          const act = (node.activation + other.activation) / 2
          const brightness = 0.2 + act * 0.8

          // vertex 1
          linePositions[vi * 3] = node.position.x
          linePositions[vi * 3 + 1] = node.position.y
          linePositions[vi * 3 + 2] = node.position.z
          _tmpColor.setHSL((node.hue + other.hue) / 720, 0.6, 0.35 + act * 0.35)
          _tmpColor.r *= brightness
          _tmpColor.g *= brightness
          _tmpColor.b *= brightness
          lineColors[vi * 3] = _tmpColor.r
          lineColors[vi * 3 + 1] = _tmpColor.g
          lineColors[vi * 3 + 2] = _tmpColor.b
          vi++

          // vertex 2
          linePositions[vi * 3] = other.position.x
          linePositions[vi * 3 + 1] = other.position.y
          linePositions[vi * 3 + 2] = other.position.z
          lineColors[vi * 3] = _tmpColor.r
          lineColors[vi * 3 + 1] = _tmpColor.g
          lineColors[vi * 3 + 2] = _tmpColor.b
          vi++
        }
      }
      // Zero out unused
      for (let i = vi * 3; i < linePositions.length; i++) {
        linePositions[i] = 0
        lineColors[i] = 0
      }

      const posAttr = lineGeo.getAttribute('position') as THREE.BufferAttribute | null
      if (!posAttr) {
        lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3))
        lineGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3))
      } else {
        posAttr.set(linePositions)
        posAttr.needsUpdate = true
        const colAttr = lineGeo.getAttribute('color') as THREE.BufferAttribute
        colAttr.set(lineColors)
        colAttr.needsUpdate = true
      }
      lineGeo.setDrawRange(0, vi)
    }

    // ── Update packets ──
    const packetMesh = packetMeshRef.current
    if (packetMesh) {
      let activeCount = 0

      for (let i = packets.length - 1; i >= 0; i--) {
        const pkt = packets[i]
        pkt.progress += pkt.speed

        if (pkt.progress >= 1) {
          const dest = nodes[pkt.toNode]
          dest.activation = Math.min(1, dest.activation + 0.15)
          dest.pulseTime = Math.min(1, dest.pulseTime + 0.3)
          packets.splice(i, 1)
          continue
        }

        const from = nodes[pkt.fromNode]
        const to = nodes[pkt.toNode]
        const t = pkt.progress
        _obj.position.set(
          from.position.x + (to.position.x - from.position.x) * t,
          from.position.y + (to.position.y - from.position.y) * t,
          from.position.z + (to.position.z - from.position.z) * t,
        )
        _obj.scale.setScalar(0.15 + Math.sin(t * Math.PI) * 0.1)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(activeCount, _obj.matrix)

        // Packets — moderate brightness
        _tmpColor.setHSL(pkt.hue / 360, 0.9, 0.7)
        _tmpColor.r *= 1.2
        _tmpColor.g *= 1.2
        _tmpColor.b *= 1.2
        packetMesh.setColorAt(activeCount, _tmpColor)

        activeCount++
      }

      // Hide unused
      for (let i = activeCount; i < MAX_PACKETS; i++) {
        _obj.position.set(0, 0, 0)
        _obj.scale.setScalar(0)
        _obj.updateMatrix()
        packetMesh.setMatrixAt(i, _obj.matrix)
      }

      packetMesh.count = Math.max(activeCount, 1)
      packetMesh.instanceMatrix.needsUpdate = true
      if (packetMesh.instanceColor) packetMesh.instanceColor.needsUpdate = true
    }
  })

  return (
    <>
      {/* ── Connections ── */}
      <lineSegments frustumCulled={false}>
        <bufferGeometry ref={lineGeoRef} />
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={1}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </lineSegments>

      {/* ── Node cores ── */}
      <instancedMesh
        ref={nodeMeshRef}
        args={[undefined, undefined, NODE_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 16, 12]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      {/* ── Glow halos ── */}
      <instancedMesh
        ref={glowMeshRef}
        args={[undefined, undefined, NODE_COUNT]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1, 10, 8]} />
        <meshBasicMaterial
          transparent
          opacity={0.1}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </instancedMesh>

      {/* ── Packets ── */}
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

/* ─── Visualizer ─── */

export function Visualizer({
  frequencyData,
  timeDomainData,
  isActive,
  width,
  height,
}: VisualizerProps) {
  return (
    <div style={{ width, height, position: 'relative', background: '#030108' }}>
      <Canvas
        camera={{ position: [0, 0, 30], fov: 60, near: 0.1, far: 200 }}
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
        <color attach="background" args={['#030108']} />

        <ambientLight intensity={0.1} />
        <pointLight position={[20, 20, 20]} intensity={0.4} color="#6366f1" />
        <pointLight position={[-20, -10, -15]} intensity={0.3} color="#ec4899" />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          autoRotate
          autoRotateSpeed={0.4}
          minDistance={8}
          maxDistance={80}
          enablePan
        />

        {isActive && (
          <NeuralScene
            frequencyData={frequencyData}
            timeDomainData={timeDomainData}
          />
        )}

        <EffectComposer>
          <Bloom
            intensity={0.8}
            luminanceThreshold={0.4}
            luminanceSmoothing={0.8}
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
          <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#a3a3a3' }}>
            {'> awaiting microphone input...'}
          </span>
        </div>
      )}
    </div>
  )
}
