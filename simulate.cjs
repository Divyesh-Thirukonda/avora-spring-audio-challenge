const fs = require('fs');
const path = require('path');

// --- CALIBRATION ---
const CALIBRATION = {
  sampleIntervalMs: 33,
  frequencyGroups: [
    [0, 2], [2, 4], [4, 8], [8, 16], [16, 32],
    [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024]
  ],
  persistenceWindowsMs: [33, 62, 117, 221, 416, 785, 1479, 2788, 5256, 9906],
};
const NUM_BANDS = CALIBRATION.frequencyGroups.length;

const TRAFFIC_DECAY_MS = CALIBRATION.persistenceWindowsMs[0];
const ACTIVATION_DECAY_MS = CALIBRATION.persistenceWindowsMs[2];
const PULSE_DECAY_MS = CALIBRATION.persistenceWindowsMs[1];
const EDGE_DECAY_MS = CALIBRATION.persistenceWindowsMs[3];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function adaptiveAlpha(windowMs, frameMs) {
  if (windowMs <= 0 || frameMs <= 0) return 0;
  return 1 - Math.exp(-frameMs / windowMs);
}

// --- STREAM STATS ---
class StreamStats {
  constructor(windowFrames) {
    this.mean = 0;
    this.varAcc = 0;
    this.count = 0;
    this.alpha = 2 / (windowFrames + 1);
  }
  push(value) {
    if (this.count === 0) {
      this.mean = value;
      this.varAcc = 0;
      this.count = 1;
      return;
    }
    this.count++;
    const delta = value - this.mean;
    this.mean += this.alpha * delta;
    this.varAcc += this.alpha * (delta * delta - this.varAcc);
  }
  get std() { return Math.sqrt(Math.max(0, this.varAcc)); }
  zScore(value) {
    const s = this.std;
    if (s < 1e-8) return value > this.mean ? 1 : 0;
    return (value - this.mean) / s;
  }
  normalize(value) {
    const s = this.std;
    if (s < 1e-8) return 0.5;
    const lo = this.mean - s;
    const hi = this.mean + 2 * s;
    if (hi <= lo) return 0.5;
    return Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
  }
}

class StreamAnalyzer {
  constructor() {
    const windowFrames = 90;
    this.bandStats = Array.from({ length: NUM_BANDS }, () => new StreamStats(windowFrames));
    this.bandOnsetStats = Array.from({ length: NUM_BANDS }, () => new StreamStats(windowFrames));
    this.energy = new StreamStats(windowFrames);
    this.centroid = new StreamStats(windowFrames);
    this.harmonicRatio = new StreamStats(windowFrames);
    this.spectralSpread = new StreamStats(windowFrames);
    this.flux = new StreamStats(windowFrames);
    this.crest = new StreamStats(windowFrames);
    this.rms = new StreamStats(windowFrames);
    this.prevBandEnergies = new Float32Array(NUM_BANDS);
    this.prevSpectrum = new Float32Array(1024);
    this.rmsRing = new Float32Array(6);
    this.rmsRingIdx = 0;
  }
}

// --- GRAPH SIMULATION ---
const GRID_SIZE = 10;
const NODE_COUNT = GRID_SIZE * GRID_SIZE * GRID_SIZE;
function nodeIndex(x, y, z) {
  return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
}

function createGraph() {
  const nodes = [];
  const edges = [];

  for (let z = 0; z < GRID_SIZE; z++) {
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        nodes.push({
          index: nodeIndex(x, y, z),
          gridX: x, gridY: y, gridZ: z,
          activation: 0, traffic: 0, pulse: 0,
          freqBand: x,
          harmonicTrait: y / (GRID_SIZE - 1),
          fluxTrait: z / (GRID_SIZE - 1),
          neighbors: [], edgeIds: [],
        });
      }
    }
  }

  const addEdge = (a, b) => {
    const edgeIndex = edges.length;
    edges.push({ a, b, traffic: 0 });
    nodes[a].neighbors.push(b);
    nodes[a].edgeIds.push(edgeIndex);
    nodes[b].neighbors.push(a);
    nodes[b].edgeIds.push(edgeIndex);
  };

  for (let z = 0; z < GRID_SIZE; z++) {
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const current = nodeIndex(x, y, z);
        if (x + 1 < GRID_SIZE) addEdge(current, nodeIndex(x + 1, y, z));
        if (y + 1 < GRID_SIZE) addEdge(current, nodeIndex(x, y + 1, z));
        if (z + 1 < GRID_SIZE) addEdge(current, nodeIndex(x, y, z + 1));
      }
    }
  }

  return { nodes, edges };
}

// --- SIMULATION ---

function runSimulation() {
  const dataPath = path.resolve(__dirname, 'test/humanvoice.json');
  const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  const frames = data.frames;

  const analyzer = new StreamAnalyzer();
  const bandEnergies = new Float32Array(NUM_BANDS);
  const bandOnsets = new Float32Array(NUM_BANDS);
  const bandOnsetZScores = new Float32Array(NUM_BANDS);
  
  const graph = createGraph();
  const nodes = graph.nodes;
  const edges = graph.edges;
  const packets = [];

  console.log(`Loaded ${frames.length} frames.`);

  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    const freqData = frame.frequency;
    
    // TimeDomain data simulation string
    const timeDomainData = frame.timeDomain || new Uint8Array(2048).fill(128); // dummy
    
    // Feature extraction logic (simplified slightly, reusing previous implementation)
    const fLen = freqData.length;
    const tLen = timeDomainData.length;
    
    let totalFreqSum = 0; let totalFreqSqSum = 0;
    for (let i = 0; i < fLen; i++) {
      const v = freqData[i] / 255;
      totalFreqSum += v; totalFreqSqSum += v * v;
    }

    let maxOnsetZ = -Infinity;
    let dominantOnsetBand = 0;

    for (let b = 0; b < NUM_BANDS; b++) {
      const [start, end] = CALIBRATION.frequencyGroups[b];
      const binStart = Math.min(start, fLen);
      const binEnd = Math.min(end, fLen);
      if (binEnd <= binStart) continue;
      let sum = 0;
      for (let i = binStart; i < binEnd; i++) { sum += freqData[i] / 255; }
      const rawBandEnergy = sum / (binEnd - binStart);

      analyzer.bandStats[b].push(rawBandEnergy);
      bandEnergies[b] = analyzer.bandStats[b].normalize(rawBandEnergy);

      const rawOnset = Math.max(0, rawBandEnergy - analyzer.prevBandEnergies[b]);
      analyzer.prevBandEnergies[b] = rawBandEnergy;

      analyzer.bandOnsetStats[b].push(rawOnset);
      bandOnsets[b] = rawOnset;

      const z = analyzer.bandOnsetStats[b].zScore(rawOnset);
      bandOnsetZScores[b] = z;

      if (z > maxOnsetZ) {
        maxOnsetZ = z;
        dominantOnsetBand = b;
      }
    }

    const rawEnergy = fLen > 0 ? Math.sqrt(totalFreqSqSum / fLen) : 0;
    analyzer.energy.push(rawEnergy);
    const energy = analyzer.energy.normalize(rawEnergy);
    
    let weightedSum = 0;
    for (let i = 0; i < fLen; i++) weightedSum += i * (freqData[i] / 255);
    const rawCentroid = totalFreqSum > 0 ? (weightedSum / totalFreqSum) / fLen : 0;
    analyzer.centroid.push(rawCentroid);
    const centroid = analyzer.centroid.normalize(rawCentroid);

    // DUMMY SOME FEATURES since we didn't implement all exact math here and they are less important for glow bug
    const harmonicRatio = 0.5;
    const spectralSpread = 0.5;
    let fluxSum = 0;
    for (let i = 0; i < fLen; i++) {
        const curr = freqData[i] / 255;
        const diff = curr - analyzer.prevSpectrum[i];
        fluxSum += diff * diff;
        analyzer.prevSpectrum[i] = curr;
    }
    const rawFlux = fluxSum / fLen;
    analyzer.flux.push(rawFlux);
    const flux = analyzer.flux.normalize(rawFlux);
    
    let sumSq = 0; let peak = 0;
    for (let i = 0; i < tLen; i++) {
      const sample = (timeDomainData[i] - 128) / 128;
      sumSq += sample * sample;
      const absSample = Math.abs(sample);
      if (absSample > peak) peak = absSample;
    }
    const rawRms = Math.sqrt(sumSq / tLen);
    analyzer.rms.push(rawRms);
    const rawCrest = rawRms > 0.001 ? peak / rawRms : 1;
    analyzer.crest.push(rawCrest);
    const crest = analyzer.crest.normalize(rawCrest);
    const attackSlope = 0.5; // dummy

    const delta = 0.033; // 33ms frame
    const frameMs = delta * 1000;
    const trafficAlpha = adaptiveAlpha(TRAFFIC_DECAY_MS, frameMs);
    const activationAlpha = adaptiveAlpha(ACTIVATION_DECAY_MS, frameMs);
    const pulseAlpha = adaptiveAlpha(PULSE_DECAY_MS, frameMs);
    const edgeDecayAlpha = adaptiveAlpha(EDGE_DECAY_MS, frameMs);

    // ======================================
    // 2. UPDATE NODE ACTIVATIONS
    // ======================================
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const onsetZ = bandOnsetZScores[node.freqBand];
      const xActivation = clamp(onsetZ * 0.5, 0, 1.2);
      const bandWarmth = bandEnergies[node.freqBand] * 0.35;
      
      const t_y = node.harmonicTrait;
      const harmonicMatch = harmonicRatio * (1 - t_y);
      const inharmonicMatch = spectralSpread * t_y;
      const yActivation = harmonicMatch + inharmonicMatch;

      const t_z = node.fluxTrait;
      const sustainedMatch = (1 - flux) * (1 - t_z);
      const transientMatch = ((flux + crest + attackSlope) / 3) * t_z;
      const zActivation = sustainedMatch + transientMatch;

      const targetActivation = clamp(
        (xActivation + bandWarmth) *
        (0.6 + 0.4 * yActivation) *
        (0.6 + 0.4 * zActivation) *
        (0.5 + energy * 0.8),
        0, 1.4
      );

      const blended = targetActivation + node.traffic * 0.25 + node.pulse * 0.3;
      node.activation += (blended - node.activation) * activationAlpha;
      node.traffic *= (1 - trafficAlpha);
      node.pulse *= (1 - pulseAlpha);
    }

    for (let i = 0; i < edges.length; i++) {
        edges[i].traffic *= (1 - edgeDecayAlpha);
    }

    // ======================================
    // 4. SPAWN PACKETS
    // ======================================
    const MAX_PACKETS = 220;
    const frameScale = Math.min(delta * 60, 1.5);
    
    let spawnedCount = 0;
    for (let band = 0; band < NUM_BANDS && packets.length < MAX_PACKETS; band++) {
      const onsetZ = bandOnsetZScores[band];
      if (onsetZ < 1.0) continue;
      const numCandidates = Math.ceil(onsetZ);
      const spawnChance = clamp(energy * 0.6 + 0.15, 0, 1) * (frameMs / CALIBRATION.sampleIntervalMs);

      for (let c = 0; c < numCandidates && packets.length < MAX_PACKETS; c++) {
        if (Math.random() >= spawnChance) continue;

        const gy = Math.floor(Math.random() * GRID_SIZE);
        const gz = Math.floor(Math.random() * GRID_SIZE);
        const nodeIdx = nodeIndex(band, gy, gz);
        const node = nodes[nodeIdx];
        
        let totalWeight = 0;
        const weights = new Array(node.neighbors.length);
        for (let j = 0; j < node.neighbors.length; j++) {
            const neighbor = nodes[node.neighbors[j]];
            const edge = edges[node.edgeIds[j]];
            const lateralPull = Math.abs(node.gridX - dominantOnsetBand) > Math.abs(neighbor.gridX - dominantOnsetBand) ? 1.5 : 0.3;
            const activationPull = neighbor.activation * neighbor.activation;
            const weight = lateralPull + activationPull * 2 + edge.traffic * 0.8 + 0.05;
            weights[j] = weight;
            totalWeight += weight;
        }
        if (totalWeight <= 0) continue;

        let selection = Math.random() * totalWeight;
        let targetSlot = 0;
        for (let j = 0; j < weights.length; j++) {
            selection -= weights[j];
            if (selection <= 0) { targetSlot = j; break; }
        }

        const edgeIndex = node.edgeIds[targetSlot];
        const toNode = node.neighbors[targetSlot];
        const edge = edges[edgeIndex];

        const packetSpeed = 0.008 + centroid * 0.032;
        const packetEnergy = clamp((onsetZ / 3) * node.activation * (0.6 + crest * 0.4), 0.05, 1.2);

        edge.traffic = clamp(edge.traffic + packetEnergy * 0.12, 0, 1.8);
        node.traffic = clamp(node.traffic + packetEnergy * 0.08, 0, 1.4);
        node.pulse = clamp(node.pulse + (onsetZ / 3) * 0.5, 0, 1.4);
        
        packets.push({ edgeIndex, fromNode: node.index, toNode, progress: 0, speed: packetSpeed, energy: packetEnergy, band });
        spawnedCount++;
      }
    }

    // ======================================
    // 7. UPDATE PACKETS
    // ======================================
    for (let i = packets.length - 1; i >= 0; i--) {
        const packet = packets[i];
        const edge = edges[packet.edgeIndex];
        packet.progress += packet.speed * frameScale;
        edge.traffic = clamp(edge.traffic + packet.energy * 0.004 * frameScale, 0, 1.8);

        if (packet.progress >= 1) {
            const destination = nodes[packet.toNode];
            destination.activation = clamp(destination.activation + packet.energy * 0.45, 0, 1.4);
            destination.traffic = clamp(destination.traffic + packet.energy * 0.35, 0, 1.5);
            destination.pulse = clamp(destination.pulse + packet.energy * crest * 0.5, 0, 1.5);
            packets.splice(i, 1);
        }
    }

    // Let's log if there was significant network activity
    let activeNodes = 0;
    let maxActivation = 0;
    let totalActivation = 0;
    for (let i = 0; i < nodes.length; i++) {
        const act = clamp(nodes[i].activation + nodes[i].traffic * 0.4 + nodes[i].pulse * 0.35, 0, 1.5);
        if (act > 0.4) activeNodes++;
        if (act > maxActivation) maxActivation = act;
        totalActivation += act;
    }

    if (spawnedCount > 0 || maxActivation > 0.5) {
        console.log(`[Frame ${f.toString().padStart(4, '0')}] Spawns: ${spawnedCount} | Packets Active: ${packets.length} | Active Nodes (>0.4): ${activeNodes} | Max Activation: ${maxActivation.toFixed(2)} | AVG Activation: ${(totalActivation / 1000).toFixed(3)}`);
    } else if (f % 50 === 0) {
        console.log(`[Frame ${f.toString().padStart(4, '0')}] (silent) Max Activation: ${maxActivation.toFixed(2)}`);
    }
  }

  console.log(`Simulation verified.`);
}

runSimulation();
