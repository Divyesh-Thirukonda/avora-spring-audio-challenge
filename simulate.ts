import fs from 'fs';
import path from 'path';

// --- CALIBRATION ---
const CALIBRATION = {
  sampleIntervalMs: 33,
  frequencyGroups: [
    [0, 2], [2, 4], [4, 8], [8, 16], [16, 32],
    [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024]
  ],
};
const NUM_BANDS = CALIBRATION.frequencyGroups.length;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

// --- STREAM STATS ---
class StreamStats {
  mean: number = 0;
  varAcc: number = 0;
  count: number = 0;
  alpha: number;

  constructor(windowFrames: number) {
    this.alpha = 2 / (windowFrames + 1);
  }

  push(value: number) {
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

  get std(): number {
    return Math.sqrt(Math.max(0, this.varAcc));
  }

  zScore(value: number): number {
    const s = this.std;
    if (s < 1e-8) return value > this.mean ? 1 : 0;
    return (value - this.mean) / s;
  }

  normalize(value: number): number {
    const s = this.std;
    if (s < 1e-8) return 0.5;
    const lo = this.mean - s;
    const hi = this.mean + 2 * s;
    if (hi <= lo) return 0.5;
    return Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
  }
}

class StreamAnalyzer {
  bandStats: StreamStats[];
  bandOnsetStats: StreamStats[];
  energy: StreamStats;
  centroid: StreamStats;
  harmonicRatio: StreamStats;
  spectralSpread: StreamStats;
  flux: StreamStats;
  crest: StreamStats;
  rms: StreamStats;

  prevBandEnergies: Float32Array;
  prevSpectrum: Float32Array;
  rmsRing: Float32Array;
  rmsRingIdx: number;

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

// --- SIMULATION ---

function runSimulation() {
  const dataPath = path.resolve(__dirname, 'test/humanvoice.json');
  const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  const frames = data.frames;

  const analyzer = new StreamAnalyzer();
  const bandEnergies = new Float32Array(NUM_BANDS);
  const bandOnsets = new Float32Array(NUM_BANDS);
  const bandOnsetZScores = new Float32Array(NUM_BANDS);

  console.log(`Loaded ${frames.length} frames.`);

  let totalPacketsSpawned = 0;
  let framesWithSpawns = 0;

  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    const freqData = frame.frequency;
    
    // Feature extraction logic
    const fLen = freqData.length;
    let totalFreqSum = 0;
    let totalFreqSqSum = 0;
    for (let i = 0; i < fLen; i++) {
      const v = freqData[i] / 255;
      totalFreqSum += v;
      totalFreqSqSum += v * v;
    }

    let maxOnsetZ = -Infinity;
    let dominantOnsetBand = 0;

    for (let b = 0; b < NUM_BANDS; b++) {
      const [start, end] = CALIBRATION.frequencyGroups[b];
      const binStart = Math.min(start, fLen);
      const binEnd = Math.min(end, fLen);
      if (binEnd <= binStart) continue;
      let sum = 0;
      for (let i = binStart; i < binEnd; i++) {
        sum += freqData[i] / 255;
      }
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
    
    // Simulate packet spawning
    let spawnedInFrame = 0;
    let spawnLog = [];
    const frameMs = 33; // approx

    for (let band = 0; band < NUM_BANDS; band++) {
      const onsetZ = bandOnsetZScores[band];
      if (onsetZ < 1.0) continue; // Threshold

      const numCandidates = Math.ceil(onsetZ);
      const spawnChance = clamp(energy * 0.6 + 0.15, 0, 1);
      
      for (let c = 0; c < numCandidates; c++) {
        if (Math.random() >= spawnChance) continue;
        spawnedInFrame++;
        totalPacketsSpawned++;
        spawnLog.push(`[Band ${band} (Z: ${onsetZ.toFixed(2)})]`);
      }
    }

    if (spawnedInFrame > 0) {
      framesWithSpawns++;
      console.log(`Frame ${f.toString().padStart(4, '0')} (t: ${(f * 33) / 1000}s): Spawned ${spawnedInFrame} packets -> ${spawnLog.join(', ')}`);
    } else if (f % 50 === 0) {
       // Just periodically log quiet frames
       console.log(`Frame ${f.toString().padStart(4, '0')} (t: ${(f * 33) / 1000}s): (quiet)`);
    }
  }

  console.log(`\nSimulation complete.`);
  console.log(`Total duration: ${(frames.length * 33) / 1000}s`);
  console.log(`Frames with spawns: ${framesWithSpawns} / ${frames.length} (${((framesWithSpawns / frames.length) * 100).toFixed(1)}%)`);
  console.log(`Total packets spawned: ${totalPacketsSpawned}`);
  const expectedPacketsPerSec = totalPacketsSpawned / ((frames.length * 33) / 1000);
  console.log(`Average packets per second: ${expectedPacketsPerSec.toFixed(2)}`);
}

runSimulation();
