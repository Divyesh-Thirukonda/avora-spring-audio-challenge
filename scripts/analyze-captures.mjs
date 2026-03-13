import fs from 'node:fs'
import path from 'node:path'

const GRID_SIZE = 10
const TEST_DIR = path.resolve('test')
const files = fs
  .readdirSync(TEST_DIR)
  .filter((file) => file.endsWith('.json'))
  .sort()

if (!files.length) {
  throw new Error('No capture files found in ./test')
}

function quantile(sorted, q) {
  if (!sorted.length) return 0
  const index = (sorted.length - 1) * q
  const low = Math.floor(index)
  const high = Math.ceil(index)
  if (low === high) return sorted[low]
  const t = index - low
  return sorted[low] * (1 - t) + sorted[high] * t
}

function summarize(values) {
  const sorted = [...values].sort((a, b) => a - b)
  return {
    min: quantile(sorted, 0),
    p10: quantile(sorted, 0.1),
    p25: quantile(sorted, 0.25),
    p50: quantile(sorted, 0.5),
    p75: quantile(sorted, 0.75),
    p90: quantile(sorted, 0.9),
    p95: quantile(sorted, 0.95),
    p99: quantile(sorted, 0.99),
    max: quantile(sorted, 1),
  }
}

function createLogFrequencyGroups(binCount, groupCount) {
  const maxBin = Math.max(2, binCount - 1)
  const boundaries = [1]
  for (let i = 1; i <= groupCount; i++) {
    const t = i / groupCount
    boundaries.push(Math.round(Math.exp(Math.log(maxBin) * t)))
  }

  const groups = []
  for (let i = 0; i < groupCount; i++) {
    const start = i === 0 ? 0 : boundaries[i]
    const end = Math.max(start + 1, boundaries[i + 1])
    groups.push([start, Math.min(end, binCount)])
  }

  groups[groupCount - 1][1] = binCount
  return groups
}

function computeFrameFeatures(frame, groups, emaState, emaAlphas) {
  const frequency = frame.frequency
  const timeDomain = frame.timeDomain
  const groupEnergy = new Array(groups.length).fill(0)

  let totalEnergy = 0
  let centroidNumerator = 0
  let spectralSquared = 0
  for (let i = 0; i < frequency.length; i++) {
    const value = frequency[i] / 255
    totalEnergy += value
    spectralSquared += value * value
    centroidNumerator += i * value
  }

  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const [start, end] = groups[groupIndex]
    let sum = 0
    for (let i = start; i < end; i++) {
      sum += frequency[i] / 255
    }
    groupEnergy[groupIndex] = sum / Math.max(1, end - start)
  }

  let waveAmp = 0
  let zeroCrossings = 0
  let crestPeak = 0
  let waveSquared = 0
  for (let i = 0; i < timeDomain.length; i++) {
    const centered = (timeDomain[i] - 128) / 128
    const absValue = Math.abs(centered)
    waveAmp += absValue
    waveSquared += centered * centered
    crestPeak = Math.max(crestPeak, absValue)
    if (i > 0) {
      const previous = (timeDomain[i - 1] - 128) / 128
      if ((previous >= 0 && centered < 0) || (previous < 0 && centered >= 0)) {
        zeroCrossings += 1
      }
    }
  }

  waveAmp /= Math.max(1, timeDomain.length)
  const waveRms = Math.sqrt(waveSquared / Math.max(1, timeDomain.length))
  const crest = waveRms > 0 ? crestPeak / waveRms : 0
  const centroid = totalEnergy > 0 ? centroidNumerator / totalEnergy / Math.max(1, frequency.length - 1) : 0

  let entropy = 0
  if (totalEnergy > 0) {
    for (let i = 0; i < frequency.length; i++) {
      const probability = frequency[i] / 255 / totalEnergy
      if (probability > 0) entropy -= probability * Math.log2(probability)
    }
    entropy /= Math.log2(frequency.length)
  }

  const residualByPersistence = new Array(emaAlphas.length).fill(0)
  const deltaByPersistence = new Array(emaAlphas.length).fill(0)
  for (let persistenceIndex = 0; persistenceIndex < emaAlphas.length; persistenceIndex++) {
    let residualSum = 0
    let deltaSum = 0
    const alpha = emaAlphas[persistenceIndex]
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
      const previous = emaState[persistenceIndex][groupIndex]
      const next = previous + (groupEnergy[groupIndex] - previous) * alpha
      deltaSum += Math.abs(next - previous)
      residualSum += Math.abs(groupEnergy[groupIndex] - next)
      emaState[persistenceIndex][groupIndex] = next
    }
    residualByPersistence[persistenceIndex] = residualSum / groups.length
    deltaByPersistence[persistenceIndex] = deltaSum / groups.length
  }

  return {
    groupEnergy,
    energy: totalEnergy / frequency.length,
    spectralRms: Math.sqrt(spectralSquared / Math.max(1, frequency.length)),
    centroid,
    entropy,
    waveAmp,
    waveRms,
    zeroCrossRate: zeroCrossings / Math.max(1, timeDomain.length),
    crest,
    residualByPersistence,
    deltaByPersistence,
  }
}

function buildCalibration() {
  const sampleDurations = []
  const datasets = []
  const firstFile = JSON.parse(fs.readFileSync(path.join(TEST_DIR, files[0]), 'utf8'))
  const sampleIntervalMs = firstFile.sampleIntervalMs
  const frequencyGroups = createLogFrequencyGroups(firstFile.frequencyBinCount, GRID_SIZE)

  for (const file of files) {
    const parsed = JSON.parse(fs.readFileSync(path.join(TEST_DIR, file), 'utf8'))
    sampleDurations.push(parsed.durationMs)
    datasets.push(parsed)
  }

  const durationSummary = summarize(sampleDurations)
  const maxPersistenceMs = Math.max(sampleIntervalMs, Math.min(12000, Math.round(durationSummary.p75 / 4)))
  const persistenceWindowsMs = Array.from({ length: GRID_SIZE }, (_, index) => {
    const t = index / (GRID_SIZE - 1)
    return Math.round(sampleIntervalMs * Math.exp(Math.log(maxPersistenceMs / sampleIntervalMs) * t))
  })
  const emaAlphas = persistenceWindowsMs.map((windowMs) => 1 - Math.exp(-sampleIntervalMs / windowMs))

  const featureSeries = {
    energy: [],
    spectralRms: [],
    centroid: [],
    entropy: [],
    waveAmp: [],
    waveRms: [],
    zeroCrossRate: [],
    crest: [],
    flux: [],
  }
  const groupEnergySeries = Array.from({ length: GRID_SIZE }, () => [])
  const residualSeries = Array.from({ length: GRID_SIZE }, () => [])
  const deltaSeries = Array.from({ length: GRID_SIZE }, () => [])

  for (const dataset of datasets) {
    const emaState = Array.from({ length: GRID_SIZE }, () => new Array(GRID_SIZE).fill(0))
    let previousGroupEnergy = null

    for (const frame of dataset.frames) {
      const features = computeFrameFeatures(frame, frequencyGroups, emaState, emaAlphas)
      featureSeries.energy.push(features.energy)
      featureSeries.spectralRms.push(features.spectralRms)
      featureSeries.centroid.push(features.centroid)
      featureSeries.entropy.push(features.entropy)
      featureSeries.waveAmp.push(features.waveAmp)
      featureSeries.waveRms.push(features.waveRms)
      featureSeries.zeroCrossRate.push(features.zeroCrossRate)
      featureSeries.crest.push(features.crest)

      for (let groupIndex = 0; groupIndex < GRID_SIZE; groupIndex++) {
        groupEnergySeries[groupIndex].push(features.groupEnergy[groupIndex])
      }

      for (let persistenceIndex = 0; persistenceIndex < GRID_SIZE; persistenceIndex++) {
        residualSeries[persistenceIndex].push(features.residualByPersistence[persistenceIndex])
        deltaSeries[persistenceIndex].push(features.deltaByPersistence[persistenceIndex])
      }

      if (previousGroupEnergy) {
        let positiveFlux = 0
        for (let groupIndex = 0; groupIndex < GRID_SIZE; groupIndex++) {
          positiveFlux += Math.max(0, features.groupEnergy[groupIndex] - previousGroupEnergy[groupIndex])
        }
        featureSeries.flux.push(positiveFlux / GRID_SIZE)
      }

      previousGroupEnergy = features.groupEnergy
    }
  }

  return {
    sampleIntervalMs,
    sourceFiles: files,
    frequencyGroups,
    persistenceWindowsMs,
    featureQuantiles: Object.fromEntries(
      Object.entries(featureSeries).map(([name, values]) => [name, summarize(values)]),
    ),
    groupEnergyQuantiles: groupEnergySeries.map((values) => summarize(values)),
    residualQuantilesByPersistence: residualSeries.map((values) => summarize(values)),
    deltaQuantilesByPersistence: deltaSeries.map((values) => summarize(values)),
  }
}

const calibration = buildCalibration()
const mode = process.argv.includes('--json') ? 'json' : 'ts'

if (mode === 'json') {
  console.log(JSON.stringify(calibration, null, 2))
} else {
  console.log(`export const CALIBRATION = ${JSON.stringify(calibration, null, 2)} as const\n`)
}
