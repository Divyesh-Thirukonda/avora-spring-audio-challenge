# 3D Audio Graph

**Live Deployment:** [https://avora-spring-audio-challenge.vercel.app](https://avora-spring-audio-challenge.vercel.app)

## Description

This project started as a simple sparse 2D graph network that looked like drifting stars. But to capture the complexity of music, it needed more depth (literally). We evolved it into a 3D 10x10x10 cube graph (reminiscent of a 3b1b-style visualization).

## Design Details

The main goal was mathematical beauty ensuring that every visual detail is purposefully tied to the audio data, avoiding arbitrary hard-coded behaviors. The visualizer is fully responsive to the nuances of the audio stream, making it almost possible to guess the song just by looking at it.

*   **Mappings:** Created a graph network (10x10x10) in the shape of a cube with nodes and edges to visualize a variety of audio features. Each axis ideally encodes a specific feature instead of physical space. With only 3 dimensions, we essentially merged the features down to the most important three (similar to PCA): the X-axis correlates to frequency bands, the Y-axis to harmonicity (timbre), and the Z-axis to time-domain temporal dynamics (like zero-crossing, crest, or pulse sharpness). The intention here was mathematical beauty: making sure every detail is accounted for. We ran simulations with sample audio to maximize coverage so that different parts of the grid glow. Rather than nodes moving, they glow based on how strongly the current audio stream matches their designated coordinate traits, with the single most concentrated audio feature extracted at that given point commanding the main focus with an intense node bloom.
*   **Packets:** Visual packets of energy travel among the nodes on beats and frequency hits. Originally meant to just indicate where the grid glows, they are sent through the system towards the most dominant frequency band of that moment. They are weighted by the significance of the audio feature (like a sharp drum hit or pitch spike), with their speed and direction symbolized by a high derivative in tempo.
*   **Adaptive Tuning:** To properly calibrate the input stream and normalize the data correctly, we implemented a simple real-time sliding-window stream rather than static thresholds. This auto-calibrates the audio distribution on the fly (used to determine other parts of the system, like where the grid lights up), ensuring that even subtle variations in notes create significant visual changes rather than being lost in the noise.

## Running Locally

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow microphone access when prompted.

## The Challenge

Edit `src/visualizers/Visualizer.tsx` to create your own visualization. You have been given a default starter template that shows audio visualized in the frequency and time domains.

## Audio Pipeline

The `useAudio` hook captures microphone input.

From the hook, you will receive:
- **frequencyData** — 1024 FFT frequency bins from low to high.
- **timeDomainData** — 2048 raw waveform samples. A value of 128 is silence, and 0 and 255 are the lowest and highest values respectively.

You SHOULD NOT update useAudio, and should instead focus on using its return values for your visualization.

## Project Structure

```
src/
├── audio/
│   └── useAudio.ts      # Audio pipeline (do not modify)
├── visualizers/
│   └── Visualizer.tsx   # YOUR CODE GOES HERE
├── App.tsx
├── App.css
├── index.css
└── main.tsx
```

## Submissions

Fork this repo and get nerdy with a visualization that you find super cool.

When you're ready, deploy your solution and send the URL + link to your submission's github repo to careers@getavora.ai.

In your submission, please also include:
- A title for your visualization
- A short description of the work you built, including some of the design or implementation decisions you made along the way

We evaluate solutions on craft and novelty. You may use any AI tools that you like in this process.

Have fun!
