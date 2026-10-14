// analysis.js
// Ported from hand_stick_figure.py -- tremor/jitter detection, tap
// detection, and spiral deviation. Pure math, no DOM/camera dependencies,
// so it can be unit-tested the same way the Python version was.

// ---------------- Direct DFT (frequency-domain analysis) ----------------
// Buffer sizes here are small (~50-300 samples) and this runs at video
// frame rate, so an O(n^2) direct DFT is plenty fast and avoids needing a
// full FFT implementation (which normally requires power-of-2 padding).
// Frequency bins are computed with the same formula numpy's rfftfreq uses,
// so results line up with the original Python behavior.
function realDFT(signal, sampleRateHz) {
  const N = signal.length;
  const nBins = Math.floor(N / 2) + 1;
  const freqs = new Array(nBins);
  const mag = new Array(nBins);
  for (let k = 0; k < nBins; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im -= signal[n] * Math.sin(angle);
    }
    freqs[k] = (k * sampleRateHz) / N;
    mag[k] = Math.sqrt(re * re + im * im);
  }
  return { freqs, mag };
}

function hanningWindow(N) {
  const w = new Array(N);
  for (let n = 0; n < N; n++) {
    w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));
  }
  return w;
}

function linspace(start, end, n) {
  if (n === 1) return [start];
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + step * i);
}

// Linear interpolation of `ys` (sampled at `xs`) onto `queryXs`. Mirrors
// numpy.interp: clamps to the first/last value outside the known range.
function interp(queryXs, xs, ys) {
  const out = new Array(queryXs.length);
  let j = 0;
  for (let i = 0; i < queryXs.length; i++) {
    const qx = queryXs[i];
    if (qx <= xs[0]) { out[i] = ys[0]; continue; }
    if (qx >= xs[xs.length - 1]) { out[i] = ys[ys.length - 1]; continue; }
    while (j < xs.length - 2 && xs[j + 1] < qx) j++;
    const x0 = xs[j], x1 = xs[j + 1];
    const y0 = ys[j], y1 = ys[j + 1];
    const t = (qx - x0) / (x1 - x0);
    out[i] = y0 + t * (y1 - y0);
  }
  return out;
}

function movingAverage(signal, windowSize) {
  const N = signal.length;
  const half = Math.floor(windowSize / 2);
  const out = new Array(N);
  let sum = 0;
  const buf = [];
  for (let i = 0; i < N; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(N - 1, i + half);
    // simple (not the fastest) windowed mean -- N is small so this is fine
    let s = 0;
    for (let k = lo; k <= hi; k++) s += signal[k];
    out[i] = s / (hi - lo + 1);
  }
  return out;
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(mean(arr.map((v) => (v - m) ** 2)));
}
function ptp(arr) { return Math.max(...arr) - Math.min(...arr); }

// ---------------- Tremor / jitter analysis ----------------

const TREMOR_WINDOW_SECONDS = 3.0;
const TREMOR_MIN_SAMPLES = 20;
const TREMOR_FREQ_MIN = 3.0;
const TREMOR_FREQ_MAX = 12.0;
const TREMOR_BAND_RATIO_THRESHOLD = 0.10;
const OVERSAMPLE_FACTOR = 4.0;
const MIN_FPS_FOR_CONFIDENCE = OVERSAMPLE_FACTOR * TREMOR_FREQ_MAX; // 48 fps

// buffer: array of {t, x, y, scale}
function analyzeTremor(buffer) {
  if (buffer.length < TREMOR_MIN_SAMPLES) return null;

  const times = buffer.map((b) => b.t);
  const xs = buffer.map((b) => b.x);
  const ys = buffer.map((b) => b.y);
  const scales = buffer.map((b) => b.scale);

  const duration = times[times.length - 1] - times[0];
  if (duration < 1.5) return null;

  const dts = [];
  for (let i = 1; i < times.length; i++) {
    const d = times[i] - times[i - 1];
    if (d > 0) dts.push(d);
  }
  if (dts.length < 2) return null;
  const sortedDts = [...dts].sort((a, b) => a - b);
  const medianDt = sortedDts[Math.floor(sortedDts.length / 2)];
  const rawFps = 1.0 / medianDt;

  const freqMaxEffective = Math.min(TREMOR_FREQ_MAX, rawFps / OVERSAMPLE_FACTOR);
  if (freqMaxEffective < TREMOR_FREQ_MIN) {
    return { status: "fps_too_low", rawFps };
  }

  const resampleHz = Math.min(60, Math.max(20, rawFps));
  const nSamples = Math.floor(duration * resampleHz);
  if (nSamples < 16) return null;

  const uniformT = linspace(times[0], times[times.length - 1], nSamples);
  const ux = interp(uniformT, times, xs);
  const uy = interp(uniformT, times, ys);
  const meanScale = mean(scales);
  if (meanScale <= 0) return null;

  // Trend removal (moving-average high-pass, matching the Python fallback
  // path -- simpler than a full Butterworth+filtfilt port, same principle).
  const trendWindow = Math.max(3, Math.floor(0.4 * resampleHz));
  const trendX = movingAverage(ux, trendWindow);
  const trendY = movingAverage(uy, trendWindow);
  const dx = ux.map((v, i) => (v - trendX[i]) / meanScale);
  const dy = uy.map((v, i) => (v - trendY[i]) / meanScale);

  const win = hanningWindow(nSamples);
  const dxWin = dx.map((v, i) => v * win[i]);
  const dyWin = dy.map((v, i) => v * win[i]);

  const { freqs, mag: magX } = realDFT(dxWin, resampleHz);
  const { mag: magY } = realDFT(dyWin, resampleHz);
  const mag = magX.map((v, i) => v + magY[i]);

  const bandIdx = [];
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= TREMOR_FREQ_MIN && freqs[i] <= freqMaxEffective) bandIdx.push(i);
  }
  if (bandIdx.length === 0) return { status: "fps_too_low", rawFps };

  let peakI = bandIdx[0];
  for (const i of bandIdx) if (mag[i] > mag[peakI]) peakI = i;
  const peakFreq = freqs[peakI];
  const peakMag = mag[peakI];
  const totalMag = mag.reduce((a, b) => a + b, 0) + 1e-9;
  const bandRatio = peakMag / totalMag;

  const radial = dx.map((v, i) => Math.sqrt(v * v + dy[i] * dy[i]));
  const amplitudePct = 2 * std(radial) * 100;

  // Deliberately uses the fixed band-ceiling requirement, not the detected
  // frequency -- an undersampled high-frequency signal can alias into a
  // clean-looking lower peak, which would otherwise pass a frequency-
  // adaptive check and produce a confidently WRONG reading. Validated by
  // simulation in the Python version before porting this rule over.
  const hasFpsHeadroom = rawFps >= MIN_FPS_FOR_CONFIDENCE;
  const confident = bandRatio >= TREMOR_BAND_RATIO_THRESHOLD && hasFpsHeadroom;

  return {
    status: "ok",
    freqHz: peakFreq,
    amplitudePct,
    bandRatio,
    confident,
    rawFps,
    hasFpsHeadroom,
  };
}

// ---------------- Glitch rejection ----------------

function rejectOutlier(prevXY, newXY, scale, maxNormJump = 0.5) {
  if (!prevXY || !scale || scale <= 0) return true;
  const jump = Math.hypot(newXY[0] - prevXY[0], newXY[1] - prevXY[1]) / scale;
  return jump < maxNormJump;
}

// ---------------- Simple peak finder (find_peaks equivalent) ----------------
// Finds local maxima in `signal` with at least `minDistance` samples
// between them and a "prominence" (isolation from surrounding lower
// values) of at least `prominence`. This is a simplified version of
// scipy.signal.find_peaks sufficient for our tap-detection use case.
function findPeaksSimple(signal, { prominence = 0, distance = 1 } = {}) {
  const N = signal.length;
  const candidates = [];
  for (let i = 1; i < N - 1; i++) {
    if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) {
      candidates.push(i);
    }
  }

  // Approximate prominence: for each candidate, look outward until the
  // signal rises above the candidate's value on both sides (or hits the
  // array edge), and take the minimum of the two "drop" depths.
  function prominenceOf(i) {
    const v = signal[i];
    let leftMin = v;
    for (let k = i - 1; k >= 0; k--) {
      leftMin = Math.min(leftMin, signal[k]);
      if (signal[k] > v) break;
    }
    let rightMin = v;
    for (let k = i + 1; k < N; k++) {
      rightMin = Math.min(rightMin, signal[k]);
      if (signal[k] > v) break;
    }
    return v - Math.max(leftMin, rightMin);
  }

  let filtered = candidates
    .map((i) => ({ i, prom: prominenceOf(i) }))
    .filter((c) => c.prom >= prominence);

  // Enforce minimum distance by greedily keeping the most prominent peak
  // in each conflicting cluster (mirrors scipy's distance-based pruning).
  filtered.sort((a, b) => b.prom - a.prom);
  const kept = [];
  for (const c of filtered) {
    if (kept.every((k) => Math.abs(k.i - c.i) >= distance)) kept.push(c);
  }
  kept.sort((a, b) => a.i - b.i);
  return kept.map((k) => k.i);
}

// ---------------- Finger tap analysis ----------------

const TAP_TEST_DURATION = 10.0;
const TAP_RESAMPLE_HZ = 60;
const TAP_MIN_PROMINENCE = 0.08;
const TAP_MIN_SEPARATION_SEC = 0.12;

// samples: array of {t, dist} (normalized pinch distance)
function analyzeTaps(samples) {
  if (samples.length < 20) return null;

  const times = samples.map((s) => s.t);
  const dist = samples.map((s) => s.dist);
  const duration = times[times.length - 1] - times[0];
  if (duration < 1.0) return null;

  const nSamples = Math.floor(duration * TAP_RESAMPLE_HZ);
  if (nSamples < 10) return null;

  const uniformT = linspace(times[0], times[times.length - 1], nSamples);
  const udist = interp(uniformT, times, dist);

  const observedRange = ptp(udist);
  const adaptiveProminence = Math.max(TAP_MIN_PROMINENCE, Math.min(0.3, 0.25 * observedRange));
  const minDistSamples = Math.max(1, Math.floor(TAP_MIN_SEPARATION_SEC * TAP_RESAMPLE_HZ));

  const negUdist = udist.map((v) => -v);
  const tapIndices = findPeaksSimple(negUdist, { prominence: adaptiveProminence, distance: minDistSamples });

  if (tapIndices.length < 2) {
    return { numTaps: tapIndices.length, insufficient: true };
  }

  const tapTimes = tapIndices.map((i) => uniformT[i]);
  const intervals = [];
  for (let i = 1; i < tapTimes.length; i++) intervals.push(tapTimes[i] - tapTimes[i - 1]);
  const avgRateHz = intervals.length ? 1.0 / mean(intervals) : 0;

  const maxIndices = findPeaksSimple(udist, { prominence: adaptiveProminence, distance: minDistSamples });

  const amplitudes = tapIndices.map((ti) => {
    const precedingMaxima = maxIndices.filter((mi) => mi < ti);
    let peakVal;
    if (precedingMaxima.length > 0) {
      peakVal = udist[precedingMaxima[precedingMaxima.length - 1]];
    } else if (ti > 0) {
      peakVal = Math.max(...udist.slice(0, ti + 1));
    } else {
      peakVal = udist[ti];
    }
    return Math.max(0, peakVal - udist[ti]);
  });

  const nEdge = Math.max(1, Math.floor(amplitudes.length / 4));
  const firstAmp = mean(amplitudes.slice(0, nEdge));
  const lastAmp = mean(amplitudes.slice(-nEdge));
  const ampDecrementPct = firstAmp > 0 ? ((firstAmp - lastAmp) / firstAmp) * 100 : 0;

  let rateSlowingPct = null;
  if (intervals.length >= 4) {
    const nEdgeI = Math.max(1, Math.floor(intervals.length / 4));
    const firstRate = 1.0 / mean(intervals.slice(0, nEdgeI));
    const lastRate = 1.0 / mean(intervals.slice(-nEdgeI));
    rateSlowingPct = firstRate > 0 ? ((firstRate - lastRate) / firstRate) * 100 : 0;
  }

  const decrementPattern = amplitudes.length >= 6 && ampDecrementPct > 20;

  return {
    numTaps: tapIndices.length,
    avgRateHz,
    ampDecrementPct,
    rateSlowingPct,
    decrementPattern,
    insufficient: false,
  };
}

// ---------------- Spiral reference ----------------

function generateSpiralPoints(center, maxRadius, turns, nPoints = 2000) {
  const pts = new Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    const theta = (turns * 2 * Math.PI * i) / (nPoints - 1);
    const r = (theta / (turns * 2 * Math.PI)) * maxRadius;
    pts[i] = [center[0] + r * Math.cos(theta), center[1] + r * Math.sin(theta)];
  }
  return pts;
}

function minDistanceToSpiral(point, spiralPts) {
  let best = Infinity;
  for (const p of spiralPts) {
    const d = Math.hypot(p[0] - point[0], p[1] - point[1]);
    if (d < best) best = d;
  }
  return best;
}

const HandAnalysis = {
  realDFT, hanningWindow, linspace, interp, movingAverage, mean, std, ptp,
  analyzeTremor, rejectOutlier, findPeaksSimple, analyzeTaps,
  generateSpiralPoints, minDistanceToSpiral,
  TREMOR_WINDOW_SECONDS, TAP_TEST_DURATION,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = HandAnalysis;
} else {
  window.HandAnalysis = HandAnalysis;
}
