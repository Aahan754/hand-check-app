import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const A = window.HandAnalysis;

// ---------------- DOM Setup ----------------

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusLine = document.getElementById("status-line");

const procCanvas = document.createElement("canvas");
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

// ---------------- Styling Constants ----------------

const COLOR = {
  bone: "#38bdf8",
  joint: "#f59e0b",
  warning: "#ef4444",
  info: "#06b6d4",
  good: "#10b981",
  disclaimer: "#64748b",
  button: "#0284c7",
  buttonHover: "#0369a1",
  buttonText: "#ffffff",
};

const DISCLAIMER = "Movement metrics only -- not a medical diagnosis";

// ---------------- App State ----------------

let appState = "loading";
let handLandmarker = null;
let mousePos = { x: -1, y: -1 };
let lastTimestamp = 0;

let W = 640, H = 480;
let aspectRatio = 4 / 3;

function fitCanvasToVideo() {
  if (video.videoWidth && video.videoHeight) {
    W = video.videoWidth;
    H = video.videoHeight;
    aspectRatio = W / H;
  } else {
    W = 640;
    H = 480;
    aspectRatio = 4 / 3;
  }
  
  canvas.width = W;
  canvas.height = H;
  procCanvas.width = W;
  procCanvas.height = H;
}

function checkHandDistance(scale) {
  if (scale < 0.10) return "Move hand closer to camera";
  if (scale > 0.40) return "Move hand further back";
  return null;
}

// ---------------- Drawing Helpers ----------------

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

function drawHand(landmarks) {
  ctx.strokeStyle = COLOR.bone;
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = landmarks[a], p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1.x * W, p1.y * H);
    ctx.lineTo(p2.x * W, p2.y * H);
    ctx.stroke();
  }
  ctx.fillStyle = COLOR.joint;
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * W, p.y * H, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function checkNearEdge(landmarks, margin = 0.03) {
  return landmarks.some(
    (lm) => lm.x < margin || lm.x > 1 - margin || lm.y < margin || lm.y > 1 - margin
  );
}

function drawText(text, x, y, color, size = 18) {
  const scaledSize = Math.max(12, Math.round(size * (H / 480)));
  ctx.font = `600 ${scaledSize}px -apple-system, sans-serif`;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function drawDisclaimer() {
  drawText(DISCLAIMER, 10, H - 15, COLOR.disclaimer, 14);
}

function drawHelpLine(text) {
  drawText(text, 10, H - 38, COLOR.disclaimer, 14);
}

function drawButton(rect, label) {
  const [x1, y1, x2, y2] = rect;
  const hovering = mousePos.x >= x1 && mousePos.x <= x2 && mousePos.y >= y1 && mousePos.y <= y2;
  
  ctx.fillStyle = hovering ? COLOR.buttonHover : COLOR.button;
  ctx.beginPath();
  ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 8);
  ctx.fill();

  const fontSize = Math.max(14, Math.round(18 * (H / 480)));
  ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
  ctx.fillStyle = COLOR.buttonText;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, (x1 + x2) / 2, (y1 + y2) / 2);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function pointInRect([x1, y1, x2, y2], x, y) {
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

// ---------------- Menu Setup ----------------

function getMenuButtons() {
  const btnW = Math.min(360, W * 0.65);
  const btnH = Math.min(50, H * 0.1);
  const centerX = W / 2;
  const startY = H * 0.35;
  const gap = btnH + 16;

  return {
    tremor: [centerX - btnW / 2, startY, centerX + btnW / 2, startY + btnH],
    tap: [centerX - btnW / 2, startY + gap, centerX + btnW / 2, startY + gap + btnH],
    spiral: [centerX - btnW / 2, startY + gap * 2, centerX + btnW / 2, startY + gap * 2 + btnH],
  };
}

function renderMenu() {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, W, H);
  
  drawText("Hand Movement Assessment", W / 2 - Math.min(180, W * 0.3), H * 0.18, "#f8fafc", 24);
  drawText("Select an evaluation test:", W / 2 - Math.min(100, W * 0.2), H * 0.27, "#94a3b8", 16);

  const buttons = getMenuButtons();
  drawButton(buttons.tremor, "1: Tremor & Jitter Evaluation");
  drawButton(buttons.tap, "2: Finger Tap Rhythm Test");
  drawButton(buttons.spiral, "3: Spiral Trace Accuracy");

  drawText("Press 1, 2, 3 or click to select", W / 2 - Math.min(110, W * 0.2), H * 0.82, "#64748b", 14);
  drawDisclaimer();
}

function handleMenuClick(x, y) {
  const buttons = getMenuButtons();
  for (const [key, rect] of Object.entries(buttons)) {
    if (pointInRect(rect, x, y)) {
      startTest(key);
      return;
    }
  }
}

function handleMenuKey(key) {
  if (key === "1") startTest("tremor");
  else if (key === "2") startTest("tap");
  else if (key === "3") startTest("spiral");
}

function startTest(name) {
  video.style.opacity = name === "spiral" ? "0" : "1";
  if (name === "tremor") tremorTest.reset();
  if (name === "tap") tapTest.reset();
  if (name === "spiral") spiralTest.reset();
  appState = name;
}

function returnToMenu() {
  video.style.opacity = "1";
  if (tremorTest) tremorTest.reset();
  if (tapTest) tapTest.reset();
  if (spiralTest) spiralTest.reset();
  appState = "menu";
}

// ============================================================
// 1. Tremor Test Module
// ============================================================

const tremorTest = {
  buffers: [[], []],
  lastValidXY: [null, null],
  warnings: [],
  readouts: [],
  prevHandCount: 0,
  lastDropTime: 0,
  lastAnalysisTime: 0,
  lastAnalysisResults: [null, null],

  reset() {
    this.buffers = [[], []];
    this.lastValidXY = [null, null];
    this.prevHandCount = 0;
    this.lastDropTime = 0;
    this.lastAnalysisTime = 0;
    this.lastAnalysisResults = [null, null];
  },

  update(handsLandmarks, worldLandmarks, now) {
    this.warnings = [];
    this.readouts = [];
    const currentHandCount = handsLandmarks.length;

    const shouldRunAnalysis = now - this.lastAnalysisTime > 0.1;
    if (shouldRunAnalysis) this.lastAnalysisTime = now;

    handsLandmarks.slice(0, 2).forEach((landmarks, idx) => {
      drawHand(landmarks);
      
      if (checkNearEdge(landmarks)) {
        this.warnings.push(`Hand ${idx+1}: Keep hand inside camera center`);
      }

      if (worldLandmarks && worldLandmarks[idx] && !A.isHandFlat(worldLandmarks[idx])) {
        this.warnings.push(`Hand ${idx+1}: Keep palm flat facing camera`);
      }
      
      const scale = A.handScaleOfAspect(landmarks, aspectRatio);
      const distWarning = checkHandDistance(scale);
      if (distWarning) this.warnings.push(`Hand ${idx+1}: ${distWarning}`);

      const tip = landmarks[8];
      const rawTip = [tip.x, tip.y];

      let tipXY = A.smoothPoint(this.lastValidXY[idx], rawTip, 0.4);

      if (!A.rejectOutlier(this.lastValidXY[idx], tipXY, scale, 1.8, aspectRatio)) {
        tipXY = this.lastValidXY[idx];
      } else {
        this.lastValidXY[idx] = tipXY;
      }

      const buf = this.buffers[idx];
      buf.push({ t: now, x: tipXY[0], y: tipXY[1], scale });
      while (buf.length && now - buf[0].t > A.TREMOR_WINDOW_SECONDS) buf.shift();

      if (shouldRunAnalysis) {
        this.lastAnalysisResults[idx] = A.analyzeTremor(buf);
      }

      const result = this.lastAnalysisResults[idx];
      if (!result) {
        this.readouts.push([`Hand ${idx+1}: Collecting samples (${buf.length})...`, COLOR.info]);
      } else if (result.status === "fps_too_low") {
        this.readouts.push([`Hand ${idx+1}: Low camera FPS (${result.rawFps.toFixed(0)})`, COLOR.warning]);
      } else if (result.confident) {
        this.readouts.push([`Hand ${idx+1}: ${result.freqHz.toFixed(1)} Hz Jitter | Amp: ${result.amplitudePct.toFixed(1)}% (${result.rawFps.toFixed(0)} FPS)`, COLOR.info]);
      } else if (!result.hasFpsHeadroom) {
        this.readouts.push([`Hand ${idx+1}: FPS borderline (${result.rawFps.toFixed(0)})`, COLOR.warning]);
      } else {
        this.readouts.push([`Hand ${idx+1}: No significant jitter detected`, COLOR.good]);
      }
    });

    if (handsLandmarks.length === 0) {
      this.buffers = [[], []];
      this.lastValidXY = [null, null];
    }

    if (currentHandCount < this.prevHandCount && this.prevHandCount > 0) this.lastDropTime = now;
    this.prevHandCount = currentHandCount;
    if (now - this.lastDropTime < 1.5) {
      this.warnings.push("Lost tracking frame -- avoid fast occlusion");
    }
  },

  render(handCount) {
    drawText(`Active Hands: ${handCount}`, 10, 28, COLOR.good, 18);
    let y = 56;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 16); y += 22; }
    for (const [text, color] of this.readouts) { drawText(text, 10, y, color, 16); y += 22; }
    drawHelpLine("Q/ESC: Main Menu  |  R: Reset Test");
    drawDisclaimer();
  },

  handleKey(key) {
    if (key === "m" || key === "q" || key === "Escape") returnToMenu();
    else if (key === "r") this.reset();
  },
};

// ============================================================
// 2. Finger Tap Module
// ============================================================

const tapTest = {
  state: "waiting",
  countdownStart: null,
  runStart: null,
  samples: [],
  result: null,
  lastValidPinch: null,
  warnings: [],
  startButton: [0,0,0,0],

  reset() {
    this.state = "waiting";
    this.samples = [];
    this.result = null;
    this.lastValidPinch = null;
    this.warnings = [];
  },

  update(handsLandmarks, now) {
    this.warnings = [];
    let pinchNorm = null;

    if (handsLandmarks.length > 0) {
      const landmarks = handsLandmarks[0];
      drawHand(landmarks);
      if (checkNearEdge(landmarks)) this.warnings.push("Keep hand centered in view");
      
      const scale = A.handScaleOfAspect(landmarks, aspectRatio);
      const thumb = landmarks[4], index = landmarks[8];
      
      if (scale > 0) {
        const rawPinch = A.distanceAspect(thumb, index, aspectRatio) / scale;
        if (this.lastValidPinch !== null && Math.abs(rawPinch - this.lastValidPinch) > 0.7) {
          pinchNorm = this.lastValidPinch;
        } else {
          pinchNorm = rawPinch;
          this.lastValidPinch = rawPinch;
        }
      }
    }

    if (this.state === "countdown") {
      const remaining = 3 - (now - this.countdownStart);
      if (remaining <= 0) { this.state = "running"; this.runStart = now; this.samples = []; }
    } else if (this.state === "running") {
      const elapsed = now - this.runStart;
      const remaining = A.TAP_TEST_DURATION - elapsed;
      if (pinchNorm !== null) this.samples.push({ t: now, dist: pinchNorm });
      if (remaining <= 0) {
        this.state = "done";
        this.result = A.analyzeTaps(this.samples);
      }
    }
  },

  render() {
    const btnW = Math.min(280, W * 0.45);
    const btnH = Math.min(46, H * 0.09);
    this.startButton = [W/2 - btnW/2, H - H*0.22, W/2 + btnW/2, H - H*0.22 + btnH];

    if (this.state === "waiting") {
      drawText("Position hand in frame and click Start", 20, 40, COLOR.good, 18);
      drawText("Tap thumb and index finger repeatedly as fast as possible", 20, 68, "#94a3b8", 15);
      drawButton(this.startButton, "Start Test");
    } else if (this.state === "countdown") {
      const remaining = Math.ceil(3 - (performance.now()/1000 - this.countdownStart));
      drawText(`Starting in ${Math.max(remaining,1)}...`, 20, 40, COLOR.info, 24);
    } else if (this.state === "running") {
      const elapsed = performance.now()/1000 - this.runStart;
      const remaining = A.TAP_TEST_DURATION - elapsed;
      drawText(`TAP NOW! Time remaining: ${remaining.toFixed(1)}s`, 20, 40, COLOR.info, 20);
      const barW = Math.max(0, (W * 0.4) * (elapsed / A.TAP_TEST_DURATION));
      ctx.fillStyle = "#334155"; ctx.fillRect(20, 55, W * 0.4, 12);
      ctx.fillStyle = COLOR.info; ctx.fillRect(20, 55, barW, 12);
    } else if (this.state === "done") {
      let y = 40;
      const r = this.result;
      if (!r || r.error || r.insufficient) {
        drawText("Insufficient tap data captured — Press R to retry", 20, y, COLOR.warning, 18);
      } else {
        const lines = [
          [`Taps Counted: ${r.numTaps}`, COLOR.good],
          [`Average Frequency: ${r.avgRateHz.toFixed(2)} Hz`, COLOR.good],
          [`Amplitude Delta: ${r.ampDecrementPct >= 0 ? "+" : ""}${r.ampDecrementPct.toFixed(1)}%`, COLOR.info],
        ];
        if (r.rateSlowingPct !== null) {
          lines.push([`Rate Change: ${r.rateSlowingPct >= 0 ? "+" : ""}${r.rateSlowingPct.toFixed(1)}%`, COLOR.info]);
        }
        lines.push(r.decrementPattern
          ? ["Decrement observed (Amplitude drop > 20%)", COLOR.warning]
          : ["Stable tap rhythm maintained", COLOR.good]);
        for (const [text, color] of lines) { drawText(text, 20, y, color, 16); y += 24; }
      }
      drawText("Press R to run test again", 20, y + 10, "#94a3b8", 14);
    }

    let y = 110;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 15); y += 20; }

    drawHelpLine("SPACE: Start  |  Q/ESC: Main Menu  |  R: Reset");
    drawDisclaimer();
  },

  handleClick(x, y) {
    if (this.state === "waiting" && pointInRect(this.startButton, x, y)) {
      this.state = "countdown";
      this.countdownStart = performance.now() / 1000;
    }
  },

  handleKey(key) {
    if (key === "m" || key === "q" || key === "Escape") returnToMenu();
    else if (key === "r") this.reset();
    else if (key === " " && this.state === "waiting") {
      this.state = "countdown";
      this.countdownStart = performance.now() / 1000;
    }
  },
};

// ============================================================
// 3. Spiral Test Module
// ============================================================

const SPIRAL_TURNS = 3;
const SPIRAL_MAX_DURATION = 25.0;

const spiralTest = {
  state: "waiting",
  runStart: null,
  tracePoints: [],
  deviationSamples: [],
  tremorBuffer: [],
  lastValidTipXY: null,
  warnings: [],
  spiralRef: null,
  spiralCenter: null,
  spiralMaxRadius: null,
  startButton: [0,0,0,0],

  reset() {
    this.state = "waiting";
    this.tracePoints = [];
    this.deviationSamples = [];
    this.tremorBuffer = [];
    this.lastValidTipXY = null;
    this.warnings = [];
    this.spiralCenter = [W/2, H/2];
    this.spiralMaxRadius = Math.floor(Math.min(W, H) * 0.35);
    this.spiralRef = A.generateSpiralPoints(this.spiralCenter, this.spiralMaxRadius, SPIRAL_TURNS);
  },

  update(handsLandmarks, now) {
    if (!this.spiralRef) this.reset();
    this.warnings = [];

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.spiralRef.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();

    if (handsLandmarks.length > 0) {
      const landmarks = handsLandmarks[0];
      drawHand(landmarks);
      if (checkNearEdge(landmarks)) this.warnings.push("Keep hand centered");
      
      const scale = A.handScaleOfAspect(landmarks, aspectRatio);
      const tip = landmarks[8];
      let tipNorm = [tip.x, tip.y];

      if (!A.rejectOutlier(this.lastValidTipXY, tipNorm, scale, 1.8, aspectRatio)) {
        tipNorm = this.lastValidTipXY;
      } else {
        this.lastValidTipXY = tipNorm;
      }

      const fingertipPx = [tipNorm[0] * W, tipNorm[1] * H];

      if (this.state === "running") {
        this.tremorBuffer.push({ t: now, x: tipNorm[0], y: tipNorm[1], scale });
        while (this.tremorBuffer.length && now - this.tremorBuffer[0].t > A.TREMOR_WINDOW_SECONDS) this.tremorBuffer.shift();

        this.tracePoints.push(fingertipPx);
        const dist = A.minDistanceToSpiral(fingertipPx, this.spiralRef);
        this.deviationSamples.push(dist);
      }
    } else if (this.state === "running") {
      this.warnings.push("Hand not detected in frame");
    }

    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.tracePoints.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();
  },

  render() {
    const btnW = Math.min(280, W * 0.45);
    const btnH = Math.min(46, H * 0.09);
    this.startButton = [W/2 - btnW/2, H - H*0.22, W/2 + btnW/2, H - H*0.22 + btnH];

    if (this.state === "waiting") {
      drawText("Position index finger over spiral center", 20, 36, COLOR.good, 18);
      drawText("Trace along gray line smoothly", 20, 62, "#94a3b8", 15);
      drawButton(this.startButton, "Start Test");
    } else if (this.state === "running") {
      const elapsed = performance.now()/1000 - this.runStart;
      const remaining = SPIRAL_MAX_DURATION - elapsed;
      drawText(`Tracing... ${remaining.toFixed(1)}s remaining (Press F when complete)`, 20, 36, COLOR.info, 16);
      if (this.deviationSamples.length) {
        const meanDev = A.mean(this.deviationSamples);
        drawText(`Average Deviation: ${meanDev.toFixed(1)}px`, 20, 58, COLOR.info, 14);
      }
      if (remaining <= 0) this.state = "done";
    } else if (this.state === "done") {
      let y = 36;
      if (this.deviationSamples.length < 10) {
        drawText("Insufficient trace samples captured — Press R to retry", 20, y, COLOR.warning, 16);
      } else {
        const meanDev = A.mean(this.deviationSamples);
        const maxDev = Math.max(...this.deviationSamples);

        const lines = [
          [`Mean Error: ${meanDev.toFixed(1)}px (${(meanDev/this.spiralMaxRadius*100).toFixed(1)}% radius)`, COLOR.good],
          [`Peak Error: ${maxDev.toFixed(1)}px (${(maxDev/this.spiralMaxRadius*100).toFixed(1)}% radius)`, COLOR.info],
        ];

        const jitterResult = this.tremorBuffer.length ? A.analyzeTremor(this.tremorBuffer) : null;
        if (jitterResult && jitterResult.confident) {
          lines.push([`Tracing Jitter Detected: ${jitterResult.freqHz.toFixed(1)} Hz`, COLOR.warning]);
        } else {
          lines.push(["Smooth tracing motion without significant jitter", COLOR.good]);
        }

        for (const [text, color] of lines) { drawText(text, 20, y, color, 16); y += 24; }
      }
      drawText("Press R to retry test", 20, y + 10, "#94a3b8", 14);
    }

    let y = 96;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 15); y += 20; }

    drawHelpLine("SPACE: Start  |  F: Finish Early  |  Q/ESC: Main Menu  |  R: Reset");
    drawDisclaimer();
  },

  handleClick(x, y) {
    if (this.state === "waiting" && pointInRect(this.startButton, x, y)) {
      this.state = "running";
      this.runStart = performance.now() / 1000;
      this.tracePoints = [];
      this.deviationSamples = [];
      this.tremorBuffer = [];
    }
  },

  handleKey(key) {
    if (key === "m" || key === "q" || key === "Escape") returnToMenu();
    else if (key === "r") this.reset();
    else if (key === " " && this.state === "waiting") {
      this.state = "running";
      this.runStart = performance.now() / 1000;
      this.tracePoints = [];
      this.deviationSamples = [];
      this.tremorBuffer = [];
    } else if ((key === "f" || key === "F") && this.state === "running") {
      this.state = "done";
    }
  },
};

// ---------------- Main Execution Loop ----------------

async function detectAndDraw() {
  if (video.readyState < 2) {
    requestAnimationFrame(detectAndDraw);
    return;
  }

  procCtx.save();
  procCtx.translate(W, 0);
  procCtx.scale(-1, 1);
  procCtx.drawImage(video, 0, 0, W, H);
  procCtx.restore();

  const now = performance.now() / 1000;
  let handsLandmarks = [];
  let worldLandmarks = [];

  if (handLandmarker) {
    let nowMs = performance.now();
    if (nowMs <= lastTimestamp) {
      nowMs = lastTimestamp + 1;
    }
    lastTimestamp = nowMs;

    try {
      const result = handLandmarker.detectForVideo(procCanvas, nowMs);
      handsLandmarks = result.landmarks || [];
      worldLandmarks = result.worldLandmarks || [];
    } catch (detectErr) {
      console.warn("Tracking detection frame skip:", detectErr);
    }
  }

  ctx.clearRect(0, 0, W, H);

  if (appState === "menu") {
    renderMenu();
  } else if (appState === "tremor") {
    tremorTest.update(handsLandmarks, worldLandmarks, now);
    tremorTest.render(handsLandmarks.length);
  } else if (appState === "tap") {
    tapTest.update(handsLandmarks, now);
    tapTest.render();
  } else if (appState === "spiral") {
    spiralTest.update(handsLandmarks, now);
    spiralTest.render();
  }

  requestAnimationFrame(detectAndDraw);
}

// ---------------- User Input Handlers ----------------

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  mousePos = {
    x: (e.clientX - rect.left) * (W / rect.width),
    y: (e.clientY - rect.top) * (H / rect.height),
  };
});

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (W / rect.width);
  const y = (e.clientY - rect.top) * (H / rect.height);
  if (appState === "menu") handleMenuClick(x, y);
  else if (appState === "tap") tapTest.handleClick(x, y);
  else if (appState === "spiral") spiralTest.handleClick(x, y);
});

window.addEventListener("keydown", (e) => {
  if (appState === "menu") handleMenuKey(e.key);
  else if (appState === "tremor") tremorTest.handleKey(e.key);
  else if (appState === "tap") tapTest.handleKey(e.key);
  else if (appState === "spiral") spiralTest.handleKey(e.key);
});

// ---------------- System Initialization ----------------

async function getCameraStreamWithFallback() {
  const profiles = [
    { name: "SD Performance 4:3", constraints: { video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60, min: 30 } } } },
    { name: "Basic Stream", constraints: { video: true } },
    { name: "Fallback Stream", constraints: { video: {} } }
  ];

  for (const profile of profiles) {
    try {
      return await navigator.mediaDevices.getUserMedia(profile.constraints);
    } catch (err) {
      console.warn(`Profile unsupported (${profile.name}):`, err.message);
    }
  }

  throw new Error("No usable camera hardware detected.");
}

async function init() {
  try {
    statusLine.textContent = "Loading tracking models...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    } catch (modelErr) {
      console.warn("CPU delegate setup warning:", modelErr);
    }

    statusLine.textContent = "Connecting camera...";
    const stream = await getCameraStreamWithFallback();

    video.srcObject = stream;

    await new Promise((resolve) => {
      if (video.readyState >= 1) {
        resolve();
      } else {
        video.onloadedmetadata = () => resolve();
      }
    });

    await video.play();
    fitCanvasToVideo();
    window.addEventListener("resize", fitCanvasToVideo);

    statusLine.textContent = "System Ready";
    appState = "menu";
    detectAndDraw();
  } catch (err) {
    console.error("Initialization failure:", err);
    statusLine.textContent = "Hardware Access Error";
    statusLine.style.color = COLOR.warning;
  }
}

init();
