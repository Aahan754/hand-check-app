import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const A = window.HandAnalysis;

// ---------------- DOM setup ----------------

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusLine = document.getElementById("status-line");

const procCanvas = document.createElement("canvas");
const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });

// ---------------- Colors ----------------

const COLOR = {
  bone: "#ffffff",
  joint: "#ffa500",
  warning: "#ff3b30",
  info: "#00e5ff",
  good: "#22c55e",
  disclaimer: "#8c8c8c",
  button: "#ff7800",
  buttonHover: "#ffa040",
  buttonText: "#ffffff",
};

const DISCLAIMER = "Movement metrics only -- not a medical diagnosis";

// ---------------- App state ----------------

let appState = "loading";
let handLandmarker = null;
let mousePos = { x: -1, y: -1 };
let lastTimestamp = 0;

let W = 1280, H = 720;
let aspectRatio = 16 / 9;

function fitCanvasToVideo() {
  if (video.videoWidth && video.videoHeight) {
    W = video.videoWidth;
    H = video.videoHeight;
    aspectRatio = W / H;
  } else {
    W = 1280;
    H = 720;
    aspectRatio = 16 / 9;
  }
  
  canvas.width = W;
  canvas.height = H;
  procCanvas.width = W;
  procCanvas.height = H;
}

// ---------------- Shared drawing helpers ----------------

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

function drawText(text, x, y, color, size = 20) {
  const scaledSize = Math.max(12, Math.round(size * (H / 720)));
  ctx.font = `${scaledSize}px -apple-system, sans-serif`;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function drawDisclaimer() {
  drawText(DISCLAIMER, 10, H - 15, COLOR.disclaimer, 15);
}

function drawHelpLine(text) {
  drawText(text, 10, H - 42, COLOR.disclaimer, 15);
}

function drawButton(rect, label) {
  const [x1, y1, x2, y2] = rect;
  const hovering = mousePos.x >= x1 && mousePos.x <= x2 && mousePos.y >= y1 && mousePos.y <= y2;
  ctx.fillStyle = hovering ? COLOR.buttonHover : COLOR.button;
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
  const fontSize = Math.max(14, Math.round(20 * (H / 720)));
  ctx.font = `${fontSize}px -apple-system, sans-serif`;
  ctx.fillStyle = COLOR.buttonText;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, (x1 + x2) / 2, (y1 + y2) / 2);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  return hovering;
}

function pointInRect([x1, y1, x2, y2], x, y) {
  return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

// ---------------- Dynamic Layout & Menu ----------------

function getMenuButtons() {
  const btnW = Math.min(400, W * 0.6);
  const btnH = Math.min(60, H * 0.08);
  const centerX = W / 2;
  const startY = H * 0.38;
  const gap = btnH + 20;

  return {
    tremor: [centerX - btnW / 2, startY, centerX + btnW / 2, startY + btnH],
    tap: [centerX - btnW / 2, startY + gap, centerX + btnW / 2, startY + gap + btnH],
    spiral: [centerX - btnW / 2, startY + gap * 2, centerX + btnW / 2, startY + gap * 2 + btnH],
  };
}

function renderMenu() {
  ctx.fillStyle = "#1e1e1e";
  ctx.fillRect(0, 0, W, H);
  
  drawText("Hand Movement Screening Tool", W / 2 - Math.min(220, W * 0.35), H * 0.2, "#ffffff", 32);
  drawText("Choose a test:", W / 2 - Math.min(80, W * 0.15), H * 0.3, "#c8c8c8", 20);

  const buttons = getMenuButtons();
  drawButton(buttons.tremor, "1: Tremor / Writing Jitter");
  drawButton(buttons.tap, "2: Finger Tap Test");
  drawButton(buttons.spiral, "3: Spiral Trace Test");

  drawText("Press 1, 2, 3 or click a button", W / 2 - Math.min(130, W * 0.2), H * 0.82, "#b4b4b4", 16);
  drawText(DISCLAIMER, W / 2 - Math.min(150, W * 0.25), H * 0.88, COLOR.disclaimer, 16);
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
// 1. Tremor / Writing Jitter Test
// ============================================================

const tremorTest = {
  buffers: [[], []],
  lastValidXY: [null, null],
  warnings: [],
  readouts: [],
  prevHandCount: 0,
  lastDropTime: 0,

  reset() {
    this.buffers = [[], []];
    this.lastValidXY = [null, null];
    this.prevHandCount = 0;
    this.lastDropTime = 0;
  },

  update(handsLandmarks, now) {
    this.warnings = [];
    this.readouts = [];
    const currentHandCount = handsLandmarks.length;

    handsLandmarks.slice(0, 2).forEach((landmarks, idx) => {
      drawHand(landmarks);
      if (checkNearEdge(landmarks)) {
        this.warnings.push(`Hand ${idx+1} not fully in frame -- move hand toward center`);
      }
      
      const scale = A.handScaleOfAspect(landmarks, aspectRatio);
      const tip = landmarks[8];
      let tipXY = [tip.x, tip.y];

      if (!A.rejectOutlier(this.lastValidXY[idx], tipXY, scale, 0.5, aspectRatio)) {
        tipXY = this.lastValidXY[idx];
      } else {
        this.lastValidXY[idx] = tipXY;
      }

      const buf = this.buffers[idx];
      buf.push({ t: now, x: tipXY[0], y: tipXY[1], scale });
      while (buf.length && now - buf[0].t > A.TREMOR_WINDOW_SECONDS) buf.shift();

      const result = A.analyzeTremor(buf);
      if (!result) {
        this.readouts.push([`Hand ${idx+1}: collecting data... (${buf.length} samples)`, COLOR.info]);
      } else if (result.status === "fps_too_low") {
        this.readouts.push([`Hand ${idx+1}: camera too slow (${result.rawFps.toFixed(0)} fps) for a reliable reading`, COLOR.warning]);
      } else if (result.confident) {
        this.readouts.push([`Hand ${idx+1}: ${result.freqHz.toFixed(1)} Hz jitter, amp ${result.amplitudePct.toFixed(1)}% (conf ${result.bandRatio.toFixed(2)}, ${result.rawFps.toFixed(0)} fps)`, COLOR.info]);
      } else if (!result.hasFpsHeadroom) {
        this.readouts.push([`Hand ${idx+1}: ${result.rawFps.toFixed(0)} fps -- borderline, treat as unreliable`, COLOR.warning]);
      } else {
        this.readouts.push([`Hand ${idx+1}: no clear jitter (peak ${result.freqHz.toFixed(1)} Hz, conf ${result.bandRatio.toFixed(2)})`, COLOR.info]);
      }
    });

    if (handsLandmarks.length === 0) {
      this.buffers = [[], []];
      this.lastValidXY = [null, null];
    }

    if (currentHandCount < this.prevHandCount && this.prevHandCount > 0) this.lastDropTime = now;
    this.prevHandCount = currentHandCount;
    if (now - this.lastDropTime < 1.5) {
      this.warnings.push("Lost track of a hand -- possible overlap or occlusion");
    }
  },

  render(handCount) {
    drawText(`Hands: ${handCount}`, 10, 30, "#22c55e", 20);
    let y = 60;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 20); y += 25; }
    for (const [text, color] of this.readouts) { drawText(text, 10, y, color, 20); y += 25; }
    drawHelpLine("q/Esc: menu    r: restart");
    drawDisclaimer();
  },

  handleKey(key) {
    if (key === "m" || key === "q" || key === "Escape") returnToMenu();
    else if (key === "r") this.reset();
  },
};

// ============================================================
// 2. Finger Tap Test
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
      if (checkNearEdge(landmarks)) this.warnings.push("Hand not fully in frame");
      
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
    const btnW = Math.min(320, W * 0.5);
    const btnH = Math.min(50, H * 0.08);
    this.startButton = [W/2 - btnW/2, H - H*0.2, W/2 + btnW/2, H - H*0.2 + btnH];

    if (this.state === "waiting") {
      drawText("Get your hand in frame, then click Start or press SPACE", 30, 50, "#22c55e", 22);
      drawText("Tap thumb and index finger together repeatedly, as fast as you can", 30, 85, "#c8c8c8", 17);
      drawButton(this.startButton, "Start Test");
    } else if (this.state === "countdown") {
      const remaining = Math.ceil(3 - (performance.now()/1000 - this.countdownStart));
      drawText(`Starting in ${Math.max(remaining,1)}...`, 30, 50, COLOR.info, 28);
    } else if (this.state === "running") {
      const elapsed = performance.now()/1000 - this.runStart;
      const remaining = A.TAP_TEST_DURATION - elapsed;
      drawText(`TAP! Time left: ${remaining.toFixed(1)}s`, 30, 50, COLOR.info, 24);
      const barW = Math.max(0, (W * 0.4) * (elapsed / A.TAP_TEST_DURATION));
      ctx.strokeStyle = "#505050"; ctx.strokeRect(30, 65, W * 0.4, 15);
      ctx.fillStyle = COLOR.info; ctx.fillRect(30, 65, barW, 15);
    } else if (this.state === "done") {
      let y = 50;
      const r = this.result;
      if (!r || r.error) {
        drawText("Not enough data captured -- press r to retry", 30, y, COLOR.warning, 20);
      } else if (r.insufficient) {
        drawText(`Only ${r.numTaps} taps detected -- press r to retry`, 30, y, COLOR.warning, 20);
      } else {
        const lines = [
          [`Taps detected: ${r.numTaps}`, COLOR.good],
          [`Average tap rate: ${r.avgRateHz.toFixed(2)} Hz`, COLOR.good],
          [`Amplitude change (first vs last): ${r.ampDecrementPct >= 0 ? "+" : ""}${r.ampDecrementPct.toFixed(1)}%`, COLOR.info],
        ];
        if (r.rateSlowingPct !== null) {
          lines.push([`Rate change (first vs last): ${r.rateSlowingPct >= 0 ? "+" : ""}${r.rateSlowingPct.toFixed(1)}%`, COLOR.info]);
        }
        lines.push(r.decrementPattern
          ? ["Decrement pattern observed (amplitude shrank >20%)", COLOR.warning]
          : ["No strong decrement pattern in this trial", COLOR.good]);
        for (const [text, color] of lines) { drawText(text, 30, y, color, 20); y += 30; }
      }
      drawText("Press r to retry", 30, y + 10, "#c8c8c8", 17);
    }

    let y = 120;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 18); y += 22; }

    drawHelpLine("SPACE: start    q/Esc: menu    r: restart");
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
// 3. Spiral Trace Test
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

    ctx.fillStyle = "#141414";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#5a5a5a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    this.spiralRef.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();

    if (handsLandmarks.length > 0) {
      const landmarks = handsLandmarks[0];
      drawHand(landmarks);
      if (checkNearEdge(landmarks)) this.warnings.push("Hand not fully in frame");
      
      const scale = A.handScaleOfAspect(landmarks, aspectRatio);
      const tip = landmarks[8];
      let tipNorm = [tip.x, tip.y];

      if (!A.rejectOutlier(this.lastValidTipXY, tipNorm, scale, 0.5, aspectRatio)) {
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
      this.warnings.push("No hand detected -- hold index fingertip over spiral");
    }

    ctx.strokeStyle = "#ffc800";
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.tracePoints.forEach((p, i) => { if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
    ctx.stroke();
  },

  render() {
    const btnW = Math.min(320, W * 0.5);
    const btnH = Math.min(50, H * 0.08);
    this.startButton = [W/2 - btnW/2, H - H*0.2, W/2 + btnW/2, H - H*0.2 + btnH];

    if (this.state === "waiting") {
      drawText("Get your fingertip in frame, then click Start or press SPACE", 30, 50, "#22c55e", 20);
      drawText("Trace the gray spiral with your index fingertip", 30, 80, "#c8c8c8", 17);
      drawButton(this.startButton, "Start Test");
    } else if (this.state === "running") {
      const elapsed = performance.now()/1000 - this.runStart;
      const remaining = SPIRAL_MAX_DURATION - elapsed;
      drawText(`Tracing... time left: ${remaining.toFixed(1)}s (press F when finished)`, 30, 40, COLOR.info, 20);
      if (this.deviationSamples.length) {
        const meanDev = A.mean(this.deviationSamples);
        drawText(`Live avg deviation: ${meanDev.toFixed(1)}px (${(meanDev/this.spiralMaxRadius*100).toFixed(1)}% of radius)`, 30, 65, COLOR.info, 16);
      }
      if (remaining <= 0) this.state = "done";
    } else if (this.state === "done") {
      let y = 40;
      if (this.deviationSamples.length < 10) {
        drawText("Not enough tracing captured -- press r to retry", 30, y, COLOR.warning, 20);
      } else {
        const meanDev = A.mean(this.deviationSamples);
        const maxDev = Math.max(...this.deviationSamples);
        const meanDevPct = meanDev / this.spiralMaxRadius * 100;
        const maxDevPct = maxDev / this.spiralMaxRadius * 100;

        const lines = [
          [`Mean deviation from spiral: ${meanDev.toFixed(1)}px (${meanDevPct.toFixed(1)}% of radius)`, COLOR.good],
          [`Max deviation from spiral: ${maxDev.toFixed(1)}px (${maxDevPct.toFixed(1)}% of radius)`, COLOR.info],
        ];

        const jitterResult = this.tremorBuffer.length ? A.analyzeTremor(this.tremorBuffer) : null;
        if (jitterResult) {
          if (jitterResult.status === "fps_too_low") {
            lines.push([`Camera too slow (${jitterResult.rawFps.toFixed(0)} fps) for a reliable jitter reading`, COLOR.warning]);
          } else if (jitterResult.confident) {
            lines.push([`Jitter while tracing: ${jitterResult.freqHz.toFixed(1)} Hz (confidence ${jitterResult.bandRatio.toFixed(2)})`, COLOR.warning]);
          } else if (!jitterResult.hasFpsHeadroom) {
            lines.push([`${jitterResult.rawFps.toFixed(0)} fps -- borderline, jitter reading may be unreliable`, COLOR.warning]);
          } else {
            lines.push(["No clear jitter detected while tracing", COLOR.good]);
          }
        }

        for (const [text, color] of lines) { drawText(text, 30, y, color, 20); y += 30; }
      }
      drawText("Press r to retry", 30, y + 10, "#c8c8c8", 17);
    }

    let y = 110;
    for (const w of this.warnings) { drawText(w, 10, y, COLOR.warning, 18); y += 22; }

    drawHelpLine("SPACE: start    F: finish early    q/Esc: menu    r: restart");
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

// ---------------- Main loop ----------------

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

  if (handLandmarker) {
    let nowMs = performance.now();
    if (nowMs <= lastTimestamp) {
      nowMs = lastTimestamp + 1;
    }
    lastTimestamp = nowMs;

    try {
      const result = handLandmarker.detectForVideo(procCanvas, nowMs);
      handsLandmarks = result.landmarks || [];
    } catch (detectErr) {
      console.warn("HandLandmarker tracking error:", detectErr);
    }
  }

  ctx.clearRect(0, 0, W, H);

  if (appState === "menu") {
    renderMenu();
  } else if (appState === "tremor") {
    tremorTest.update(handsLandmarks, now);
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

// ---------------- Input handling ----------------

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

// ---------------- Startup with Fallbacks ----------------

async function getCameraStreamWithFallback() {
  const profiles = [
    { name: "Ideal HD 16:9", constraints: { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } } },
    { name: "Ideal SD 4:3", constraints: { video: { width: { ideal: 640 }, height: { ideal: 480 } } } },
    { name: "Basic Video", constraints: { video: true } },
    { name: "Bare Minimum Constraints", constraints: { video: {} } }
  ];

  for (const profile of profiles) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(profile.constraints);
      return stream;
    } catch (err) {
      console.warn(`Profile failed (${profile.name}):`, err.name, err.message);
    }
  }

  throw new Error("No compatible camera stream could be established.");
}

async function init() {
  try {
    statusLine.textContent = "Loading hand tracking model...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    } catch (gpuErr) {
      console.warn("GPU delegate failed on this hardware, falling back to CPU:", gpuErr);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    }

    statusLine.textContent = "Requesting camera access...";
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

    statusLine.textContent = "";
    appState = "menu";
    detectAndDraw();
  } catch (err) {
    console.error("Camera/Initialization error:", err);
    statusLine.innerHTML =
      `Error: ${err.message}<br>` +
      `<small style="color: #b4b4b4;">Camera hardware not detected or access blocked. Please verify permissions and ensure no other application is using the camera.</small><br>` +
      `<button onclick="location.reload()" style="margin-top:10px; padding:6px 16px; background:#ff7800; color:#fff; border:none; border-radius:4px; cursor:pointer;">Retry Camera Access</button>`;
  }
}

init();
