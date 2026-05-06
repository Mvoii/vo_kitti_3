"""
Visual Odometry — KITTI Dataset
================================
Runs the VO pipeline in a background thread and serves a live dashboard
over HTTP with Server-Sent Events (SSE) for real-time frame streaming.

Usage:
    python VO.py                          # default: KITTI_sequence_1, port 8080
    python VO.py --seq KITTI_sequence_2
    python VO.py --seq KITTI_sequence_1 --port 9090
    python VO.py --no-browser             # suppress auto-opening browser

Standard deps only: cv2, numpy, tqdm — no Flask / aiohttp needed.
"""

import argparse
import json
import math
import os
import queue
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from tqdm import tqdm


#  EMBEDDED DASHBOARD  (served at GET /)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VO Mission Control — KITTI Live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#080c10; --bg2:#0d1219; --bg3:#111820;
    --border:#1e2d3d; --border2:#243447;
    --cyan:#00d4ff; --cyan-dim:#0099bb;
    --amber:#ffb020; --amber-dim:#cc8800;
    --green:#00ff88; --green-dim:#00aa55;
    --red:#ff4466; --red-dim:#cc2244;
    --white:#e8f0f8; --muted:#4a6070;
    --mono:'Share Tech Mono',monospace;
    --display:'Orbitron',monospace;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--white);font-family:var(--mono);
       height:100vh;overflow:hidden;display:flex;flex-direction:column}
  body::after{content:'';position:fixed;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);
    pointer-events:none;z-index:9999}

  /* ── HEADER ── */
  header{display:flex;align-items:center;gap:18px;padding:9px 18px;
         background:var(--bg2);border-bottom:1px solid var(--border2);flex-shrink:0}
  .logo{font-family:var(--display);font-size:11px;font-weight:900;
        letter-spacing:4px;color:var(--cyan);text-shadow:0 0 20px rgba(0,212,255,.5)}
  .logo span{color:var(--muted)}
  .hdr-divider{width:1px;height:20px;background:var(--border2)}
  .status-bar{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted)}
  .status-dot{width:6px;height:6px;border-radius:50%;background:var(--muted)}
  .status-dot.live{background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 1.2s ease-in-out infinite}
  .status-dot.done{background:var(--cyan);box-shadow:0 0 6px var(--cyan);animation:none}
  .status-dot.warn{background:var(--amber);animation:pulse 2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}

  .seq-label{font-size:9px;color:var(--muted);letter-spacing:1px}
  .seq-val{font-size:10px;color:var(--amber);letter-spacing:1px}

  .seq-tabs{display:flex;gap:4px;margin-left:auto}
  .seq-tab{padding:5px 14px;font-family:var(--display);font-size:9px;letter-spacing:2px;
           border:1px solid var(--border2);background:transparent;color:var(--muted);
           cursor:pointer;transition:all .2s}
  .seq-tab.active{background:rgba(0,212,255,.1);border-color:var(--cyan);
                   color:var(--cyan);box-shadow:0 0 12px rgba(0,212,255,.2)}
  .seq-tab:hover:not(.active){color:var(--white)}
  .seq-tab:disabled{opacity:.4;cursor:not-allowed}

  .done-badge{display:none;padding:3px 8px;background:rgba(0,255,136,.1);
              border:1px solid var(--green);color:var(--green);font-size:8px;
              letter-spacing:2px;font-family:var(--display)}
  .done-badge.show{display:inline-block;animation:fadeIn .5s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}

  /* MAIN GRID */
  .main{display:grid;grid-template-columns:1fr 310px;grid-template-rows:1fr 190px;
        gap:1px;flex:1;background:var(--border);overflow:hidden}
  .panel{background:var(--bg2);display:flex;flex-direction:column;overflow:hidden}
  .panel-header{display:flex;align-items:center;gap:10px;padding:8px 14px;
                background:var(--bg3);border-bottom:1px solid var(--border);flex-shrink:0}
  .panel-title{font-family:var(--display);font-size:8px;letter-spacing:3px;color:var(--cyan-dim)}

  /* ── MAP ── */
  #map-panel{grid-row:1;grid-column:1;position:relative}
  #path-canvas{flex:1;display:block;cursor:crosshair}

  /* ── STATS ── */
  #stats-panel{grid-row:1/3;grid-column:2;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border2) transparent}
  #stats-panel::-webkit-scrollbar{width:4px}
  #stats-panel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

  .stat-group{padding:11px 14px;border-bottom:1px solid var(--border)}
  .stat-group-title{font-family:var(--display);font-size:7px;letter-spacing:3px;color:var(--muted);margin-bottom:10px}
  .stat-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
  .stat-label{font-size:10px;color:var(--muted)}
  .stat-value{font-size:13px;font-weight:bold;color:var(--white);text-align:right;transition:color .3s}
  .stat-value.cyan{color:var(--cyan)} .stat-value.amber{color:var(--amber)}
  .stat-value.green{color:var(--green)} .stat-value.red{color:var(--red)}
  .stat-unit{font-size:9px;color:var(--muted);margin-left:2px}

  .error-meter{margin-top:8px}
  .error-meter-bar{height:4px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:4px}
  .error-meter-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--green),var(--amber),var(--red));transition:width .4s ease;width:0%}

  .progress-ring-wrap{display:flex;justify-content:center;align-items:center;padding:12px 0;position:relative}
  .progress-ring{transform:rotate(-90deg)}
  .ring-bg{fill:none;stroke:var(--border2);stroke-width:4}
  .ring-fill{fill:none;stroke:var(--cyan);stroke-width:4;stroke-linecap:round;transition:stroke-dashoffset .4s}
  .ring-center{position:absolute;text-align:center;display:flex;flex-direction:column;align-items:center}
  .ring-pct{font-family:var(--display);font-size:20px;color:var(--cyan)}
  .ring-sub{font-size:9px;color:var(--muted)}

  .frame-bars{display:flex;gap:2px;margin-top:6px}
  .frame-bar-seg{height:3px;flex:1;border-radius:1px;background:var(--border2);transition:background .3s}

  .toggle-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0}
  .toggle-label{font-size:10px;color:var(--muted);display:flex;align-items:center;gap:6px}
  .toggle-dot{width:8px;height:8px;border-radius:50%}
  .dot-cyan{background:var(--cyan);box-shadow:0 0 5px var(--cyan)}
  .dot-amber{background:var(--amber);box-shadow:0 0 5px var(--amber)}
  .dot-red{background:var(--red)} .dot-green{background:var(--green)}

  .toggle{width:32px;height:16px;border-radius:8px;background:var(--border2);
          border:none;cursor:pointer;position:relative;transition:background .2s;flex-shrink:0}
  .toggle::after{content:'';position:absolute;top:2px;left:2px;width:12px;height:12px;
                 border-radius:50%;background:var(--muted);transition:transform .2s,background .2s}
  .toggle.on{background:rgba(0,212,255,.2)}
  .toggle.on::after{transform:translateX(16px);background:var(--cyan);box-shadow:0 0 5px var(--cyan)}

  .pose-matrix{font-size:9px;line-height:1.7;color:var(--muted);font-family:var(--mono);
               background:var(--bg);padding:8px 10px;border:1px solid var(--border);
               border-radius:2px;white-space:pre}
  .pose-matrix .hi{color:var(--cyan)} .pose-matrix .pos{color:var(--amber)}

  /* match quality bar */
  .match-bar{height:3px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:6px}
  .match-fill{height:100%;background:linear-gradient(90deg,var(--amber),var(--green));border-radius:2px;transition:width .4s}

  /* ── ERROR CHART ── */
  #error-panel{grid-row:2;grid-column:1;border-top:1px solid var(--border)}
  #error-canvas{flex:1;display:block}

  /* ── CONTROLS ── */
  .controls{display:flex;align-items:center;gap:8px;padding:7px 14px;
            background:var(--bg);border-top:1px solid var(--border);flex-shrink:0}
  .ctrl-btn{width:30px;height:30px;border:1px solid var(--border2);background:transparent;
            color:var(--muted);cursor:pointer;font-size:12px;display:flex;
            align-items:center;justify-content:center;transition:all .15s;font-family:var(--mono)}
  .ctrl-btn:hover{border-color:var(--cyan);color:var(--cyan)}
  .ctrl-btn.active{background:rgba(0,212,255,.15);border-color:var(--cyan);color:var(--cyan)}
  .speed-slider{-webkit-appearance:none;width:72px;height:2px;background:var(--border2);
                border:none;outline:none;cursor:pointer}
  .speed-slider::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;
    border-radius:50%;background:var(--cyan);box-shadow:0 0 5px var(--cyan)}
  .frame-counter{font-family:var(--display);font-size:11px;color:var(--amber);margin-left:auto}
  .reset-btn{padding:5px 10px;border:1px solid var(--border2);background:transparent;
             color:var(--muted);font-family:var(--mono);font-size:9px;cursor:pointer;transition:all .15s}
  .reset-btn:hover{border-color:var(--amber);color:var(--amber)}

  /* ── WAITING OVERLAY ── */
  #wait-overlay{position:absolute;inset:0;background:rgba(8,12,16,.92);
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                gap:16px;z-index:10;transition:opacity .5s}
  #wait-overlay.hidden{opacity:0;pointer-events:none}
  .wait-spinner{width:48px;height:48px;border:2px solid var(--border2);
                border-top-color:var(--cyan);border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .wait-text{font-family:var(--display);font-size:9px;letter-spacing:3px;color:var(--cyan)}
  .wait-sub{font-size:10px;color:var(--muted)}

  /* tooltip */
  #tooltip{position:fixed;background:rgba(13,18,25,.96);border:1px solid var(--cyan);
           padding:6px 10px;font-size:10px;color:var(--white);pointer-events:none;
           opacity:0;transition:opacity .15s;z-index:100;white-space:nowrap}
</style>
</head>
<body>

<header>
  <div class="logo">VO<span>//</span>MISSION<span>·</span>CTRL</div>
  <div class="hdr-divider"></div>
  <div class="status-bar">
    <div class="status-dot warn" id="statusDot"></div>
    <span id="statusText">CONNECTING…</span>
  </div>
  <div class="hdr-divider"></div>
  <div class="seq-label">SEQ &nbsp;</div>
  <div class="seq-val" id="hdrSeq">—</div>
  <div class="seq-tabs" id="seqTabs">
    <button class="seq-tab" id="tabSeq1" onclick="requestSeq('KITTI_sequence_1')">SEQ · 1</button>
    <button class="seq-tab" id="tabSeq2" onclick="requestSeq('KITTI_sequence_2')">SEQ · 2</button>
  </div>
  <div class="done-badge" id="doneBadge">■ COMPLETE</div>
</header>

<div class="main">

  <!-- MAP -->
  <div class="panel" id="map-panel" style="position:relative">
    <div class="panel-header">
      <div class="panel-title">live trajectory map</div>
      <div style="display:flex;gap:12px;align-items:center;margin-left:auto">
        <span style="font-size:9px;color:var(--muted)">ERR VEC</span>
        <button class="toggle on" id="errVecToggle" onclick="toggleLayer('errVec')"></button>
        <span style="font-size:9px;color:var(--muted)">GHOST</span>
        <button class="toggle on" id="ghostToggle" onclick="toggleLayer('ghost')"></button>
      </div>
    </div>
    <canvas id="path-canvas"></canvas>
    <div class="controls">
      <button class="ctrl-btn" id="playBtn" onclick="toggleReplay()">▶</button>
      <button class="ctrl-btn" onclick="replayStep(-1)">◀</button>
      <button class="ctrl-btn" onclick="replayStep(1)">▶▶</button>
      <span style="font-size:9px;color:var(--muted);margin-left:4px">SPD</span>
      <input class="speed-slider" type="range" min="1" max="20" value="10" id="speedSlider" oninput="updateSpeed()">
      <span id="speedVal" style="font-size:9px;color:var(--cyan);min-width:18px">10x</span>
      <div class="frame-counter" id="frameCounter">000 / 000</div>
      <button class="reset-btn" onclick="replayReset()">↺ RESET</button>
    </div>

    <!-- waiting overlay -->
    <div id="wait-overlay">
      <div class="wait-spinner"></div>
      <div class="wait-text">AWAITING DATA</div>
      <div class="wait-sub" id="waitSub">Connecting to VO pipeline…</div>
    </div>
  </div>

  <!-- STATS -->
  <div class="panel" id="stats-panel">
    <div class="stat-group">
      <div class="stat-group-title">pipeline progress</div>
      <div class="progress-ring-wrap">
        <svg class="progress-ring" width="80" height="80" viewBox="0 0 80 80">
          <circle class="ring-bg" cx="40" cy="40" r="34"/>
          <circle class="ring-fill" cx="40" cy="40" r="34" id="ringFill"
            stroke-dasharray="213.6" stroke-dashoffset="213.6"/>
        </svg>
        <div class="ring-center">
          <div class="ring-pct" id="ringPct">0%</div>
          <div class="ring-sub">FRAMES</div>
        </div>
      </div>
      <div class="frame-bars" id="frameBars"></div>
    </div>

    <div class="stat-group">
      <div class="stat-group-title">live metrics</div>
      <div class="stat-row">
        <div class="stat-label">Frame</div>
        <div class="stat-value cyan" id="mFrame">—</div>
      </div>
      <div class="stat-row">
        <div class="stat-label">Matches</div>
        <div class="stat-value" id="mMatches">—</div>
      </div>
      <div class="stat-row">
        <div class="stat-label">Current Error</div>
        <div class="stat-value" id="mCurErr">—</div>
      </div>
      <div class="stat-row">
        <div class="stat-label">Running RMSE</div>
        <div class="stat-value amber" id="mRmse">—</div>
      </div>
      <div class="stat-row">
        <div class="stat-label">Peak Error</div>
        <div class="stat-value red" id="mPeak">—</div>
      </div>
      <div class="stat-row">
        <div class="stat-label">Frame Time</div>
        <div class="stat-value" id="mTime" style="font-size:11px;color:var(--muted)">—</div>
      </div>
      <div class="error-meter">
        <div style="font-size:9px;color:var(--muted);margin-bottom:2px">ERROR MAGNITUDE</div>
        <div class="error-meter-bar"><div class="error-meter-fill" id="errFill"></div></div>
      </div>
      <div style="font-size:9px;color:var(--muted);margin-top:8px;margin-bottom:2px">MATCH QUALITY</div>
      <div class="match-bar"><div class="match-fill" id="matchFill"></div></div>
    </div>

    <!-- final (hidden until done) -->
    <div class="stat-group" id="finalGroup" style="opacity:.3">
      <div class="stat-group-title">final results</div>
      <div class="stat-row"><div class="stat-label">Total RMSE</div>
        <div class="stat-value green" id="fRmse">—</div></div>
      <div class="stat-row"><div class="stat-label">Max Error</div>
        <div class="stat-value" id="fMax">—</div></div>
      <div class="stat-row"><div class="stat-label">Mean Error</div>
        <div class="stat-value" id="fMean">—</div></div>
      <div class="stat-row"><div class="stat-label">Path Length</div>
        <div class="stat-value" id="fPath">—</div></div>
      <div class="stat-row"><div class="stat-label">Total Time</div>
        <div class="stat-value" id="fTime">—</div></div>
    </div>

    <!-- layer toggles -->
    <div class="stat-group">
      <div class="stat-group-title">display layers</div>
      <div class="toggle-row">
        <span class="toggle-label"><span class="toggle-dot dot-cyan"></span>Ground Truth</span>
        <button class="toggle on" id="gtToggle" onclick="toggleLayer('gt')"></button>
      </div>
      <div class="toggle-row">
        <span class="toggle-label"><span class="toggle-dot dot-amber"></span>Estimated Path</span>
        <button class="toggle on" id="estToggle" onclick="toggleLayer('est')"></button>
      </div>
      <div class="toggle-row">
        <span class="toggle-label"><span class="toggle-dot dot-red"></span>Error Vectors</span>
        <button class="toggle on" id="errToggle2" onclick="toggleLayer('errVec')"></button>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">Waypoints</span>
        <button class="toggle on" id="wpToggle" onclick="toggleLayer('wp')"></button>
      </div>
    </div>

    <!-- pose -->
    <div class="stat-group">
      <div class="stat-group-title">estimated pose (T_cur)</div>
      <div class="pose-matrix" id="poseMatrix">[ <span class="hi">—</span>  —  — | <span class="pos">—</span> ]
[ —  <span class="hi">—</span>  — | <span class="pos">—</span> ]
[ —  —  <span class="hi">—</span> | <span class="pos">—</span> ]</div>
    </div>
  </div>

  <!-- ERROR CHART -->
  <div class="panel" id="error-panel">
    <div class="panel-header">
      <div class="panel-title">error propagation — live</div>
      <span id="errBadge" style="font-size:9px;color:var(--muted);margin-left:auto">Δ translational error vs frame</span>
    </div>
    <canvas id="error-canvas"></canvas>
  </div>

</div>

<div id="tooltip"></div>

<script>

//  DATA STORE
const frames = [];          // received from SSE
let totalFrames = 0;
let maxError = 0;
let peakError = 0;
let seqName = '';
let isDone = false;
let maxMatches = 500;

// replay state (separate from live cursor)
let replayFrame = 0;        // what the user is viewing
let replayMode = false;     // true when user has taken manual control
let replayPlaying = false;
let replayTimer = null;
let replaySpeed = 10;

// layer visibility
const layers = { gt: true, est: true, errVec: true, ghost: true, wp: true };

const mapCanvas = document.getElementById('path-canvas');
const mapCtx = mapCanvas.getContext('2d');
const errCanvas = document.getElementById('error-canvas');
const errCtx = errCanvas.getContext('2d');


//  SSE  —  LIVE DATA
let sse = null;

function connectSSE() {
  if (sse) sse.close();
  sse = new EventSource('/stream');

  sse.onopen = () => setStatus('LIVE', 'live');

  sse.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    seqName = d.seq;
    totalFrames = d.total;
    isDone = false;
    document.getElementById('hdrSeq').textContent = d.seq || '—';
    document.getElementById('doneBadge').className = 'done-badge';
    document.getElementById('finalGroup').style.opacity = '0.3';
    if (d.state === 'loading') {
      setStatus('LOADING DATA…', 'warn');
      document.getElementById('waitSub').textContent = 'Loading ' + d.seq + '…';
    } else if (d.state === 'processing') {
      setStatus('LIVE', 'live');
      hideWaitOverlay();
    }
    // highlight active seq tab
    document.getElementById('tabSeq1').classList.toggle('active', d.seq === 'KITTI_sequence_1');
    document.getElementById('tabSeq2').classList.toggle('active', d.seq === 'KITTI_sequence_2');
    initFrameBars();
    resize();
  });

  sse.addEventListener('frame', e => {
    const d = JSON.parse(e.data);
    frames[d.i] = d;
    if (d.error > maxError) maxError = d.error;
    if (d.error > peakError) peakError = d.error;
    if (d.matches > maxMatches) maxMatches = d.matches;
    if (d.i === 0) {
      frames.length = 1;
      peakError = d.error;
      maxError = 0.001;
    }
    if (!replayMode) replayFrame = d.i;
    hideWaitOverlay();
    if (d.i === 0) setStatus('PROCESSING', 'live');
    updateUI();
  });

  sse.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    isDone = true;
    setStatus('COMPLETE', 'done');
    document.getElementById('doneBadge').className = 'done-badge show';
    document.getElementById('finalGroup').style.opacity = '1';
    document.getElementById('fRmse').innerHTML = d.rmse.toFixed(4) + '<span class="stat-unit">m</span>';
    document.getElementById('fMax').innerHTML  = d.max_error.toFixed(4) + '<span class="stat-unit">m</span>';
    document.getElementById('fMean').innerHTML = d.mean_error.toFixed(4) + '<span class="stat-unit">m</span>';
    document.getElementById('fPath').innerHTML = d.path_length.toFixed(2) + '<span class="stat-unit">m</span>';
    document.getElementById('fTime').innerHTML = d.elapsed_s.toFixed(1) + '<span class="stat-unit">s</span>';
    // re-enable seq tabs
    document.querySelectorAll('.seq-tab').forEach(t => t.disabled = false);
    updateUI();
  });

  sse.onerror = () => {
    setStatus('RECONNECTING…', 'warn');
    document.getElementById('waitSub').textContent = 'Lost connection — retrying…';
  };
}


//  RESIZE
function resize() {
  const mapPanel = document.getElementById('map-panel');
  const errPanel = document.getElementById('error-panel');
  const r1 = mapPanel.getBoundingClientRect();
  const r2 = errPanel.getBoundingClientRect();
  mapCanvas.width = r1.width;
  mapCanvas.height = r1.height - 46;   // controls bar height
  errCanvas.width  = r2.width;
  errCanvas.height = r2.height - 36;
  drawAll();
}
window.addEventListener('resize', resize);


//  COORDINATE TRANSFORM
function getTransform() {
  const pad = 44;
  const W = mapCanvas.width - pad * 2;
  const H = mapCanvas.height - pad * 2;
  const n = replayFrame + 1;
  if (n === 0 || frames.length === 0) return null;

  const allX = [], allZ = [];
  for (let i = 0; i < frames.length; i++) {
    if (!frames[i]) continue;
    allX.push(frames[i].gt_x, frames[i].est_x);
    allZ.push(frames[i].gt_z, frames[i].est_z);
  }
  if (!allX.length) return null;

  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minZ = Math.min(...allZ), maxZ = Math.max(...allZ);
  const rX = maxX - minX || 1, rZ = maxZ - minZ || 1;
  const scale = Math.min(W / rX, H / rZ);
  return {
    scale,
    offX: pad + (W - rX * scale) / 2 - minX * scale,
    offZ: pad + (H - rZ * scale) / 2,
    maxZ
  };
}

function toCanvas(x, z, t) {
  return { cx: x * t.scale + t.offX, cy: (t.maxZ - z) * t.scale + t.offZ };
}


//  DRAW MAP
function drawMap() {
  const ctx = mapCtx;
  const W = mapCanvas.width, H = mapCanvas.height;
  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.strokeStyle = 'rgba(30,45,60,.8)'; ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 50) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
  for (let y = 0; y < H; y += 50) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

  // corner marks
  const C = [[10,10],[W-10,10],[10,H-10],[W-10,H-10]];
  ctx.strokeStyle = 'rgba(0,212,255,.25)'; ctx.lineWidth = 1;
  C.forEach(([cx,cy]) => {
    const s = 10;
    ctx.beginPath(); ctx.moveTo(cx-s,cy); ctx.lineTo(cx,cy); ctx.lineTo(cx,cy+(cy<H/2?s:-s)); ctx.stroke();
  });

  if (frames.length === 0) return;
  const t = getTransform();
  if (!t) return;

  const n = Math.min(replayFrame + 1, frames.length);

  // ── ghost (full future path, very faint) ──
  if (layers.ghost && frames.length > 1) {
    ctx.strokeStyle = 'rgba(0,212,255,.08)';
    ctx.lineWidth = 1.5; ctx.setLineDash([4,8]);
    ctx.beginPath();
    frames.forEach((f,i) => { if (!f) return; const p = toCanvas(f.gt_x, f.gt_z, t); i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy); });
    ctx.stroke(); ctx.setLineDash([]);
  }

  // error vectors
  if (layers.errVec) {
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const gp = toCanvas(f.gt_x, f.gt_z, t);
      const ep = toCanvas(f.est_x, f.est_z, t);
      const ratio = peakError > 0 ? f.error / peakError : 0;
      ctx.strokeStyle = `rgba(255,68,102,${0.2 + ratio * 0.6})`;
      ctx.lineWidth = 1; ctx.setLineDash([2,3]);
      ctx.beginPath(); ctx.moveTo(gp.cx, gp.cy); ctx.lineTo(ep.cx, ep.cy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // GT path
  if (layers.gt && n > 0) {
    const grad = ctx.createLinearGradient(0,0,0,H);
    grad.addColorStop(0,'rgba(0,212,255,.3)'); grad.addColorStop(1,'rgba(0,212,255,1)');
    ctx.strokeStyle = grad; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const p = toCanvas(f.gt_x, f.gt_z, t);
      i===0 ? ctx.moveTo(p.cx,p.cy) : ctx.lineTo(p.cx,p.cy);
    }
    ctx.stroke();
  }

  // estimated path
  if (layers.est && n > 0) {
    ctx.strokeStyle = '#ffb020'; ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(255,176,32,.5)'; ctx.shadowBlur = 6;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const p = toCanvas(f.est_x, f.est_z, t);
      i===0 ? ctx.moveTo(p.cx,p.cy) : ctx.lineTo(p.cx,p.cy);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
  }

  // waypoints
  if (layers.wp) {
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const isLast = i === n - 1;
      const gp = toCanvas(f.gt_x, f.gt_z, t);
      ctx.fillStyle = isLast ? '#00d4ff' : 'rgba(0,180,220,.5)';
      ctx.beginPath(); ctx.arc(gp.cx,gp.cy,isLast?4:2,0,Math.PI*2); ctx.fill();
      const ep = toCanvas(f.est_x, f.est_z, t);
      ctx.fillStyle = isLast ? '#ffb020' : 'rgba(200,140,20,.5)';
      ctx.beginPath(); ctx.arc(ep.cx,ep.cy,isLast?4:2,0,Math.PI*2); ctx.fill();
    }
  }

  // ── current frame rings ──
  const cf = frames[replayFrame]; if (!cf) return;
  const cgp = toCanvas(cf.gt_x, cf.gt_z, t);
  const cep = toCanvas(cf.est_x, cf.est_z, t);
  [[cgp,'#00d4ff'],[cep,'#ffb020']].forEach(([pt,col]) => {
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(pt.cx, pt.cy, 9, 0, Math.PI*2); ctx.stroke();
  });

  // labels
  if (n > 0 && frames[0]) {
    const sp = toCanvas(frames[0].gt_x, frames[0].gt_z, t);
    ctx.fillStyle = 'rgba(0,212,255,.8)'; ctx.font = '9px Share Tech Mono';
    ctx.fillText('START', sp.cx + 8, sp.cy + 3);
  }
  ctx.fillStyle = 'rgba(74,96,112,.7)'; ctx.font = '9px Share Tech Mono';
  ctx.fillText('X →', W - 28, H - 5);
  ctx.fillText('Z ↑', 5, 18);
}


//  DRAW ERROR CHART
function drawErrorChart() {
  const ctx = errCtx;
  const W = errCanvas.width, H = errCanvas.height;
  const pad = {l:46, r:18, t:14, b:28};
  ctx.clearRect(0, 0, W, H);

  const n = Math.min(replayFrame + 1, frames.length);
  const total = Math.max(totalFrames, frames.length, 1);
  const maxE = Math.max(peakError * 1.1, 0.1);

  // grid
  const gL = 4;
  ctx.strokeStyle = 'rgba(30,45,60,.8)'; ctx.lineWidth = 1;
  for (let i = 0; i <= gL; i++) {
    const y = pad.t + (H - pad.t - pad.b) * i / gL;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = 'rgba(74,96,112,.8)'; ctx.font = '8px Share Tech Mono';
    ctx.fillText((maxE * (1 - i/gL)).toFixed(2), 2, y + 3);
  }
  for (let i = 0; i <= 5; i++) {
    const x = pad.l + (W - pad.l - pad.r) * i / 5;
    ctx.strokeStyle = 'rgba(30,45,60,.5)';
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.fillStyle = 'rgba(74,96,112,.7)'; ctx.font = '8px Share Tech Mono';
    ctx.fillText(Math.round(total * i / 5), x - 4, H - 4);
  }

  // ghost
  if (frames.length > 1) {
    ctx.strokeStyle = 'rgba(255,68,102,.1)'; ctx.lineWidth = 1; ctx.setLineDash([3,5]);
    ctx.beginPath();
    frames.forEach((f,i) => {
      if (!f) return;
      const x = pad.l + (W-pad.l-pad.r)*i/(total-1);
      const y = pad.t + (H-pad.t-pad.b)*(1-f.error/maxE);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }

  // fill
  if (n > 1) {
    const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
    grad.addColorStop(0,'rgba(255,68,102,.3)'); grad.addColorStop(1,'rgba(255,68,102,.02)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const x = pad.l + (W-pad.l-pad.r)*i/(total-1);
      const y = pad.t + (H-pad.t-pad.b)*(1-f.error/maxE);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    const lx = pad.l + (W-pad.l-pad.r)*(n-1)/(total-1);
    ctx.lineTo(lx, H-pad.b); ctx.lineTo(pad.l, H-pad.b); ctx.closePath(); ctx.fill();
  }

  // line
  if (n > 0) {
    ctx.strokeStyle = '#ff4466'; ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(255,68,102,.5)'; ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const f = frames[i]; if (!f) continue;
      const x = pad.l + (W-pad.l-pad.r)*i/(total-1);
      const y = pad.t + (H-pad.t-pad.b)*(1-f.error/maxE);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
  }

  // current dot
  const cf = frames[replayFrame];
  if (cf) {
    const x = pad.l + (W-pad.l-pad.r)*replayFrame/(total-1);
    const y = pad.t + (H-pad.t-pad.b)*(1-cf.error/maxE);
    ctx.fillStyle = '#ff4466'; ctx.shadowColor = '#ff4466'; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,68,102,.3)'; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x, H-pad.b); ctx.stroke(); ctx.setLineDash([]);
  }

  // axes
  ctx.strokeStyle = 'rgba(30,45,60,1)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();

  ctx.fillStyle = 'rgba(74,96,112,.7)'; ctx.font = '8px Share Tech Mono';
  ctx.save(); ctx.translate(10, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('ERROR (m)', -22, 0); ctx.restore();
  ctx.fillText('FRAME', W/2-16, H-1);
}


//  STATS UPDATE
function updateStats() {
  const n = Math.min(replayFrame + 1, frames.length);
  const cf = frames[replayFrame];
  const total = Math.max(totalFrames, frames.length, 1);

  document.getElementById('frameCounter').textContent =
    `${String(replayFrame).padStart(3,'0')} / ${String(total-1).padStart(3,'0')}`;

  // ring
  const pct = n / total;
  document.getElementById('ringFill').style.strokeDashoffset = 213.6 * (1 - pct);
  document.getElementById('ringPct').textContent = Math.round(pct * 100) + '%';

  // frame bars
  const bars = document.getElementById('frameBars').children;
  const barN = bars.length;
  for (let i = 0; i < barN; i++) {
    const fi = Math.floor(i * frames.length / barN);
    const f = frames[fi];
    bars[i].style.background = fi < n && f
      ? `rgb(${Math.round(255*(f.error/peakError))},${Math.round(255*(1-f.error/peakError))},68)`
      : 'var(--border2)';
  }

  if (!cf) return;

  // live metrics
  document.getElementById('mFrame').textContent = replayFrame;
  document.getElementById('mMatches').textContent = cf.matches;

  const errCol = cf.error < 0.5 ? 'green' : cf.error < 1.5 ? 'amber' : 'red';
  document.getElementById('mCurErr').className = 'stat-value ' + errCol;
  document.getElementById('mCurErr').innerHTML = cf.error.toFixed(3) + '<span class="stat-unit">m</span>';

  // running RMSE over displayed frames
  let sumSq = 0, cnt = 0;
  for (let i = 0; i < n; i++) { if (frames[i]) { sumSq += frames[i].error**2; cnt++; } }
  const rmse = cnt > 0 ? Math.sqrt(sumSq / cnt) : 0;
  document.getElementById('mRmse').innerHTML = rmse.toFixed(3) + '<span class="stat-unit">m</span>';
  document.getElementById('mPeak').innerHTML = peakError.toFixed(3) + '<span class="stat-unit">m</span>';
  document.getElementById('mTime').innerHTML = cf.processing_ms + '<span class="stat-unit">ms</span>';

  // meters
  document.getElementById('errFill').style.width = (cf.error / Math.max(peakError, 0.001) * 100) + '%';
  document.getElementById('matchFill').style.width = Math.min(cf.matches / maxMatches * 100, 100) + '%';

  // pose matrix
  if (cf.pose) {
    const p = cf.pose;
    const fmt = v => v.toFixed(3).padStart(7);
    document.getElementById('poseMatrix').innerHTML =
      `[ <span class="hi">${fmt(p[0][0])}</span> ${fmt(p[0][1])} ${fmt(p[0][2])} | <span class="pos">${fmt(p[0][3])}</span> ]\n` +
      `[ ${fmt(p[1][0])} <span class="hi">${fmt(p[1][1])}</span> ${fmt(p[1][2])} | <span class="pos">${fmt(p[1][3])}</span> ]\n` +
      `[ ${fmt(p[2][0])} ${fmt(p[2][1])} <span class="hi">${fmt(p[2][2])}</span> | <span class="pos">${fmt(p[2][3])}</span> ]`;
  }
}

function updateUI() {
  drawAll();
}

function drawAll() {
  drawMap();
  drawErrorChart();
  updateStats();
}


//  REPLAY CONTROLS  (works on already-received data)
function toggleReplay() {
  if (replayPlaying) {
    pauseReplay();
  } else {
    replayMode = true;
    if (replayFrame >= frames.length - 1) replayFrame = 0;
    replayPlaying = true;
    document.getElementById('playBtn').textContent = '⏸';
    document.getElementById('playBtn').classList.add('active');
    scheduleReplay();
  }
}

function scheduleReplay() {
  if (!replayPlaying) return;
  replayTimer = setTimeout(() => {
    replayFrame = Math.min(replayFrame + 1, frames.length - 1);
    drawAll();
    if (replayFrame < frames.length - 1) scheduleReplay();
    else pauseReplay();
  }, Math.round(1000 / replaySpeed));
}

function pauseReplay() {
  replayPlaying = false;
  clearTimeout(replayTimer);
  document.getElementById('playBtn').textContent = '▶';
  document.getElementById('playBtn').classList.remove('active');
}

function replayStep(d) {
  pauseReplay(); replayMode = true;
  replayFrame = Math.max(0, Math.min(frames.length - 1, replayFrame + d));
  drawAll();
}

function replayReset() {
  pauseReplay(); replayMode = false;
  replayFrame = Math.max(0, frames.length - 1);
  drawAll();
}

function updateSpeed() {
  replaySpeed = parseInt(document.getElementById('speedSlider').value);
  document.getElementById('speedVal').textContent = replaySpeed + 'x';
}


//  LAYER TOGGLES
function toggleLayer(name) {
  layers[name] = !layers[name];
  const ids = {
    gt: 'gtToggle', est: 'estToggle', errVec: ['errVecToggle','errToggle2'], wp: 'wpToggle', ghost: 'ghostToggle'
  };
  const id = ids[name];
  const set = id => { document.getElementById(id).className = 'toggle' + (layers[name]?' on':''); };
  Array.isArray(id) ? id.forEach(set) : set(id);
  drawMap();
}

//  SEQ REQUEST
function requestSeq(seq) {
  if (!isDone) return; // only allow switch when previous run is done
  frames.length = 0;
  replayFrame = 0; replayMode = false; peakError = 0; maxError = 0;
  document.getElementById('doneBadge').className = 'done-badge';
  document.getElementById('finalGroup').style.opacity = '0.3';
  document.getElementById('hdrSeq').textContent = '…';
  document.querySelectorAll('.seq-tab').forEach(t => t.disabled = true);
  fetch('/run?seq=' + seq).catch(() => {});
}


//  STATUS / OVERLAY HELPERS
function setStatus(text, state) {
  document.getElementById('statusText').textContent = text;
  document.getElementById('statusDot').className = 'status-dot ' + state;
}

function hideWaitOverlay() {
  document.getElementById('wait-overlay').classList.add('hidden');
}

function initFrameBars() {
  const el = document.getElementById('frameBars');
  el.innerHTML = '';
  const count = 50;
  for (let i = 0; i < count; i++) {
    const d = document.createElement('div');
    d.className = 'frame-bar-seg';
    el.appendChild(d);
  }
}


//  HOVER TOOLTIP
mapCanvas.addEventListener('mousemove', e => {
  const t = getTransform(); if (!t) return;
  const r = mapCanvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const n = Math.min(replayFrame + 1, frames.length);
  let best = -1, bestD = 18;
  for (let i = 0; i < n; i++) {
    const f = frames[i]; if (!f) continue;
    const gp = toCanvas(f.gt_x, f.gt_z, t);
    const ep = toCanvas(f.est_x, f.est_z, t);
    const d = Math.min(Math.hypot(mx-gp.cx,my-gp.cy), Math.hypot(mx-ep.cx,my-ep.cy));
    if (d < bestD) { bestD = d; best = i; }
  }
  const tt = document.getElementById('tooltip');
  if (best >= 0) {
    const f = frames[best];
    tt.style.opacity = '1';
    tt.style.left = (e.clientX + 12) + 'px';
    tt.style.top  = (e.clientY - 32) + 'px';
    tt.innerHTML = `Frame ${best} &nbsp;|&nbsp; Err: <span style="color:#ff4466">${f.error.toFixed(4)}m</span>&nbsp;|&nbsp;Matches:${f.matches}<br>GT(${f.gt_x.toFixed(3)},${f.gt_z.toFixed(3)}) &nbsp; Est(${f.est_x.toFixed(3)},${f.est_z.toFixed(3)})`;
  } else { tt.style.opacity = '0'; }
});
mapCanvas.addEventListener('mouseleave', () => { document.getElementById('tooltip').style.opacity='0'; });

errCanvas.addEventListener('mousemove', e => {
  const pad = {l:46,r:18,t:14,b:28};
  const r = errCanvas.getBoundingClientRect();
  const mx = e.clientX - r.left;
  const total = Math.max(totalFrames, frames.length, 1);
  const fi = Math.round((mx-pad.l)/(errCanvas.width-pad.l-pad.r)*(total-1));
  if (fi < 0 || fi >= frames.length || !frames[fi]) return;
  const tt = document.getElementById('tooltip');
  tt.style.opacity='1'; tt.style.left=(e.clientX+12)+'px'; tt.style.top=(e.clientY-32)+'px';
  tt.innerHTML = `Frame ${fi} &nbsp;|&nbsp; Error: <span style="color:#ff4466">${frames[fi].error.toFixed(4)}m</span>`;
});
errCanvas.addEventListener('mouseleave', () => { document.getElementById('tooltip').style.opacity='0'; });

//  INIT
window.addEventListener('load', () => {
  initFrameBars();
  resize();
  connectSSE();
});
</script>
</body>
</html>"""

#
#  GLOBAL SERVER STATE
#
_lock        = threading.Lock()
_subscribers = []           # list[queue.Queue]  — one per SSE client
_frame_history = []         # list[dict]          — every emitted frame event
_status_event  = None       # dict                — last 'status' event
_done_event    = None       # dict | None
_vo_thread     = None       # current VO worker thread
_stop_flag     = threading.Event()   # signal VO thread to abort


def _broadcast(event_type: str, payload: dict):
    """Put an event on every active subscriber queue."""
    msg = {"type": event_type, "data": payload}
    with _lock:
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


def push_status(state: str, seq: str, total: int):
    global _status_event, _frame_history, _done_event
    evt = {"state": state, "seq": seq, "total": total}
    with _lock:
        _status_event  = evt
        _frame_history = []
        _done_event    = None
    _broadcast("status", evt)


def push_frame(frame_dict: dict):
    with _lock:
        _frame_history.append(frame_dict)
    _broadcast("frame", frame_dict)


def push_done(summary: dict):
    global _done_event
    with _lock:
        _done_event = summary
    _broadcast("done", summary)


#
#  HTTP REQUEST HANDLER
#
class VOHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress per-request access log noise; keep errors
        pass

    def log_error(self, fmt, *args):
        print(f"[server] ERROR: {fmt % args}")

    # routing
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/stream":
            self._serve_sse()
        elif path == "/run":
            qs  = parse_qs(parsed.query)
            seq = qs.get("seq", ["KITTI_sequence_1"])[0]
            self._handle_run(seq)
        else:
            self.send_error(404)

    # dashboard HTML
    def _serve_dashboard(self):
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # SSE stream
    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue(maxsize=512)

        # Catch up: replay everything already emitted so reconnects
        # and new tabs see the full picture immediately.
        with _lock:
            _subscribers.append(q)
            catch_status = _status_event
            catch_frames = list(_frame_history)
            catch_done   = _done_event

        try:
            if catch_status:
                self._write_event("status", catch_status)
            for f in catch_frames:
                self._write_event("frame", f)
            if catch_done:
                self._write_event("done", catch_done)

            # Stream new events as they arrive
            while True:
                try:
                    msg = q.get(timeout=25)       # 25 s heartbeat
                except queue.Empty:
                    self._write_raw(b": heartbeat\n\n")
                    continue
                self._write_event(msg["type"], msg["data"])

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _lock:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass

    def _write_event(self, event_type: str, payload: dict):
        data = json.dumps(payload, separators=(",", ":"))
        raw  = f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")
        self._write_raw(raw)

    def _write_raw(self, raw: bytes):
        self.wfile.write(raw)
        self.wfile.flush()

    # /run?seq=...
    def _handle_run(self, seq: str):
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
        # Start a new VO run in the background
        _restart_vo(seq)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles each connection in a new thread."""
    daemon_threads = True


#
#  VISUAL ODOMETRY CLASS  (original, unchanged)
#
class VisualOdometry:
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, "calib.txt"))
        self.gt_poses   = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images     = self._load_images(os.path.join(data_dir, "image_l"))
        self.orb        = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params    = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                               key_size=12, multi_probe_level=1)
        search_params   = dict(checks=50)
        self.flann      = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, "r") as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=" ")
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=" ")
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        paths = [os.path.join(filepath, f) for f in sorted(os.listdir(filepath))]
        return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T

    def get_matches(self, i):
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i],     None)
        matches   = self.flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.5 * n.distance]
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(E, q1, q2)
        return self._form_transf(R, t)

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t_flat     = np.ndarray.flatten(t)
        candidates = [
            self._form_transf(R1,  t_flat),
            self._form_transf(R2,  t_flat),
            self._form_transf(R1, -t_flat),
            self._form_transf(R2, -t_flat),
        ]
        K_hom = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        scores = []
        for T_cand in candidates:
            P_cand = K_hom @ T_cand
            hom_Q1 = cv2.triangulatePoints(self.P, P_cand, q1.T, q2.T)
            hom_Q2 = T_cand @ hom_Q1
            Q1 = hom_Q1[:3] / hom_Q1[3]
            Q2 = hom_Q2[:3] / hom_Q2[3]
            pos   = sum(Q2[2] > 0) + sum(Q1[2] > 0)
            scale = np.mean(
                np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            )
            scores.append(pos + scale)
        best = int(np.argmax(scores))
        R_out = R1 if best in (0, 2) else R2
        t_out = t_flat if best in (0, 1) else -t_flat
        return R_out, t_out


#
#  VO PIPELINE WORKER
#
def _run_vo(data_dir: str):
    """Run the full VO loop; push each frame to the dashboard via SSE."""
    global _stop_flag

    seq_name = os.path.basename(data_dir)
    print(f"\n[vo] Loading data from '{data_dir}' …")
    push_status("loading", seq_name, 0)

    try:
        vo = VisualOdometry(data_dir)
    except Exception as exc:
        print(f"[vo] ERROR during init: {exc}")
        return

    n_frames = len(vo.gt_poses)
    push_status("processing", seq_name, n_frames)
    print(f"[vo] Starting pipeline: {n_frames} frames")

    cur_pose    = vo.gt_poses[0]
    gt_path     = []
    est_path    = []
    errors      = []
    start_time  = time.time()

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose", ncols=72)):
        if _stop_flag.is_set():
            print("[vo] Stopped by new run request.")
            return

        frame_start = time.perf_counter()

        if i == 0:
            cur_pose = gt_pose
            n_matches = 0
        else:
            try:
                q1, q2   = vo.get_matches(i)
                n_matches = len(q1)
                transf   = vo.get_pose(q1, q2)
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            except Exception as exc:
                print(f"[vo] Frame {i} error: {exc}")
                n_matches = 0

        proc_ms = round((time.perf_counter() - frame_start) * 1000)

        gt_x, gt_z    = float(gt_pose[0, 3]),   float(gt_pose[2, 3])
        est_x, est_z  = float(cur_pose[0, 3]),  float(cur_pose[2, 3])
        error         = math.hypot(gt_x - est_x, gt_z - est_z)

        gt_path.append((gt_x, gt_z))
        est_path.append((est_x, est_z))
        errors.append(error)

        # 3×4 pose matrix for the dashboard
        pose_rows = cur_pose[:3].tolist()

        push_frame({
            "i":             i,
            "gt_x":          gt_x,
            "gt_z":          gt_z,
            "est_x":         est_x,
            "est_z":         est_z,
            "error":         error,
            "matches":       n_matches,
            "processing_ms": proc_ms,
            "pose":          pose_rows,
        })

    # summary
    errors_arr   = np.array(errors)
    rmse         = float(np.sqrt(np.mean(errors_arr ** 2)))
    max_error    = float(np.max(errors_arr))
    mean_error   = float(np.mean(errors_arr))
    path_length  = sum(
        math.hypot(gt_path[i][0] - gt_path[i-1][0],
                   gt_path[i][1] - gt_path[i-1][1])
        for i in range(1, len(gt_path))
    )
    elapsed_s    = time.time() - start_time

    print(f"\n[vo] Done  |  RMSE={rmse:.4f}m  max={max_error:.4f}m  "
          f"mean={mean_error:.4f}m  path={path_length:.1f}m  "
          f"time={elapsed_s:.1f}s")

    push_done({
        "rmse":         rmse,
        "max_error":    max_error,
        "mean_error":   mean_error,
        "path_length":  path_length,
        "elapsed_s":    elapsed_s,
    })


def _restart_vo(data_dir: str):
    """Abort any running VO thread and start a new one."""
    global _vo_thread, _stop_flag
    _stop_flag.set()
    if _vo_thread and _vo_thread.is_alive():
        _vo_thread.join(timeout=3)
    _stop_flag = threading.Event()
    _vo_thread = threading.Thread(target=_run_vo, args=(data_dir,), daemon=True)
    _vo_thread.start()


#
#  MAIN
#
def main():
    parser = argparse.ArgumentParser(description="Visual Odometry — live dashboard")
    parser.add_argument("--seq",        default="KITTI_sequence_1",
                        help="KITTI sequence directory (default: KITTI_sequence_1)")
    parser.add_argument("--port",       type=int, default=8080,
                        help="HTTP port for the dashboard (default: 8080)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open browser automatically")
    args = parser.parse_args()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), VOHandler)

    # Start the VO pipeline in the background immediately
    _restart_vo(args.seq)

    url = f"http://localhost:{args.port}"
    print(f"\n{'─'*52}")
    print(f"  VO Mission Control")
    print(f"  Dashboard : {url}")
    print(f"  Sequence  : {args.seq}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'─'*52}\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
        _stop_flag.set()
        server.shutdown()


if __name__ == "__main__":
    main()
