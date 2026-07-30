"""
Visual Odometry — KITTI Dataset
================================
Runs the VO pipeline in a background thread and serves a live dashboard
over HTTP with Server-Sent Events (SSE) for real-time frame streaming,
live camera feed with keypoint overlay, and instant restart capability.

Usage:
    python VO.py                           # default: KITTI_sequence_1, port 8080
    python VO.py --seq KITTI_sequence_2
    python VO.py --seq KITTI_sequence_1 --port 9090
    python VO.py --no-browser              # suppress auto-opening browser
    python VO.py --video-width 480         # smaller feed (faster)
    python VO.py --jpeg-quality 55         # lower quality (less bandwidth)

Standard deps only: cv2, numpy, tqdm — no Flask / aiohttp needed.

MODIFICATIONS vs original VO.py
════════════════════════════════
1. encode_frame()   — new helper: resizes grayscale frame, draws matched
                      keypoints colour-coded by current error magnitude,
                      returns base64-encoded JPEG string.

2. push_image()     — new broadcaster: sends an 'image' SSE event with the
                      encoded frame; only the latest image is kept in memory
                      for catch-up on reconnect (keeps memory flat).

3. _run_vo()        — now calls encode_frame() and push_image() each iteration.
                      Rolling FPS is computed from a deque of timestamps.

4. VOHandler        — added /run route for restart; SSE catch-up now also
                      replays _last_image so reconnecting clients see the
                      last camera frame immediately.

5. main()           — added --video-width and --jpeg-quality CLI flags.

6. DASHBOARD_HTML   — rewritten: 3-column layout (map | video | stats),

                      video panel with HUD overlays, keypoint legend,
                      ↺ RESTART button in header (works mid-run or post-done),
                      SEQ·1 / SEQ·2 tabs always usable.
"""

import argparse
import base64
import json
import math
import os
import queue
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from tqdm import tqdm


#  CONFIG  (overridable via CLI)
_VIDEO_WIDTH  = 620   # output frame width in pixels
_JPEG_QUALITY = 65    # JPEG encode quality (1-100)


#  EMBEDDED DASHBOARD HTML
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VO Mission Control — KITTI Live</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#080c10;--bg2:#0d1219;--bg3:#111820;
    --border:#1e2d3d;--border2:#243447;
    --cyan:#00d4ff;--cyan-dim:#0099bb;
    --amber:#ffb020;--green:#00ff88;--red:#ff4466;
    --white:#e8f0f8;--muted:#4a6070;
    --mono:'Share Tech Mono',monospace;--display:'Orbitron',monospace;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--white);font-family:var(--mono);
       height:100vh;overflow:hidden;display:flex;flex-direction:column}
  body::after{content:'';position:fixed;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,
      rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);
    pointer-events:none;z-index:9999}

  /* HEADER */
  header{display:flex;align-items:center;gap:12px;padding:8px 16px;
         background:var(--bg2);border-bottom:1px solid var(--border2);flex-shrink:0}
  .logo{font-family:var(--display);font-size:11px;font-weight:900;
        letter-spacing:4px;color:var(--cyan);text-shadow:0 0 20px rgba(0,212,255,.5)}
  .logo span{color:var(--muted)}
  .hdiv{width:1px;height:20px;background:var(--border2)}
  .status-bar{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted)}
  .sdot{width:6px;height:6px;border-radius:50%;background:var(--muted)}
  .sdot.live{background:var(--green);box-shadow:0 0 6px var(--green);animation:blink 1.2s infinite}
  .sdot.done{background:var(--cyan);box-shadow:0 0 6px var(--cyan)}
  .sdot.warn{background:var(--amber);animation:blink 2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
  .seq-val{font-size:10px;color:var(--amber)}
  .tabs{display:flex;gap:4px}
  .tab{padding:5px 12px;font-family:var(--display);font-size:8px;letter-spacing:2px;
       border:1px solid var(--border2);background:transparent;color:var(--muted);
       cursor:pointer;transition:all .2s}
  .tab.active{background:rgba(0,212,255,.1);border-color:var(--cyan);color:var(--cyan)}
  .tab:hover:not(.active){color:var(--white)}
  .done-badge{display:none;padding:3px 8px;background:rgba(0,255,136,.1);
              border:1px solid var(--green);color:var(--green);font-size:8px;
              letter-spacing:2px;font-family:var(--display)}
  .done-badge.show{display:inline-block;animation:fadein .5s ease}
  @keyframes fadein{from{opacity:0;transform:translateY(-4px)}to{opacity:1}}

  /* RESTART BUTTON */
  .restart-btn{padding:5px 14px;font-family:var(--display);font-size:8px;
               letter-spacing:2px;border:1px solid var(--border2);
               background:transparent;color:var(--muted);cursor:pointer;
               transition:all .2s;margin-left:auto}
  .restart-btn:hover{border-color:var(--amber);color:var(--amber);
                     box-shadow:0 0 12px rgba(255,176,32,.2)}
  .restart-btn.spinning{border-color:var(--green);color:var(--green);
                         animation:rp .8s infinite}
  @keyframes rp{0%,100%{box-shadow:0 0 0 rgba(0,255,136,0)}
                50%{box-shadow:0 0 14px rgba(0,255,136,.4)}}

  /* MAIN GRID
     Columns : [map 1fr] [video 310px] [stats 268px]
     Rows    : [top 1fr] [error 180px]
 */
  .main{display:grid;
        grid-template-columns:1fr 310px 268px;
        grid-template-rows:1fr 180px;
        gap:1px;flex:1;background:var(--border);overflow:hidden}
  .panel{background:var(--bg2);display:flex;flex-direction:column;overflow:hidden}
  .ph{display:flex;align-items:center;gap:8px;padding:7px 12px;
      background:var(--bg3);border-bottom:1px solid var(--border);flex-shrink:0}
  .pt{font-family:var(--display);font-size:7px;letter-spacing:3px;color:var(--cyan-dim)}

  /* MAP PANEL */
  #map-panel{grid-row:1;grid-column:1;position:relative}
  #path-canvas{flex:1;display:block;cursor:crosshair}

  /* VIDEO PANEL - Spans both rows in column 2 */
  #video-panel{grid-row:1/3;grid-column:2;border-left:1px solid var(--border)}
  .video-viewport{flex:1;background:#000;position:relative;overflow:hidden;
                  display:flex;align-items:center;justify-content:center;min-height:0}
  /* scanline effect on video */
  .video-viewport::after{content:'';position:absolute;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 3px,
      rgba(0,255,100,.03) 3px,rgba(0,255,100,.03) 4px);
    pointer-events:none;z-index:2}
  #videoImg{max-width:100%;max-height:100%;object-fit:contain;
            display:block;image-rendering:pixelated}
  /* "no signal" placeholder */
  .no-sig{position:absolute;inset:0;display:flex;flex-direction:column;
          align-items:center;justify-content:center;gap:10px;
          color:var(--muted);font-size:10px}
  .no-sig .icon{font-size:28px;opacity:.3}
  /* flash on new frame */
  .fflash{position:absolute;inset:0;background:rgba(0,212,255,.06);
          opacity:0;pointer-events:none;z-index:3;transition:opacity .05s}
  .fflash.lit{opacity:1}
  /* HUD overlays */
  .hud-tl{position:absolute;top:6px;left:6px;z-index:4;
           display:flex;flex-direction:column;gap:3px}
  .hud-br{position:absolute;bottom:6px;right:6px;z-index:4;text-align:right}
  .htag{font-size:8px;padding:1px 5px;background:rgba(8,12,16,.8);
        border:1px solid rgba(0,212,255,.3);color:var(--cyan)}
  .htag-a{font-size:8px;padding:1px 5px;background:rgba(8,12,16,.8);
           border:1px solid rgba(255,176,32,.3);color:var(--amber)}
  .htag-r{font-size:8px;padding:1px 5px;background:rgba(8,12,16,.8);
           border:1px solid rgba(255,68,102,.3);color:var(--red)}
  /* stat strip under video */
  .vstats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;
          background:var(--border);flex-shrink:0}
  .vs{background:var(--bg3);padding:5px 8px;text-align:center}
  .vsl{font-size:7px;color:var(--muted);letter-spacing:1px}
  .vsv{font-size:12px;font-weight:bold;color:var(--white);margin-top:2px}
  .vsv.c{color:var(--cyan)}.vsv.a{color:var(--amber)}.vsv.g{color:var(--green)}
  /* keypoint legend */
  .kp-legend{display:flex;align-items:center;gap:12px;padding:5px 10px;
             border-top:1px solid var(--border);flex-shrink:0}
  .kp-dot{width:6px;height:6px;border-radius:50%}
  .kp-item{display:flex;align-items:center;gap:4px;font-size:8px;color:var(--muted)}

  /* STATS PANEL - Spans both rows in column 3            */
  #stats-panel{grid-row:1/3;grid-column:3;overflow-y:auto;
               scrollbar-width:thin;scrollbar-color:var(--border2) transparent}
  #stats-panel::-webkit-scrollbar{width:4px}
  #stats-panel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
  .sg{padding:10px 12px;border-bottom:1px solid var(--border)}
  .sgt{font-family:var(--display);font-size:7px;letter-spacing:2px;
       color:var(--muted);margin-bottom:8px}
  .sr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}
  .sl{font-size:9px;color:var(--muted)}
  .sv{font-size:12px;font-weight:bold;color:var(--white);
      text-align:right;transition:color .3s}
  .sv.c{color:var(--cyan)}.sv.a{color:var(--amber)}
  .sv.g{color:var(--green)}.sv.r{color:var(--red)}
  .su{font-size:8px;color:var(--muted);margin-left:2px}
  .ebar{height:4px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:3px}
  .efill{height:100%;border-radius:2px;
         background:linear-gradient(90deg,var(--green),var(--amber),var(--red));
         transition:width .4s;width:0%}
  .mbar{height:3px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:5px}
  .mfill{height:100%;background:linear-gradient(90deg,var(--amber),var(--green));
         border-radius:2px;transition:width .4s}
  .prw{display:flex;justify-content:center;align-items:center;padding:10px 0;position:relative}
  .pr{transform:rotate(-90deg)}
  .rb{fill:none;stroke:var(--border2);stroke-width:4}
  .rf{fill:none;stroke:var(--cyan);stroke-width:4;
      stroke-linecap:round;transition:stroke-dashoffset .4s}
  .rc{position:absolute;text-align:center;display:flex;flex-direction:column;align-items:center}
  .rp{font-family:var(--display);font-size:18px;color:var(--cyan)}
  .rs{font-size:8px;color:var(--muted)}
  .fbars{display:flex;gap:2px;margin-top:5px}
  .fbar{height:3px;flex:1;border-radius:1px;background:var(--border2);transition:background .3s}
  .tr{display:flex;justify-content:space-between;align-items:center;padding:4px 0}
  .tl{font-size:9px;color:var(--muted);display:flex;align-items:center;gap:5px}
  .td{width:7px;height:7px;border-radius:50%}
  .td-c{background:var(--cyan);box-shadow:0 0 4px var(--cyan)}
  .td-a{background:var(--amber);box-shadow:0 0 4px var(--amber)}
  .td-r{background:var(--red)}.td-g{background:var(--green)}
  .tg{width:30px;height:15px;border-radius:8px;background:var(--border2);
      border:none;cursor:pointer;position:relative;transition:background .2s;flex-shrink:0}
  .tg::after{content:'';position:absolute;top:2px;left:2px;width:11px;height:11px;
             border-radius:50%;background:var(--muted);
             transition:transform .2s,background .2s}
  .tg.on{background:rgba(0,212,255,.2)}
  .tg.on::after{transform:translateX(15px);background:var(--cyan);box-shadow:0 0 4px var(--cyan)}
  .pm{font-size:8px;line-height:1.7;color:var(--muted);font-family:var(--mono);
      background:var(--bg);padding:7px 9px;border:1px solid var(--border);
      border-radius:2px;white-space:pre}
  .pm .hi{color:var(--cyan)}.pm .po{color:var(--amber)}

  /* ERROR CHART
    Row 2, column 1 only                   */
  #error-panel{grid-row:2;grid-column:1}
  #error-canvas{flex:1;display:block}

  /* ── MAP CONTROLS ───────────────────────── */
  .ctrls{display:flex;align-items:center;gap:7px;padding:6px 12px;
         background:var(--bg);border-top:1px solid var(--border);flex-shrink:0}
  .cb{width:28px;height:28px;border:1px solid var(--border2);background:transparent;
      color:var(--muted);cursor:pointer;font-size:11px;display:flex;
      align-items:center;justify-content:center;transition:all .15s;font-family:var(--mono)}
  .cb:hover{border-color:var(--cyan);color:var(--cyan)}
  .cb.active{background:rgba(0,212,255,.15);border-color:var(--cyan);color:var(--cyan)}
  .spd{-webkit-appearance:none;width:66px;height:2px;background:var(--border2);
       border:none;outline:none;cursor:pointer}
  .spd::-webkit-slider-thumb{-webkit-appearance:none;width:9px;height:9px;
    border-radius:50%;background:var(--cyan);box-shadow:0 0 5px var(--cyan)}
  .fc{font-family:var(--display);font-size:10px;color:var(--amber);margin-left:auto}
  .rb2{padding:4px 8px;border:1px solid var(--border2);background:transparent;
       color:var(--muted);font-family:var(--mono);font-size:8px;
       cursor:pointer;transition:all .15s}
  .rb2:hover{border-color:var(--amber);color:var(--amber)}

  /* WAIT OVERLAY */
  #wov{position:absolute;inset:0;background:rgba(8,12,16,.93);
       display:flex;flex-direction:column;align-items:center;
       justify-content:center;gap:14px;z-index:10;transition:opacity .4s}
  #wov.hidden{opacity:0;pointer-events:none}
  .wspin{width:44px;height:44px;border:2px solid var(--border2);
         border-top-color:var(--cyan);border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .wt{font-family:var(--display);font-size:8px;letter-spacing:3px;color:var(--cyan)}
  .ws{font-size:9px;color:var(--muted)}

  /* TOOLTIP */
  #tooltip{position:fixed;background:rgba(13,18,25,.96);border:1px solid var(--cyan);
           padding:5px 9px;font-size:9px;color:var(--white);pointer-events:none;
           opacity:0;transition:opacity .15s;z-index:100;white-space:nowrap}
</style>
</head>
<body>

<header>
  <div class="logo">VO<span>//</span>MISSION<span>·</span>CTRL</div>
  <div class="hdiv"></div>
  <div class="status-bar">
    <div class="sdot warn" id="sDot"></div>
    <span id="sTxt">CONNECTING…</span>
  </div>
  <div class="hdiv"></div>
  <span style="font-size:9px;color:var(--muted)">SEQ &nbsp;</span>
  <span class="seq-val" id="hSeq">—</span>
  <div class="hdiv"></div>
  <div class="tabs">
    <button class="tab" id="tab1" onclick="switchSeq('KITTI_sequence_1')">SEQ · 1</button>
    <button class="tab" id="tab2" onclick="switchSeq('KITTI_sequence_2')">SEQ · 2</button>
  </div>
  <div class="done-badge" id="doneBadge">■ COMPLETE</div>
  <!-- ↺ RESTART — works at any time, mid-run or post-completion -->
  <button class="restart-btn" id="restartBtn" onclick="restartRun()">↺ &nbsp;RESTART</button>
</header>

<div class="main">

  <!-- MAP -->
  <div class="panel" id="map-panel" style="position:relative">
    <div class="ph">
      <span class="pt">trajectory map</span>
      <div style="display:flex;gap:10px;align-items:center;margin-left:auto">
        <span style="font-size:8px;color:var(--muted)">ERR VEC</span>
        <button class="tg on" id="tEV"  onclick="toggleLayer('errVec')"></button>
        <span style="font-size:8px;color:var(--muted)">GHOST</span>
        <button class="tg on" id="tGH"  onclick="toggleLayer('ghost')"></button>
      </div>
    </div>
    <canvas id="path-canvas"></canvas>
    <div class="ctrls">
      <button class="cb" id="playBtn" onclick="toggleReplay()">▶</button>
      <button class="cb" onclick="replayStep(-1)">◀</button>
      <button class="cb" onclick="replayStep(1)">▶▶</button>
      <span style="font-size:8px;color:var(--muted);margin-left:3px">SPD</span>
      <input class="spd" type="range" min="1" max="20" value="10"
             id="spdSlider" oninput="updateSpeed()">
      <span id="spdVal" style="font-size:8px;color:var(--cyan);min-width:18px">10x</span>
      <div class="fc" id="fCtr">000 / 000</div>
      <button class="rb2" onclick="replayReset()">↺ RESET</button>
    </div>
    <div id="wov">
      <div class="wspin"></div>
      <div class="wt">AWAITING DATA</div>
      <div class="ws" id="wSub">Connecting to VO pipeline…</div>
    </div>
  </div>

  <!-- LIVE CAMERA FEED -->
  <div class="panel" id="video-panel">
    <div class="ph">
      <span class="pt">camera feed · keypoints</span>
      <span id="vFTag" style="font-size:8px;color:var(--muted);margin-left:auto">FRAME —</span>
    </div>
    <div class="video-viewport" id="vp">
      <!-- placeholder before first image arrives -->
      <div class="no-sig" id="noSig">
        <div class="icon">⊡</div>
        <div>NO SIGNAL</div>
      </div>
      <img id="videoImg" style="display:none" alt="camera feed">
      <div class="fflash" id="fflash"></div>
      <!-- top-left HUD -->
      <div class="hud-tl">
        <div class="htag"   id="hudSeq">—</div>
        <div class="htag"   id="hudFps">— fps</div>
      </div>
      <!-- bottom-right HUD -->
      <div class="hud-br">
        <div class="htag-a" id="hudKp">— kp</div>
        <div class="htag-r" id="hudErr" style="margin-top:3px">err — m</div>
      </div>
    </div>
    <!-- 3-metric strip -->
    <div class="vstats">
      <div class="vs"><div class="vsl">FRAME</div>  <div class="vsv c" id="vsFrm">—</div></div>
      <div class="vs"><div class="vsl">MATCHES</div><div class="vsv a" id="vsMatch">—</div></div>
      <div class="vs"><div class="vsl">PROC ms</div><div class="vsv"   id="vsMs">—</div></div>
    </div>
    <!-- keypoint colour legend -->
    <div class="kp-legend">
      <div class="kp-item"><div class="kp-dot" style="background:#00ff64"></div>Matched · low err</div>
      <div class="kp-item"><div class="kp-dot" style="background:#ffb020"></div>Mid err</div>
      <div class="kp-item"><div class="kp-dot" style="background:#ff4466"></div>High err</div>
    </div>
  </div>

  <!-- STATS -->
  <div class="panel" id="stats-panel">
    <!-- pipeline progress -->
    <div class="sg">
      <div class="sgt">pipeline progress</div>
      <div class="prw">
        <svg class="pr" width="76" height="76" viewBox="0 0 76 76">
          <circle class="rb" cx="38" cy="38" r="32"/>
          <circle class="rf" cx="38" cy="38" r="32" id="ring"
            stroke-dasharray="201.1" stroke-dashoffset="201.1"/>
        </svg>
        <div class="rc"><div class="rp" id="rPct">0%</div><div class="rs">FRAMES</div></div>
      </div>
      <div class="fbars" id="fBars"></div>
    </div>
    <!-- live metrics -->
    <div class="sg">
      <div class="sgt">live metrics</div>
      <div class="sr"><span class="sl">Frame</span>      <span class="sv c" id="mFrm">—</span></div>
      <div class="sr"><span class="sl">Camera FPS</span> <span class="sv"   id="mFps">—</span></div>
      <div class="sr"><span class="sl">Current Error</span><span class="sv"  id="mErr">—</span></div>
      <div class="sr"><span class="sl">Running RMSE</span><span class="sv a" id="mRmse">—</span></div>
      <div class="sr"><span class="sl">Peak Error</span> <span class="sv r" id="mPeak">—</span></div>
      <div class="sr"><span class="sl">Frame Time</span> <span class="sv"   id="mMs" style="font-size:10px;color:var(--muted)">—</span></div>
      <div style="font-size:8px;color:var(--muted);margin-top:6px;margin-bottom:2px">ERROR MAGNITUDE</div>
      <div class="ebar"><div class="efill" id="eFill"></div></div>
      <div style="font-size:8px;color:var(--muted);margin-top:7px;margin-bottom:2px">MATCH QUALITY</div>
      <div class="mbar"><div class="mfill" id="mFill"></div></div>
    </div>
    <!-- final results (unlocks when done) -->
    <div class="sg" id="finalSg" style="opacity:.3">
      <div class="sgt">final results</div>
      <div class="sr"><span class="sl">Total RMSE</span>  <span class="sv g" id="fRmse">—</span></div>
      <div class="sr"><span class="sl">Max Error</span>   <span class="sv"   id="fMax">—</span></div>
      <div class="sr"><span class="sl">Mean Error</span>  <span class="sv"   id="fMean">—</span></div>
      <div class="sr"><span class="sl">Path Length</span> <span class="sv"   id="fPath">—</span></div>
      <div class="sr"><span class="sl">Total Time</span>  <span class="sv"   id="fTime">—</span></div>
    </div>
    <!-- display layer toggles -->
    <div class="sg">
      <div class="sgt">display layers</div>
      <div class="tr"><span class="tl"><span class="td td-c"></span>Ground Truth</span>   <button class="tg on" id="tGT"  onclick="toggleLayer('gt')"></button></div>
      <div class="tr"><span class="tl"><span class="td td-a"></span>Estimated Path</span> <button class="tg on" id="tEST" onclick="toggleLayer('est')"></button></div>
      <div class="tr"><span class="tl"><span class="td td-r"></span>Error Vectors</span>  <button class="tg on" id="tEV2" onclick="toggleLayer('errVec')"></button></div>
      <div class="tr"><span class="tl">Waypoints</span>                                   <button class="tg on" id="tWP"  onclick="toggleLayer('wp')"></button></div>
    </div>
    <!-- live pose matrix -->
    <div class="sg">
      <div class="sgt">estimated pose (T_cur)</div>
      <div class="pm" id="pMat">[ <span class="hi">—</span>   —    — | <span class="po">—</span> ]
[  —   <span class="hi">—</span>   — | <span class="po">—</span> ]
[  —    —  <span class="hi">—</span> | <span class="po">—</span> ]</div>
    </div>
  </div>

  <!-- ERROR CHART -->
  <div class="panel" id="error-panel">
    <div class="ph">
      <span class="pt">error propagation — live</span>
      <span style="font-size:8px;color:var(--muted);margin-left:auto">Δ translational error vs frame</span>
    </div>
    <canvas id="error-canvas"></canvas>
  </div>

</div><!-- .main -->

<div id="tooltip"></div>

<script>

//  CLIENT STATE
const frames = [];
let totalFrames = 0, peakErr = 0, maxMatch = 500;
let seqName = '', isDone = false;

// replay controls
let replayFrame = 0, replayMode = false, replayPlaying = false;
let replayTimer = null, replaySpeed = 10;

// layer visibility flags
const L = {gt:true, est:true, errVec:true, ghost:true, wp:true};

// canvas refs
const mapCv  = document.getElementById('path-canvas');
const mapCtx = mapCv.getContext('2d');
const errCv  = document.getElementById('error-canvas');
const errCtx = errCv.getContext('2d');


//  SSE — LIVE DATA
let sse = null;
function connectSSE() {
  if (sse) sse.close();
  sse = new EventSource('/stream');
  sse.onopen = () => setStatus('CONNECTED','live');

  // Pipeline state change (loading / processing / seq switch)
  sse.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    seqName     = d.seq;
    totalFrames = d.total;
    isDone      = false;
    document.getElementById('hSeq').textContent = d.seq || '—';
    document.getElementById('doneBadge').className = 'done-badge';
    document.getElementById('finalSg').style.opacity = '0.3';
    document.getElementById('tab1').classList.toggle('active', d.seq==='KITTI_sequence_1');
    document.getElementById('tab2').classList.toggle('active', d.seq==='KITTI_sequence_2');
    if (d.state==='loading')     { setStatus('LOADING DATA…','warn'); document.getElementById('wSub').textContent='Loading '+d.seq+'…'; }
    else if (d.state==='processing') { setStatus('PROCESSING','live'); hideWait(); }
    initFBars(); resize();
  });

  // Navigation frame event — pose + error data (no image)
  sse.addEventListener('frame', e => {
    const d = JSON.parse(e.data);
    frames[d.i] = d;
    if (d.error  > peakErr)  peakErr  = d.error;
    if (d.matches > maxMatch) maxMatch = d.matches;
    if (d.i === 0) { frames.length = 1; peakErr = d.error; }
    if (!replayMode) replayFrame = d.i;
    hideWait();
    updateUI();
  });

  // Camera image event — base64 JPEG with keypoint overlay
  sse.addEventListener('image', e => {
    const d = JSON.parse(e.data);
    showVideoFrame(d.b64, d.i, d.fps);
  });

  // Run complete
  sse.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    isDone = true;
    setStatus('COMPLETE','done');
    document.getElementById('doneBadge').className  = 'done-badge show';
    document.getElementById('finalSg').style.opacity = '1';
    document.getElementById('fRmse').innerHTML  = d.rmse.toFixed(4)        + '<span class="su">m</span>';
    document.getElementById('fMax').innerHTML   = d.max_error.toFixed(4)   + '<span class="su">m</span>';
    document.getElementById('fMean').innerHTML  = d.mean_error.toFixed(4)  + '<span class="su">m</span>';
    document.getElementById('fPath').innerHTML  = d.path_length.toFixed(2) + '<span class="su">m</span>';
    document.getElementById('fTime').innerHTML  = d.elapsed_s.toFixed(1)   + '<span class="su">s</span>';
    stopRestartSpin();
    updateUI();
  });

  sse.onerror = () => setStatus('RECONNECTING…','warn');
}


//  VIDEO FRAME DISPLAY
function showVideoFrame(b64, frameIdx, fps) {
  const img   = document.getElementById('videoImg');
  const noSig = document.getElementById('noSig');
  const flash = document.getElementById('fflash');

  // Swap in new frame
  img.src           = 'data:image/jpeg;base64,' + b64;
  img.style.display = 'block';
  noSig.style.display = 'none';

  // Brief cyan flash to show a new frame arrived
  flash.classList.add('lit');
  setTimeout(() => flash.classList.remove('lit'), 60);

  // HUD overlays
  const f = frames[frameIdx];
  document.getElementById('hudSeq').textContent  = seqName || '—';
  document.getElementById('hudFps').textContent  = fps > 0 ? fps.toFixed(1)+' fps' : '— fps';
  document.getElementById('hudKp').textContent   = (f ? f.matches : '—') + ' kp';
  document.getElementById('hudErr').textContent  = 'err ' + (f ? f.error.toFixed(3) : '—') + ' m';
  document.getElementById('vFTag').textContent   = 'FRAME ' + frameIdx;

  // Metric strip
  document.getElementById('vsFrm').textContent   = frameIdx;
  document.getElementById('vsMatch').textContent  = f ? f.matches   : '—';
  document.getElementById('vsMs').textContent     = f ? f.processing_ms : '—';
  document.getElementById('mFps').textContent     = fps > 0 ? fps.toFixed(1) : '—';
}


//  RESIZE
function resize() {
  const r1 = document.getElementById('map-panel').getBoundingClientRect();
  const r2 = document.getElementById('error-panel').getBoundingClientRect();
  mapCv.width  = r1.width;
  mapCv.height = r1.height - 42;   // subtract controls bar
  errCv.width  = r2.width;
  errCv.height = r2.height - 34;   // subtract panel header
  drawAll();
}
window.addEventListener('resize', resize);


//  COORDINATE TRANSFORM  (world → canvas pixels)
function getT() {
  const pad=40, W=mapCv.width-pad*2, H=mapCv.height-pad*2;
  if (!frames.length) return null;
  const allX=[], allZ=[];
  frames.forEach(f=>{ if(f){ allX.push(f.gt_x,f.est_x); allZ.push(f.gt_z,f.est_z); } });
  if (!allX.length) return null;
  const minX=Math.min(...allX), maxX=Math.max(...allX);
  const minZ=Math.min(...allZ), maxZ=Math.max(...allZ);
  const rX=maxX-minX||1, rZ=maxZ-minZ||1;
  const scale=Math.min(W/rX, H/rZ);
  return { scale, offX:pad+(W-rX*scale)/2-minX*scale, offZ:pad+(H-rZ*scale)/2, maxZ };
}
function toCv(x,z,t){ return { cx:x*t.scale+t.offX, cy:(t.maxZ-z)*t.scale+t.offZ }; }


//  DRAW MAP
function drawMap() {
  const ctx=mapCtx, W=mapCv.width, H=mapCv.height;
  ctx.clearRect(0,0,W,H);

  // background grid
  ctx.strokeStyle='rgba(30,45,60,.8)'; ctx.lineWidth=1;
  for(let x=0;x<W;x+=50){ ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
  for(let y=0;y<H;y+=50){ ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }

  // corner brackets
  [[10,10],[W-10,10],[10,H-10],[W-10,H-10]].forEach(([cx,cy])=>{
    const s=10, d=cy<H/2?s:-s;
    ctx.strokeStyle='rgba(0,212,255,.25)'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(cx-s,cy);ctx.lineTo(cx,cy);ctx.lineTo(cx,cy+d);ctx.stroke();
  });

  if (!frames.length) return;
  const t=getT(); if(!t) return;
  const n=Math.min(replayFrame+1, frames.length);

  // ghost — full future path, very faint
  if (L.ghost && frames.length>1) {
    ctx.strokeStyle='rgba(0,212,255,.07)';ctx.lineWidth=1.5;ctx.setLineDash([4,8]);
    ctx.beginPath();
    frames.forEach((f,i)=>{ if(!f)return; const p=toCv(f.gt_x,f.gt_z,t); i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy); });
    ctx.stroke(); ctx.setLineDash([]);
  }

  // error vectors
  if (L.errVec) {
    for(let i=0;i<n;i++){
      const f=frames[i]; if(!f)continue;
      const gp=toCv(f.gt_x,f.gt_z,t), ep=toCv(f.est_x,f.est_z,t);
      const ratio=peakErr>0?f.error/peakErr:0;
      ctx.strokeStyle=`rgba(255,68,102,${.18+ratio*.6})`;
      ctx.lineWidth=1; ctx.setLineDash([2,3]);
      ctx.beginPath();ctx.moveTo(gp.cx,gp.cy);ctx.lineTo(ep.cx,ep.cy);ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // ground-truth path
  if (L.gt && n>0) {
    const g=ctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0,'rgba(0,212,255,.3)'); g.addColorStop(1,'rgba(0,212,255,1)');
    ctx.strokeStyle=g; ctx.lineWidth=2; ctx.beginPath();
    for(let i=0;i<n;i++){ const f=frames[i];if(!f)continue; const p=toCv(f.gt_x,f.gt_z,t); i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy); }
    ctx.stroke();
  }

  // estimated path
  if (L.est && n>0) {
    ctx.strokeStyle='#ffb020'; ctx.lineWidth=2;
    ctx.shadowColor='rgba(255,176,32,.5)'; ctx.shadowBlur=6;
    ctx.beginPath();
    for(let i=0;i<n;i++){ const f=frames[i];if(!f)continue; const p=toCv(f.est_x,f.est_z,t); i===0?ctx.moveTo(p.cx,p.cy):ctx.lineTo(p.cx,p.cy); }
    ctx.stroke(); ctx.shadowBlur=0;
  }

  // waypoints
  if (L.wp) {
    for(let i=0;i<n;i++){
      const f=frames[i]; if(!f)continue; const last=i===n-1;
      const gp=toCv(f.gt_x,f.gt_z,t);
      ctx.fillStyle=last?'#00d4ff':'rgba(0,180,220,.5)';
      ctx.beginPath();ctx.arc(gp.cx,gp.cy,last?4:2,0,Math.PI*2);ctx.fill();
      const ep=toCv(f.est_x,f.est_z,t);
      ctx.fillStyle=last?'#ffb020':'rgba(200,140,20,.5)';
      ctx.beginPath();ctx.arc(ep.cx,ep.cy,last?4:2,0,Math.PI*2);ctx.fill();
    }
  }

  // current-frame rings
  const cf=frames[replayFrame];
  if(cf){
    [toCv(cf.gt_x,cf.gt_z,t),toCv(cf.est_x,cf.est_z,t)].forEach((p,idx)=>{
      ctx.strokeStyle=idx===0?'#00d4ff':'#ffb020'; ctx.lineWidth=1.5;
      ctx.beginPath();ctx.arc(p.cx,p.cy,9,0,Math.PI*2);ctx.stroke();
    });
  }

  // labels
  if(frames[0]){
    const sp=toCv(frames[0].gt_x,frames[0].gt_z,t);
    ctx.fillStyle='rgba(0,212,255,.8)';ctx.font='9px Share Tech Mono';
    ctx.fillText('START',sp.cx+8,sp.cy+3);
  }
  ctx.fillStyle='rgba(74,96,112,.6)';ctx.font='8px Share Tech Mono';
  ctx.fillText('X →',W-26,H-4); ctx.fillText('Z ↑',5,16);
}


//  DRAW ERROR CHART
function drawError() {
  const ctx=errCtx, W=errCv.width, H=errCv.height;
  const P={l:44,r:16,t:12,b:26};
  ctx.clearRect(0,0,W,H);
  const n=Math.min(replayFrame+1,frames.length);
  const tot=Math.max(totalFrames,frames.length,1);
  const maxE=Math.max(peakErr*1.1,.1);

  // grid
  for(let i=0;i<=4;i++){
    const y=P.t+(H-P.t-P.b)*i/4;
    ctx.strokeStyle='rgba(30,45,60,.8)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(W-P.r,y);ctx.stroke();
    ctx.fillStyle='rgba(74,96,112,.8)';ctx.font='7px Share Tech Mono';
    ctx.fillText((maxE*(1-i/4)).toFixed(2),2,y+3);
  }
  for(let i=0;i<=5;i++){
    const x=P.l+(W-P.l-P.r)*i/5;
    ctx.strokeStyle='rgba(30,45,60,.5)';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(x,P.t);ctx.lineTo(x,H-P.b);ctx.stroke();
    ctx.fillStyle='rgba(74,96,112,.7)';ctx.font='7px Share Tech Mono';
    ctx.fillText(Math.round(tot*i/5),x-4,H-3);
  }

  // ghost (full future)
  if(frames.length>1){
    ctx.strokeStyle='rgba(255,68,102,.1)';ctx.lineWidth=1;ctx.setLineDash([3,5]);
    ctx.beginPath();
    frames.forEach((f,i)=>{ if(!f)return;
      const x=P.l+(W-P.l-P.r)*i/(tot-1), y=P.t+(H-P.t-P.b)*(1-f.error/maxE);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();ctx.setLineDash([]);
  }

  // fill under curve
  if(n>1){
    const g=ctx.createLinearGradient(0,P.t,0,H-P.b);
    g.addColorStop(0,'rgba(255,68,102,.3)');g.addColorStop(1,'rgba(255,68,102,.02)');
    ctx.fillStyle=g;ctx.beginPath();
    for(let i=0;i<n;i++){ const f=frames[i];if(!f)continue;
      const x=P.l+(W-P.l-P.r)*i/(tot-1), y=P.t+(H-P.t-P.b)*(1-f.error/maxE);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    const lx=P.l+(W-P.l-P.r)*(n-1)/(tot-1);
    ctx.lineTo(lx,H-P.b);ctx.lineTo(P.l,H-P.b);ctx.closePath();ctx.fill();
  }

  // live error line
  if(n>0){
    ctx.strokeStyle='#ff4466';ctx.lineWidth=2;
    ctx.shadowColor='rgba(255,68,102,.5)';ctx.shadowBlur=8;
    ctx.beginPath();
    for(let i=0;i<n;i++){ const f=frames[i];if(!f)continue;
      const x=P.l+(W-P.l-P.r)*i/(tot-1), y=P.t+(H-P.t-P.b)*(1-f.error/maxE);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();ctx.shadowBlur=0;
  }

  // current-frame dot
  const cf=frames[replayFrame];
  if(cf){
    const x=P.l+(W-P.l-P.r)*replayFrame/(tot-1);
    const y=P.t+(H-P.t-P.b)*(1-cf.error/maxE);
    ctx.fillStyle='#ff4466';ctx.shadowColor='#ff4466';ctx.shadowBlur=10;
    ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
    ctx.strokeStyle='rgba(255,68,102,.3)';ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x,H-P.b);ctx.stroke();ctx.setLineDash([]);
  }

  // axes
  ctx.strokeStyle='rgba(30,45,60,1)';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(P.l,P.t);ctx.lineTo(P.l,H-P.b);ctx.lineTo(W-P.r,H-P.b);ctx.stroke();
  ctx.fillStyle='rgba(74,96,112,.6)';ctx.font='7px Share Tech Mono';
  ctx.save();ctx.translate(9,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('ERROR (m)',-20,0);ctx.restore();
  ctx.fillText('FRAME',W/2-14,H-1);
}


//  STATS UPDATE
function updateStats() {
  const n=Math.min(replayFrame+1,frames.length);
  const cf=frames[replayFrame];
  const tot=Math.max(totalFrames,frames.length,1);

  document.getElementById('fCtr').textContent =
    String(replayFrame).padStart(3,'0')+' / '+String(tot-1).padStart(3,'0');

  // progress ring
  const pct=n/tot;
  document.getElementById('ring').style.strokeDashoffset=201.1*(1-pct);
  document.getElementById('rPct').textContent=Math.round(pct*100)+'%';

  // frame colour bar strip
  const bars=document.getElementById('fBars').children;
  for(let i=0;i<bars.length;i++){
    const fi=Math.floor(i*frames.length/bars.length);
    const f=frames[fi];
    bars[i].style.background=fi<n&&f
      ?`rgb(${Math.round(255*(f.error/Math.max(peakErr,.001)))},${Math.round(255*(1-f.error/Math.max(peakErr,.001)))},68)`
      :'var(--border2)';
  }

  if(!cf) return;

  document.getElementById('mFrm').textContent=replayFrame;
  const ec=cf.error<.5?'c':cf.error<1.5?'a':'r';
  document.getElementById('mErr').className='sv '+ec;
  document.getElementById('mErr').innerHTML=cf.error.toFixed(3)+'<span class="su">m</span>';

  let ss=0,cnt=0;
  for(let i=0;i<n;i++){ if(frames[i]){ss+=frames[i].error**2;cnt++;} }
  document.getElementById('mRmse').innerHTML=(cnt>0?Math.sqrt(ss/cnt):0).toFixed(3)+'<span class="su">m</span>';
  document.getElementById('mPeak').innerHTML=peakErr.toFixed(3)+'<span class="su">m</span>';
  document.getElementById('mMs').innerHTML=cf.processing_ms+'<span class="su">ms</span>';
  document.getElementById('eFill').style.width=(cf.error/Math.max(peakErr,.001)*100)+'%';
  document.getElementById('mFill').style.width=Math.min(cf.matches/maxMatch*100,100)+'%';

  if(cf.pose){
    const p=cf.pose, f=v=>v.toFixed(3).padStart(7);
    document.getElementById('pMat').innerHTML=
      `[ <span class="hi">${f(p[0][0])}</span> ${f(p[0][1])} ${f(p[0][2])} | <span class="po">${f(p[0][3])}</span> ]\n`+
      `[ ${f(p[1][0])} <span class="hi">${f(p[1][1])}</span> ${f(p[1][2])} | <span class="po">${f(p[1][3])}</span> ]\n`+
      `[ ${f(p[2][0])} ${f(p[2][1])} <span class="hi">${f(p[2][2])}</span> | <span class="po">${f(p[2][3])}</span> ]`;
  }
}

function drawAll(){ drawMap(); drawError(); updateStats(); }
function updateUI(){ drawAll(); }


//  REPLAY CONTROLS
function toggleReplay(){
  if(replayPlaying){ pauseReplay(); }
  else {
    replayMode=true;
    if(replayFrame>=frames.length-1) replayFrame=0;
    replayPlaying=true;
    document.getElementById('playBtn').textContent='⏸';
    document.getElementById('playBtn').classList.add('active');
    schedReplay();
  }
}
function schedReplay(){
  if(!replayPlaying) return;
  replayTimer=setTimeout(()=>{
    replayFrame=Math.min(replayFrame+1,frames.length-1);
    drawAll();
    if(replayFrame<frames.length-1) schedReplay(); else pauseReplay();
  }, Math.round(1000/replaySpeed));
}
function pauseReplay(){
  replayPlaying=false; clearTimeout(replayTimer);
  document.getElementById('playBtn').textContent='▶';
  document.getElementById('playBtn').classList.remove('active');
}
function replayStep(d){
  pauseReplay(); replayMode=true;
  replayFrame=Math.max(0,Math.min(frames.length-1,replayFrame+d));
  drawAll();
}
function replayReset(){
  pauseReplay(); replayMode=false;
  replayFrame=Math.max(0,frames.length-1);
  drawAll();
}
function updateSpeed(){
  replaySpeed=parseInt(document.getElementById('spdSlider').value);
  document.getElementById('spdVal').textContent=replaySpeed+'x';
}


//  RESTART / SEQ SWITCH


// ↺ RESTART — kills the current VO thread server-side and starts fresh.
// Works at any point: during a run, when complete, or mid-error.
function restartRun(){
  const seq = seqName || 'KITTI_sequence_1';
  resetClientState();
  startRestartSpin();
  fetch('/run?seq='+seq).catch(()=>{});
}

function switchSeq(seq){
  if(seq===seqName && !isDone) return;  // already running this seq, ignore
  resetClientState();
  startRestartSpin();
  fetch('/run?seq='+seq).catch(()=>{});
}

function resetClientState(){
  pauseReplay();
  frames.length=0;
  replayFrame=0; replayMode=false; peakErr=0; maxMatch=500; isDone=false;
  document.getElementById('doneBadge').className='done-badge';
  document.getElementById('finalSg').style.opacity='0.3';
  document.getElementById('videoImg').style.display='none';
  document.getElementById('noSig').style.display='flex';
  document.getElementById('hSeq').textContent='…';
  document.getElementById('wSub').textContent='Restarting pipeline…';
  document.getElementById('wov').classList.remove('hidden');
  setStatus('RESTARTING…','warn');
  drawAll();
}

function startRestartSpin(){ document.getElementById('restartBtn').classList.add('spinning'); }
function stopRestartSpin() { document.getElementById('restartBtn').classList.remove('spinning'); }


//  LAYER TOGGLES
function toggleLayer(name){
  L[name]=!L[name];
  const on=L[name];
  const map={gt:'tGT',est:'tEST',errVec:['tEV','tEV2'],ghost:'tGH',wp:'tWP'};
  const ids=map[name];
  (Array.isArray(ids)?ids:[ids]).forEach(id=>{
    document.getElementById(id).className='tg'+(on?' on':'');
  });
  drawMap();
}


//  STATUS / OVERLAY HELPERS
function setStatus(t,s){
  document.getElementById('sTxt').textContent=t;
  document.getElementById('sDot').className='sdot '+s;
}
function hideWait(){ document.getElementById('wov').classList.add('hidden'); }
function initFBars(){
  const el=document.getElementById('fBars'); el.innerHTML='';
  for(let i=0;i<50;i++){ const d=document.createElement('div');d.className='fbar';el.appendChild(d); }
}


//  HOVER TOOLTIPS
mapCv.addEventListener('mousemove', e=>{
  const t=getT(); if(!t) return;
  const r=mapCv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const n=Math.min(replayFrame+1,frames.length);
  let best=-1, bestD=20;
  for(let i=0;i<n;i++){
    const f=frames[i]; if(!f) continue;
    const gp=toCv(f.gt_x,f.gt_z,t), ep=toCv(f.est_x,f.est_z,t);
    const d=Math.min(Math.hypot(mx-gp.cx,my-gp.cy),Math.hypot(mx-ep.cx,my-ep.cy));
    if(d<bestD){bestD=d;best=i;}
  }
  const tt=document.getElementById('tooltip');
  if(best>=0){
    const f=frames[best];
    tt.style.opacity='1';tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY-32)+'px';
    tt.innerHTML=`Frame ${best} | Err:<span style="color:#ff4466"> ${f.error.toFixed(4)}m</span> KP:${f.matches}<br>GT(${f.gt_x.toFixed(3)},${f.gt_z.toFixed(3)}) Est(${f.est_x.toFixed(3)},${f.est_z.toFixed(3)})`;
  } else { tt.style.opacity='0'; }
});
mapCv.addEventListener('mouseleave',()=>{ document.getElementById('tooltip').style.opacity='0'; });

errCv.addEventListener('mousemove', e=>{
  const P={l:44,r:16}; const tot=Math.max(totalFrames,frames.length,1);
  const r=errCv.getBoundingClientRect();
  const fi=Math.round((e.clientX-r.left-P.l)/(errCv.width-P.l-P.r)*(tot-1));
  if(fi<0||fi>=frames.length||!frames[fi]) return;
  const tt=document.getElementById('tooltip');
  tt.style.opacity='1';tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY-32)+'px';
  tt.innerHTML=`Frame ${fi} | Err:<span style="color:#ff4466"> ${frames[fi].error.toFixed(4)}m</span>`;
});
errCv.addEventListener('mouseleave',()=>{ document.getElementById('tooltip').style.opacity='0'; });


//  INIT
window.addEventListener('load',()=>{ initFBars(); resize(); connectSSE(); });
</script>
</body>
</html>"""


#  GLOBAL SERVER STATE
_lock          = threading.Lock()
_subscribers   = []      # list[queue.Queue]  — one per SSE client
_frame_history = []      # list[dict]          — lean nav data (no images)
_last_image    = None    # dict {b64,i,fps}    — latest camera frame only
_status_event  = None    # dict                — most recent 'status' event
_done_event    = None    # dict | None
_vo_thread     = None
_stop_flag     = threading.Event()


def _broadcast(event_type: str, payload: dict):
    """Put an event on every active subscriber queue, pruning dead ones."""
    msg = {"type": event_type, "data": payload}
    with _lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def push_status(state: str, seq: str, total: int):
    global _status_event, _frame_history, _done_event, _last_image
    evt = {"state": state, "seq": seq, "total": total}
    with _lock:
        _status_event  = evt
        _frame_history = []
        _done_event    = None
        _last_image    = None
    _broadcast("status", evt)


def push_frame(frame_dict: dict):
    """Push navigation data. No image — keeps frame history lean."""
    with _lock:
        _frame_history.append(frame_dict)
    _broadcast("frame", frame_dict)


# NEW: push_image
def push_image(b64: str, frame_idx: int, fps: float):
    """
    Push a single camera frame as a base64 JPEG.
    Only the *latest* image is held in memory for catch-up on reconnect —
    this keeps memory usage flat regardless of sequence length.
    """
    global _last_image
    payload = {"b64": b64, "i": frame_idx, "fps": round(fps, 2)}
    with _lock:
        _last_image = payload
    _broadcast("image", payload)


def push_done(summary: dict):
    global _done_event
    with _lock:
        _done_event = summary
    _broadcast("done", summary)



#  NEW: encode_frame — resize + keypoint overlay -> base64 JPEG
def encode_frame(img_gray: np.ndarray,
                 matched_kp,          # np.ndarray (N,2) or None
                 error: float,
                 peak_error: float) -> str:
    """
    1. Resize grayscale frame to _VIDEO_WIDTH (preserving aspect ratio).
    2. Convert to BGR for coloured overlay.
    3. Draw matched keypoints colour-coded by error/peak_error ratio:
         green  → low error
         amber  → mid error
         red    → high error
    4. Draw a subtle green border (telemetry aesthetic).
    5. JPEG-encode and return base64 string.
    """
    h, w = img_gray.shape
    out_w = _VIDEO_WIDTH
    out_h = int(h * out_w / w)
    small = cv2.resize(img_gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    color = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    if matched_kp is not None and len(matched_kp) > 0:
        ratio = min(error / max(peak_error, 0.001), 1.0)
        sx, sy = out_w / w, out_h / h
        # Interpolate green → amber → red
        if ratio < 0.5:
            r_c = int(ratio * 2 * 255)
            g_c = 200
            b_c = 60
        else:
            r_c = 255
            g_c = int((1.0 - (ratio - 0.5) * 2) * 200)
            b_c = 30
        kp_color = (b_c, g_c, r_c)   # OpenCV uses BGR

        for pt in matched_kp[:800]:   # cap to avoid overdraw on dense matches
            x = int(pt[0] * sx)
            y = int(pt[1] * sy)
            cv2.circle(color, (x, y), 2, kp_color, -1, cv2.LINE_AA)

    # Subtle green border
    cv2.rectangle(color, (0, 0), (out_w - 1, out_h - 1), (0, 160, 60), 2)

    ok, buf = cv2.imencode(
        ".jpg", color,
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    )
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""



#  HTTP REQUEST HANDLER
class VOHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress per-request noise

    def log_error(self, fmt, *args):
        print(f"[server] {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/stream":
            self._serve_sse()
        elif path == "/run":
            qs  = parse_qs(parsed.query)
            seq = qs.get("seq", ["KITTI_sequence_1"])[0]
            self._handle_run(seq)
        else:
            self.send_error(404)

    def _serve_html(self):
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type",  "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue(maxsize=256)

        # Catch-up: replay all history so reconnects / new tabs see full state.
        # Note: only _last_image (not all images) is replayed — by design.
        with _lock:
            _subscribers.append(q)
            catch_status = _status_event
            catch_frames = list(_frame_history)
            catch_image  = _last_image
            catch_done   = _done_event

        try:
            if catch_status:
                self._emit("status", catch_status)
            for f in catch_frames:
                self._emit("frame", f)
            if catch_image:
                self._emit("image", catch_image)
            if catch_done:
                self._emit("done", catch_done)

            while True:
                try:
                    msg = q.get(timeout=20)
                except queue.Empty:
                    self._raw(b": heartbeat\n\n")
                    continue
                self._emit(msg["type"], msg["data"])

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _lock:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass

    def _emit(self, event_type: str, payload: dict):
        data = json.dumps(payload, separators=(",", ":"))
        self._raw(f"event: {event_type}\ndata: {data}\n\n".encode("utf-8"))

    def _raw(self, b: bytes):
        self.wfile.write(b)
        self.wfile.flush()

    def _handle_run(self, seq: str):
        """Trigger a new VO run. Responds immediately, starts worker async."""
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
        _restart_vo(seq)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Each connection handled in its own thread — essential for SSE."""
    daemon_threads = True



#  VISUAL ODOMETRY CLASS  (core algorithm unchanged)
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
        good      = [m for m, n in matches if m.distance < 0.5 * n.distance]
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
        K_hom  = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        scores = []
        for T_c in candidates:
            P_c    = K_hom @ T_c
            hom_Q1 = cv2.triangulatePoints(self.P, P_c, q1.T, q2.T)
            hom_Q2 = T_c @ hom_Q1
            Q1     = hom_Q1[:3] / hom_Q1[3]
            Q2     = hom_Q2[:3] / hom_Q2[3]
            pos    = sum(Q2[2] > 0) + sum(Q1[2] > 0)
            scale  = np.mean(
                np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            )
            scores.append(pos + scale)
        best  = int(np.argmax(scores))
        R_out = R1 if best in (0, 2) else R2
        t_out = t_flat if best in (0, 1) else -t_flat
        return R_out, t_out



#  VO PIPELINE WORKER  ← modified to stream camera frames
def _run_vo(data_dir: str):
    global _stop_flag

    seq_name = os.path.basename(data_dir)
    print(f"\n[vo] Loading '{data_dir}' …")
    push_status("loading", seq_name, 0)

    try:
        vo = VisualOdometry(data_dir)
    except Exception as exc:
        print(f"[vo] Init error: {exc}")
        return

    n_frames = len(vo.gt_poses)
    push_status("processing", seq_name, n_frames)
    print(f"[vo] Pipeline started: {n_frames} frames")

    cur_pose   = vo.gt_poses[0]
    gt_path, est_path, errors = [], [], []
    peak_error = 0.0

    # Rolling FPS — sliding window of frame wall-clock timestamps
    ts_buf: deque = deque(maxlen=8)
    start_time = time.time()

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose", ncols=72)):
        if _stop_flag.is_set():
            print("[vo] Aborted.")
            return

        t0   = time.perf_counter()
        q1 = q2 = None

        if i == 0:
            cur_pose  = gt_pose
            n_matches = 0
        else:
            try:
                q1, q2    = vo.get_matches(i)
                n_matches = len(q1)
                transf    = vo.get_pose(q1, q2)
                cur_pose  = np.matmul(cur_pose, np.linalg.inv(transf))
            except Exception as exc:
                print(f"[vo] Frame {i}: {exc}")
                n_matches = 0

        proc_ms = round((time.perf_counter() - t0) * 1000)

        gt_x,  gt_z  = float(gt_pose[0, 3]),  float(gt_pose[2, 3])
        est_x, est_z = float(cur_pose[0, 3]), float(cur_pose[2, 3])
        error        = math.hypot(gt_x - est_x, gt_z - est_z)
        if error > peak_error:
            peak_error = error

        gt_path.append((gt_x, gt_z))
        est_path.append((est_x, est_z))
        errors.append(error)

        # push navigation data (pose, error, matches)
        push_frame({
            "i":             i,
            "gt_x":          gt_x,
            "gt_z":          gt_z,
            "est_x":         est_x,
            "est_z":         est_z,
            "error":         error,
            "matches":       n_matches,
            "processing_ms": proc_ms,
            "pose":          cur_pose[:3].tolist(),
        })

        # encode and push camera frame with keypoint overlay
        # q2 contains matched keypoint coords in the CURRENT image,
        # so the dots align precisely with what is displayed.
        ts_buf.append(time.perf_counter())
        fps = ((len(ts_buf) - 1) / (ts_buf[-1] - ts_buf[0])
               if len(ts_buf) > 1 else 0.0)

        b64 = encode_frame(
            vo.images[i],                          # current grayscale frame
            q2 if q2 is not None else None,        # matched keypoints (or None for frame 0)
            error,
            peak_error,
        )
        if b64:
            push_image(b64, i, fps)

    # final summary
    errors_arr  = np.array(errors)
    rmse        = float(np.sqrt(np.mean(errors_arr ** 2)))
    max_error   = float(np.max(errors_arr))
    mean_error  = float(np.mean(errors_arr))
    path_length = sum(
        math.hypot(gt_path[j][0] - gt_path[j-1][0],
                   gt_path[j][1] - gt_path[j-1][1])
        for j in range(1, len(gt_path))
    )
    elapsed_s = time.time() - start_time

    print(f"\n[vo] Done  RMSE={rmse:.4f}m  max={max_error:.4f}m  "
          f"path={path_length:.1f}m  time={elapsed_s:.1f}s")

    push_done({
        "rmse":         rmse,
        "max_error":    max_error,
        "mean_error":   mean_error,
        "path_length":  path_length,
        "elapsed_s":    elapsed_s,
    })


def _restart_vo(data_dir: str):
    """Signal the current VO thread to stop then launch a fresh one."""
    global _vo_thread, _stop_flag
    _stop_flag.set()
    if _vo_thread and _vo_thread.is_alive():
        _vo_thread.join(timeout=4)
    _stop_flag = threading.Event()
    _vo_thread = threading.Thread(target=_run_vo, args=(data_dir,), daemon=True)
    _vo_thread.start()



#  MAIN
def main():
    parser = argparse.ArgumentParser(description="VO Mission Control — live dashboard + camera feed")
    parser.add_argument("--seq",          default="KITTI_sequence_1",
                        help="KITTI sequence directory (default: KITTI_sequence_1)")
    parser.add_argument("--port",         type=int, default=8080,
                        help="HTTP port (default: 8080)")
    parser.add_argument("--no-browser",   action="store_true",
                        help="Do not auto-open browser")
    parser.add_argument("--video-width",  type=int, default=620,
                        help="Camera feed output width in pixels (default: 620)")
    parser.add_argument("--jpeg-quality", type=int, default=65,
                        help="JPEG quality for camera stream, 1-100 (default: 65)")
    args = parser.parse_args()

    global _VIDEO_WIDTH, _JPEG_QUALITY
    _VIDEO_WIDTH  = args.video_width
    _JPEG_QUALITY = args.jpeg_quality

    server = ThreadedHTTPServer(("0.0.0.0", args.port), VOHandler)
    _restart_vo(args.seq)

    url = f"http://localhost:{args.port}"
    print(f"\n{'─'*56}")
    print(f"  VO Mission Control  —  Live Camera + Map")
    print(f"  Dashboard    : {url}")
    print(f"  Sequence     : {args.seq}")
    print(f"  Camera feed  : {_VIDEO_WIDTH}px  JPEG q={_JPEG_QUALITY}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'─'*56}\n")

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
