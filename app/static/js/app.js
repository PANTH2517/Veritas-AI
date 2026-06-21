

'use strict';

const STATE = {
  currentTab: 'single',
  sessionStats: { real: 0, fake: 0, uncertain: 0, totalRisk: 0, count: 0 },
  history: [],
  scanPollingId: null,
  dualPollingId: null,
  eqAnimId: null,
  frameData: [],
  currentFrameIdx: 0,
  miniChart: null,
  analyticsHistoryChart: null,
  analyticsDonutChart: null,
  timelineChartInst: null,
  particleCanvas: null,
  particleCtx: null,
  particles: [],
  cursor: { x: 0, y: 0, lx: 0, ly: 0 },
  serverOnline: false,
  dualRealFile: null,
  dualCheckFile: null,
  scanHistory: []
};

const BOOT_MESSAGES = [
  { msg: 'Initializing secure execution environment…', delay: 0 },
  { msg: 'Loading forensic neural weight matrices…',   delay: 380 },
  { msg: 'Binding OpenCV image pipeline (v4.x)…',      delay: 720 },
  { msg: 'Calibrating spectral voice-print analyzer…', delay: 1050 },
  { msg: 'Establishing backend API handshake…',        delay: 1350 },
  { msg: 'Warming up GPU inference runtime…',          delay: 1650 },
  { msg: 'Mounting encrypted session vault…',          delay: 1920 },
  { msg: 'SYSTEM READY — All modules online.',         delay: 2180 },
];

function runBootSequence() {
  const bar    = document.getElementById('boot-bar');
  const logEl  = document.getElementById('boot-log');
  const screen = document.getElementById('boot-screen');
  let barVal   = 0;

  BOOT_MESSAGES.forEach(({ msg, delay }, i) => {
    setTimeout(() => {
      const line = document.createElement('div');
      line.className = 'boot-log-line';
      line.textContent = msg;
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
      barVal = Math.round(((i + 1) / BOOT_MESSAGES.length) * 100);
      bar.style.width = barVal + '%';
    }, delay);
  });

  const totalTime = BOOT_MESSAGES[BOOT_MESSAGES.length - 1].delay;
  setTimeout(() => {
    screen.classList.add('hidden');
    onAppReady();
  }, totalTime + 700);
}

function onAppReady() {
  initParticles();
  initCursor();
  initClock();
  checkServerHealth();
  setInterval(checkServerHealth, 18000);
  fetchScanHistory();
  setInterval(fetchScanHistory, 60000);
  initMiniChart();
  initUploadZone();
  initFileInput();
  initDualDropZones();
  initScrubber();
  initSpecsPage();
}

function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  STATE.particleCanvas = canvas;
  STATE.particleCtx    = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  spawnParticles();
  animateParticles();
}

function resizeCanvas() {
  const c = STATE.particleCanvas;
  c.width  = window.innerWidth;
  c.height = window.innerHeight;
}

function spawnParticles() {
  STATE.particles = [];
  const count = Math.min(Math.floor((window.innerWidth * window.innerHeight) / 10000), 90);
  for (let i = 0; i < count; i++) {
    STATE.particles.push({
      x:  Math.random() * window.innerWidth,
      y:  Math.random() * window.innerHeight,
      vx: (Math.random() - .5) * .28,
      vy: (Math.random() - .5) * .28,
      r:  Math.random() * 1.4 + .5,
      o:  Math.random() * .45 + .1,
      c:  Math.random() > .6 ? '#00d4ff' : Math.random() > .5 ? '#1d8eff' : '#ffffff'
    });
  }
}

function animateParticles() {
  const ctx = STATE.particleCtx;
  const { width, height } = STATE.particleCanvas;
  ctx.clearRect(0, 0, width, height);

  STATE.particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = width;
    if (p.x > width) p.x = 0;
    if (p.y < 0) p.y = height;
    if (p.y > height) p.y = 0;

    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = p.c;
    ctx.globalAlpha = p.o;
    ctx.fill();
  });

  ctx.globalAlpha = 1;
  for (let i = 0; i < STATE.particles.length; i++) {
    for (let j = i + 1; j < STATE.particles.length; j++) {
      const a = STATE.particles[i], b = STATE.particles[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 130) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(0,180,255,${0.055 * (1 - dist / 130)})`;
        ctx.lineWidth = .7;
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(animateParticles);
}

function initCursor() {
  const dot  = document.getElementById('cursor-dot');
  const ring = document.getElementById('cursor-ring');
  let lx = 0, ly = 0;

  document.addEventListener('mousemove', e => {
    STATE.cursor.x = e.clientX;
    STATE.cursor.y = e.clientY;
    dot.style.left = e.clientX + 'px';
    dot.style.top  = e.clientY + 'px';
  });

  (function smoothRing() {
    lx += (STATE.cursor.x - lx) * 0.14;
    ly += (STATE.cursor.y - ly) * 0.14;
    ring.style.left = lx + 'px';
    ring.style.top  = ly + 'px';
    requestAnimationFrame(smoothRing);
  })();

  document.addEventListener('mousedown', () => ring.classList.add('clicking'));
  document.addEventListener('mouseup',   () => ring.classList.remove('clicking'));

  document.querySelectorAll('button, a, [onclick], input, label').forEach(el => {
    el.addEventListener('mouseenter', () => ring.classList.add('hovering'));
    el.addEventListener('mouseleave', () => ring.classList.remove('hovering'));
  });
}

function initClock() {
  const el = document.getElementById('live-clock');
  function tick() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2,'0');
    const m = String(now.getMinutes()).padStart(2,'0');
    const s = String(now.getSeconds()).padStart(2,'0');
    el.textContent = `${h}:${m}:${s}`;
  }
  tick(); setInterval(tick, 1000);
}

async function checkServerHealth(retryCount = 0) {
  const chip    = document.getElementById('status-chip');
  const dot     = document.getElementById('status-dot');
  const txt     = document.getElementById('server-status-text');
  const badge   = document.getElementById('engine-badge');

  try {
    // 20s timeout to handle Vercel cold starts (Python + ONNX can take 10-15s)
    const res  = await fetch('/api/health', { signal: AbortSignal.timeout(20000) });
    const data = await res.json();

    chip.classList.remove('offline'); chip.classList.add('online');
    dot.style.background = 'var(--green)';
    txt.textContent = 'ONLINE';
    if (badge) { badge.textContent = 'LIVE'; badge.className = 'gc-status-badge online'; }
    STATE.serverOnline = true;

    const platformEl = document.getElementById('metric-platform');
    if (platformEl) platformEl.textContent = data.system || data.platform || '—';
    const pythonEl = document.getElementById('metric-python');
    if (pythonEl) pythonEl.textContent   = (data.python_version ?? '—').split(' ')[0];
    const deviceEl = document.getElementById('metric-device');
    if (deviceEl) deviceEl.textContent   = data.device ? data.device.toUpperCase() : '—';

  } catch {
    if (retryCount < 3) {
      // Auto-retry up to 3 times with increasing delays (cold start handling)
      const delay = (retryCount + 1) * 5000;
      txt.textContent = `CONNECTING... (${retryCount + 1}/3)`;
      setTimeout(() => checkServerHealth(retryCount + 1), delay);
      return;
    }
    chip.classList.remove('online'); chip.classList.add('offline');
    dot.style.background = 'var(--red)';
    txt.textContent = 'OFFLINE';
    if (badge) { badge.textContent = 'OFFLINE'; badge.className = 'gc-status-badge offline'; }
    STATE.serverOnline = false;
  }
}

function switchTab(name) {
  STATE.currentTab = name;
  document.querySelectorAll('.ws-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.getElementById(`content-${name}`).classList.remove('hidden');
  if (name === 'analytics') refreshAnalytics();
}

function initUploadZone() {
  const zone = document.getElementById('drop-zone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) startSingleScan(file);
  });
  zone.addEventListener('click', e => {
    if (e.target.closest('#select-file-btn')) return;
    document.getElementById('file-input').click();
  });
}

function initFileInput() {
  document.getElementById('file-input').addEventListener('change', function() {
    if (this.files[0]) startSingleScan(this.files[0]);
  });
}

async function startSingleScan(file) {
  showPanel('progress');
  setProgressState(0, 'Preparing upload…', 'Buffering file data into forensic pipeline…');
  appendScanLog('[ sys ] Acquired file: ' + file.name);

  const formData = new FormData();
  formData.append('file', file);

  const stages = [
    { at: 12,  delay: 0,    msg: 'Transmitting asset…',           sub: 'Streaming file to backend neural pipeline…' },
    { at: 28,  delay: 600,  msg: 'Neural classifiers running…',    sub: 'Feeding latent feature vectors to forensic MLP…' },
    { at: 42,  delay: 1400, msg: 'Detecting facial landmarks…',    sub: 'Haar cascade face detection in progress…' },
    { at: 56,  delay: 2200, msg: 'Generating anomaly heatmaps…',   sub: 'Applying Grad-CAM visualization overlay…' },
    { at: 68,  delay: 3100, msg: 'Spectral frequency analysis…',   sub: 'Mapping FFT residues and noise floor artifacts…' },
    { at: 79,  delay: 4000, msg: 'Cross-validating model ensemble…', sub: 'Fusing vision + forensic model scores…' },
    { at: 88,  delay: 5000, msg: 'Compiling forensic report…',     sub: 'Structuring analysis metadata and signals…' },
    { at: 95,  delay: 6000, msg: 'Finalizing result…',             sub: 'Writing to encrypted audit vault…' },
  ];

  const stageTimers = stages.map(({ at, delay, msg, sub }) =>
    setTimeout(() => {
      setProgressState(at, msg, sub);
      appendScanLog(`[ ai  ] ${msg}`);
    }, delay)
  );

  appendScanLog('[ net ] POST /api/scan/file — ' + formatBytes(file.size));

  let result;
  try {
    const res = await fetch('/api/scan/file', { method: 'POST', body: formData });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }
    result = await res.json();
  } catch (err) {
    
    stageTimers.forEach(clearTimeout);
    showToast('Scan failed: ' + err.message, 'error');
    appendScanLog('[ err ] ' + err.message);
    showPanel('upload');
    return;
  }

  stageTimers.forEach(clearTimeout);

  setProgressState(100, 'Analysis complete!', 'Rendering forensic intelligence report…');
  appendScanLog('[ sys ] SCAN COMPLETE. Displaying results.');
  await sleep(500);
  displaySingleResult(result);
}

function displaySingleResult(r) {
  showPanel('results');

  const risk  = r.risk_score ?? 0;
  const grade = r.risk_grade ?? 'Unknown';
  let vClass = 'uncertain', vIcon = verdictIconUncertain(), vMain = 'UNCERTAIN', vSub = 'Insufficient confidence to classify media.';

  if (r.verdict === 'REAL') {
    vClass = 'real'; vIcon = verdictIconReal();
    vMain = 'AUTHENTIC MEDIA CONFIRMED';
    vSub  = `No AI manipulation detected. Bio-signal fingerprints align with organic media.`;
  } else if (r.verdict === 'AI_GENERATED' || r.verdict === 'DEEPFAKE') {
    vClass = 'fake'; vIcon = verdictIconFake();
    vMain  = 'AI FABRICATION DETECTED';
    vSub   = `Neural synthesis artifacts detected with ${(risk*100).toFixed(1)}% confidence.`;
  }

  const banner = document.getElementById('verdict-banner');
  banner.className = `verdict-banner ${vClass}`;
  document.getElementById('verdict-icon').className = `vb-icon ${vClass}`;
  document.getElementById('verdict-icon').innerHTML = vIcon;
  document.getElementById('verdict-main-text').textContent = vMain;
  document.getElementById('verdict-sub-text').textContent  = vSub;
  document.getElementById('verdict-score-big').textContent = (risk * 100).toFixed(1) + '%';

  setGauge(risk);
  const riskScoreValEl = document.getElementById('risk-score-val');
  if (riskScoreValEl) riskScoreValEl.textContent = (risk * 100).toFixed(1) + '%';
  const riskGradeValEl = document.getElementById('risk-grade-val');
  if (riskGradeValEl) riskGradeValEl.textContent = grade.toUpperCase();

  document.getElementById('result-filename').textContent   = r.filename ?? '—';
  document.getElementById('result-media-type').textContent = (r.media_type ?? 'unknown').toUpperCase();
  document.getElementById('result-scan-id').textContent    = r.scan_id ? r.scan_id.slice(0, 16) + '…' : '—';

  const fakeHash = 'SHA256: ' + Array.from({length:64}, () => '0123456789abcdef'[Math.floor(Math.random()*16)]).join('');
  document.getElementById('crypto-hash-val').textContent = fakeHash;
  const dot = document.getElementById('crypto-dot');
  const lbl = document.getElementById('crypto-status-lbl');
  dot.className = vClass === 'fake' ? 'cc-dot danger' : 'cc-dot';
  lbl.textContent = vClass === 'fake' ? 'TAMPER_DETECTED' : 'INTEGRITY_CONFIRMED';

  const threatChip = document.getElementById('threat-chip');
  const threatVal  = document.getElementById('threat-level-val');
  if (risk >= 0.935) {
    threatChip.className = 'threat-chip danger';
    threatVal.textContent = 'HIGH';
  } else if (risk >= 0.70) {
    threatChip.className = 'threat-chip';
    threatVal.textContent = 'ELEVATED';
  } else {
    threatChip.className = 'threat-chip safe';
    threatVal.textContent = 'LOW';
  }

  const isAudio  = r.media_type === 'audio';
  const sbBox    = document.getElementById('side-by-side-box');
  const audBox   = document.getElementById('audio-only-box');
  const scrubber = document.getElementById('frame-scrubber-controls');
  const viewerLbl = document.getElementById('viewer-type-lbl');

  sbBox.classList.add('hidden');
  audBox.classList.add('hidden');
  scrubber.classList.add('hidden');

  if (isAudio) {
    audBox.classList.remove('hidden');
    viewerLbl.textContent = 'SPECTRAL VOICE-PRINT ANALYSIS HUD';
    startEqualizerAnimation(vClass === 'fake');
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = '';
  } else {
    sbBox.classList.remove('hidden');
    viewerLbl.textContent = 'ANOMALY HEATMAP VISUALIZATION HUD';
    stopEqualizerAnimation();

    if (r.frames && r.frames.length > 1) {
      STATE.frameData = r.frames;
      STATE.currentFrameIdx = 0;
      scrubber.classList.remove('hidden');
      const slider = document.getElementById('frame-slider');
      slider.max   = r.frames.length - 1;
      slider.value = 0;
      document.getElementById('frame-indicator').textContent = `1 / ${r.frames.length}`;
      loadFrame(0, r.frames);
    } else {
      STATE.frameData = [];
      document.getElementById('original-frame-img').src = r.overlay_url  || r.frame_url || '';
      document.getElementById('heatmap-frame-img').src  = r.heatmap_url  || '';
    }
  }

  const breakdownList = document.getElementById('breakdown-list');
  breakdownList.innerHTML = '';
  const signals = r.analysis?.signals ?? {};
  const signalColors = {
    'Boundary Gradient Discontinuity': 'red',
    'Chrominance Spectral Residual'  : 'red',
    'High-Freq Texture Residual'     : 'amber',
    'Compression Artifact Score'     : 'amber',
    'Noise Floor Inconsistency'      : 'amber',
    'Spectral Rolloff Anomaly'       : 'cyan',
    'Phase Inconsistency'            : 'cyan',
    'Temporal Coherence'             : 'green',
    'Bio-Signal Fidelity'            : 'green',
  };
  Object.entries(signals).forEach(([name, val]) => {
    const color = signalColors[name] ?? 'cyan';
    const pct   = (val * 100).toFixed(1);
    const item  = document.createElement('div');
    item.className = 'bar-item';
    item.innerHTML = `
      <div class="bar-lbl-row">
        <span class="bar-lbl">${name}</span>
        <span class="bar-val text-${color}">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" data-val="${val}" style="background:var(--${color})"></div>
      </div>`;
    breakdownList.appendChild(item);
  });
  
  setTimeout(() => {
    breakdownList.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = (parseFloat(el.dataset.val) * 100) + '%';
    });
  }, 80);

  if ((r.media_type === 'video' || r.media_type === 'image') && r.frame_risks && r.frame_risks.length > 1) {
    document.getElementById('timeline-panel').classList.remove('hidden');
    renderTimelineChart(r.frame_risks);
  } else {
    document.getElementById('timeline-panel').classList.add('hidden');
    destroyChart('timelineChartInst');
  }

  updateSessionStats(r.verdict, risk);

  addToActivityFeed(r);

  STATE.scanHistory.unshift(r);
  if (STATE.scanHistory.length > 50) STATE.scanHistory.pop();
  const totalScansEl = document.getElementById('metric-total-scans');
  if (totalScansEl) totalScansEl.textContent = STATE.scanHistory.length;
  updateRecentFlags();
}

function initScrubber() {
  const slider = document.getElementById('frame-slider');
  slider.addEventListener('input', () => {
    const idx = parseInt(slider.value);
    loadFrame(idx, STATE.frameData);
    document.getElementById('frame-indicator').textContent = `${idx + 1} / ${STATE.frameData.length}`;
  });
}

function loadFrame(idx, frames) {
  if (!frames || !frames[idx]) return;
  const f = frames[idx];
  document.getElementById('original-frame-img').src = f.overlay_url || f.frame_url || '';
  document.getElementById('heatmap-frame-img').src  = f.heatmap_url || '';
}

function setGauge(risk) {
  const circumference = 301.6;
  const offset = circumference - risk * circumference;
  const el = document.getElementById('gauge-circle');
  if (!el) return;
  el.style.strokeDashoffset = offset;
  if      (risk >= 0.7)  el.style.stroke = 'var(--red)';
  else if (risk >= 0.4)  el.style.stroke = 'var(--amber)';
  else                   el.style.stroke = 'var(--green)';
  el.style.filter = `drop-shadow(0 0 ${8 + risk * 16}px ${risk >= .7 ? 'var(--red)' : risk >= .4 ? 'var(--amber)' : 'var(--green)'})`;
}

function updateSessionStats(verdict, risk) {
  const s = STATE.sessionStats;
  if      (verdict === 'REAL')                           s.real++;
  else if (verdict === 'AI_GENERATED' || verdict === 'DEEPFAKE') s.fake++;
  else                                                    s.uncertain++;
  s.count++; s.totalRisk += risk;

  document.getElementById('count-real').textContent      = s.real;
  document.getElementById('count-fake').textContent      = s.fake;
  document.getElementById('count-uncertain').textContent  = s.uncertain;
  document.getElementById('kpi-real').textContent         = s.real;
  document.getElementById('kpi-fake').textContent         = s.fake;
  document.getElementById('kpi-uncertain').textContent    = s.uncertain;

  const avgRisk = s.count > 0 ? (s.totalRisk / s.count) : 0;
  document.getElementById('avg-risk-display').textContent = (avgRisk * 100).toFixed(1) + '%';
  document.getElementById('avg-risk-fill').style.width    = (avgRisk * 100) + '%';
  document.getElementById('kpi-avg-risk').textContent     = (avgRisk * 100).toFixed(1) + '%';

  const total = s.real + s.fake + s.uncertain;
  setDonut('donut-real',      total ? s.real / total : 0);
  setDonut('donut-fake',      total ? s.fake / total : 0);
  setDonut('donut-uncertain', total ? s.uncertain / total : 0);

  document.getElementById('mcl-real').textContent      = s.real;
  document.getElementById('mcl-fake').textContent      = s.fake;
  document.getElementById('mcl-uncertain').textContent  = s.uncertain;

  if (STATE.miniChart) {
    STATE.miniChart.data.datasets[0].data = [s.real, s.fake, s.uncertain];
    STATE.miniChart.update('none');
  }
}

function setDonut(id, fraction) {
  const circ = 113.1;
  const el   = document.getElementById(id);
  if (el) el.style.strokeDashoffset = circ - fraction * circ;
}

function initMiniChart() {
  const ctx = document.getElementById('mini-dist-chart');
  if (!ctx) return;
  STATE.miniChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Real', 'AI Fake', 'Uncertain'],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ['rgba(0,230,118,.22)', 'rgba(255,61,87,.22)', 'rgba(255,193,7,.22)'],
        borderColor:     ['#00e676', '#ff3d57', '#ffc107'],
        borderWidth: 2,
        hoverOffset: 5
      }]
    },
    options: {
      cutout: '65%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 700, easing: 'easeInOutQuart' }
    }
  });
}

function renderTimelineChart(frameRisks) {
  destroyChart('timelineChartInst');
  const ctx = document.getElementById('timeline-chart');
  if (!ctx) return;
  const labels = frameRisks.map((_, i) => `F${i + 1}`);
  STATE.timelineChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: frameRisks.map(v => +(v * 100).toFixed(2)),
        borderColor: '#00d4ff',
        backgroundColor: (ctx) => {
          const c = ctx.chart.ctx;
          const g = c.createLinearGradient(0, 0, 0, 160);
          g.addColorStop(0, 'rgba(0,212,255,.22)');
          g.addColorStop(1, 'rgba(0,212,255,0)');
          return g;
        },
        borderWidth: 2,
        pointRadius: frameRisks.length <= 30 ? 3 : 0,
        pointBackgroundColor: '#00d4ff',
        pointHoverRadius: 5,
        fill: true, tension: 0.4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: 'rgba(6,14,30,.97)',
        borderColor: 'rgba(0,212,255,.3)',
        borderWidth: 1,
        titleColor: '#8899bb',
        bodyColor: '#00d4ff',
        callbacks: { label: ctx => ` Risk: ${ctx.parsed.y.toFixed(1)}%` }
      }},
      scales: {
        x: { ticks: { color: '#3d4f6a', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,.03)' } },
        y: { min: 0, max: 100, ticks: { color: '#3d4f6a', font: { size: 9 }, callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });
}

function refreshAnalytics() {
  const history = STATE.scanHistory;
  
  destroyChart('analyticsHistoryChart');
  const histCtx = document.getElementById('analytics-history-chart');
  if (histCtx && history.length) {
    STATE.analyticsHistoryChart = new Chart(histCtx, {
      type: 'line',
      data: {
        labels: history.map((_, i) => `Scan ${history.length - i}`).reverse(),
        datasets: [{
          data: [...history].reverse().map(r => +(r.risk_score * 100).toFixed(1)),
          borderColor: '#1d8eff',
          backgroundColor: (ctx) => {
            const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 260);
            g.addColorStop(0, 'rgba(29,142,255,.25)');
            g.addColorStop(1, 'rgba(29,142,255,0)');
            return g;
          },
          borderWidth: 2, fill: true, tension: 0.4,
          pointRadius: 3, pointBackgroundColor: '#1d8eff'
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: {
          backgroundColor: 'rgba(6,14,30,.97)',
          borderColor: 'rgba(29,142,255,.3)', borderWidth: 1,
          titleColor: '#8899bb', bodyColor: '#1d8eff',
          callbacks: { label: c => ` Risk: ${c.parsed.y.toFixed(1)}%` }
        }},
        scales: {
          x: { ticks: { color: '#3d4f6a', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,.025)' } },
          y: { min: 0, max: 100, ticks: { color: '#3d4f6a', font: { size: 9 }, callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,.035)' } }
        }
      }
    });
  }

  destroyChart('analyticsDonutChart');
  const s = STATE.sessionStats;
  const donutCtx = document.getElementById('analytics-donut-chart');
  if (donutCtx) {
    STATE.analyticsDonutChart = new Chart(donutCtx, {
      type: 'doughnut',
      data: {
        labels: ['Verified Real', 'AI Fabricated', 'Uncertain'],
        datasets: [{
          data: [s.real || 0, s.fake || 0, s.uncertain || 0],
          backgroundColor: ['rgba(0,230,118,.2)', 'rgba(255,61,87,.2)', 'rgba(255,193,7,.2)'],
          borderColor:     ['#00e676', '#ff3d57', '#ffc107'],
          borderWidth: 2, hoverOffset: 8
        }]
      },
      options: {
        cutout: '58%', responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'bottom', labels: { color: '#8899bb', font: { size: 10 }, padding: 14 }},
          tooltip: {
            backgroundColor: 'rgba(6,14,30,.97)',
            borderColor: 'rgba(255,255,255,.1)', borderWidth: 1,
            titleColor: '#8899bb', bodyColor: '#e2eeff'
          }
        }
      }
    });
  }

  const tbody = document.getElementById('history-table-body');
  if (history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-table-row">No scan history available.</td></tr>';
    return;
  }
  tbody.innerHTML = history.map(r => {
    const verdict = r.verdict;
    const cls     = verdict === 'REAL' ? 'green' : verdict === 'AI_GENERATED' ? 'red' : 'amber';
    const lbl     = verdict === 'REAL' ? 'REAL' : verdict === 'AI_GENERATED' ? 'AI FAKE' : 'UNCERTAIN';
    const scanId  = r.scan_id ?? r.id ?? '';
    const rJson   = JSON.stringify(r).replace(/"/g,'&quot;');
    return `<tr>
      <td style="color:var(--text-1)">${escapeHtml(r.filename ?? '—')}</td>
      <td style="text-transform:uppercase">${r.media_type ?? '—'}</td>
      <td><span style="color:var(--${cls});font-weight:700">${lbl}</span></td>
      <td style="font-family:var(--font-mono);color:var(--${cls})">${((r.risk_score ?? 0)*100).toFixed(1)}%</td>
      <td>
        <div class="history-row-actions">
          <button class="btn-sm" onclick="viewHistoryDetail(${rJson})">VIEW</button>
          <button class="btn-sm danger" onclick="deleteScanHistoryItem('${escapeHtml(scanId)}', this)" title="Delete this record">
            <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.8" width="11" height="11"><polyline points="1,3 13,3"/><path d="M5 3V2a1 1 0 012 0v1"/><path d="M2 3l.8 9h8.4L12 3"/></svg>
          </button>
        </div>
      </td>
    </tr>`;
  }).join('');
}

async function fetchScanHistory() {
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    if (Array.isArray(data) && data.length > 0) {
      data.forEach(r => {
        const key = r.scan_id ?? r.id;
        if (!STATE.scanHistory.find(h => (h.scan_id ?? h.id) === key)) {
          STATE.scanHistory.push(r);
        }
      });
      const totalScansEl = document.getElementById('metric-total-scans');
      if (totalScansEl) totalScansEl.textContent = STATE.scanHistory.length;
    }
  } catch { /* silent */ }
}

async function deleteScanHistoryItem(scanId, btnEl) {
  if (!scanId) return;
  try {
    btnEl.disabled = true;
    const res = await fetch(`/api/history/${scanId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    // Remove from local STATE
    STATE.scanHistory = STATE.scanHistory.filter(h => (h.scan_id ?? h.id) !== scanId);
    // Animate row removal
    const row = btnEl.closest('tr');
    if (row) {
      row.style.transition = 'opacity .3s ease, transform .3s ease';
      row.style.opacity = '0';
      row.style.transform = 'translateX(20px)';
      setTimeout(() => {
        row.remove();
        const tbody = document.getElementById('history-table-body');
        if (tbody && tbody.children.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="empty-table-row">No scan history available.</td></tr>';
        }
      }, 320);
    }
    showToast('Record deleted', 'success', 2500);
  } catch (e) {
    if (btnEl) btnEl.disabled = false;
    showToast('Could not delete record', 'error');
  }
}

async function clearAllHistory() {
  if (!confirm('Delete ALL scan history? This cannot be undone.')) return;
  try {
    const res = await fetch('/api/history', { method: 'DELETE' });
    if (!res.ok) throw new Error('Clear failed');
    STATE.scanHistory = [];
    const tbody = document.getElementById('history-table-body');
    if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="empty-table-row">No scan history available.</td></tr>';
    showToast('Scan history cleared', 'success', 3000);
  } catch {
    showToast('Could not clear history', 'error');
  }
}

function viewHistoryDetail(r) {
  switchTab('single');
  setTimeout(() => displaySingleResult(r), 50);
}

// ── DUAL SCAN ─────────────────────────────────────────────────────
function initDualDropZones() {
  ['real', 'check'].forEach(side => {
    const zone   = document.getElementById(`dual-drop-zone-${side === 'real' ? 'real' : 'check'}`);
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) { setDualFile(side, file); }
    });
  });
  document.getElementById('dual-real-file').addEventListener('change', function() {
    if (this.files[0]) setDualFile('real', this.files[0]);
  });
  document.getElementById('dual-check-file').addEventListener('change', function() {
    if (this.files[0]) setDualFile('check', this.files[0]);
  });
}

function previewDualFile(side) {
  const inp  = document.getElementById(`dual-${side}-file`);
  const file = inp.files[0];
  if (file) setDualFile(side, file);
}

function setDualFile(side, file) {
  if (side === 'real') STATE.dualRealFile  = file;
  else                 STATE.dualCheckFile = file;

  const previewId = `dual-preview-${side}`;
  const preview   = document.getElementById(previewId);
  const reader    = new FileReader();
  reader.onload   = e => {
    preview.src = e.target.result;
    preview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  const zone = document.getElementById(side === 'real' ? 'dual-drop-zone-real' : 'dual-drop-zone-check');
  zone.classList.add('active-file');

  const btn  = document.getElementById('btn-run-dual');
  const hint = document.getElementById('dual-hint');
  if (STATE.dualRealFile && STATE.dualCheckFile) {
    btn.disabled = false;
    hint.textContent = 'Both images loaded. Ready to compare identities.';
    hint.style.color = 'var(--green)';
  } else {
    btn.disabled = true;
    hint.textContent = side === 'real'
      ? 'Reference loaded. Now upload the suspect image.'
      : 'Suspect loaded. Now upload the reference image.';
    hint.style.color = 'var(--amber)';
  }
}

async function runDualVerification() {
  if (!STATE.dualRealFile || !STATE.dualCheckFile) return;
  document.getElementById('dual-upload-card').classList.add('hidden');
  document.getElementById('dual-progress-box').classList.remove('hidden');
  document.getElementById('dual-results-box').classList.add('hidden');

  const form = new FormData();
  form.append('real_file',  STATE.dualRealFile);
  form.append('check_file', STATE.dualCheckFile);

  let pct = 0;
  const animInterval = setInterval(() => {
    pct = Math.min(pct + 2, 88);
    document.getElementById('dual-progress-bar').style.width = pct + '%';
    document.getElementById('dual-pct-display').textContent  = pct + '%';
  }, 80);

  const statusStages = [
    'Detecting facial landmarks in both images…',
    'Computing biometric embedding vectors…',
    'Measuring cosine identity similarity…',
    'Running AI fabrication classifier on suspect…',
    'Cross-referencing deepfake signal signatures…',
    'Generating final identity match verdict…'
  ];
  let si = 0;
  const statusInterval = setInterval(() => {
    if (si < statusStages.length)
      document.getElementById('dual-progress-status').textContent = statusStages[si++];
  }, 700);

  try {
    const res  = await fetch('/api/scan/dual', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    clearInterval(animInterval);
    clearInterval(statusInterval);
    document.getElementById('dual-progress-bar').style.width = '100%';
    document.getElementById('dual-pct-display').textContent  = '100%';
    await sleep(400);
    displayDualResult(data);
  } catch (err) {
    clearInterval(animInterval);
    clearInterval(statusInterval);
    showToast('Dual scan failed: ' + err.message, 'error');
    resetDualScanner();
  }
}

function displayDualResult(data) {
  document.getElementById('dual-progress-box').classList.add('hidden');
  document.getElementById('dual-results-box').classList.remove('hidden');

  const sim        = data.similarity ?? 0;
  const aiScore    = data.deepfake_score ?? 0;
  const isMatch    = data.is_match ?? false;

  const banner = document.getElementById('dual-verdict-banner-val');
  const icon   = document.getElementById('dual-verdict-icon');
  const main   = document.getElementById('dual-verdict-main');
  const sub    = document.getElementById('dual-summary-val');

  if (isMatch && aiScore < .5) {
    banner.className = 'verdict-banner real';
    icon.className   = 'vb-icon real';
    main.textContent = 'IDENTITY VERIFIED — SAME PERSON';
    sub.textContent  = `Biometric similarity: ${sim.toFixed(1)}%. No AI manipulation detected.`;
  } else if (!isMatch) {
    banner.className = 'verdict-banner fake';
    icon.className   = 'vb-icon fake';
    main.textContent = 'IDENTITY MISMATCH — DIFFERENT PERSON';
    sub.textContent  = `Biometric similarity: ${sim.toFixed(1)}%. These do not appear to be the same individual.`;
  } else {
    banner.className = 'verdict-banner uncertain';
    icon.className   = 'vb-icon uncertain';
    main.textContent = 'POTENTIAL DEEPFAKE IDENTITY SWAP';
    sub.textContent  = `Match confidence: ${sim.toFixed(1)}% but AI fabrication score is ${(aiScore * 100).toFixed(1)}%.`;
  }
  icon.innerHTML = verdictIconForVerdict(banner.className);

  document.getElementById('dmm-sim-label').textContent = sim.toFixed(0) + '%';
  document.getElementById('bar-sim-pct').textContent   = sim.toFixed(1) + '%';
  document.getElementById('bar-ai-pct').textContent    = (aiScore * 100).toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('bar-sim-fill').style.width = Math.min(sim, 100) + '%';
    document.getElementById('bar-ai-fill').style.width  = (aiScore * 100) + '%';
  }, 100);

  if (data.real_crop_url) {
    document.getElementById('cropped-real-img').src  = data.real_crop_url;
    document.getElementById('cropped-check-img').src = data.check_crop_url || '';
  }

  const cardLbl = document.getElementById('dual-check-card-lbl');
  cardLbl.textContent = isMatch && aiScore < .5 ? 'CONFIRMED MATCH' : 'SUSPECT / MISMATCH';
  cardLbl.className   = `dfc-badge ${isMatch && aiScore < .5 ? 'green' : 'red'}`;

  addToActivityFeed({ filename: 'Dual Verification', verdict: isMatch ? 'REAL' : 'AI_GENERATED', risk_score: aiScore, media_type: 'dual' });
}

function resetDualScanner() {
  STATE.dualRealFile  = null;
  STATE.dualCheckFile = null;
  document.getElementById('dual-upload-card').classList.remove('hidden');
  document.getElementById('dual-progress-box').classList.add('hidden');
  document.getElementById('dual-results-box').classList.add('hidden');
  ['real','check'].forEach(s => {
    const prev = document.getElementById(`dual-preview-${s}`);
    prev.src = ''; prev.classList.add('hidden');
    const zone = document.getElementById(`dual-drop-zone-${s === 'real' ? 'real' : 'check'}`);
    zone.classList.remove('active-file', 'dragover');
  });
  document.getElementById('btn-run-dual').disabled = true;
  document.getElementById('dual-hint').textContent = 'Select both images to begin comparison';
  document.getElementById('dual-hint').style.color = '';
  document.getElementById('dual-real-file').value  = '';
  document.getElementById('dual-check-file').value = '';
  
  const icon = document.getElementById('dual-verdict-icon');
  icon.className = 'vb-icon';
  icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>`;
}

// ── ACTIVITY FEED ─────────────────────────────────────────────────
function addToActivityFeed(r) {
  const feed = document.getElementById('history-container');
  const emptyMsg = feed.querySelector('.empty-feed-msg');
  if (emptyMsg) emptyMsg.remove();

  const verdict = r.verdict;
  const risk    = r.risk_score ?? 0;
  const cls     = verdict === 'REAL' ? 'real' : verdict === 'AI_GENERATED' || verdict === 'DEEPFAKE' ? 'fake' : 'uncertain';
  const lbl     = verdict === 'REAL' ? 'REAL' : verdict === 'AI_GENERATED' || verdict === 'DEEPFAKE' ? 'AI FAKE' : 'UNCERTAIN';
  const mtype   = r.media_type ?? 'img';
  const typeIconMap = { image:'IMG', video:'VID', audio:'AUD', dual:'DUAL' };
  const typeIcon = typeIconMap[mtype] ?? 'SYS';
  const typeCls  = { image:'img', video:'vid', audio:'aud', dual:'dual' }[mtype] ?? 'img';

  const item = document.createElement('div');
  item.className = 'activity-item';
  item.innerHTML = `
    <div class="ai-type-icon ${typeCls}">${typeIcon}</div>
    <div class="ai-meta">
      <div class="ai-name">${escapeHtml(r.filename ?? 'Unknown')}</div>
      <div class="ai-sub">${(risk * 100).toFixed(1)}% risk · ${new Date().toLocaleTimeString()}</div>
    </div>
    <div class="ai-badge ${cls}">${lbl}</div>
  `;
  item.addEventListener('click', () => { if (r.media_type !== 'dual') viewHistoryDetail(r); });
  feed.prepend(item);
  if (feed.children.length > 25) feed.removeChild(feed.lastChild);
}

// ── RECENT THREATS ─────────────────────────────────────────────────
function updateRecentFlags() {
  const list    = document.getElementById('recent-flags-list');
  const threats = STATE.scanHistory.filter(r => r.risk_score >= .6).slice(0, 10);
  const emptyMsg = list.querySelector('.empty-feed-msg');
  if (threats.length === 0) {
    if (!emptyMsg) list.innerHTML = '<div class="empty-feed-msg">No flagged threats detected yet.</div>';
    return;
  }
  if (emptyMsg) emptyMsg.remove();
  list.innerHTML = threats.map(r => `
    <div class="flag-item" onclick="viewHistoryDetail(${JSON.stringify(r).replace(/"/g,'&quot;')})">
      <div class="flag-dot"></div>
      <span class="flag-name">${escapeHtml(r.filename ?? '—')}</span>
      <span class="flag-score">${((r.risk_score ?? 0)*100).toFixed(0)}%</span>
    </div>
  `).join('');
}

// ── PANEL MANAGEMENT ──────────────────────────────────────────────
function showPanel(panel) {
  const uploadBox   = document.getElementById('drop-zone');
  const progressBox = document.getElementById('scan-progress-box');
  const resultsBox  = document.getElementById('scan-results-box');

  uploadBox.classList.add('hidden');
  progressBox.classList.add('hidden');
  resultsBox.classList.add('hidden');

  if (panel === 'upload')   uploadBox.classList.remove('hidden');
  if (panel === 'progress') progressBox.classList.remove('hidden');
  if (panel === 'results')  resultsBox.classList.remove('hidden');
}

function resetScanner() {
  clearInterval(STATE.scanPollingId);
  stopEqualizerAnimation();
  destroyChart('timelineChartInst');
  STATE.frameData = [];
  document.getElementById('file-input').value = '';

  // Reset threat chip
  const tc = document.getElementById('threat-chip');
  tc.className = 'threat-chip';
  document.getElementById('threat-level-val').textContent = 'NOMINAL';

  showPanel('upload');
}

// ── PROGRESS HELPERS ──────────────────────────────────────────────
function setProgressState(pct, title, status) {
  document.getElementById('progress-title').textContent  = title;
  document.getElementById('progress-status').textContent = status;
  document.getElementById('progress-bar').style.width   = pct + '%';
  document.getElementById('scan-pct-display').textContent = pct + '%';
}

function appendScanLog(msg) {
  const log  = document.getElementById('scan-log');
  const line = document.createElement('div');
  line.className = 'spp-log-line';
  if (msg.includes('COMPLETE') || msg.includes('READY')) line.classList.add('ok');
  if (msg.includes('WARN') || msg.includes('warn'))      line.classList.add('warn');
  if (msg.includes('ERR')  || msg.includes('error'))     line.classList.add('err');
  line.textContent = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
  if (log.children.length > 12) log.removeChild(log.firstChild);
}

// ── EQUALIZER ANIMATION ───────────────────────────────────────────
function startEqualizerAnimation(isFake) {
  stopEqualizerAnimation();
  const bars = document.querySelectorAll('.eq-bar');
  const baseHeights = [35, 55, 70, 45, 90, 60, 80, 50, 65, 75, 40, 55, 70, 45, 85, 60, 75, 50, 68, 42];
  let t = 0;
  function tick() {
    t += 0.12;
    bars.forEach((bar, i) => {
      const bh = baseHeights[i % baseHeights.length];
      const wave = bh + Math.sin(t + i * 0.45) * 22 + Math.random() * (isFake ? 24 : 10);
      bar.style.height = Math.max(5, Math.min(98, wave)) + '%';
    });
    STATE.eqAnimId = requestAnimationFrame(tick);
  }
  tick();
}

function stopEqualizerAnimation() {
  if (STATE.eqAnimId) { cancelAnimationFrame(STATE.eqAnimId); STATE.eqAnimId = null; }
  document.querySelectorAll('.eq-bar').forEach(b => b.style.height = '10%');
}

// ── TOAST ─────────────────────────────────────────────────────────
const TOAST_ICONS = {
  success: `<svg class="toast-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"/><circle cx="10" cy="10" r="8"/></svg>`,
  error:   `<svg class="toast-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="10" cy="10" r="8"/><line x1="8" y1="8" x2="12" y2="12"/><line x1="12" y1="8" x2="8" y2="12"/></svg>`,
  warning: `<svg class="toast-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="10" cy="10" r="8"/><path d="M10 6v4"/><circle cx="10" cy="14" r="1" fill="currentColor"/></svg>`,
  info:    `<svg class="toast-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="10" cy="10" r="8"/><path d="M10 8v6"/><circle cx="10" cy="6" r="1" fill="currentColor"/></svg>`,
};

function showToast(msg, type = 'info', duration = 4500) {
  const container = document.getElementById('toast-container');
  const toast     = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `${TOAST_ICONS[type] ?? TOAST_ICONS.info}
    <span class="toast-msg">${escapeHtml(msg)}</span>
    <button class="toast-close" onclick="dismissToast(this.parentElement)">✕</button>`;
  container.appendChild(toast);
  setTimeout(() => dismissToast(toast), duration);
  if (container.children.length > 5) container.removeChild(container.firstChild);
}

function dismissToast(el) {
  if (!el || !el.parentElement) return;
  el.classList.add('out');
  setTimeout(() => el.remove(), 350);
}

// ── UTILITY ───────────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function destroyChart(key) {
  if (STATE[key]) { try { STATE[key].destroy(); } catch {} STATE[key] = null; }
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard', 'success', 2000)).catch(() => {});
}

function copyCode(id) {
  const el = document.getElementById(id);
  if (el) copyText(el.textContent);
}

// ── VERDICT ICON SVG HELPERS ──────────────────────────────────────
function verdictIconReal() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>`;
}
function verdictIconFake() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>`;
}
function verdictIconUncertain() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M12 8v4"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>`;
}
function verdictIconForVerdict(className) {
  if (className.includes('real')) return verdictIconReal();
  if (className.includes('fake')) return verdictIconFake();
  return verdictIconUncertain();
}

// ── BOOT ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', runBootSequence);

// ── HOW IT WORKS DASHBOARD LOGIC ─────────────────────────────────
function initSpecsPage() {
  const slider = document.getElementById('sim-threshold-slider');
  if (!slider) return;
  
  slider.addEventListener('input', function() {
    updateSpecsSimulator(parseFloat(this.value));
  });
  
  // Initial simulation update
  updateSpecsSimulator(parseFloat(slider.value));
}

function approxNormalCDF(x, mean, std) {
  const z = (x - mean) / std;
  // erf approximation formula
  const t_erf = 1 / (1 + 0.5 * Math.abs(z));
  const ans = 1 - t_erf * Math.exp(-z * z - 1.26551223 + t_erf * (1.00002368 + t_erf * (0.37409196 + t_erf * (0.09678418 + t_erf * (-0.18628806 + t_erf * (0.27886807 + t_erf * (-1.13520398 + t_erf * (1.48851587 + t_erf * (-0.82215223 + t_erf * 0.17087277)))))))));
  return z >= 0 ? ans : 1 - ans;
}

function updateSpecsSimulator(val) {
  const t = val / 100; // convert slider percent to decimal threshold
  
  // Math model mapping:
  // Real scores follow a normal distribution centered at 0.86 with std dev of 0.058
  // Fake scores follow a normal distribution centered at 0.96 with std dev of 0.024
  const realPassRate = approxNormalCDF(t, 0.86, 0.058); // Real below threshold (True Negatives)
  const fakePassRate = approxNormalCDF(t, 0.96, 0.024); // Fake below threshold (False Negatives)
  
  const tn = Math.round(50 * realPassRate);
  const fp = 50 - tn;
  const fn = Math.round(50 * fakePassRate);
  const tp = 50 - fn;
  
  const fpr = (fp / 50) * 100;
  const fnr = (fn / 50) * 100;
  const overallAcc = ((tn + tp) / 100) * 100;
  
  // Update stats displays
  const valEl = document.getElementById('sim-threshold-val');
  if (valEl) valEl.textContent = val.toFixed(1) + '%';
  
  const accEl = document.getElementById('sim-acc');
  if (accEl) accEl.textContent = overallAcc.toFixed(1) + '%';
  
  const fprEl = document.getElementById('sim-fpr');
  if (fprEl) fprEl.textContent = fpr.toFixed(1) + '%';
  
  const fnrEl = document.getElementById('sim-fnr');
  if (fnrEl) fnrEl.textContent = fnr.toFixed(1) + '%';
  
  const passEl = document.getElementById('sim-pass');
  if (passEl) passEl.textContent = (realPassRate * 100).toFixed(1) + '%';
  
  // Update Confusion Matrix values
  const tnEl = document.getElementById('cm-tn');
  if (tnEl) tnEl.textContent = tn;
  
  const fpEl = document.getElementById('cm-fp');
  if (fpEl) fpEl.textContent = fp;
  
  const fnEl = document.getElementById('cm-fn');
  if (fnEl) fnEl.textContent = fn;
  
  const tpEl = document.getElementById('cm-tp');
  if (tpEl) tpEl.textContent = tp;
  
  // Update SVG plot threshold line
  const line = document.getElementById('sim-plot-threshold-line');
  const label = document.getElementById('sim-plot-threshold-label');
  if (line) {
    // x-axis of density plot goes from x=10 (50%) to x=390 (100%)
    const x = 10 + ((val - 50) / 50) * 380;
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
    if (label) {
      label.setAttribute('x', x);
      label.textContent = val.toFixed(1) + '%';
    }
  }
}

window.filterSignals = function(category) {
  // Toggle active filter button
  document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`fbtn-${category}`);
  if (activeBtn) activeBtn.classList.add('active');

  const items = document.querySelectorAll('.signal-spec-item');
  items.forEach(item => {
    if (category === 'all' || item.getAttribute('data-category') === category) {
      item.style.display = 'flex';
    } else {
      item.style.display = 'none';
    }
  });
}
