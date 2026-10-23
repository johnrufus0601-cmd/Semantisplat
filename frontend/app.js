/**
 * SemantiSplat — Frontend Application Logic
 *
 * Handles: image/video upload, pipeline status tracking (REST + WebSocket),
 * open-vocabulary 3D search, spatial reasoning queries, and 3D viewer interaction.
 */

const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

// ===== State =====
let currentJobId = null;
let currentSceneId = 'e3b3020d';
let statusWs = null;
let reprojectionFrames = [];
let currentFrameIndex = 0;
let lastReprojectionSceneId = null;

window.addEventListener('DOMContentLoaded', () => {
  updateQuickChips('indoor_room');
  if (currentSceneId) {
    handlePipelineComplete({ status: 'completed', scene_id: currentSceneId });
  }
});

// ===== DOM Elements =====
const navTabs = document.querySelectorAll('.nav-tab');
const panels = document.querySelectorAll('.panel');
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const processingPanel = document.getElementById('processing-panel');
const progressBar = document.getElementById('progress-bar');
const progressMessage = document.getElementById('progress-message');
const queryInput = document.getElementById('query-input');
const queryBtn = document.getElementById('query-btn');
const topKInput = document.getElementById('top-k-input');
const thresholdInput = document.getElementById('threshold-input');
const resultsSection = document.getElementById('results-section');
const resultsList = document.getElementById('results-list');
const resultsTitle = document.getElementById('results-title');
const resultsMeta = document.getElementById('results-meta');
const noMatchSection = document.getElementById('no-match-section');
const noMatchMessage = document.getElementById('no-match-message');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const retryBtn = document.getElementById('retry-btn');

// ===== Navigation =====
navTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.panel;
    navTabs.forEach(t => t.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panelEl = document.getElementById(`panel-${target}`);
    if (panelEl) panelEl.classList.add('active');

    // If switching to upload tab and not actively uploading, ensure upload zone is visible
    if (target === 'upload' && uploadZone) {
      const progressPanel = document.getElementById('upload-progress-panel');
      const isUploading = progressPanel && progressPanel.style.display !== 'none';
      if (!isUploading) {
        uploadZone.style.display = 'block';
      }
    }

    // If switching to 3D viewer tab, re-frame camera on detected boxes
    if (target === 'viewer' && window.splatViewer) {
      setTimeout(() => {
        if (window.splatViewer.boundingBoxes && window.splatViewer.boundingBoxes.length > 0) {
          window.splatViewer.frameBoundingBoxes();
        }
      }, 60);
    }
  });
});

// ===== Upload Triggers & Actions =====
function triggerNewUpload() {
  if (fileInput) {
    fileInput.value = ''; // Reset so the same file or a new file can be selected
    fileInput.click();
  }
}

const btnHeaderUpload = document.getElementById('btn-header-upload');
const btnQueryUpload = document.getElementById('btn-query-upload');
const btnViewerUpload = document.getElementById('btn-viewer-upload');
const btnUploadAnother = document.getElementById('btn-upload-another');

if (btnHeaderUpload) btnHeaderUpload.addEventListener('click', triggerNewUpload);
if (btnQueryUpload) btnQueryUpload.addEventListener('click', triggerNewUpload);
if (btnViewerUpload) btnViewerUpload.addEventListener('click', triggerNewUpload);
if (btnUploadAnother) btnUploadAnother.addEventListener('click', triggerNewUpload);

if (browseBtn) {
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    triggerNewUpload();
  });
}

if (uploadZone) {
  uploadZone.addEventListener('click', () => triggerNewUpload());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  });
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
  });
}

// Clear previous scene data, 3D boxes, query results, and images
function clearPreviousResults() {
  // Clear query results
  if (queryInput) queryInput.value = '';
  if (resultsSection) resultsSection.style.display = 'none';
  if (resultsList) resultsList.innerHTML = '';
  if (resultsTitle) resultsTitle.textContent = '';
  if (resultsMeta) resultsMeta.textContent = '';
  if (noMatchSection) noMatchSection.style.display = 'none';
  if (noMatchMessage) noMatchMessage.textContent = '';
  const spatialCard = document.getElementById('spatial-card');
  if (spatialCard) spatialCard.style.display = 'none';

  // Clear 3D viewer state
  if (window.splatViewer) {
    window.splatViewer.setBoundingBoxes([], false);
    window.splatViewer.clearHighlight();
    window.splatViewer.gaussians = null;
    window.splatViewer._initialFramedDone = false;
  }
  window.detectedObjects = [];
  const viewerInfo = document.getElementById('viewer-info-text');
  if (viewerInfo) viewerInfo.textContent = 'Uploading new file...';

  // Clear heatmap
  const heatmapImg = document.getElementById('heatmap-img');
  const heatmapPlaceholder = document.getElementById('heatmap-placeholder');
  if (heatmapImg) { heatmapImg.src = ''; heatmapImg.style.display = 'none'; }
  if (heatmapPlaceholder) heatmapPlaceholder.style.display = 'block';

  // Clear reprojection frames
  reprojectionFrames = [];
  currentFrameIndex = 0;
  lastReprojectionSceneId = null;
  const reprojectionImg = document.getElementById('reprojection-img');
  const reprojectionControls = document.getElementById('reprojection-controls');
  const reprojectionPlaceholder = document.getElementById('reprojection-placeholder');
  if (reprojectionImg) { reprojectionImg.src = ''; reprojectionImg.style.display = 'none'; }
  if (reprojectionControls) reprojectionControls.style.display = 'none';
  if (reprojectionPlaceholder) reprojectionPlaceholder.style.display = 'block';

  // Reset pipeline stages UI
  document.querySelectorAll('.stage').forEach(el => {
    el.classList.remove('active', 'completed');
    const st = el.querySelector('.stage-status');
    if (st) st.textContent = 'Pending';
    const sp = el.querySelector('.stage-percent');
    if (sp) sp.textContent = '';
  });
  if (progressBar) {
    progressBar.style.width = '0%';
    progressBar.style.background = '';
  }
  if (progressMessage) {
    progressMessage.textContent = 'Initializing pipeline...';
    progressMessage.style.color = '';
  }
  if (retryBtn) retryBtn.style.display = 'none';
  const completedActions = document.getElementById('pipeline-completed-actions');
  if (completedActions) completedActions.style.display = 'none';

  // Close previous status WebSocket
  if (statusWs) {
    try { statusWs.close(); } catch (e) {}
    statusWs = null;
  }
}

function uploadFile(file) {
  // Clear previous scene results & reset state
  clearPreviousResults();

  // Switch to upload tab to track pipeline
  const uploadNav = document.getElementById('nav-upload');
  if (uploadNav) uploadNav.click();

  setStatus('uploading', 'Uploading file...');
  
  const progressPanel = document.getElementById('upload-progress-panel');
  const uploadBar = document.getElementById('upload-progress-bar');
  const uploadText = document.getElementById('upload-progress-text');
  
  if (uploadZone) uploadZone.style.display = 'none';
  if (processingPanel) processingPanel.style.display = 'none';
  if (progressPanel) {
    progressPanel.style.display = 'block';
    uploadBar.style.width = '0%';
    uploadText.textContent = '0%';
  }

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${API_BASE}/api/upload`, true);

  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable && progressPanel) {
      const percentComplete = Math.round((event.loaded / event.total) * 100);
      uploadBar.style.width = `${percentComplete}%`;
      uploadText.textContent = `${percentComplete}%`;
    }
  };

  xhr.onload = () => {
    if (progressPanel) progressPanel.style.display = 'none';
    
    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        const data = JSON.parse(xhr.responseText);
        if (data.status === 'error') {
          setStatus('error', data.message);
          alert(data.message);
          if (uploadZone) uploadZone.style.display = 'block';
          return;
        }
        currentJobId = data.job_id;
        showProcessingPanel();
        trackStatus(data.job_id);
        setStatus('processing', 'Processing...');
      } catch (err) {
        setStatus('error', 'Invalid response from server.');
        if (uploadZone) uploadZone.style.display = 'block';
      }
    } else {
      setStatus('error', 'Upload failed: HTTP ' + xhr.status);
      if (uploadZone) uploadZone.style.display = 'block';
    }
  };

  xhr.onerror = () => {
    if (progressPanel) progressPanel.style.display = 'none';
    setStatus('error', 'Upload failed due to a network error.');
    if (uploadZone) uploadZone.style.display = 'block';
  };

  xhr.send(formData);
}

function showProcessingPanel() {
  if (uploadZone) uploadZone.style.display = 'none';
  if (processingPanel) processingPanel.style.display = 'block';
}

if (retryBtn) {
  retryBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    retryBtn.disabled = true;
    retryBtn.textContent = 'Retrying...';
    try {
      const res = await fetch(`${API_BASE}/api/retry/${currentJobId}`, { method: 'POST' });
      if (!res.ok) throw new Error('Retry failed');
      const data = await res.json();
      retryBtn.style.display = 'none';
      retryBtn.disabled = false;
      retryBtn.textContent = 'Retry Failed Pipeline';
      setStatus('processing', 'Restarting pipeline...');
      trackStatus(currentJobId);
    } catch (err) {
      alert('Failed to retry pipeline. Please try re-uploading the file.');
      retryBtn.disabled = false;
      retryBtn.textContent = 'Retry Failed Pipeline';
    }
  });
}

// ===== Status Tracking =====
function trackStatus(jobId) {
  try {
    statusWs = new WebSocket(`${WS_BASE}/api/ws/status/${jobId}`);

    statusWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updatePipelineUI(data);

      if (data.status === 'completed' || data.status === 'failed') {
        statusWs.close();
        handlePipelineComplete(data);
      }
    };

    statusWs.onerror = () => {
      pollStatus(jobId);
    };
  } catch {
    pollStatus(jobId);
  }
}

async function pollStatus(jobId) {
  const poll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status/${jobId}`);
      const data = await res.json();
      updatePipelineUI(data);

      if (data.status === 'completed' || data.status === 'failed') {
        handlePipelineComplete(data);
        return;
      }

      setTimeout(poll, 1500);
    } catch {
      setTimeout(poll, 3000);
    }
  };
  poll();
}

function updatePipelineUI(data) {
  if (progressBar) progressBar.style.width = `${Math.round((data.progress || 0) * 100)}%`;
  if (progressMessage) progressMessage.textContent = data.message || '';

  const stageOrder = ['frame_extraction', 'pose_estimation', 'segmentation', 'coverage_heatmap', 'reconstruction'];
  const currentIdx = stageOrder.indexOf(data.stage);

  document.querySelectorAll('.stage').forEach((el, idx) => {
    el.classList.remove('active', 'completed', 'failed');
    const stageStatus = el.querySelector('.stage-status');
    const stagePercent = el.querySelector('.stage-percent');
    if (stagePercent) stagePercent.textContent = '';

    if (idx < currentIdx) {
      el.classList.add('completed');
      if (stageStatus) stageStatus.textContent = '✓ Done';
      if (stagePercent) stagePercent.textContent = '100%';
    } else if (idx === currentIdx) {
      el.classList.add('active');
      if (stageStatus) stageStatus.textContent = '● Running';
      if (stagePercent && data.stage_progress != null) {
        stagePercent.textContent = `${Math.round(data.stage_progress * 100)}%`;
      }
    } else {
      if (stageStatus) stageStatus.textContent = 'Pending';
    }
  });

  if (data.status === 'completed') {
    document.querySelectorAll('.stage').forEach(el => {
      el.classList.remove('active');
      el.classList.add('completed');
      const st = el.querySelector('.stage-status');
      if (st) st.textContent = '✓ Done';
      const sp = el.querySelector('.stage-percent');
      if (sp) sp.textContent = '100%';
    });
  }

  if (data.status === 'failed') {
    if (progressBar) progressBar.style.background = 'var(--accent-red)';
    if (progressMessage) progressMessage.style.color = 'var(--accent-red)';
    if (retryBtn) retryBtn.style.display = 'block';
  } else {
    if (progressBar) progressBar.style.background = '';
    if (progressMessage) progressMessage.style.color = '';
    if (retryBtn) retryBtn.style.display = 'none';
  }
}

function handlePipelineComplete(data) {
  if (data.status === 'completed') {
    currentSceneId = data.scene_id || currentJobId;
    setStatus('ready', 'Scene ready');
    
    // Show completion actions and keep upload dropzone visible for new uploads
    const completedActions = document.getElementById('pipeline-completed-actions');
    if (completedActions) completedActions.style.display = 'block';
    if (uploadZone) uploadZone.style.display = 'block';

    // Refresh heatmap
    const heatmapImg = document.getElementById('heatmap-img');
    const heatmapPlaceholder = document.getElementById('heatmap-placeholder');
    if (heatmapImg && heatmapPlaceholder) {
      heatmapImg.src = `${API_BASE}/api/heatmap/${currentSceneId}?t=${Date.now()}`;
      heatmapImg.style.display = 'inline-block';
      heatmapPlaceholder.style.display = 'none';
    }

    // Refresh 3D scene in viewer
    if (window.splatViewer) {
      window.splatViewer.loadScene(`${API_BASE}/api/scene/${currentSceneId}`);
      fetchDetectedObjects(currentSceneId);
    }

    // Clear reprojection frames cache for new scene
    reprojectionFrames = [];
    lastReprojectionSceneId = null;

    setTimeout(() => {
      const qTab = document.getElementById('nav-query');
      if (qTab) qTab.click();
    }, 1200);
  } else {
    setStatus('error', 'Pipeline failed');
    if (uploadZone) uploadZone.style.display = 'block';
  }
}

// ===== Query & Search =====
if (queryBtn) queryBtn.addEventListener('click', submitQuery);
if (queryInput) {
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitQuery();
  });
}

async function submitQuery() {
  const text = queryInput.value.trim();
  if (!text) return;

  queryBtn.disabled = true;
  queryBtn.textContent = 'Searching...';
  if (resultsSection) resultsSection.style.display = 'none';
  if (noMatchSection) noMatchSection.style.display = 'none';
  const spatialCard = document.getElementById('spatial-card');
  if (spatialCard) spatialCard.style.display = 'none';

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        top_k: parseInt(topKInput.value) || 5,
        similarity_threshold: parseFloat(thresholdInput.value) || 0.22,
        scene_id: currentSceneId || 'default',
      }),
    });

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    if (noMatchSection) noMatchSection.style.display = 'block';
    if (noMatchMessage) noMatchMessage.textContent = 'Query failed: ' + err.message;
  }

  queryBtn.disabled = false;
  queryBtn.textContent = 'Search';
}

function renderResults(data) {
  // 1. Spatial Reasoning Result Card
  const spatialCard = document.getElementById('spatial-card');
  const spatialMsg = document.getElementById('spatial-message');
  const spatialTitle = document.getElementById('spatial-title');
  const spatialIcon = document.getElementById('spatial-icon');
  const spatialMetrics = document.getElementById('spatial-metrics');

  if (data.spatial_query && spatialCard) {
    spatialCard.style.display = 'block';
    const fits = data.spatial_query.fits;
    spatialCard.className = fits ? 'spatial-card fit-yes' : 'spatial-card fit-no';
    if (spatialIcon) spatialIcon.textContent = fits ? '✅' : '❌';
    if (spatialTitle) spatialTitle.textContent = fits ? 'Space Available' : 'Insufficient Space';
    if (spatialMsg) spatialMsg.textContent = data.spatial_query.message;
    if (spatialMetrics) {
      spatialMetrics.innerHTML = `
        <div style="display:flex; gap: 1.5rem; margin-top: 0.6rem; font-size: 0.88rem; font-family: monospace;">
          <span>Required: <strong style="color:var(--accent);">${data.spatial_query.required_m}m</strong></span>
          <span>Clearance: <strong style="color:var(--accent-green);">${data.spatial_query.available_m}m</strong></span>
          <span>Margin: <strong style="color:${fits ? 'var(--accent-green)' : 'var(--accent-red)'};">${data.spatial_query.margin_m > 0 ? '+' : ''}${data.spatial_query.margin_m}m</strong></span>
        </div>
      `;
    }
  } else if (spatialCard) {
    spatialCard.style.display = 'none';
  }

  if (!data.match_found) {
    if (noMatchSection) noMatchSection.style.display = 'block';
    if (noMatchMessage) noMatchMessage.textContent = data.message;
    if (resultsSection) resultsSection.style.display = 'none';
    return;
  }

  if (noMatchSection) noMatchSection.style.display = 'none';
  if (resultsSection) resultsSection.style.display = 'block';

  if (resultsTitle) resultsTitle.textContent = `Results for "${data.query}"`;
  if (resultsMeta) {
    resultsMeta.textContent = `${data.result_count} 3D box(es) found • Best score: ${data.max_similarity.toFixed(3)}`;
  }

  resultsList.innerHTML = '';

  data.results.forEach(match => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.onclick = () => highlightInViewer(match);

    const centroid = match.centroid;
    let sizeBadge = '';
    if (match.size_str) {
      sizeBadge = `<div style="color:#7ee787; font-weight:600; margin-top:4px; font-size:0.85rem;">📐 Size: ${match.size_str}</div>`;
    } else if (match.bbox_3d) {
      const dx = Math.abs(match.bbox_3d.max.x - match.bbox_3d.min.x);
      const dy = Math.abs(match.bbox_3d.max.y - match.bbox_3d.min.y);
      const dz = Math.abs(match.bbox_3d.max.z - match.bbox_3d.min.z);
      sizeBadge = `<div style="color:#7ee787; font-weight:600; margin-top:4px; font-size:0.85rem;">📐 Est. Size: ${dx.toFixed(2)}m × ${dy.toFixed(2)}m × ${dz.toFixed(2)}m</div>`;
    }

    card.innerHTML = `
      <div class="result-rank">${match.rank}</div>
      <div class="result-info">
        <div class="result-label">${match.region_id.toUpperCase()}</div>
        <div class="result-details">
          Center: (${centroid.x.toFixed(2)}m, ${centroid.y.toFixed(2)}m, ${centroid.z.toFixed(2)}m)
          ${sizeBadge}
        </div>
      </div>
      <div class="result-score">${match.similarity.toFixed(2)}</div>
    `;

    resultsList.appendChild(card);
  });
}

function highlightInViewer(match) {
  // Switch to viewer panel
  const viewerTab = document.getElementById('nav-viewer');
  if (viewerTab) viewerTab.click();

  if (window.splatViewer) {
    // Focus camera on centroid
    if (match.centroid) {
      window.splatViewer.camera.target = [match.centroid.x, match.centroid.y, match.centroid.z];
      window.splatViewer.distance = 2.0;
      window.splatViewer.updateCamera();
    }

    // Highlight the bounding box in the 3D viewer
    if (window.detectedObjects && window.detectedObjects.length > 0) {
      const matchCenter = match.centroid;
      const boxes = window.detectedObjects.map(obj => {
        const b = JSON.parse(JSON.stringify(obj.obb_3d || obj));
        const dist = Math.hypot(b.center[0] - matchCenter.x, b.center[1] - matchCenter.y, b.center[2] - matchCenter.z);
        b.highlighted = (dist < 0.45 || b.label.toLowerCase() === match.region_id.toLowerCase());
        return b;
      });
      window.splatViewer.setBoundingBoxes(boxes, false);
    } else if (match.obb_3d) {
      const highlightedBox = JSON.parse(JSON.stringify(match.obb_3d));
      highlightedBox.highlighted = true;
      window.splatViewer.setBoundingBoxes([highlightedBox], false);
    }
  }
}

function updateQuickChips(domain = 'indoor_room') {
  const container = document.querySelector('.quick-chips');
  if (!container) return;

  const isVehicle = (domain === 'vehicle');
  const chips = isVehicle ? [
    { query: 'steering wheel', label: '🚗 steering wheel' },
    { query: 'touchscreen display', label: '📟 screen' },
    { query: 'car seat', label: '💺 car seat' },
    { query: 'dashboard', label: '🎛️ dashboard' },
    { query: 'air vent', label: '💨 air vent' },
    { query: 'center console', label: '🕹️ center console' },
    { query: 'Can a 2-meter table fit in this car?', label: '📐 Can a 2-meter table fit?', isSpatial: true },
    { query: 'Can a 6-foot sofa fit here?', label: '📏 Can a 6-foot sofa fit?', isSpatial: true },
  ] : [
    { query: 'chair', label: '🪑 chair' },
    { query: 'sofa', label: '🛋️ sofa' },
    { query: 'table', label: '🪵 table' },
    { query: 'fan', label: '💨 fan' },
    { query: 'ladder', label: '🪜 ladder' },
    { query: 'curtain', label: '🪟 curtain' },
    { query: 'Can a 6-foot sofa fit here?', label: '📐 Can a 6-foot sofa fit here?', isSpatial: true },
    { query: 'Can a 2-meter table fit?', label: '📏 Can a 2-meter table fit?', isSpatial: true },
  ];

  let html = `<span class="chip-label" style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Quick queries (${isVehicle ? 'Vehicle Interior' : 'Indoor Room'}):</span>`;
  chips.forEach(c => {
    const cls = c.isSpatial ? 'chip-btn chip-spatial' : 'chip-btn';
    html += `<button class="${cls}" data-query="${c.query}">${c.label}</button>`;
  });
  container.innerHTML = html;

  container.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.dataset.query;
      if (q && queryInput) {
        queryInput.value = q;
        submitQuery();
      }
    });
  });
}

async function fetchDetectedObjects(sceneId) {
  try {
    const res = await fetch(`${API_BASE}/api/detect/${sceneId}`);
    const data = await res.json();
    if (data && data.objects) {
      window.detectedObjects = data.objects;
      const domain = data.domain || 'indoor_room';
      updateQuickChips(domain);
      if (window.splatViewer && typeof window.splatViewer.setBoundingBoxes === 'function') {
        const boxes = data.objects.map(obj => obj.obb_3d || obj);
        window.splatViewer.setBoundingBoxes(boxes, true);
        
        const infoEl = document.getElementById('viewer-info-text');
        if (infoEl) {
          infoEl.textContent = `Scene: ${sceneId} (${domain}) • ${boxes.length} Metric 3D Boxes • Metric Grid (0.5m)`;
        }
      }
    }
  } catch (err) {
    console.error("Failed to fetch detected objects:", err);
  }
}

// ===== Status Indicator =====
function setStatus(type, text) {
  if (statusText) statusText.textContent = text;
  if (statusDot) {
    statusDot.style.background = {
      ready: 'var(--accent-green)',
      processing: 'var(--accent)',
      uploading: 'var(--accent-amber)',
      error: 'var(--accent-red)',
    }[type] || 'var(--text-muted)';
    statusDot.style.boxShadow = `0 0 8px ${getComputedStyle(statusDot).background}`;
  }
}

// ===== Reprojection Viewer =====
// ===== Reprojection Viewer =====
const reprojectionNav = document.getElementById('nav-reprojection');
let reprojectionTopK = 8;
let reprojectionLabelMode = 'clean';
let isPlayingVideo = false;
let playInterval = null;

if (reprojectionNav) {
  reprojectionNav.addEventListener('click', async () => {
    if (!currentSceneId) return;
    if (reprojectionFrames.length > 0 && lastReprojectionSceneId === currentSceneId) return;

    try {
      const res = await fetch(`${API_BASE}/api/reprojection/${currentSceneId}/info`);
      if (!res.ok) throw new Error('Info failed');
      const data = await res.json();
      reprojectionFrames = data.images || [];
      lastReprojectionSceneId = currentSceneId;
      
      const controls = document.getElementById('reprojection-controls');
      const imgEl = document.getElementById('reprojection-img');
      const placeholder = document.getElementById('reprojection-placeholder');
      const slider = document.getElementById('reproj-frame-slider');

      if (reprojectionFrames.length > 0) {
        if (controls) controls.style.display = 'flex';
        if (imgEl) imgEl.style.display = 'inline-block';
        if (placeholder) placeholder.style.display = 'none';
        if (slider) {
          slider.max = reprojectionFrames.length - 1;
          slider.value = 0;
        }
        currentFrameIndex = 0;
        loadReprojectionFrame();
      } else {
        if (placeholder) placeholder.style.display = 'block';
        if (imgEl) imgEl.style.display = 'none';
        if (controls) controls.style.display = 'none';
      }
    } catch (err) {
      console.error("Failed to load reprojection info:", err);
    }
  });
}

function loadReprojectionFrame() {
  if (reprojectionFrames.length === 0) return;
  const frame = reprojectionFrames[currentFrameIndex];
  const imgEl = document.getElementById('reprojection-img');
  const counter = document.getElementById('frame-counter');
  const slider = document.getElementById('reproj-frame-slider');
  
  if (slider) {
    slider.value = currentFrameIndex;
  }
  if (imgEl) {
    imgEl.src = `${API_BASE}/api/reprojection/${currentSceneId}/frame/${frame.id}?top_k=${reprojectionTopK}&label_mode=${reprojectionLabelMode}&t=${Date.now()}`;
  }
  if (counter) {
    counter.textContent = `Frame: ${currentFrameIndex + 1} / ${reprojectionFrames.length}`;
  }
}

const prevFrameBtn = document.getElementById('btn-prev-frame');
if (prevFrameBtn) {
  prevFrameBtn.addEventListener('click', () => {
    if (reprojectionFrames.length === 0) return;
    stopPlayback();
    if (currentFrameIndex > 0) {
      currentFrameIndex--;
    } else {
      currentFrameIndex = reprojectionFrames.length - 1; // wrap around
    }
    loadReprojectionFrame();
  });
}

const nextFrameBtn = document.getElementById('btn-next-frame');
if (nextFrameBtn) {
  nextFrameBtn.addEventListener('click', () => {
    if (reprojectionFrames.length === 0) return;
    stopPlayback();
    if (currentFrameIndex < reprojectionFrames.length - 1) {
      currentFrameIndex++;
    } else {
      currentFrameIndex = 0; // wrap around
    }
    loadReprojectionFrame();
  });
}

const frameSlider = document.getElementById('reproj-frame-slider');
if (frameSlider) {
  frameSlider.addEventListener('input', (e) => {
    stopPlayback();
    currentFrameIndex = parseInt(e.target.value) || 0;
    loadReprojectionFrame();
  });
}

const topkSelect = document.getElementById('reproj-topk-select');
if (topkSelect) {
  topkSelect.addEventListener('change', (e) => {
    reprojectionTopK = parseInt(e.target.value) || 8;
    loadReprojectionFrame();
  });
}

const labelModeSelect = document.getElementById('reproj-label-mode');
if (labelModeSelect) {
  labelModeSelect.addEventListener('change', (e) => {
    reprojectionLabelMode = e.target.value || 'clean';
    loadReprojectionFrame();
  });
}

const playBtn = document.getElementById('btn-play-frame');
function stopPlayback() {
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
  }
  isPlayingVideo = false;
  if (playBtn) playBtn.textContent = '▶ Play';
}

if (playBtn) {
  playBtn.addEventListener('click', () => {
    if (reprojectionFrames.length <= 1) return;
    if (isPlayingVideo) {
      stopPlayback();
    } else {
      isPlayingVideo = true;
      playBtn.textContent = '⏸ Pause';
      playInterval = setInterval(() => {
        currentFrameIndex = (currentFrameIndex + 1) % reprojectionFrames.length;
        loadReprojectionFrame();
      }, 150); // ~6.6 FPS smooth replay
    }
  });
}
