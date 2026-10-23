/**
 * SemantiSplat — 3D Gaussian Splat WebGL Viewer
 *
 * Renders a trained .ply Gaussian splat scene as a point cloud in the browser
 * with support for highlighting matched regions from text queries.
 */

class SplatViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.gl = null;
    this.gaussians = null;
    this.highlightedIndices = new Set();
    this.isQueryActive = false;

    // Camera (OpenCV coordinates: -Y is world UP, +Y is ground, +Z is forward)
    this.camera = {
      position: [0, -1.2, 0.5],
      target: [0, 0, 2.5],
      up: [0, -1, 0],
      fov: 60,
      near: 0.05,
      far: 1000,
    };

    // Orbit & Pan interaction
    this.isDragging = false;
    this.isPanning = false;
    this.lastMouse = { x: 0, y: 0 };
    this.rotation = { theta: 0.45, phi: -0.35 }; // elevated perspective
    this.distance = 4.0;
    this.sceneRadius = 3.0; // will be computed from bounding boxes / scene
    this.pointSizeMultiplier = 1.0;

    // 3D Object Detection & Visualization States
    this.showBoundingBoxes = true;
    this.showPointCloud = false; // Clean structured 3D boxes over raw noisy points
    this.isTopDown = false;
    this.boundingBoxes = [];
    this.labelsOverlay = null;

    // Touch
    this.lastTouchDist = 0;
    this.lastTouchCenter = { x: 0, y: 0 };

    this.program = null;
    this.buffers = {};
    this.locations = {};

    // Grid / axis helpers
    this.helperProgram = null;
    this.helperLocations = {};
    this.gridBuffer = null;
    this.gridVertCount = 0;
    this.axisBuffer = null;
    this.axisColorBuffer = null;

    this._resizeScheduled = false;

    this.init();
  }

  init() {
    if (!this.canvas) return;

    this.gl = this.canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
    });

    if (!this.gl) {
      console.warn('WebGL2 not available. 3D viewer disabled.');
      this.showFallback();
      return;
    }

    this.setupCanvas();
    this.setupEventListeners();
    this.setupWebGL();
    this.setupHelpers();
    this.buildGrid(10, 1); // default grid until scene loads
    this.render();
  }

  /* ------------------------------------------------------------------
   * Canvas Sizing — handles the hidden-panel problem
   * ----------------------------------------------------------------*/
  setupCanvas() {
    const resize = () => {
      const parent = this.canvas.parentElement;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      if (w === 0 || h === 0) return; // panel is hidden, skip

      this.canvas.width = w * window.devicePixelRatio;
      this.canvas.height = h * window.devicePixelRatio;
      this.canvas.style.width = w + 'px';
      this.canvas.style.height = h + 'px';
      if (this.gl) {
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      }

      // Auto-frame bounding boxes on initial view display once size is valid
      if (!this._initialFramedDone && this.boundingBoxes && this.boundingBoxes.length > 0) {
        this._initialFramedDone = true;
        this.frameBoundingBoxes();
      }
    };

    resize();
    window.addEventListener('resize', resize);

    // Observe panel visibility changes so we resize when the viewer tab activates
    const observer = new MutationObserver(() => {
      if (!this._resizeScheduled) {
        this._resizeScheduled = true;
        requestAnimationFrame(() => {
          resize();
          this._resizeScheduled = false;
        });
      }
    });

    // Watch the panel element that wraps us
    const panel = this.canvas.closest('.panel');
    if (panel) {
      observer.observe(panel, { attributes: true, attributeFilter: ['class', 'style'] });
    }
    // Also observe any parent up to body
    if (panel && panel.parentElement) {
      observer.observe(panel.parentElement, { childList: true, subtree: false, attributes: true });
    }

    // Extra: force a resize whenever our panel becomes active (click event delegation)
    document.addEventListener('click', (e) => {
      const tab = e.target.closest('[data-panel="viewer"]');
      if (tab) {
        // Small delay to let CSS display:block propagate
        setTimeout(resize, 50);
        setTimeout(resize, 200);
      }
    });
  }

  /* ------------------------------------------------------------------
   * Camera Orientation Vectors for Pan & Orbit
   * ----------------------------------------------------------------*/
  getCameraVectors() {
    // Direction from camera position to target
    const fx = this.camera.target[0] - this.camera.position[0];
    const fy = this.camera.target[1] - this.camera.position[1];
    const fz = this.camera.target[2] - this.camera.position[2];
    const lenF = Math.hypot(fx, fy, fz) || 1e-6;
    const fwdX = fx / lenF, fwdY = fy / lenF, fwdZ = fz / lenF;

    // Camera view backward vector: z = -fwd
    const zx = -fwdX, zy = -fwdY, zz = -fwdZ;

    // World up in OpenCV space (-Y is UP)
    // Horizontal right vector = cross([0, -1, 0], z) = [-zz, 0, zx]
    let rx = -zz;
    let rz = zx;
    const lenR = Math.hypot(rx, rz) || 1e-6;
    rx /= lenR;
    rz /= lenR;

    // Screen up vector in world coordinates = cross(z, right)
    const ux = zy * rz;
    const uy = zz * rx - zx * rz;
    const uz = -zy * rx;
    const lenU = Math.hypot(ux, uy, uz) || 1e-6;

    return {
      forward: [fwdX, fwdY, fwdZ],
      right: [rx, 0.0, rz],
      up: [ux / lenU, uy / lenU, uz / lenU],
    };
  }

  /* ------------------------------------------------------------------
   * Orbit, Pan, and Zoom Implementations
   * ----------------------------------------------------------------*/
  orbit(dx, dy) {
    const orbitSpeed = 0.005;
    // Turntable azimuth rotation around target
    this.rotation.theta -= dx * orbitSpeed;

    // Elevation angle phi (clamped smoothly to avoid pole singularity flip)
    // Negative phi = elevated view looking down; positive = looking slightly up
    this.rotation.phi = Math.max(
      -Math.PI / 2 + 0.08,
      Math.min(0.55, this.rotation.phi + dy * orbitSpeed)
    );
    this.updateCamera();
  }

  pan(dx, dy) {
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const canvasH = (this.canvas && this.canvas.clientHeight) || 600;
    const worldUnitsPerPixel = (2.0 * this.distance * Math.tan(fovRad / 2.0)) / canvasH;

    const { right, up } = this.getCameraVectors();

    // 1:1 pixel pinning:
    // Moving mouse right (dx > 0) moves target in -right direction so scene moves right
    // Moving mouse up (dy < 0) moves target in +up direction so scene moves up
    const deltaX = (-dx * right[0] + dy * up[0]) * worldUnitsPerPixel;
    const deltaY = (-dx * right[1] + dy * up[1]) * worldUnitsPerPixel;
    const deltaZ = (-dx * right[2] + dy * up[2]) * worldUnitsPerPixel;

    this.camera.target[0] += deltaX;
    this.camera.target[1] += deltaY;
    this.camera.target[2] += deltaZ;
    this.updateCamera();
  }

  zoom(deltaY, deltaMode = 0) {
    let delta = deltaY;
    if (deltaMode === 1) delta *= 24; // lines
    else if (deltaMode === 2) delta *= 200; // pages
    delta = Math.max(-100, Math.min(100, delta));

    const zoomSpeed = 0.0018;
    const factor = Math.exp(delta * zoomSpeed);

    const minD = 0.2;
    const maxD = Math.max(30.0, this.sceneRadius * 8.0);
    this.distance = Math.max(minD, Math.min(maxD, this.distance * factor));
    this.updateCamera();
  }

  /* ------------------------------------------------------------------
   * Event Listeners — container-level orbit, zoom, pan (mouse, touch & pointers)
   * ----------------------------------------------------------------*/
  setupEventListeners() {
    const container = this.container || this.canvas.parentElement || this.canvas;

    let isPointerDown = false;
    let downPos = { x: 0, y: 0 };
    let downTime = 0;
    let activePointerButton = 0;
    let clickTargetBadge = null;

    // --- Pointer down on viewer container ---
    container.addEventListener('pointerdown', (e) => {
      // Don't intercept clicks on viewer top toolbar controls
      if (e.target.closest('.viewer-controls') || e.target.closest('button')) {
        return;
      }

      isPointerDown = true;
      downPos = { x: e.clientX, y: e.clientY };
      downTime = Date.now();
      activePointerButton = e.button;
      this.lastMouse = { x: e.clientX, y: e.clientY };
      clickTargetBadge = e.target.closest('.viewer-3d-badge');

      if (e.pointerType === 'mouse' && container.setPointerCapture) {
        try {
          container.setPointerCapture(e.pointerId);
        } catch (_) {}
      }
      e.preventDefault();
    });

    window.addEventListener('pointermove', (e) => {
      if (!isPointerDown) return;
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      const totalMoved = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);

      if (totalMoved > 3) {
        // Active drag motion
        if (!this.isDragging && !this.isPanning) {
          if (activePointerButton === 0 && !e.shiftKey) {
            this.isDragging = true;
            this.isPanning = false;
          } else {
            this.isPanning = true;
            this.isDragging = false;
          }
          container.classList.add('is-dragging');
        }

        if (this.isDragging) {
          this.orbit(dx, dy);
        } else if (this.isPanning) {
          this.pan(dx, dy);
        }

        this.lastMouse = { x: e.clientX, y: e.clientY };
      }
    });

    window.addEventListener('pointerup', (e) => {
      if (!isPointerDown) return;
      const totalMoved = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
      const isQuickClick = totalMoved < 4 && (Date.now() - downTime < 450);

      // If user quickly clicked on a 3D badge without dragging, focus that box
      if (isQuickClick && clickTargetBadge) {
        const boxIdx = parseInt(clickTargetBadge.dataset.boxIndex);
        if (!isNaN(boxIdx)) {
          this.focusBox(boxIdx);
        }
      }

      isPointerDown = false;
      this.isDragging = false;
      this.isPanning = false;
      container.classList.remove('is-dragging');
      clickTargetBadge = null;
    });

    window.addEventListener('pointercancel', () => {
      isPointerDown = false;
      this.isDragging = false;
      this.isPanning = false;
      container.classList.remove('is-dragging');
      clickTargetBadge = null;
    });

    // Prevent context menu on right click so right-drag pan works seamlessly
    container.addEventListener('contextmenu', (e) => e.preventDefault());

    // --- Wheel = smooth zoom centered on target ---
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom(e.deltaY, e.deltaMode);
    }, { passive: false });

    // --- Pinch-to-zoom on touch screens ---
    container.addEventListener('touchstart', (e) => {
      if (e.touches.length === 2) {
        this.lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        if (this.lastTouchDist > 0 && dist > 0) {
          const ratio = this.lastTouchDist / dist;
          const minD = 0.2;
          const maxD = Math.max(30.0, this.sceneRadius * 8.0);
          this.distance = Math.max(minD, Math.min(maxD, this.distance * ratio));
          this.updateCamera();
        }
        this.lastTouchDist = dist;
      }
    }, { passive: true });

    // Reset camera button
    const resetBtn = document.getElementById('btn-reset-camera');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => this.resetCamera());
    }

    // Toggle Top-Down / 3D Orbit view button
    const toggleViewBtn = document.getElementById('btn-toggle-view');
    if (toggleViewBtn) {
      toggleViewBtn.addEventListener('click', () => this.toggleViewMode());
    }

    // Toggle 3D Bounding Boxes button
    const toggleBoxesBtn = document.getElementById('btn-toggle-boxes');
    if (toggleBoxesBtn) {
      toggleBoxesBtn.addEventListener('click', () => this.toggleBoxes());
    }

    // Toggle Raw Point Cloud button
    const togglePointsBtn = document.getElementById('btn-toggle-points');
    if (togglePointsBtn) {
      togglePointsBtn.addEventListener('click', () => this.togglePoints());
    }

    // Screenshot button
    const screenshotBtn = document.getElementById('btn-screenshot');
    if (screenshotBtn) {
      screenshotBtn.addEventListener('click', () => this.takeScreenshot());
    }
  }

  updateCamera() {
    const x = this.distance * Math.cos(this.rotation.phi) * Math.sin(this.rotation.theta);
    const y = this.distance * Math.sin(this.rotation.phi);
    const z = this.distance * Math.cos(this.rotation.phi) * Math.cos(this.rotation.theta);
    this.camera.position = [
      this.camera.target[0] + x,
      this.camera.target[1] + y,
      this.camera.target[2] + z,
    ];

    // Maintain dynamic clipping planes proportional to distance
    this.camera.near = Math.max(0.01, this.distance * 0.005);
    this.camera.far = Math.max(250.0, this.distance * 30.0 + 100.0);
  }

  frameBoundingBoxes() {
    if (!this.boundingBoxes || this.boundingBoxes.length === 0) {
      if (this.gaussians && this.gaussians.count > 0) {
        this.centerCameraOnScene();
      } else {
        this.camera.target = [0, 0, 2.5];
        this.distance = 3.8;
        this.rotation = { theta: 0.45, phi: -0.35 };
        this.updateCamera();
      }
      return;
    }

    // Robust centroid & outlier rejection for video sequences
    const boxes = this.boundingBoxes;
    const centers = boxes.map(b => b.center);
    const xs = centers.map(c => c[0]).sort((a, b) => a - b);
    const ys = centers.map(c => c[1]).sort((a, b) => a - b);
    const zs = centers.map(c => c[2]).sort((a, b) => a - b);

    const medX = xs[Math.floor(xs.length * 0.5)];
    const medY = ys[Math.floor(ys.length * 0.5)];
    const medZ = zs[Math.floor(zs.length * 0.5)];

    // Filter out extreme spatial outliers (> 12m from median)
    const inliers = boxes.filter(b => {
      const d = Math.hypot(b.center[0] - medX, b.center[1] - medY, b.center[2] - medZ);
      return d < 12.0;
    });

    const activeBoxes = inliers.length > 0 ? inliers : boxes;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    activeBoxes.forEach(b => {
      const c = b.center;
      const ex = b.extents || [0.35, 0.35, 0.35];
      minX = Math.min(minX, c[0] - ex[0]);
      maxX = Math.max(maxX, c[0] + ex[0]);
      minY = Math.min(minY, c[1] - ex[1]);
      maxY = Math.max(maxY, c[1] + ex[1]);
      minZ = Math.min(minZ, c[2] - ex[2]);
      maxZ = Math.max(maxZ, c[2] + ex[2]);
    });

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    const spanX = Math.max(0.6, maxX - minX);
    const spanY = Math.max(0.6, maxY - minY);
    const spanZ = Math.max(0.6, maxZ - minZ);
    const radius = Math.hypot(spanX, spanY, spanZ) / 2;

    this.sceneRadius = Math.max(radius, 1.2);
    this.camera.target = [cx, cy, cz];

    // Framing distance to fit all boxes in view comfortably with 25% padding
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const fitDist = (this.sceneRadius / Math.sin(fovRad / 2)) * 1.25;
    this.distance = Math.max(2.0, Math.min(18.0, fitDist));

    // Natural angled overview
    this.rotation = { theta: 0.45, phi: -0.35 };
    this.isTopDown = false;
    const btn = document.getElementById('btn-toggle-view');
    if (btn) {
      btn.textContent = '⬇️ Top-Down';
      btn.classList.remove('active-toggle');
    }

    // Rebuild floor grid under the boxes (floor is maxY in OpenCV coords)
    const floorY = maxY + 0.02;
    const gridSize = Math.max(5, Math.ceil(this.sceneRadius * 1.5));
    this.buildGrid(gridSize, 0.5, floorY, cx, cz);

    this.updateCamera();
  }

  resetCamera() {
    this.frameBoundingBoxes();
  }

  /* ------------------------------------------------------------------
   * WebGL Shader Programs
   * ----------------------------------------------------------------*/
  setupWebGL() {
    const gl = this.gl;

    /* ----- Point cloud shaders ----- */
    const vsSource = `#version 300 es
      uniform mat4 u_viewProjection;
      uniform float u_pointSizeMultiplier;
      uniform float u_screenHeight;
      in vec3 a_position;
      in vec3 a_color;
      in vec3 a_scale;
      in float a_opacity;
      in float a_highlight;
      out vec3 v_color;
      out float v_opacity;
      out float v_highlight;
      void main() {
        gl_Position = u_viewProjection * vec4(a_position, 1.0);
        
        // Use the maximum scale to determine point size
        float maxScale = max(a_scale.x, max(a_scale.y, a_scale.z));
        
        // Project world size to screen size
        float depth = max(gl_Position.w, 0.001);
        float pointSize = (maxScale * 2.5 * u_screenHeight) / depth;
        
        gl_PointSize = clamp(pointSize * u_pointSizeMultiplier, 2.0, 200.0);
        
        v_color = a_color;
        v_opacity = a_opacity;
        v_highlight = a_highlight;
      }
    `;

    const fsSource = `#version 300 es
      precision highp float;
      in vec3 v_color;
      in float v_opacity;
      in float v_highlight;
      out vec4 outColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float r2 = dot(coord, coord);
        if (r2 > 0.25) discard;

        // Gaussian falloff
        float alpha = exp(-4.0 * r2) * v_opacity;

        vec3 color = v_color;
        if (v_highlight > 0.5) {
           color = mix(color, vec3(1.0, 1.0, 0.0), 0.8); // Bright Yellow highlight
        } else if (v_highlight < -0.5) {
           color *= 0.2; // Dim unselected
           alpha *= 0.2;
        }
        outColor = vec4(color, alpha);
      }
    `;

    const compileShader = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        return null;
      }
      return shader;
    };

    const vs = compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Shader link error:', gl.getProgramInfoLog(this.program));
    }

    this.locations = {
      position: gl.getAttribLocation(this.program, 'a_position'),
      color: gl.getAttribLocation(this.program, 'a_color'),
      scale: gl.getAttribLocation(this.program, 'a_scale'),
      opacity: gl.getAttribLocation(this.program, 'a_opacity'),
      highlight: gl.getAttribLocation(this.program, 'a_highlight'),
      viewProjection: gl.getUniformLocation(this.program, 'u_viewProjection'),
      pointSizeMultiplier: gl.getUniformLocation(this.program, 'u_pointSizeMultiplier'),
      screenHeight: gl.getUniformLocation(this.program, 'u_screenHeight'),
    };

    this.buffers = {
      position: gl.createBuffer(),
      color: gl.createBuffer(),
      scale: gl.createBuffer(),
      opacity: gl.createBuffer(),
      highlight: gl.createBuffer(),
    };
  }

  /* ----- Helper lines shader (grid + axes) ----- */
  setupHelpers() {
    const gl = this.gl;

    const vsLine = `#version 300 es
      uniform mat4 u_viewProjection;
      in vec3 a_position;
      in vec3 a_color;
      out vec3 v_color;
      void main() {
        gl_Position = u_viewProjection * vec4(a_position, 1.0);
        v_color = a_color;
      }
    `;
    const fsLine = `#version 300 es
      precision highp float;
      in vec3 v_color;
      out vec4 outColor;
      void main() {
        outColor = vec4(v_color, 0.35);
      }
    `;

    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, vsLine);
    const fs = compile(gl.FRAGMENT_SHADER, fsLine);
    this.helperProgram = gl.createProgram();
    gl.attachShader(this.helperProgram, vs);
    gl.attachShader(this.helperProgram, fs);
    gl.linkProgram(this.helperProgram);

    this.helperLocations = {
      position: gl.getAttribLocation(this.helperProgram, 'a_position'),
      color: gl.getAttribLocation(this.helperProgram, 'a_color'),
      viewProjection: gl.getUniformLocation(this.helperProgram, 'u_viewProjection'),
    };
  }

  buildGrid(halfSize, step, groundY = 0, centerX = 0, centerZ = 0) {
    const gl = this.gl;
    const lines = [];
    const colors = [];
    const gridColor = [0.18, 0.24, 0.35];
    const gridCenterColor = [0.35, 0.45, 0.65];

    const minX = centerX - halfSize;
    const maxX = centerX + halfSize;
    const minZ = centerZ - halfSize;
    const maxZ = centerZ + halfSize;

    for (let x = minX; x <= maxX; x += step) {
      lines.push(x, groundY, minZ, x, groundY, maxZ);
      const isCenter = Math.abs(x - centerX) < step * 0.5;
      const col = isCenter ? gridCenterColor : gridColor;
      colors.push(...col, ...col);
    }

    for (let z = minZ; z <= maxZ; z += step) {
      lines.push(minX, groundY, z, maxX, groundY, z);
      const isCenter = Math.abs(z - centerZ) < step * 0.5;
      const col = isCenter ? gridCenterColor : gridColor;
      colors.push(...col, ...col);
    }

    this.gridVertCount = lines.length / 3;
    if (!this.gridBuffer) this.gridBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);

    if (!this.gridColorBuffer) this.gridColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    // Axes located at (centerX, groundY, centerZ)
    const axLen = halfSize * 0.35;
    const axVerts = [
      centerX, groundY, centerZ, centerX + axLen, groundY, centerZ, // X
      centerX, groundY, centerZ, centerX, groundY - axLen, centerZ, // -Y (up into room)
      centerX, groundY, centerZ, centerX, groundY, centerZ + axLen, // Z
    ];
    const axColors = [
      1.0, 0.25, 0.25, 1.0, 0.25, 0.25, // X red
      0.25, 1.0, 0.25, 0.25, 1.0, 0.25, // Y green
      0.35, 0.55, 1.0, 0.35, 0.55, 1.0, // Z blue
    ];

    if (!this.axisBuffer) this.axisBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.axisBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(axVerts), gl.STATIC_DRAW);

    if (!this.axisColorBuffer) this.axisColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.axisColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(axColors), gl.STATIC_DRAW);
  }

  /* ------------------------------------------------------------------
   * Scene Loading
   * ----------------------------------------------------------------*/
  async loadScene(url) {
    const viewerInfo = document.getElementById('viewer-info');
    if (viewerInfo) viewerInfo.innerHTML = '<span>Loading scene...</span>';

    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('Scene not found');
      const buffer = await res.arrayBuffer();

      this.gaussians = this.parsePLY(buffer);
      this.updateHighlightBuffer();

      if (viewerInfo && this.gaussians) {
        viewerInfo.innerHTML = `<span>${this.gaussians.count.toLocaleString()} Points loaded — Drag to orbit, scroll to zoom, right-drag to pan</span>`;
      }
      
      // Compute bounding box to center camera
      this.centerCameraOnScene();

    } catch (err) {
      console.error('Failed to load scene:', err);
      if (viewerInfo) {
        viewerInfo.innerHTML = '<span>Failed to load scene</span>';
      }
    }
  }
  
  centerCameraOnScene() {
    if (!this.gaussians || this.gaussians.count === 0) return;
    const pos = this.gaussians.positions;
    const N = this.gaussians.count;

    // Compute robust center (median) and range (5th–95th percentile)
    let xs = [], ys = [], zs = [];
    for (let i = 0; i < N; i++) {
      xs.push(pos[i * 3]);
      ys.push(pos[i * 3 + 1]);
      zs.push(pos[i * 3 + 2]);
    }
    xs.sort((a, b) => a - b);
    ys.sort((a, b) => a - b);
    zs.sort((a, b) => a - b);
    
    const p5  = Math.floor(N * 0.05);
    const p50 = Math.floor(N * 0.50);
    const p95 = Math.floor(N * 0.95);

    // Center on the median for robustness against outliers
    const cx = xs[p50];
    const cy = ys[p50];
    const cz = zs[p50];
    this.camera.target = [cx, cy, cz];
    
    // Scene size from 5th–95th percentile spread
    const rangeX = xs[p95] - xs[p5];
    const rangeY = ys[p95] - ys[p5];
    const rangeZ = zs[p95] - zs[p5];
    this.sceneRadius = Math.max(rangeX, rangeY, rangeZ, 0.5);

    // Position camera at 1.5× the scene radius for a good overview
    this.distance = this.sceneRadius * 1.5;
    
    // Set a nice initial viewing angle (slightly above and to the side)
    this.rotation = { theta: 0.5, phi: -0.35 };

    // Adjust near/far planes to scene scale
    this.camera.near = this.sceneRadius * 0.001;
    this.camera.far = this.sceneRadius * 50;

    // Rebuild the grid to match scene scale
    const gridHalf = Math.ceil(this.sceneRadius * 0.75);
    const gridStep = Math.max(1, Math.round(gridHalf / 10));
    this.buildGrid(gridHalf, gridStep);

    this.updateCamera();

    console.log(`Camera centered: target=(${cx.toFixed(2)}, ${cy.toFixed(2)}, ${cz.toFixed(2)}), ` +
                `distance=${this.distance.toFixed(2)}, sceneRadius=${this.sceneRadius.toFixed(2)}`);
  }

  /* ------------------------------------------------------------------
   * PLY Parser
   * ----------------------------------------------------------------*/
  parsePLY(buffer) {
    const textDecoder = new TextDecoder();
    const view = new Uint8Array(buffer);
    let headerEnd = -1;

    // Search for "\nend_header\n"
    for (let i = 0; i < view.length - 11; i++) {
      if (view[i]   === 0x0A &&  // \n
          view[i+1] === 0x65 &&  // e
          view[i+2] === 0x6E &&  // n
          view[i+3] === 0x64 &&  // d
          view[i+4] === 0x5F &&  // _
          view[i+5] === 0x68 &&  // h
          view[i+6] === 0x65 &&  // e
          view[i+7] === 0x61 &&  // a
          view[i+8] === 0x64 &&  // d
          view[i+9] === 0x65 &&  // e
          view[i+10]=== 0x72 &&  // r
          view[i+11]=== 0x0A) {  // \n
        headerEnd = i + 12;
        break;
      }
    }

    if (headerEnd === -1) {
      console.error("Invalid PLY file: could not find end_header");
      return null;
    }

    const headerStr = textDecoder.decode(view.subarray(0, headerEnd));
    const lines = headerStr.split('\n');
    let vertexCount = 0;
    let properties = [];

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('element vertex')) {
        vertexCount = parseInt(trimmed.split(/\s+/)[2]);
      } else if (trimmed.startsWith('property')) {
        const parts = trimmed.split(/\s+/);
        properties.push({ type: parts[1], name: parts[2] });
      }
    }

    // Type → byte size mapping (covers standard PLY type names)
    const typeSizes = {
      'char': 1, 'int8': 1,
      'uchar': 1, 'uint8': 1,
      'short': 2, 'int16': 2,
      'ushort': 2, 'uint16': 2,
      'int': 4, 'int32': 4,
      'uint': 4, 'uint32': 4,
      'float': 4, 'float32': 4, 'f4': 4,
      'double': 8, 'float64': 8,
    };

    let vertexSize = 0;
    let offsets = {};

    properties.forEach(p => {
      offsets[p.name] = vertexSize;
      vertexSize += typeSizes[p.type] || 4;
    });

    console.log(`PLY: ${vertexCount} vertices, ${properties.length} properties, ${vertexSize} bytes/vertex`);
    console.log(`PLY property offsets:`, offsets);

    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);
    const scales = new Float32Array(vertexCount * 3);
    const opacities = new Float32Array(vertexCount);
    
    const dataView = new DataView(buffer, headerEnd);
    const SH_C0 = 0.28209479177387814;

    const hasXYZ = offsets.x !== undefined && offsets.y !== undefined && offsets.z !== undefined;
    const hasSH  = offsets.sh_0_0 !== undefined && offsets.sh_0_1 !== undefined && offsets.sh_0_2 !== undefined;
    const hasRGB = offsets.red !== undefined && offsets.green !== undefined && offsets.blue !== undefined;
    const hasScale = offsets.scale_0 !== undefined;
    const hasOpacity = offsets.opacity !== undefined;

    if (!hasXYZ) {
      console.error("PLY file missing x/y/z properties");
      return null;
    }

    for (let i = 0; i < vertexCount; i++) {
      const offset = i * vertexSize;
      positions[i * 3 + 0] = dataView.getFloat32(offset + offsets.x, true);
      positions[i * 3 + 1] = dataView.getFloat32(offset + offsets.y, true);
      positions[i * 3 + 2] = dataView.getFloat32(offset + offsets.z, true);
      
      if (hasScale) {
        scales[i * 3 + 0] = Math.exp(dataView.getFloat32(offset + offsets.scale_0, true));
        scales[i * 3 + 1] = Math.exp(dataView.getFloat32(offset + offsets.scale_1, true));
        scales[i * 3 + 2] = Math.exp(dataView.getFloat32(offset + offsets.scale_2, true));
      } else {
        scales[i * 3 + 0] = 0.01; scales[i * 3 + 1] = 0.01; scales[i * 3 + 2] = 0.01;
      }
      
      if (hasOpacity) {
        const op = dataView.getFloat32(offset + offsets.opacity, true);
        opacities[i] = 1.0 / (1.0 + Math.exp(-op)); // sigmoid
      } else {
        opacities[i] = 1.0;
      }

      if (hasSH) {
        // Decode SH DC component to color: color = 0.5 + SH_C0 * sh
        const sh0 = dataView.getFloat32(offset + offsets.sh_0_0, true);
        const sh1 = dataView.getFloat32(offset + offsets.sh_0_1, true);
        const sh2 = dataView.getFloat32(offset + offsets.sh_0_2, true);
        colors[i * 3 + 0] = Math.max(0, Math.min(1, 0.5 + SH_C0 * sh0));
        colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * sh1));
        colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * sh2));
      } else if (hasRGB) {
        // Direct RGB as uchar [0-255]
        const rType = properties.find(p => p.name === 'red')?.type;
        if (rType === 'uchar' || rType === 'uint8') {
          const r = dataView.getUint8(offset + offsets.red);
          const g = dataView.getUint8(offset + offsets.green);
          const b = dataView.getUint8(offset + offsets.blue);
          colors[i * 3 + 0] = r / 255;
          colors[i * 3 + 1] = g / 255;
          colors[i * 3 + 2] = b / 255;
        } else {
          colors[i * 3 + 0] = dataView.getFloat32(offset + offsets.red, true);
          colors[i * 3 + 1] = dataView.getFloat32(offset + offsets.green, true);
          colors[i * 3 + 2] = dataView.getFloat32(offset + offsets.blue, true);
        }
      } else {
        // Fallback: white
        colors[i * 3 + 0] = 0.8;
        colors[i * 3 + 1] = 0.8;
        colors[i * 3 + 2] = 0.8;
      }
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
    gl.bufferData(gl.ARRAY_BUFFER, scales, gl.STATIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.opacity);
    gl.bufferData(gl.ARRAY_BUFFER, opacities, gl.STATIC_DRAW);

    return { count: vertexCount, positions, colors };
  }

  /* ------------------------------------------------------------------
   * Highlighting
   * ----------------------------------------------------------------*/
  updateHighlightBuffer() {
    if (!this.gaussians || !this.gl) return;
    
    const count = this.gaussians.count;
    const highlights = new Float32Array(count);
    
    if (this.isQueryActive) {
      for (let i = 0; i < count; i++) {
        highlights[i] = this.highlightedIndices.has(i) ? 1.0 : -1.0;
      }
    } else {
      highlights.fill(0.0);
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.highlight);
    gl.bufferData(gl.ARRAY_BUFFER, highlights, gl.DYNAMIC_DRAW);
  }

  highlight(indices, centroid) {
    this.highlightedIndices = new Set(indices);
    this.isQueryActive = true;
    this.updateHighlightBuffer();

    if (centroid) {
      this.camera.target = [centroid.x, centroid.y, centroid.z];
      this.distance = this.sceneRadius * 0.3;
      this.updateCamera();
    }
  }

  clearHighlight() {
    this.highlightedIndices.clear();
    this.isQueryActive = false;
    this.updateHighlightBuffer();
  }
  
  toggleViewMode() {
    this.isTopDown = !this.isTopDown;
    const btn = document.getElementById('btn-toggle-view');
    if (btn) {
      btn.textContent = this.isTopDown ? '🌐 3D Orbit' : '⬇️ Top-Down';
      btn.classList.toggle('active-toggle', this.isTopDown);
    }
    if (this.isTopDown) {
      this.rotation.theta = 0.0;
      this.rotation.phi = -Math.PI / 2 + 0.001;
      this.distance = Math.max(3.0, this.sceneRadius * 1.8);
    } else {
      this.rotation.theta = 0.45;
      this.rotation.phi = -0.35;
      this.distance = Math.max(2.5, this.sceneRadius * 1.2);
    }
    this.updateCamera();
  }

  toggleBoxes() {
    this.showBoundingBoxes = !this.showBoundingBoxes;
    const btn = document.getElementById('btn-toggle-boxes');
    if (btn) {
      btn.textContent = this.showBoundingBoxes ? '📦 3D Boxes: ON' : '📦 3D Boxes: OFF';
      btn.classList.toggle('active-toggle', this.showBoundingBoxes);
    }
  }

  togglePoints() {
    this.showPointCloud = !this.showPointCloud;
    const btn = document.getElementById('btn-toggle-points');
    if (btn) {
      btn.textContent = this.showPointCloud ? '☁️ Points: ON' : '☁️ Points: OFF';
      btn.classList.toggle('active-toggle', this.showPointCloud);
    }
  }

  takeScreenshot() {
    if (!this.canvas) return;
    const link = document.createElement('a');
    link.download = `semantisplat_3d_view_${Date.now()}.png`;
    link.href = this.canvas.toDataURL('image/png');
    link.click();
  }

  focusBox(index) {
    if (!this.boundingBoxes || !this.boundingBoxes[index]) return;
    const b = this.boundingBoxes[index];
    const c = b.center;
    const ex = b.extents || [0.4, 0.4, 0.4];
    const boxRadius = Math.hypot(ex[0], ex[1], ex[2]);
    this.camera.target = [c[0], c[1], c[2]];
    this.distance = Math.max(1.2, boxRadius * 2.8);
    this.updateCamera();
  }

  setBoundingBoxes(boxes, autoCenter = true) {
    if (!this.gl) return;
    const gl = this.gl;
    this.boundingBoxes = boxes || [];

    if (!boxes || boxes.length === 0) {
      this.boxVertCount = 0;
      return;
    }

    const lines = [];
    const colors = [];

    boxes.forEach(box => {
      const c = box.center;
      const axes = box.axes || [[1,0,0],[0,1,0],[0,0,1]];
      const ex = box.extents || [0.4, 0.4, 0.4];

      const v = [];
      for (let i = 0; i < 8; i++) {
        const x = (i & 1) ? ex[0] : -ex[0];
        const y = (i & 2) ? ex[1] : -ex[1];
        const z = (i & 4) ? ex[2] : -ex[2];

        const vx = c[0] + x*axes[0][0] + y*axes[0][1] + z*axes[0][2];
        const vy = c[1] + x*axes[1][0] + y*axes[1][1] + z*axes[1][2];
        const vz = c[2] + x*axes[2][0] + y*axes[2][1] + z*axes[2][2];
        v.push([vx, vy, vz]);
      }

      const edges = [
        0,1, 1,3, 3,2, 2,0, // Bottom
        4,5, 5,7, 7,6, 6,4, // Top
        0,4, 1,5, 2,6, 3,7  // Sides
      ];

      // Highlighted = bright glowing gold/yellow; Normal = class color or vibrant teal/green
      const baseCol = box.color || [0.2, 0.8, 0.3];
      const color = box.highlighted ? [1.0, 0.88, 0.1] : baseCol;

      for (let i = 0; i < edges.length; i++) {
        const pt = v[edges[i]];
        lines.push(pt[0], pt[1], pt[2]);
        colors.push(...color);
      }
    });

    this.boxVertCount = lines.length / 3;

    if (!this.boxBuffer) this.boxBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);

    if (!this.boxColorBuffer) this.boxColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    if (autoCenter) {
      this.frameBoundingBoxes();
    }
  }

  updateLabelsOverlay(vp) {
    if (!this.labelsOverlay) {
      this.labelsOverlay = document.getElementById('viewer-labels-overlay');
      if (!this.labelsOverlay) return;
    }
    if (!this._badgePool) {
      this._badgePool = [];
    }

    if (!this.showBoundingBoxes || !this.boundingBoxes || this.boundingBoxes.length === 0) {
      this._badgePool.forEach(el => el.style.display = 'none');
      return;
    }

    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;

    // Filter boxes: always include highlighted search matches, plus top-8 highest confidence
    const indexedBoxes = this.boundingBoxes.map((b, idx) => ({ box: b, idx }));
    const highlighted = indexedBoxes.filter(ib => ib.box.highlighted);
    const nonHighlighted = indexedBoxes
      .filter(ib => !ib.box.highlighted)
      .sort((a, b) => (b.box.confidence || 0) - (a.box.confidence || 0));

    const maxBadges = 8;
    const candidates = [...highlighted, ...nonHighlighted.slice(0, Math.max(0, maxBadges - highlighted.length))];

    // Project candidates to screen coordinates
    const projectedBadges = [];
    candidates.forEach(({ box, idx }) => {
      const c = box.center;
      const x = c[0], y = c[1], z = c[2];
      const clipX = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
      const clipY = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
      const clipZ = vp[2] * x + vp[6] * y + vp[10] * z + vp[14];
      const clipW = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];

      // Depth check: in front of camera
      if (clipW > 0.05 && clipZ / clipW > 0 && clipZ / clipW < 1.0) {
        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        const sx = ((ndcX + 1) / 2) * w;
        const sy = ((1 - ndcY) / 2) * h;

        if (sx >= -40 && sx <= w + 40 && sy >= -20 && sy <= h + 20) {
          projectedBadges.push({
            box,
            idx,
            sx: Math.round(sx),
            sy: Math.round(sy),
            dist: clipW,
          });
        }
      }
    });

    // Screen collision avoidance: if two badges are too close in Y and X, nudge one vertically
    projectedBadges.sort((a, b) => a.sy - b.sy);
    for (let i = 0; i < projectedBadges.length; i++) {
      for (let j = i + 1; j < projectedBadges.length; j++) {
        const b1 = projectedBadges[i];
        const b2 = projectedBadges[j];
        if (Math.abs(b1.sx - b2.sx) < 115 && Math.abs(b1.sy - b2.sy) < 24) {
          b2.sy = b1.sy + 26;
        }
      }
    }

    // Reuse or create DOM elements from pool (zero innerHTML thrashing)
    while (this._badgePool.length < projectedBadges.length) {
      const el = document.createElement('div');
      el.className = 'viewer-3d-badge';
      this.labelsOverlay.appendChild(el);
      this._badgePool.push(el);
    }

    projectedBadges.forEach((item, i) => {
      const el = this._badgePool[i];
      const box = item.box;
      const lbl = (box.label || 'Object').toUpperCase();
      let size = box.short_size_str || (box.size ? `${Number(box.size[0]).toFixed(2)}×${Number(box.size[1]).toFixed(2)}m` : '');
      if (size && size.includes('(')) size = size.split('(')[0].trim();
      const isHigh = box.highlighted;
      const prefix = isHigh ? '⭐ ' : '📦 ';

      el.className = isHigh ? 'viewer-3d-badge highlighted' : 'viewer-3d-badge';
      el.textContent = `${prefix}${lbl} • ${size}`;
      el.dataset.boxIndex = item.idx;
      el.style.display = 'block';
      el.style.left = `${item.sx}px`;
      el.style.top = `${item.sy - 12}px`;
    });

    // Hide remaining unused pool badges
    for (let i = projectedBadges.length; i < this._badgePool.length; i++) {
      this._badgePool[i].style.display = 'none';
    }
  }

  /* ------------------------------------------------------------------
   * Matrix Helpers
   * ----------------------------------------------------------------*/
  computeViewProjection() {
    const fov = this.camera.fov * Math.PI / 180;
    const aspect = (this.canvas.width || 1) / (this.canvas.height || 1);
    const f = 1.0 / Math.tan(fov / 2);
    const near = this.camera.near;
    const far = this.camera.far;
    const proj = new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0
    ]);

    // lookAt
    const eye = this.camera.position;
    const target = this.camera.target;

    const z0 = eye[0] - target[0], z1 = eye[1] - target[1], z2 = eye[2] - target[2];
    const lenZ = Math.hypot(z0, z1, z2) || 1e-6;
    const zx = z0 / lenZ, zy = z1 / lenZ, zz = z2 / lenZ;

    // Stable horizontal right vector from world up [0, -1, 0]
    let x0 = -zz, x1 = 0.0, x2 = zx;
    let lenX = Math.hypot(x0, x2) || 1e-6;
    x0 /= lenX;
    x2 /= lenX;

    // Screen up vector
    const y0 = zy * x2;
    const y1 = zz * x0 - zx * x2;
    const y2 = -zy * x0;

    const viewMat = new Float32Array([
      x0, y0, zx, 0,
      x1, y1, zy, 0,
      x2, y2, zz, 0,
      -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
      -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
      1
    ]);

    // Multiply proj * view (column-major)
    const vp = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        vp[c * 4 + r] = proj[0 * 4 + r] * viewMat[c * 4 + 0] +
                         proj[1 * 4 + r] * viewMat[c * 4 + 1] +
                         proj[2 * 4 + r] * viewMat[c * 4 + 2] +
                         proj[3 * 4 + r] * viewMat[c * 4 + 3];
      }
    }
    return vp;
  }

  /* ------------------------------------------------------------------
   * Render Loop
   * ----------------------------------------------------------------*/
  render() {
    if (!this.gl) return;

    const gl = this.gl;

    // Skip render if canvas has zero size (panel hidden)
    if (this.canvas.width === 0 || this.canvas.height === 0) {
      requestAnimationFrame(() => this.render());
      return;
    }

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.024, 0.031, 0.059, 1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const vp = this.computeViewProjection();

    // --- Draw grid + axes (behind points) ---
    if (this.helperProgram && this.gridBuffer) {
      gl.useProgram(this.helperProgram);
      gl.uniformMatrix4fv(this.helperLocations.viewProjection, false, vp);

      // Grid
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffer);
      gl.enableVertexAttribArray(this.helperLocations.position);
      gl.vertexAttribPointer(this.helperLocations.position, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridColorBuffer);
      gl.enableVertexAttribArray(this.helperLocations.color);
      gl.vertexAttribPointer(this.helperLocations.color, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.LINES, 0, this.gridVertCount);

      // Axes
      gl.bindBuffer(gl.ARRAY_BUFFER, this.axisBuffer);
      gl.vertexAttribPointer(this.helperLocations.position, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.axisColorBuffer);
      gl.vertexAttribPointer(this.helperLocations.color, 3, gl.FLOAT, false, 0, 0);
      gl.lineWidth(2);
      gl.drawArrays(gl.LINES, 0, 6);
      
      // Bounding Boxes
      if (this.showBoundingBoxes && this.boxVertCount > 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.boxBuffer);
        gl.vertexAttribPointer(this.helperLocations.position, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.boxColorBuffer);
        gl.vertexAttribPointer(this.helperLocations.color, 3, gl.FLOAT, false, 0, 0);
        gl.lineWidth(3);
        gl.drawArrays(gl.LINES, 0, this.boxVertCount);
      }

      gl.disableVertexAttribArray(this.helperLocations.position);
      gl.disableVertexAttribArray(this.helperLocations.color);
    }

    // --- Draw point cloud (only if enabled by user) ---
    if (this.showPointCloud && this.gaussians && this.gaussians.count > 0 && this.program) {
      gl.useProgram(this.program);

      gl.uniformMatrix4fv(this.locations.viewProjection, false, vp);

      gl.uniform1f(this.locations.pointSizeMultiplier, this.pointSizeMultiplier * window.devicePixelRatio);
      gl.uniform1f(this.locations.screenHeight, this.canvas.height);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
      gl.enableVertexAttribArray(this.locations.position);
      gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.color);
      gl.enableVertexAttribArray(this.locations.color);
      gl.vertexAttribPointer(this.locations.color, 3, gl.FLOAT, false, 0, 0);
      
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.scale);
      if (this.locations.scale !== -1) {
          gl.enableVertexAttribArray(this.locations.scale);
          gl.vertexAttribPointer(this.locations.scale, 3, gl.FLOAT, false, 0, 0);
      }
      
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.opacity);
      if (this.locations.opacity !== -1) {
          gl.enableVertexAttribArray(this.locations.opacity);
          gl.vertexAttribPointer(this.locations.opacity, 1, gl.FLOAT, false, 0, 0);
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.highlight);
      gl.enableVertexAttribArray(this.locations.highlight);
      gl.vertexAttribPointer(this.locations.highlight, 1, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, this.gaussians.count);
    }

    // Update screen-space 3D bounding box labels
    this.updateLabelsOverlay(vp);

    requestAnimationFrame(() => this.render());
  }

  showFallback() {
    if (this.canvas) this.canvas.style.display = 'none';
    const container = document.getElementById('viewer-container');
    if (container) {
      const msg = document.createElement('div');
      msg.style.cssText = `display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 1.1rem;`;
      msg.textContent = 'WebGL2 is required for the 3D viewer.';
      container.appendChild(msg);
    }
  }
}

// Initialize viewer
window.splatViewer = new SplatViewer('splat-canvas');
