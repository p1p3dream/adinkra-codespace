/**
 * the_168_render.mjs
 *
 * WebGL2 / Three.js scene for The 168. One BufferGeometry of points
 * (40320 vertices for S8) and one LineSegments geometry (282240
 * endpoints). The 7D SO(7) rotation and the 7D to 3D orthographic
 * projection run in the vertex shader: the 7D coordinates ride along
 * as vertex attributes and the first three rows of the current SO(7)
 * matrix arrive as 21 uniform floats.
 *
 * What the viewer sees is an exact 3D orthographic slice of the
 * genuine 7D object. Nothing here fakes depth or precomputes a
 * flattened layout.
 */

import {
  DIM,
  embedPermutations,
  composeSO7,
  idleAngles,
  sliceCentroids,
  PLANE_COUNT,
} from "./rotor7.mjs";

const VERTEX_SHADER = /* glsl */ `
  precision highp float;

  attribute vec4 aPa;        // 7D coords, dims 0..3
  attribute vec3 aPb;        // 7D coords, dims 4..6
  attribute vec4 aCa;        // right-coset centroid, dims 0..3
  attribute vec3 aCb;        // right-coset centroid, dims 4..6
  attribute float aRightHue; // hashed right slice id in [0,1)
  attribute float aLeftHue;  // hashed left slice id in [0,1)
  attribute float aGold;     // 1.0 iff member of one of the 168
  attribute float aFilter;   // 1.0 kept, 0.0 filtered out (isolate mode)
  #ifdef POINTS
  attribute float aId;        // vertex rank, for GPU picking
  attribute float aHighlight; // 0 none, 1 white, 2 green, 3 amber
  #endif

  uniform float uRot[21];   // first 3 rows of the SO(7) matrix, row major
  uniform float uScale;
  uniform float uGather;    // 0 = true positions, 1 = coset centroids
  uniform float uFog;       // weight of the undifferentiated pale fog
  uniform float uRight;     // weight of right-coset coloring
  uniform float uLeft;      // weight of left-coset coloring
  uniform float uMoire;     // left/right interference amount
  uniform float uMoirePhase;
  uniform float uForeshadow; // brightness lift on the 168 before gold
  uniform float uGold;      // 0 = normal palette, 1 = gold ignition
  uniform float uAlpha;
  uniform float uIsolate;   // 0 = show all, 1 = isolate to aFilter members
  #ifdef POINTS
  uniform float uPointSize;
  uniform float uPixelRatio;
  #endif

  varying vec3 vColor;
  varying float vAlpha;
  varying float vKeep;   // 0 iff this vertex is filtered out in isolate mode
  #ifdef POINTS
  varying float vId;
  #endif

  vec3 hsl2rgb(float h, float s, float l) {
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return l + s * (rgb - 0.5) * (1.0 - abs(2.0 * l - 1.0));
  }

  void main() {
    // Gather toward the right-coset centroid (money shot 1), then
    // rotate in 7D and take the first three rotated coordinates.
    float p[7];
    p[0] = mix(aPa.x, aCa.x, uGather);
    p[1] = mix(aPa.y, aCa.y, uGather);
    p[2] = mix(aPa.z, aCa.z, uGather);
    p[3] = mix(aPa.w, aCa.w, uGather);
    p[4] = mix(aPb.x, aCb.x, uGather);
    p[5] = mix(aPb.y, aCb.y, uGather);
    p[6] = mix(aPb.z, aCb.z, uGather);

    vec3 rotated = vec3(0.0);
    for (int c = 0; c < 7; c += 1) {
      rotated.x += uRot[c] * p[c];
      rotated.y += uRot[7 + c] * p[c];
      rotated.z += uRot[14 + c] * p[c];
    }

    vec4 mvPosition = modelViewMatrix * vec4(rotated * uScale, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Palette. Color is scarce and semantic: fog is pale monochrome,
    // right cosets live in cyan to violet, left cosets in green to
    // teal, and gold is reserved for the 168 alone.
    vec3 fogColor = vec3(0.58, 0.63, 0.72);
    vec3 rightColor = hsl2rgb(0.55 + 0.23 * aRightHue, 0.62, 0.60);
    vec3 leftColor = hsl2rgb(0.34 + 0.16 * aLeftHue, 0.55, 0.55);

    float shimmer = 0.5 + 0.5 * sin(uMoirePhase + (aRightHue - aLeftHue) * 31.0);
    vec3 cosetColor = mix(rightColor, leftColor, uMoire * shimmer);

    vec3 base = fogColor * uFog
      + cosetColor * uRight
      + leftColor * uLeft;

    base += vec3(0.30, 0.30, 0.34) * (uForeshadow * aGold);

    // Gold ignition: everything else desaturates to near monochrome.
    float grey = dot(base, vec3(0.299, 0.587, 0.114));
    vec3 muted = mix(base, vec3(grey) * 0.52, uGold);
    vec3 goldColor = vec3(1.0, 0.80, 0.28);
    vColor = mix(muted, goldColor, uGold * aGold);
    vAlpha = uAlpha * mix(1.0, mix(0.5, 1.35, aGold), uGold);

    // Isolate overlay. In isolate mode keep is aFilter (0 hides), else 1.
    // Applied to alpha here so both points and edges fade filtered-out
    // members. Reapplied after the highlight overrides in the POINTS
    // block so a filtered-out highlighted node still stays hidden.
    float keep = mix(1.0, aFilter, uIsolate);
    vAlpha *= keep;
    vKeep = keep;

    #ifdef POINTS
    vId = aId;
    float boost = 1.0 + 0.9 * uGold * aGold;
    // Enlarge kept nodes while isolating so the meaningful subset is easy
    // to click without the surrounding fog.
    boost *= 1.0 + 1.6 * uIsolate * aFilter;
    float hl = aHighlight;
    if (hl > 0.5 && hl < 1.5) { vColor = vec3(0.96, 0.97, 1.0); boost = 2.1; vAlpha = 1.0; }
    if (hl > 1.5 && hl < 2.5) { vColor = vec3(0.22, 0.83, 0.55); boost = 2.1; vAlpha = 1.0; }
    if (hl > 2.5) { vColor = vec3(0.85, 0.55, 0.13); boost = 2.1; vAlpha = 1.0; }
    vAlpha *= keep;
    #ifdef PICKING
    gl_PointSize = max(5.0, uPointSize * boost) * uPixelRatio * (26.0 / -mvPosition.z) * keep;
    #else
    gl_PointSize = uPointSize * boost * uPixelRatio * (26.0 / -mvPosition.z) * keep;
    #endif
    #endif
  }
`;

const POINT_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKeep;
  varying float vId;

  void main() {
    // Isolate gate. Some drivers clamp gl_PointSize to a minimum of 1,
    // so a size-0 filtered node can still emit a fragment. Discarding
    // here makes a hidden node truly unrendered AND unpickable, since the
    // pick color is written below this guard.
    if (vKeep < 0.5) discard;
    vec2 uv = gl_PointCoord - 0.5;
    float r2 = dot(uv, uv);
    if (r2 > 0.25) discard;
    #ifdef PICKING
    float id = vId;
    float rc = floor(id / 65536.0);
    float gc = floor(mod(id, 65536.0) / 256.0);
    float bc = mod(id, 256.0);
    gl_FragColor = vec4(rc / 255.0, gc / 255.0, bc / 255.0, 1.0);
    #else
    float soft = smoothstep(0.25, 0.05, r2);
    gl_FragColor = vec4(vColor, vAlpha * soft);
    #endif
  }
`;

const EDGE_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKeep;
  void main() {
    // An edge fragment is kept only if its endpoint survives the isolate
    // gate; vAlpha already carries keep, this discards the fully hidden ones.
    if (vKeep < 0.5) discard;
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

function hashHue(id) {
  const x = id * 0.6180339887498949;
  return x - Math.floor(x);
}

/** Visual state keys tweened every frame toward their targets. */
const STATE_KEYS = [
  "fog", "right", "left", "moire", "foreshadow", "gold", "gather",
  "pointSize", "pointAlpha", "edgeAlpha", "scale", "rotorSpeed",
];

const DEFAULT_STATE = {
  fog: 1, right: 0, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0,
  pointSize: 2.6, pointAlpha: 0.55, edgeAlpha: 0.05, scale: 1.6,
  rotorSpeed: 1,
};

export class Instrument168 {
  constructor({ canvas, THREE }) {
    this.THREE = THREE;
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false, powerPreference: "high-performance",
    });
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setClearColor(new THREE.Color(0x04070f), 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
    this.camera.position.set(0, 0, 26);

    this.pickScene = new THREE.Scene();
    this.pickTarget = new THREE.WebGLRenderTarget(1, 1);
    this.pickBuffer = new Uint8Array(4);

    this.state = { ...DEFAULT_STATE };
    this.target = { ...DEFAULT_STATE };
    this.userYaw = 0;
    this.userPitch = 0;
    this.rotorTime = 0;
    this.moirePhase = 0;
    this.frozen = false; // when true, idle SO(7) rotation halts; manual drag still works
    this.lastFrame = null;
    this.datasetName = null;
    this.dataset = null;
    this.onPick = null;
    this.frameCallback = null;
    this.isolate = 0;
    this.filterAttr = null;

    this._buildMaterials();
    this._bindResize();
    this._bindPointer();
  }

  _buildMaterials() {
    const { THREE } = this;
    const uniforms = () => ({
      uRot: { value: new Float32Array(21) },
      uScale: { value: 1.6 },
      uGather: { value: 0 },
      uFog: { value: 1 },
      uRight: { value: 0 },
      uLeft: { value: 0 },
      uMoire: { value: 0 },
      uMoirePhase: { value: 0 },
      uForeshadow: { value: 0 },
      uGold: { value: 0 },
      uAlpha: { value: 0.55 },
      uIsolate: { value: 0 },
      uPointSize: { value: 2.6 },
      uPixelRatio: { value: this.pixelRatio },
    });

    this.pointMaterial = new THREE.ShaderMaterial({
      uniforms: uniforms(),
      vertexShader: VERTEX_SHADER,
      fragmentShader: POINT_FRAGMENT_SHADER,
      defines: { POINTS: 1 },
      transparent: true,
      depthWrite: false,
    });
    this.edgeMaterial = new THREE.ShaderMaterial({
      uniforms: uniforms(),
      vertexShader: VERTEX_SHADER,
      fragmentShader: EDGE_FRAGMENT_SHADER,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.pickMaterial = new THREE.ShaderMaterial({
      uniforms: uniforms(),
      vertexShader: VERTEX_SHADER,
      fragmentShader: POINT_FRAGMENT_SHADER,
      defines: { POINTS: 1, PICKING: 1 },
      transparent: false,
      depthWrite: true,
    });
  }

  /**
   * Load a permutahedron atlas (S4 or S8 shape) into GPU buffers.
   * Computes the 7D embedding and right-coset centroids on the spot.
   */
  setDataset(name, atlas) {
    const { THREE } = this;
    this._disposeGeometry();

    const n = atlas.metadata.n;
    const perms = atlas.permutations;
    const count = perms.length;
    const { coords } = embedPermutations(perms, n);
    const rightBy = atlas.right_slice_by_rank;
    const leftBy = atlas.left_slice_by_rank;
    const sliceCount = atlas.right_slices.length;
    const centroids = sliceCentroids(coords, rightBy, sliceCount);
    const goldSet = new Set(atlas.abnormal_right_slices);

    const pa = new Float32Array(count * 4);
    const pb = new Float32Array(count * 3);
    const ca = new Float32Array(count * 4);
    const cb = new Float32Array(count * 3);
    const rightHue = new Float32Array(count);
    const leftHue = new Float32Array(count);
    const gold = new Float32Array(count);
    const filter = new Float32Array(count);
    filter.fill(1);
    const ids = new Float32Array(count);
    const highlight = new Float32Array(count);
    const positions = new Float32Array(count * 3);

    for (let v = 0; v < count; v += 1) {
      const b7 = v * DIM;
      pa[v * 4] = coords[b7]; pa[v * 4 + 1] = coords[b7 + 1];
      pa[v * 4 + 2] = coords[b7 + 2]; pa[v * 4 + 3] = coords[b7 + 3];
      pb[v * 3] = coords[b7 + 4]; pb[v * 3 + 1] = coords[b7 + 5]; pb[v * 3 + 2] = coords[b7 + 6];
      const s = rightBy[v];
      const c7 = s * DIM;
      ca[v * 4] = centroids[c7]; ca[v * 4 + 1] = centroids[c7 + 1];
      ca[v * 4 + 2] = centroids[c7 + 2]; ca[v * 4 + 3] = centroids[c7 + 3];
      cb[v * 3] = centroids[c7 + 4]; cb[v * 3 + 1] = centroids[c7 + 5]; cb[v * 3 + 2] = centroids[c7 + 6];
      rightHue[v] = hashHue(s);
      leftHue[v] = hashHue(leftBy[v]);
      gold[v] = goldSet.has(s) ? 1 : 0;
      ids[v] = v;
      positions[v * 3] = coords[b7];
      positions[v * 3 + 1] = coords[b7 + 1];
      positions[v * 3 + 2] = coords[b7 + 2];
    }

    const pointGeometry = new THREE.BufferGeometry();
    const attrPos = new THREE.BufferAttribute(positions, 3);
    const attrPa = new THREE.BufferAttribute(pa, 4);
    const attrPb = new THREE.BufferAttribute(pb, 3);
    const attrCa = new THREE.BufferAttribute(ca, 4);
    const attrCb = new THREE.BufferAttribute(cb, 3);
    const attrRightHue = new THREE.BufferAttribute(rightHue, 1);
    const attrLeftHue = new THREE.BufferAttribute(leftHue, 1);
    const attrGold = new THREE.BufferAttribute(gold, 1);
    const attrFilter = new THREE.BufferAttribute(filter, 1);
    pointGeometry.setAttribute("position", attrPos);
    pointGeometry.setAttribute("aPa", attrPa);
    pointGeometry.setAttribute("aPb", attrPb);
    pointGeometry.setAttribute("aCa", attrCa);
    pointGeometry.setAttribute("aCb", attrCb);
    pointGeometry.setAttribute("aRightHue", attrRightHue);
    pointGeometry.setAttribute("aLeftHue", attrLeftHue);
    pointGeometry.setAttribute("aGold", attrGold);
    pointGeometry.setAttribute("aFilter", attrFilter);
    pointGeometry.setAttribute("aId", new THREE.BufferAttribute(ids, 1));
    pointGeometry.setAttribute("aHighlight", new THREE.BufferAttribute(highlight, 1));

    // Edge geometry shares the point attribute buffers and draws them indexed.
    // This replaces duplicating about 19 floats per endpoint (roughly 21 MB for
    // S8) and a 282,240-iteration copy loop that stalled first paint with a
    // single pass that fills one index buffer.
    const edges = atlas.edges;
    const eCount = edges.length;
    const edgeIndex = new Uint32Array(eCount * 2);
    for (let e = 0; e < eCount; e += 1) {
      edgeIndex[e * 2] = edges[e][0];
      edgeIndex[e * 2 + 1] = edges[e][1];
    }
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", attrPos);
    edgeGeometry.setAttribute("aPa", attrPa);
    edgeGeometry.setAttribute("aPb", attrPb);
    edgeGeometry.setAttribute("aCa", attrCa);
    edgeGeometry.setAttribute("aCb", attrCb);
    edgeGeometry.setAttribute("aRightHue", attrRightHue);
    edgeGeometry.setAttribute("aLeftHue", attrLeftHue);
    edgeGeometry.setAttribute("aGold", attrGold);
    edgeGeometry.setAttribute("aFilter", attrFilter);
    edgeGeometry.setIndex(new THREE.BufferAttribute(edgeIndex, 1));

    this.points = new THREE.Points(pointGeometry, this.pointMaterial);
    this.points.frustumCulled = false;
    this.lines = new THREE.LineSegments(edgeGeometry, this.edgeMaterial);
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);
    this.scene.add(this.points);

    this.pickPoints = new THREE.Points(pointGeometry, this.pickMaterial);
    this.pickPoints.frustumCulled = false;
    this.pickScene.add(this.pickPoints);

    this.datasetName = name;
    this.dataset = atlas;
    this.highlightAttr = pointGeometry.getAttribute("aHighlight");
    this.filterAttr = attrFilter;
    this.isolate = 0;
    this.vertexCount = count;
  }

  _disposeGeometry() {
    if (this.points) {
      this.scene.remove(this.points);
      this.scene.remove(this.lines);
      this.pickScene.remove(this.pickPoints);
      this.points.geometry.dispose();
      this.lines.geometry.dispose();
      this.points = null;
      this.lines = null;
      this.pickPoints = null;
    }
  }

  /** Tween the visual state toward these values. */
  setTargets(partial) {
    for (const key of Object.keys(partial)) {
      if (STATE_KEYS.includes(key)) this.target[key] = partial[key];
    }
  }

  /** Jump the visual state instantly (used at dataset swaps). */
  snapTargets(partial) {
    this.setTargets(partial);
    for (const key of Object.keys(partial)) {
      if (STATE_KEYS.includes(key)) this.state[key] = partial[key];
    }
  }

  /** map is rank -> code (1 white, 2 green, 3 amber). Replaces all. */
  setHighlights(map) {
    if (!this.highlightAttr) return;
    const arr = this.highlightAttr.array;
    arr.fill(0);
    if (map) {
      for (const [rank, code] of map) arr[rank] = code;
    }
    this.highlightAttr.needsUpdate = true;
  }

  /**
   * Isolate the display to a set of vertex ranks. Pass null to clear the
   * isolate overlay and show everything again. Filtered-out nodes get
   * point size 0 in every material, so they are neither drawn nor pickable.
   */
  setFilter(rankSetOrNull) {
    if (!this.filterAttr) return;
    const arr = this.filterAttr.array;
    if (rankSetOrNull === null || rankSetOrNull === undefined) {
      arr.fill(1);
      this.isolate = 0;
    } else {
      arr.fill(0);
      for (const r of rankSetOrNull) {
        if (r >= 0 && r < arr.length) arr[r] = 1;
      }
      this.isolate = 1;
    }
    this.filterAttr.needsUpdate = true;
  }

  /** GPU color-ID picking. Returns a vertex rank or null. */
  pickAt(clientX, clientY) {
    if (!this.pickPoints) return null;
    const { renderer, camera, pickTarget } = this;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * this.pixelRatio));
    const h = Math.max(1, Math.floor(rect.height * this.pixelRatio));
    if (pickTarget.width !== w || pickTarget.height !== h) pickTarget.setSize(w, h);
    const prevClear = renderer.getClearColor(new this.THREE.Color());
    const prevAlpha = renderer.getClearAlpha();
    renderer.setRenderTarget(pickTarget);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear();
    renderer.render(this.pickScene, camera);
    const x = Math.floor((clientX - rect.left) * this.pixelRatio);
    const y = Math.floor((rect.height - (clientY - rect.top)) * this.pixelRatio);
    renderer.readRenderTargetPixels(pickTarget, x, y, 1, 1, this.pickBuffer);
    renderer.setRenderTarget(null);
    renderer.setClearColor(prevClear, prevAlpha);
    const [r, g, b] = this.pickBuffer;
    const id = r * 65536 + g * 256 + b;
    return id < this.vertexCount ? id : null;
  }

  _bindResize() {
    const resize = () => {
      const w = this.canvas.clientWidth || window.innerWidth;
      const h = this.canvas.clientHeight || window.innerHeight;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", resize);
    resize();
  }

  _bindPointer() {
    let dragging = false;
    let moved = 0;
    let last = null;
    this.canvas.addEventListener("pointerdown", event => {
      dragging = true; moved = 0; last = [event.clientX, event.clientY];
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", event => {
      if (!dragging) return;
      const dx = event.clientX - last[0];
      const dy = event.clientY - last[1];
      moved += Math.abs(dx) + Math.abs(dy);
      last = [event.clientX, event.clientY];
      // Dragging rotates the object inside SO(7): plane (0,2) for
      // horizontal, plane (1,2) for vertical. Still an exact slice.
      this.userYaw += dx * 0.005;
      this.userPitch += dy * 0.005;
    });
    this.canvas.addEventListener("pointerup", event => {
      dragging = false;
      if (moved < 6 && this.onPick) {
        const rank = this.pickAt(event.clientX, event.clientY);
        if (rank !== null) this.onPick(rank);
      }
    });
    this.canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const factor = Math.exp(event.deltaY * 0.001);
      this.camera.position.z = Math.min(80, Math.max(8, this.camera.position.z * factor));
    }, { passive: false });
  }

  _applyUniforms(rotFirst3) {
    const s = this.state;
    for (const mat of [this.pointMaterial, this.edgeMaterial, this.pickMaterial]) {
      const u = mat.uniforms;
      u.uRot.value.set(rotFirst3);
      u.uScale.value = s.scale;
      u.uGather.value = s.gather;
      u.uFog.value = s.fog;
      u.uRight.value = s.right;
      u.uLeft.value = s.left;
      u.uMoire.value = s.moire;
      u.uMoirePhase.value = this.moirePhase;
      u.uForeshadow.value = s.foreshadow;
      u.uGold.value = s.gold;
      u.uIsolate.value = this.isolate;
      u.uPointSize.value = s.pointSize;
      u.uPixelRatio.value = this.pixelRatio;
    }
    this.pointMaterial.uniforms.uAlpha.value = s.pointAlpha;
    this.edgeMaterial.uniforms.uAlpha.value = s.edgeAlpha;
  }

  /** One animation step. Public so tests and scripts can drive it. */
  _tick(now) {
      const t = now * 0.001;
      const dt = this.lastFrame === null ? 0.016 : Math.min(0.1, t - this.lastFrame);
      this.lastFrame = t;

      // Exponential tween of every state key toward its target.
      const k = 1 - Math.exp(-dt * 2.6);
      const kGold = 1 - Math.exp(-dt * 3.4);
      for (const key of STATE_KEYS) {
        const rate = key === "gold" ? kGold : k;
        this.state[key] += (this.target[key] - this.state[key]) * rate;
      }

      if (!this.frozen) this.rotorTime += dt * this.state.rotorSpeed;
      this.moirePhase += dt * 2.2;

      const angles = idleAngles(this.rotorTime, 1);
      angles[1] += this.userYaw;   // plane (0,2): horizontal drag
      angles[6] += this.userPitch; // plane (1,2): vertical drag
      const rot = composeSO7(angles);
      const first3 = new Float32Array(21);
      for (let i = 0; i < 21; i += 1) first3[i] = rot[i];
      this._applyUniforms(first3);

      this.renderer.render(this.scene, this.camera);
      if (this.frameCallback) this.frameCallback(t, dt);
  }

  start() {
    const loop = now => {
      requestAnimationFrame(loop);
      this._tick(now);
    };
    requestAnimationFrame(loop);
    // Fallback driver: requestAnimationFrame is suspended when the
    // page is hidden or occluded. Keep the instrument alive at a low
    // rate so it is never frozen when it comes back on screen.
    setInterval(() => {
      const now = performance.now();
      if (this.lastFrame === null || now * 0.001 - this.lastFrame > 0.25) this._tick(now);
    }, 120);
  }
}

export { PLANE_COUNT };
