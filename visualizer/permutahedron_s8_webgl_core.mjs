/**
 * Dependency-free WebGL2 renderer for a three-dimensional projection of the
 * S8 permutahedron.
 *
 * Minimal use:
 *
 *   const geometry = buildS8Geometry();
 *   const view = new S8WebGLRenderer(canvas, { geometry });
 *   view.attachControls();
 *   view.resizeToDisplaySize();
 *   view.render();
 *
 * Geometry may also be supplied by the caller. Edge indices must be grouped
 * by generator so each of the seven visibility switches costs one draw call.
 */

export const S8_VERTEX_COUNT = 40320;
export const S8_EDGE_COUNT = 141120;
export const S8_GENERATOR_COUNT = 7;
export const S8_EDGES_PER_GENERATOR = 20160;

export const DEFAULT_GENERATOR_COLORS = Object.freeze([
  "#e53935", "#f39c12", "#d4c51c", "#22a35a",
  "#1597a8", "#3267d6", "#8e44ad",
]);

const FACTORIAL = Object.freeze([1, 1, 2, 6, 24, 120, 720, 5040, 40320]);

function rankPermutation8(p) {
  let rank = 0;
  for (let i = 0; i < 8; i += 1) {
    let smaller = 0;
    for (let j = i + 1; j < 8; j += 1) if (p[j] < p[i]) smaller += 1;
    rank += smaller * FACTORIAL[7 - i];
  }
  return rank;
}

function nextPermutation(values) {
  let i = values.length - 2;
  while (i >= 0 && values[i] >= values[i + 1]) i -= 1;
  if (i < 0) return false;
  let j = values.length - 1;
  while (values[j] <= values[i]) j -= 1;
  [values[i], values[j]] = [values[j], values[i]];
  for (let a = i + 1, b = values.length - 1; a < b; a += 1, b -= 1) {
    [values[a], values[b]] = [values[b], values[a]];
  }
  return true;
}

/**
 * Return an orthonormal 3 by 8 projection matrix stored row-major.
 * The rows are low Fourier modes in the seven-dimensional sum-zero space.
 */
export function defaultS8ProjectionMatrix() {
  const matrix = new Float32Array(24);
  const scale = Math.sqrt(2 / 8);
  for (let i = 0; i < 8; i += 1) {
    const angle = 2 * Math.PI * i / 8;
    matrix[i] = scale * Math.cos(angle);
    matrix[8 + i] = scale * Math.sin(angle);
    matrix[16 + i] = scale * Math.cos(2 * angle);
  }
  return matrix;
}

/** Project one S8 permutation into three dimensions. */
export function projectS8Permutation(permutation, projectionMatrix = defaultS8ProjectionMatrix()) {
  if (!permutation || permutation.length !== 8) throw new RangeError("an S8 permutation must have eight entries");
  if (!projectionMatrix || projectionMatrix.length !== 24) throw new RangeError("projectionMatrix must contain 24 values");
  const output = new Float32Array(3);
  for (let axis = 0; axis < 3; axis += 1) {
    let sum = 0;
    for (let i = 0; i < 8; i += 1) sum += (Number(permutation[i]) - 4.5) * projectionMatrix[axis * 8 + i];
    output[axis] = sum;
  }
  return output;
}

/**
 * Build all projected positions and adjacent-transposition edges in rank order.
 * edgeIndices contains seven consecutive blocks, one block per generator.
 */
export function buildS8Geometry({ projectionMatrix = defaultS8ProjectionMatrix(), scale = 1 } = {}) {
  if (!projectionMatrix || projectionMatrix.length !== 24) throw new RangeError("projectionMatrix must contain 24 values");
  if (!Number.isFinite(scale) || scale <= 0) throw new RangeError("scale must be positive");

  const positions = new Float32Array(S8_VERTEX_COUNT * 3);
  const blocks = Array.from({ length: S8_GENERATOR_COUNT }, () => new Uint16Array(S8_EDGES_PER_GENERATOR * 2));
  const cursors = new Uint32Array(S8_GENERATOR_COUNT);
  const permutation = [1, 2, 3, 4, 5, 6, 7, 8];

  for (let rank = 0; rank < S8_VERTEX_COUNT; rank += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      let coordinate = 0;
      const row = axis * 8;
      for (let i = 0; i < 8; i += 1) coordinate += (permutation[i] - 4.5) * projectionMatrix[row + i];
      positions[rank * 3 + axis] = coordinate * scale;
    }

    for (let generator = 0; generator < S8_GENERATOR_COUNT; generator += 1) {
      [permutation[generator], permutation[generator + 1]] = [permutation[generator + 1], permutation[generator]];
      const neighbor = rankPermutation8(permutation);
      [permutation[generator], permutation[generator + 1]] = [permutation[generator + 1], permutation[generator]];
      if (rank < neighbor) {
        const cursor = cursors[generator];
        blocks[generator][cursor] = rank;
        blocks[generator][cursor + 1] = neighbor;
        cursors[generator] += 2;
      }
    }
    if (rank + 1 < S8_VERTEX_COUNT) nextPermutation(permutation);
  }

  const edgeIndices = new Uint16Array(S8_EDGE_COUNT * 2);
  const edgeRanges = [];
  let offset = 0;
  for (let generator = 0; generator < S8_GENERATOR_COUNT; generator += 1) {
    if (cursors[generator] !== S8_EDGES_PER_GENERATOR * 2) throw new Error(`generator ${generator + 1} edge count is incorrect`);
    edgeIndices.set(blocks[generator], offset);
    edgeRanges.push(Object.freeze({ offset, count: blocks[generator].length }));
    offset += blocks[generator].length;
  }
  return { positions, edgeIndices, edgeRanges };
}

function shader(gl, type, source) {
  const value = gl.createShader(type);
  gl.shaderSource(value, source);
  gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(value) || "shader compilation failed";
    gl.deleteShader(value);
    throw new Error(message);
  }
  return value;
}

function program(gl, vertexSource, fragmentSource) {
  const vertex = shader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = shader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const value = gl.createProgram();
  gl.attachShader(value, vertex);
  gl.attachShader(value, fragment);
  gl.linkProgram(value);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(value, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(value) || "program link failed";
    gl.deleteProgram(value);
    throw new Error(message);
  }
  return value;
}

function color4(value, fallback) {
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    const color = Array.from(value, Number);
    if (color.length === 3) color.push(1);
    if (color.length === 4 && color.every(Number.isFinite)) return new Float32Array(color);
  }
  if (typeof value === "string") {
    const match = value.trim().match(/^#([0-9a-f]{6}|[0-9a-f]{8})$/i);
    if (match) {
      const raw = match[1];
      return new Float32Array([
        parseInt(raw.slice(0, 2), 16) / 255,
        parseInt(raw.slice(2, 4), 16) / 255,
        parseInt(raw.slice(4, 6), 16) / 255,
        raw.length === 8 ? parseInt(raw.slice(6, 8), 16) / 255 : 1,
      ]);
    }
  }
  return color4(fallback, [0, 0, 0, 1]);
}

function identity4() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function multiply4(a, b) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[column * 4 + k];
      output[column * 4 + row] = sum;
    }
  }
  return output;
}

function translation4(x, y, z) {
  const matrix = identity4();
  matrix[12] = x; matrix[13] = y; matrix[14] = z;
  return matrix;
}

function rotation4(x, y, z) {
  const cx = Math.cos(x), sx = Math.sin(x);
  const cy = Math.cos(y), sy = Math.sin(y);
  const cz = Math.cos(z), sz = Math.sin(z);
  const rx = new Float32Array([1, 0, 0, 0, 0, cx, sx, 0, 0, -sx, cx, 0, 0, 0, 0, 1]);
  const ry = new Float32Array([cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1]);
  const rz = new Float32Array([cz, sz, 0, 0, -sz, cz, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  return multiply4(rz, multiply4(ry, rx));
}

function perspective4(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const inverse = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0, 0, f, 0, 0,
    0, 0, (far + near) * inverse, -1,
    0, 0, 2 * far * near * inverse, 0,
  ]);
}

function validateGeometry(geometry) {
  if (!geometry || !geometry.positions || geometry.positions.length !== S8_VERTEX_COUNT * 3) {
    throw new RangeError(`positions must contain ${S8_VERTEX_COUNT * 3} values`);
  }
  if (!geometry.edgeIndices || geometry.edgeIndices.length !== S8_EDGE_COUNT * 2) {
    throw new RangeError(`edgeIndices must contain ${S8_EDGE_COUNT * 2} values`);
  }
  if (!geometry.edgeRanges || geometry.edgeRanges.length !== S8_GENERATOR_COUNT) {
    throw new RangeError("edgeRanges must contain seven entries");
  }
  let expectedOffset = 0;
  for (const range of geometry.edgeRanges) {
    if (range.offset !== expectedOffset || range.count !== S8_EDGES_PER_GENERATOR * 2) {
      throw new RangeError("each edge range must be a consecutive 20,160-edge generator block");
    }
    expectedOffset += range.count;
  }
}

function geometryFromInputs(positionsInput, edgesInput) {
  let positions;
  if (positionsInput instanceof Float32Array) {
    positions = positionsInput;
  } else if (positionsInput?.length === S8_VERTEX_COUNT &&
             (Array.isArray(positionsInput[0]) || typeof positionsInput[0] === "object")) {
    positions = new Float32Array(S8_VERTEX_COUNT * 3);
    for (let rank = 0; rank < S8_VERTEX_COUNT; rank += 1) {
      const point = positionsInput[rank];
      positions[rank * 3] = Number(point[0] ?? point.x);
      positions[rank * 3 + 1] = Number(point[1] ?? point.y);
      positions[rank * 3 + 2] = Number(point[2] ?? point.z);
    }
  } else {
    positions = new Float32Array(positionsInput);
  }
  if (edgesInput?.edgeIndices && edgesInput?.edgeRanges) {
    return { positions, edgeIndices: edgesInput.edgeIndices, edgeRanges: edgesInput.edgeRanges };
  }
  if (!edgesInput || typeof edgesInput[Symbol.iterator] !== "function") {
    throw new TypeError("edges must be iterable or contain edgeIndices and edgeRanges");
  }

  const edges = Array.from(edgesInput);
  if (edges.length !== S8_EDGE_COUNT) throw new RangeError(`edges must contain ${S8_EDGE_COUNT} entries`);
  const parsed = edges.map(edge => {
    if (Array.isArray(edge) || ArrayBuffer.isView(edge)) {
      return { source: Number(edge[0]), target: Number(edge[1]), generator: Number(edge[2]) };
    }
    return {
      source: Number(edge.source ?? edge.from ?? edge.a),
      target: Number(edge.target ?? edge.to ?? edge.b),
      generator: Number(edge.generator ?? edge.generatorIndex ?? edge.color ?? edge.g),
    };
  });
  const oneBased = parsed.every(edge => edge.generator >= 1 && edge.generator <= 7);
  const blocks = Array.from({ length: 7 }, () => []);
  for (const edge of parsed) {
    const generator = edge.generator - (oneBased ? 1 : 0);
    if (!Number.isInteger(edge.source) || !Number.isInteger(edge.target) ||
        edge.source < 0 || edge.target < 0 || edge.source >= S8_VERTEX_COUNT || edge.target >= S8_VERTEX_COUNT) {
      throw new RangeError("edge endpoint is outside S8");
    }
    if (!Number.isInteger(generator) || generator < 0 || generator >= 7) throw new RangeError("edge generator is outside one through seven");
    blocks[generator].push(edge.source, edge.target);
  }
  const edgeIndices = new Uint16Array(S8_EDGE_COUNT * 2);
  const edgeRanges = [];
  let offset = 0;
  for (const block of blocks) {
    edgeIndices.set(block, offset);
    edgeRanges.push({ offset, count: block.length });
    offset += block.length;
  }
  return { positions, edgeIndices, edgeRanges };
}

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(a_position, 1.0); }
`;

const LINE_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 outColor;
void main() { outColor = u_color; }
`;

const POINT_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
layout(location=1) in float a_state;
uniform mat4 u_mvp;
uniform vec3 u_sizes;
uniform float u_pixelRatio;
uniform bool u_isolate;
flat out int v_state;
void main() {
  gl_Position = u_mvp * vec4(a_position, 1.0);
  v_state = int(a_state + 0.5);
  gl_PointSize = u_sizes[v_state] * u_pixelRatio;
  if (u_isolate && v_state == 0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
  }
}
`;

const POINT_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
flat in int v_state;
uniform vec4 u_baseColor;
uniform vec4 u_highlightColor;
uniform vec4 u_selectedColor;
out vec4 outColor;
void main() {
  vec2 delta = gl_PointCoord - vec2(0.5);
  float radius = length(delta);
  if (radius > 0.5) discard;
  vec4 color = v_state == 2 ? u_selectedColor : (v_state == 1 ? u_highlightColor : u_baseColor);
  float edge = smoothstep(0.50, 0.40, radius);
  outColor = vec4(color.rgb, color.a * edge);
}
`;

/** WebGL2 renderer specialized for the full S8 adjacent-transposition graph. */
export class S8WebGLRenderer {
  constructor(canvas, options = {}, edges, generatorColors) {
    if (!canvas || typeof canvas.getContext !== "function") throw new TypeError("canvas must provide getContext");
    const positionalGeometry = Array.isArray(options) || ArrayBuffer.isView(options);
    if (positionalGeometry) {
      options = { geometry: geometryFromInputs(options, edges), generatorColors };
    }
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      antialias: options.antialias !== false,
      alpha: options.alpha !== false,
      preserveDrawingBuffer: options.preserveDrawingBuffer !== false,
    });
    if (!this.gl) throw new Error("WebGL2 is unavailable");

    this.rotation = new Float32Array(options.rotation || [-0.42, 0.58, 0]);
    this.zoom = Number(options.zoom ?? 1);
    this.fov = Number(options.fov ?? Math.PI / 4);
    this.generatorVisible = Array.from({ length: 7 }, (_, i) => options.generatorVisible?.[i] !== false);
    this.generatorColors = DEFAULT_GENERATOR_COLORS.map((color, i) => color4(options.generatorColors?.[i], color));
    this.clearColor = color4(options.clearColor, [0.969, 0.965, 0.945, 1]);
    this.baseColor = color4(options.baseColor, [0.015, 0.02, 0.025, 0.95]);
    this.highlightColor = color4(options.highlightColor, [0.05, 0.72, 0.92, 1]);
    this.selectedColor = color4(options.selectedColor, [1, 0.72, 0.08, 1]);
    this.lineAlpha = Math.max(0, Math.min(1, Number(options.lineAlpha ?? 0.24)));
    this.pointSizes = new Float32Array(options.pointSizes || [2.25, 6, 9]);
    this.showEdges = true;
    this.showVertices = true;
    this.isolate = false;
    this.pixelRatio = 1;
    this.center = new Float32Array(3);
    this.radius = 1;
    this.geometry = null;
    this.pointStates = new Uint8Array(S8_VERTEX_COUNT);
    this.selectedRanks = new Set();
    this.highlightedRanks = new Set();
    this._controlsAbort = null;
    this._disposed = false;
    this._needsRender = true;
    this._frameHandle = 0;
    this._lastMvp = null;

    this._initializeGL();
    if (options.geometry) this.setGeometry(options.geometry);

    this._onContextLost = event => { event.preventDefault(); };
    this._onContextRestored = () => {
      this._initializeGL();
      if (this.geometry) this._uploadGeometry();
      this.requestRender();
    };
    canvas.addEventListener("webglcontextlost", this._onContextLost);
    canvas.addEventListener("webglcontextrestored", this._onContextRestored);
  }

  _initializeGL() {
    const gl = this.gl;
    this.lineProgram = program(gl, VERTEX_SHADER, LINE_FRAGMENT_SHADER);
    this.pointProgram = program(gl, POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER);
    this.positionBuffer = gl.createBuffer();
    this.edgeBuffer = gl.createBuffer();
    this.stateBuffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.UNSIGNED_BYTE, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeBuffer);
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  _uploadGeometry() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.geometry.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.geometry.edgeIndices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.pointStates, gl.DYNAMIC_DRAW);
  }

  setGeometry(geometry) {
    validateGeometry(geometry);
    const positions = geometry.positions instanceof Float32Array ? geometry.positions : new Float32Array(geometry.positions);
    const edgeIndices = geometry.edgeIndices instanceof Uint16Array ? geometry.edgeIndices : new Uint16Array(geometry.edgeIndices);
    this.geometry = { positions, edgeIndices, edgeRanges: geometry.edgeRanges.map(range => ({ ...range })) };

    const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], positions[i + axis]);
        maximum[axis] = Math.max(maximum[axis], positions[i + axis]);
      }
    }
    for (let axis = 0; axis < 3; axis += 1) this.center[axis] = (minimum[axis] + maximum[axis]) / 2;
    this.radius = 0;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i] - this.center[0], y = positions[i + 1] - this.center[1], z = positions[i + 2] - this.center[2];
      this.radius = Math.max(this.radius, Math.hypot(x, y, z));
    }
    this.radius = Math.max(this.radius, 1e-4);
    this._lastMvp = null;
    this._uploadGeometry();
    this.requestRender();
    return this;
  }

  setGeneratorVisible(generator, visible) {
    if (!Number.isInteger(generator) || generator < 1 || generator > 7) throw new RangeError("generator must be one through seven");
    this.generatorVisible[generator - 1] = Boolean(visible);
    this.requestRender();
    return this;
  }

  setGeneratorVisibility(flags) {
    if (!flags || flags.length !== 7) throw new RangeError("generator visibility needs seven flags");
    for (let i = 0; i < 7; i += 1) this.generatorVisible[i] = Boolean(flags[i]);
    this.requestRender();
    return this;
  }

  setSelectedRanks(ranks) { return this._setRanks(ranks, 2); }
  setHighlightedRanks(ranks) { return this._setRanks(ranks, 1); }
  setHighlightRanks(ranks) { return this.setHighlightedRanks(ranks); }

  _setRanks(ranks, state) {
    const destination = state === 2 ? this.selectedRanks : this.highlightedRanks;
    destination.clear();
    if (ranks != null) {
      for (const value of ranks) {
        const rank = Number(value);
        if (!Number.isInteger(rank) || rank < 0 || rank >= S8_VERTEX_COUNT) throw new RangeError(`rank ${value} is outside S8`);
        destination.add(rank);
      }
    }
    this.pointStates.fill(0);
    for (const rank of this.highlightedRanks) this.pointStates[rank] = 1;
    for (const rank of this.selectedRanks) this.pointStates[rank] = 2;
    if (this.geometry) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.stateBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.pointStates);
    }
    this.requestRender();
    return this;
  }

  setRotation(x, y, z = this.rotation[2]) {
    if (Array.isArray(x) || ArrayBuffer.isView(x)) [x, y, z] = x;
    if (![x, y, z].every(Number.isFinite)) throw new TypeError("rotation values must be finite");
    this.rotation.set([x, y, z]);
    this._lastMvp = null;
    this.requestRender();
    return this;
  }

  rotate(deltaX, deltaY, deltaZ = 0) {
    return this.setRotation(this.rotation[0] + deltaX, this.rotation[1] + deltaY, this.rotation[2] + deltaZ);
  }

  setZoom(zoom) {
    if (!Number.isFinite(zoom) || zoom <= 0) throw new RangeError("zoom must be positive");
    this.zoom = Math.max(0.12, Math.min(24, zoom));
    this._lastMvp = null;
    this.requestRender();
    return this;
  }

  setCamera(yaw, elevation, zoom = this.zoom) {
    this.setRotation(elevation, yaw, this.rotation[2]);
    return this.setZoom(zoom);
  }

  zoomBy(factor) {
    if (!Number.isFinite(factor) || factor <= 0) throw new RangeError("zoom factor must be positive");
    return this.setZoom(this.zoom * factor);
  }

  /** Resize the backing buffer while preserving CSS layout size. */
  resize(width, height, pixelRatio = globalThis.devicePixelRatio || 1) {
    if (width == null || height == null) return this.resizeToDisplaySize(pixelRatio);
    if (![width, height, pixelRatio].every(Number.isFinite) || width <= 0 || height <= 0 || pixelRatio <= 0) {
      throw new RangeError("resize dimensions and pixel ratio must be positive");
    }
    this.pixelRatio = Math.min(pixelRatio, 3);
    const physicalWidth = Math.max(1, Math.round(width * this.pixelRatio));
    const physicalHeight = Math.max(1, Math.round(height * this.pixelRatio));
    if (this.canvas.width !== physicalWidth) this.canvas.width = physicalWidth;
    if (this.canvas.height !== physicalHeight) this.canvas.height = physicalHeight;
    this.gl.viewport(0, 0, physicalWidth, physicalHeight);
    this._lastMvp = null;
    this.requestRender();
    return this;
  }

  resizeToDisplaySize(pixelRatio = globalThis.devicePixelRatio || 1) {
    const rect = this.canvas.getBoundingClientRect();
    return this.resize(Math.max(1, rect.width), Math.max(1, rect.height), pixelRatio);
  }

  /** Install drag rotation and wheel zoom. Returns a function that removes them. */
  attachControls({ rotationSpeed = 0.008, wheelSpeed = 0.0015 } = {}) {
    this.detachControls();
    const abort = new AbortController();
    const signal = abort.signal;
    let pointer = null;
    this.canvas.addEventListener("pointerdown", event => {
      pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
    }, { signal });
    this.canvas.addEventListener("pointermove", event => {
      if (!pointer || event.pointerId !== pointer.id) return;
      const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
      pointer.x = event.clientX; pointer.y = event.clientY;
      this.rotate(dy * rotationSpeed, dx * rotationSpeed);
    }, { signal });
    const release = event => { if (pointer?.id === event.pointerId) pointer = null; };
    this.canvas.addEventListener("pointerup", release, { signal });
    this.canvas.addEventListener("pointercancel", release, { signal });
    this.canvas.addEventListener("wheel", event => {
      event.preventDefault();
      this.zoomBy(Math.exp(-event.deltaY * wheelSpeed));
    }, { passive: false, signal });
    this._controlsAbort = abort;
    return () => this.detachControls();
  }

  detachControls() {
    this._controlsAbort?.abort();
    this._controlsAbort = null;
    return this;
  }

  requestRender() {
    this._needsRender = true;
    if (!this._frameHandle && typeof globalThis.requestAnimationFrame === "function") {
      this._frameHandle = globalThis.requestAnimationFrame(() => {
        this._frameHandle = 0;
        this.render();
      });
    }
    return this;
  }

  _makeMvp() {
    const gl = this.gl;
    const aspect = Math.max(1e-6, gl.drawingBufferWidth / gl.drawingBufferHeight);
    const distance = this.radius * 3.2;
    const near = Math.max(this.radius * 0.001, distance - this.radius * 1.7);
    const far = Math.max(near + 1, distance + this.radius * 1.7);
    const zoomedFov = 2 * Math.atan(Math.tan(this.fov / 2) / this.zoom);
    const projection = perspective4(zoomedFov, aspect, near, far);
    const view = translation4(0, 0, -distance);
    const model = multiply4(rotation4(...this.rotation), translation4(-this.center[0], -this.center[1], -this.center[2]));
    return multiply4(projection, multiply4(view, model));
  }

  _getMvp() {
    if (!this._lastMvp) this._lastMvp = this._makeMvp();
    return this._lastMvp;
  }

  /** Return the current CSS-pixel projection of a lexicographic vertex rank. */
  projectRank(rank) {
    if (!Number.isInteger(rank) || rank < 0 || rank >= S8_VERTEX_COUNT) throw new RangeError(`rank ${rank} is outside S8`);
    if (!this.geometry) return null;
    const matrix = this._getMvp();
    const offset = rank * 3;
    const x = this.geometry.positions[offset], y = this.geometry.positions[offset + 1], z = this.geometry.positions[offset + 2];
    const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const clipZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
    const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    const nx = clipX / clipW, ny = clipY / clipW, nz = clipZ / clipW;
    const width = this.canvas.width / this.pixelRatio, height = this.canvas.height / this.pixelRatio;
    return {
      x: (nx * 0.5 + 0.5) * width,
      y: (1 - (ny * 0.5 + 0.5)) * height,
      depth: nz,
      visible: clipW > 0 && nx >= -1 && nx <= 1 && ny >= -1 && ny <= 1 && nz >= -1 && nz <= 1,
    };
  }

  render(options = false) {
    let force = options === true;
    if (options && typeof options === "object") {
      if (options.highlightRanks !== undefined) this.setHighlightedRanks(options.highlightRanks);
      if (options.selectedRanks !== undefined) this.setSelectedRanks(options.selectedRanks);
      if (options.generatorVisible !== undefined) this.setGeneratorVisibility(options.generatorVisible);
      if (options.showEdges !== undefined) this.showEdges = Boolean(options.showEdges);
      if (options.showVertices !== undefined) this.showVertices = Boolean(options.showVertices);
      if (options.isolate !== undefined) this.isolate = Boolean(options.isolate);
      force = options.force === true;
    }
    if (this._disposed || (!force && !this._needsRender) || !this.geometry || this.gl.isContextLost()) return false;
    const gl = this.gl;
    const mvp = this._getMvp();

    gl.clearColor(...this.clearColor);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.bindVertexArray(this.vao);

    gl.useProgram(this.lineProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProgram, "u_mvp"), false, mvp);
    const lineColorLocation = gl.getUniformLocation(this.lineProgram, "u_color");
    if (this.showEdges && !this.isolate) {
      for (let generator = 0; generator < 7; generator += 1) {
        if (!this.generatorVisible[generator]) continue;
        const color = this.generatorColors[generator];
        gl.uniform4f(lineColorLocation, color[0], color[1], color[2], color[3] * this.lineAlpha);
        const range = this.geometry.edgeRanges[generator];
        gl.drawElements(gl.LINES, range.count, gl.UNSIGNED_SHORT, range.offset * Uint16Array.BYTES_PER_ELEMENT);
      }
    }

    if (this.showVertices || this.isolate) {
      gl.useProgram(this.pointProgram);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.pointProgram, "u_mvp"), false, mvp);
      gl.uniform3fv(gl.getUniformLocation(this.pointProgram, "u_sizes"), this.pointSizes);
      gl.uniform1f(gl.getUniformLocation(this.pointProgram, "u_pixelRatio"), this.pixelRatio);
      gl.uniform1i(gl.getUniformLocation(this.pointProgram, "u_isolate"), this.isolate ? 1 : 0);
      gl.uniform4fv(gl.getUniformLocation(this.pointProgram, "u_baseColor"), this.baseColor);
      gl.uniform4fv(gl.getUniformLocation(this.pointProgram, "u_highlightColor"), this.highlightColor);
      gl.uniform4fv(gl.getUniformLocation(this.pointProgram, "u_selectedColor"), this.selectedColor);
      gl.drawArrays(gl.POINTS, 0, S8_VERTEX_COUNT);
    }

    gl.bindVertexArray(null);
    this._needsRender = false;
    return true;
  }

  dispose() {
    if (this._disposed) return;
    this.detachControls();
    if (this._frameHandle && typeof globalThis.cancelAnimationFrame === "function") globalThis.cancelAnimationFrame(this._frameHandle);
    this._frameHandle = 0;
    this.canvas.removeEventListener("webglcontextlost", this._onContextLost);
    this.canvas.removeEventListener("webglcontextrestored", this._onContextRestored);
    const gl = this.gl;
    gl.deleteBuffer(this.positionBuffer);
    gl.deleteBuffer(this.edgeBuffer);
    gl.deleteBuffer(this.stateBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.lineProgram);
    gl.deleteProgram(this.pointProgram);
    this._disposed = true;
  }
}
