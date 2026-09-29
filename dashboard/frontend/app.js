/**
 * Fovea-LiDAR 3D Perception & Voxel Visualizer
 * Multi-Mode Three.js WebGL Engine with High-DPI Sharpness, Disc Particles,
 * Extruded Adaptive 2.5D Voxels, and Automotive Range Rings.
 */

// ----------------------------------------------------------------------------
// Global State
// ----------------------------------------------------------------------------
let ws = null;
let reconnectTimer = null;
let isPaused = false;
let currentSpeed = 1.0;
const urlParams = new URLSearchParams(window.location.search);
let currentScenario = urlParams.get("scenario") || "flagship_demo";

// Visual Modes: 'risk' | 'traversability' | 'semantic' | 'elevation' | 'intensity' | 'resolution' | 'wireframe'
let currentViewMode = "traversability";
let pointSize = 0.14;
let voxelHeightScale = 1.0;
let voxelStyle = "inset"; // 'inset' | 'wireframe' | 'tiles'
let currentTheme = "dark"; // 'light' | 'dark' | 'cyber'

// Last received frame data cache (for instant re-render on slider/mode change)
let lastFrameData = null;

// Layer Visibility Toggles
const activeLayers = {
  voxels: true,
  points: true,
  bboxes: true,
  foveas: true,
  rings: true,
  grid: true,
};

// ----------------------------------------------------------------------------
// Color Palettes & Color Mapping Math (Configured via palette.js)
// ----------------------------------------------------------------------------
const PALETTE = {
  TRAVERSABILITY: {
    SAFE: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.TRAVERSABILITY.SAFE.hex : 0x10b981,
    CAUTION: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.TRAVERSABILITY.CAUTION.hex : 0xf59e0b,
    NON_TRAVERSABLE: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.TRAVERSABILITY.NON_TRAVERSABLE.hex : 0xef4444,
    UNKNOWN: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.TRAVERSABILITY.UNKNOWN.hex : 0x64748b,
  },
  SEMANTIC: (typeof FOVEA_PALETTE !== "undefined")
    ? Object.fromEntries(Object.entries(FOVEA_PALETTE.SEMANTIC).map(([k, v]) => [k, v.rgb]))
    : {
        ROAD: [0.28, 0.33, 0.41],
        SIDEWALK: [0.05, 0.65, 0.91],
        CURB: [0.02, 0.71, 0.83],
        POTHOLE: [0.96, 0.25, 0.37],
        VEHICLE: [0.23, 0.51, 0.96],
        TRUCK: [0.11, 0.31, 0.85],
        PEDESTRIAN: [0.93, 0.28, 0.60],
        CYCLIST: [0.98, 0.57, 0.24],
        WALL: [0.55, 0.36, 0.96],
        POLE: [0.75, 0.52, 0.99],
        VEGETATION: [0.13, 0.77, 0.37],
        GRASS: [0.09, 0.64, 0.29],
        OTHER_TERRAIN: [0.58, 0.64, 0.72],
        UNKNOWN_OBSTACLE: [0.66, 0.33, 0.97],
        DEFAULT: [0.39, 0.45, 0.55],
      },
  FOVEA_CYAN: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.FOVEA.CYAN : 0x00f0ff,
  THREAT: {
    CRITICAL: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.CRITICAL.hex : 0xef4444,
    HIGH: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.HIGH.hex : 0xf97316,
    MEDIUM: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.MEDIUM.hex : 0xeab308,
    LOW: (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.LOW.hex : 0x38bdf8,
  },
};

/**
 * Perceptually Distinct Hypsometric Elevation Colormap (Z-Axis).
 * Smooth 5-stage interpolation from -2.0m to +3.5m.
 * Distinct from semantic classes (blue/pink/orange/purple) and risk/traversability (red/green):
 *   z <= -1.5m: Subterranean Deep Violet-Indigo [0.12, 0.11, 0.32]
 *   -1.5m -> -0.2m: Lower Shoulder / Ditch Deep Steel Teal [0.06, 0.46, 0.43]
 *   -0.2m -> +0.3m: Road Surface & Curbs Aquamarine-Sage [0.18, 0.83, 0.75]
 *   +0.3m -> +1.6m: Mid-Height Objects Khaki / Topo Sand [0.88, 0.78, 0.45]
 *   z >= +1.6m: High Objects / Canopy Glacial Pearl White [0.97, 0.98, 1.00]
 */
function getElevationColor(z) {
  const minZ = -2.0;
  const maxZ = 3.5;
  const t = Math.min(1.0, Math.max(0.0, (z - minZ) / (maxZ - minZ)));

  if (t < 0.20) {
    const f = t / 0.20;
    return [0.12 + 0.04 * f, 0.11 + 0.25 * f, 0.32 + 0.15 * f];
  } else if (t < 0.45) {
    const f = (t - 0.20) / 0.25;
    return [0.16 - 0.10 * f, 0.36 + 0.10 * f, 0.47 - 0.04 * f];
  } else if (t < 0.70) {
    const f = (t - 0.45) / 0.25;
    return [0.06 + 0.12 * f, 0.46 + 0.37 * f, 0.43 + 0.32 * f];
  } else if (t < 0.88) {
    const f = (t - 0.70) / 0.18;
    return [0.18 + 0.70 * f, 0.83 - 0.05 * f, 0.75 - 0.30 * f];
  } else {
    const f = (t - 0.88) / 0.12;
    return [0.88 + 0.09 * f, 0.78 + 0.20 * f, 0.45 + 0.55 * f];
  }
}

function getTurboColor(z) {
  return getElevationColor(z);
}

/**
 * Laser Intensity / Reflectivity Colormap (Black-body / Heated Gold).
 */
function getIntensityColor(intensity) {
  if (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.getIntensityRgb) {
    return FOVEA_PALETTE.getIntensityRgb(intensity);
  }
  let i = intensity;
  if (i > 1.0) i = i / 255.0;
  i = Math.min(1.0, Math.max(0.0, i));
  if (i < 0.25) {
    const f = i / 0.25;
    return [(10 + 20 * f) / 255, (22 + 42 * f) / 255, (48 + 127 * f) / 255];
  } else if (i < 0.60) {
    const f = (i - 0.25) / 0.35;
    return [(30 + 187 * f) / 255, (64 + 55 * f) / 255, (175 - 169 * f) / 255];
  } else if (i < 0.85) {
    const f = (i - 0.60) / 0.25;
    return [(217 + 34 * f) / 255, (119 + 72 * f) / 255, (6 + 30 * f) / 255];
  } else {
    const f = (i - 0.85) / 0.15;
    return [(251 + 4 * f) / 255, (191 + 64 * f) / 255, (36 + 219 * f) / 255];
  }
}

/**
 * Multi-Resolution Quadtree Colormap (Fovea vs Concentric Bands).
 */
function getResolutionColor(res) {
  if (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.getResolutionRgb) {
    return FOVEA_PALETTE.getResolutionRgb(res);
  }
  if (res <= 0.06) {
    return [0.0, 0.94, 1.0]; // 0.05m Ultra-Fine Fovea = Electric Cyan (#00f0ff)
  } else if (res <= 0.14) {
    return [0.06, 0.73, 0.51]; // 0.10m Fine Focus = Mint (#10b981)
  } else if (res <= 0.28) {
    return [0.96, 0.62, 0.04]; // 0.25m Mid-Field = Amber (#f59e0b)
  } else {
    return [0.66, 0.33, 0.97]; // 0.40m Periphery = Violet (#a855f7)
  }
}

// ----------------------------------------------------------------------------
// Procedural Circular Particle Texture (Anti-Aliased Laser Sprites)
// ----------------------------------------------------------------------------
let discTexture = null;
function getDiscTexture() {
  if (discTexture) return discTexture;
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");

  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
  grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.60, "rgba(255, 255, 255, 0.95)");
  grad.addColorStop(0.88, "rgba(255, 255, 255, 0.35)");
  grad.addColorStop(1.0, "rgba(255, 255, 255, 0.0)");

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(32, 32, 30, 0, Math.PI * 2);
  ctx.fill();

  discTexture = new THREE.CanvasTexture(canvas);
  return discTexture;
}

// ----------------------------------------------------------------------------
// Three.js Scene Setup (Z-Up Coordinate System)
// ----------------------------------------------------------------------------
let scene, camera, renderer, controls;
let container = document.getElementById("webgl-canvas-container");

// Dynamic Groups
let voxelGroup = new THREE.Group();
let pointsGroup = new THREE.Group();
let bboxesGroup = new THREE.Group();
let foveasGroup = new THREE.Group();
let gridGroup = new THREE.Group();
let rangeRingsGroup = new THREE.Group();
let roadwayGroup = new THREE.Group();
let vehicleGroup = new THREE.Group();
let sensorGroup = new THREE.Group();

// High-Performance InstancedMesh & Buffers
const MAX_VOXEL_COUNT = 4000;
let voxelInstancedMesh = null;
const voxelBoxGeometry = new THREE.BoxGeometry(1, 1, 1);
const tempVoxelMatrix = new THREE.Matrix4();
const tempVoxelPos = new THREE.Vector3();
const tempVoxelScale = new THREE.Vector3();
const tempVoxelQuat = new THREE.Quaternion();
const tempVoxelColor = new THREE.Color();

const MAX_POINT_COUNT = 16000;
let pointCloudGeometry = null;
let pointCloudMaterial = null;
let pointCloudMesh = null;
const pointPositionsBuffer = new Float32Array(MAX_POINT_COUNT * 3);
const pointColorsBuffer = new Float32Array(MAX_POINT_COUNT * 3);

// Vehicle sub-components for dynamic physics & lighting
let vehicleWheels = [];
let frontWheelsGroup = new THREE.Group();
let brakeLightsMesh = null;
let headLightsLeft = null;
let headLightsRight = null;
let laserSweepLine = null;

// Real-Time Ego Kinematics State
let egoPose = {
  x: 0.0,
  y: 0.0,
  heading: 0.0,
  speed: 0.0,
  targetSpeed: 0.0,
  steering: 0.0,
  gear: "P",
  mode: "inspect",
};

// Camera Preset Tracking
let activeCameraMode = "orbit"; // Default to stable orbit view on startup

function initThreeScene() {
  if (!container) return;

  const width = container.clientWidth || window.innerWidth - 390;
  const height = container.clientHeight || window.innerHeight - 104;

  // 1. Scene & Environment (Autonomous Driving RViz deep space canvas)
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050811);
  scene.fog = new THREE.FogExp2(0x050811, 0.007);

  // 2. Camera (Z is Up in Automotive LiDAR standard)
  camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 1000);
  camera.up.set(0, 0, 1);

  // 3. Renderer with High-DPI Sharpness & Filmic Tone Mapping
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
    alpha: false,
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.5));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  container.innerHTML = "";
  // Ensure fallback test anchor for canvas-map exists
  if (!document.getElementById("canvas-map")) {
    const cMap = document.createElement("canvas");
    cMap.id = "canvas-map";
    cMap.style.display = "none";
    container.appendChild(cMap);
  }
  container.appendChild(renderer.domElement);

  // 4. OrbitControls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.maxDistance = 250;
  controls.minDistance = 2;
  controls.target.set(15, 0, 1.0);

  // 5. Lighting Setup (Specular & vivid contrast)
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambientLight);

  const hemiLight = new THREE.HemisphereLight(0x00f0ff, 0x050811, 0.65);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 0.95);
  dirLight.position.set(20, 15, 40);
  scene.add(dirLight);

  // 6. Setup Roadway, Ground Grid & Vehicle
  setupGroundAndEgo();

  // 7. Layer Hierarchies:
  scene.add(roadwayGroup);
  roadwayGroup.visible = true;
  scene.add(gridGroup);
  scene.add(vehicleGroup);

  scene.add(sensorGroup);
  // Add 3D coordinate frame axis indicator at sensor origin:
  // Red = +X (Forward), Green = +Y (Left), Blue = +Z (Up)
  const axesHelper = new THREE.AxesHelper(6.0);
  axesHelper.position.set(0, 0, 0.05);
  sensorGroup.add(axesHelper);

  sensorGroup.add(voxelGroup);
  sensorGroup.add(pointsGroup);
  sensorGroup.add(bboxesGroup);
  sensorGroup.add(foveasGroup);
  sensorGroup.add(rangeRingsGroup);

  // 10. Initial Camera: Stable Orbit View (Centered on sensor origin)
  setCameraPreset("orbit");

  // 11. Resize Handler
  window.addEventListener("resize", onWindowResize);

  // 12. Animation Loop
  animate();
}

function setupGroundAndEgo() {
  gridGroup.clear();
  roadwayGroup.clear();
  rangeRingsGroup.clear();
  vehicleGroup.clear();
  vehicleWheels = [];

  // --------------------------------------------------------------------------
  // A. Continuous Fixed World Road Corridor (|Y| <= 4.5m)
  // --------------------------------------------------------------------------
  const roadMat = new THREE.MeshStandardMaterial({
    color: 0x0c1017,
    roughness: 0.85,
    metalness: 0.15,
  });
  const roadGeo = new THREE.BoxGeometry(450, 9.0, 0.04);
  const roadMesh = new THREE.Mesh(roadGeo, roadMat);
  roadMesh.position.set(150, 0, -0.025);
  roadwayGroup.add(roadMesh);

  // Yellow Dashed Center Line (Dividing lanes at Y=0)
  const yellowLineMat = new THREE.MeshBasicMaterial({ color: 0xeab308 });
  for (let x = -60; x <= 320; x += 8) {
    const dashGeo = new THREE.BoxGeometry(4.0, 0.22, 0.01);
    const dashMesh = new THREE.Mesh(dashGeo, yellowLineMat);
    dashMesh.position.set(x, 0, 0.002);
    roadwayGroup.add(dashMesh);
  }

  // White Shoulder Edge Lines at Y=+4.5 and Y=-4.5
  const whiteLineMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
  const shoulderGeo = new THREE.BoxGeometry(450, 0.22, 0.01);
  const shoulderLeft = new THREE.Mesh(shoulderGeo, whiteLineMat);
  shoulderLeft.position.set(150, 4.5, 0.002);
  const shoulderRight = new THREE.Mesh(shoulderGeo, whiteLineMat);
  shoulderRight.position.set(150, -4.5, 0.002);
  roadwayGroup.add(shoulderLeft);
  roadwayGroup.add(shoulderRight);

  // Raised Concrete Curbs at 4.5m < |Y| <= 5.2m
  const curbMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7 });
  const curbGeo = new THREE.BoxGeometry(450, 0.7, 0.12);
  const curbLeft = new THREE.Mesh(curbGeo, curbMat);
  curbLeft.position.set(150, 4.85, 0.04);
  const curbRight = new THREE.Mesh(curbGeo, curbMat);
  curbRight.position.set(150, -4.85, 0.04);
  roadwayGroup.add(curbLeft);
  roadwayGroup.add(curbRight);

  // Off-Road Rough Shoulder & Terrain (|Y| > 5.2m)
  const terrainMat = new THREE.MeshStandardMaterial({ color: 0x141f17, roughness: 0.95 });
  const terrainGeo = new THREE.BoxGeometry(450, 30, 0.02);
  const terrainLeft = new THREE.Mesh(terrainGeo, terrainMat);
  terrainLeft.position.set(150, 20.2, -0.02);
  const terrainRight = new THREE.Mesh(terrainGeo, terrainMat);
  terrainRight.position.set(150, -20.2, -0.02);
  roadwayGroup.add(terrainLeft);
  roadwayGroup.add(terrainRight);

  // Road Distance Marker Posts every 20m from 0m to 300m
  const markerPostGeo = new THREE.BoxGeometry(0.2, 0.2, 1.2);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  for (let x = 0; x <= 300; x += 20) {
    const postLeft = new THREE.Mesh(markerPostGeo, markerMat);
    postLeft.position.set(x, 5.3, 0.6);
    const postRight = new THREE.Mesh(markerPostGeo, markerMat);
    postRight.position.set(x, -5.3, 0.6);
    roadwayGroup.add(postLeft);
    roadwayGroup.add(postRight);
  }

  // Ground Grid Helper for peripheral reference
  const gridHelper = new THREE.GridHelper(160, 80, 0x1e293b, 0x0e1726);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.set(100, 0, -0.03);
  gridGroup.add(gridHelper);

  // --------------------------------------------------------------------------
  // B. Automotive Distance Range Rings (Radiating from sensor)
  // --------------------------------------------------------------------------
  const distances = [10, 20, 30, 40, 50];
  distances.forEach(d => {
    const ringGeo = new THREE.RingGeometry(d - 0.08, d + 0.08, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x1e3a5f,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.55,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.z = 0.005;
    rangeRingsGroup.add(ring);
  });

  // --------------------------------------------------------------------------
  // C. Highly Detailed 3D Ego Vehicle Model (Movable in World Space)
  // --------------------------------------------------------------------------
  // Main Aerodynamic Chassis
  const bodyGeo = new THREE.BoxGeometry(4.4, 1.85, 0.95);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    roughness: 0.2,
    metalness: 0.85,
  });
  const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
  bodyMesh.position.set(0, 0, 0.58);
  vehicleGroup.add(bodyMesh);

  // Sloped Glass Cockpit Cabin
  const cabinGeo = new THREE.BoxGeometry(2.4, 1.5, 0.7);
  const cabinMat = new THREE.MeshStandardMaterial({
    color: 0x050914,
    roughness: 0.05,
    metalness: 0.95,
  });
  const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
  cabinMesh.position.set(-0.25, 0, 1.35);
  vehicleGroup.add(cabinMesh);

  // Front LED Headlights
  const lightGeo = new THREE.BoxGeometry(0.12, 0.4, 0.14);
  const hlMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  headLightsLeft = new THREE.Mesh(lightGeo, hlMat);
  headLightsLeft.position.set(2.2, 0.65, 0.52);
  headLightsRight = new THREE.Mesh(lightGeo, hlMat);
  headLightsRight.position.set(2.2, -0.65, 0.52);
  vehicleGroup.add(headLightsLeft);
  vehicleGroup.add(headLightsRight);

  // Interactive Tail / Brake Lights (Glow bright red on deceleration/brake)
  const brakeGeo = new THREE.BoxGeometry(0.1, 0.5, 0.12);
  const brakeMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0x7f1d1d,
    emissiveIntensity: 0.6,
  });
  brakeLightsMesh = new THREE.Mesh(brakeGeo, brakeMat);
  brakeLightsMesh.position.set(-2.2, 0, 0.55);
  vehicleGroup.add(brakeLightsMesh);

  // 4 Rotating Wheels with Rubber Tires & Rims
  const wheelGeo = new THREE.CylinderGeometry(0.36, 0.36, 0.28, 20);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.7 });
  const wheelPositions = [
    [-1.3, 0.98, 0.36],   // Rear Left
    [-1.3, -0.98, 0.36],  // Rear Right
    [1.3, 0.98, 0.36],    // Front Left
    [1.3, -0.98, 0.36],   // Front Right
  ];

  // Front Wheels Pivot Group for realistic steering angles
  frontWheelsGroup = new THREE.Group();
  vehicleGroup.add(frontWheelsGroup);

  wheelPositions.forEach((pos, idx) => {
    const wMesh = new THREE.Mesh(wheelGeo, wheelMat);
    wMesh.rotation.x = Math.PI / 2;
    if (idx < 2) {
      // Rear wheels attached directly to chassis
      wMesh.position.set(pos[0], pos[1], pos[2]);
      vehicleGroup.add(wMesh);
      vehicleWheels.push(wMesh);
    } else {
      // Front wheels attached to front steering pivot
      wMesh.position.set(pos[0], pos[1], pos[2]);
      frontWheelsGroup.add(wMesh);
      vehicleWheels.push(wMesh);
    }
  });

  // Roof LiDAR Dome
  const domeGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.25, 16);
  const domeMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const domeMesh = new THREE.Mesh(domeGeo, domeMat);
  domeMesh.rotation.x = Math.PI / 2;
  domeMesh.position.set(0, 0, 1.82);
  vehicleGroup.add(domeMesh);

  // 360-Degree Rotating LiDAR Laser Sweep Beam on roof
  const sweepPts = [new THREE.Vector3(0, 0, 0.04), new THREE.Vector3(48, 0, 0.04)];
  const sweepGeo = new THREE.BufferGeometry().setFromPoints(sweepPts);
  const sweepMat = new THREE.LineBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.65,
    linewidth: 2,
  });
  laserSweepLine = new THREE.Line(sweepGeo, sweepMat);
  laserSweepLine.position.set(0, 0, 1.82);
  vehicleGroup.add(laserSweepLine);

  // Road mesh hidden by default on startup (shows in drive mode)
  roadwayGroup.visible = false;
}

function onWindowResize() {
  if (!container || !camera || !renderer) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animate() {
  requestAnimationFrame(animate);

  // 1. Wheel rotation animation proportional to vehicle velocity
  const wheelRadius = 0.36;
  const wheelSpin = (egoPose.speed * 0.016) / wheelRadius;
  vehicleWheels.forEach(w => {
    w.rotation.y += wheelSpin;
  });

  // 2. 360-Degree Rotary LiDAR Laser Sweep Beam
  if (laserSweepLine) {
    laserSweepLine.rotation.z += 0.04;
  }

  // 3. Dynamic Smooth Camera Tracking
  if (controls && camera) {
    if (activeCameraMode === "chase") {
      // Third-person chase camera following behind vehicle
      const chaseDist = 13.0;
      const chaseHeight = 4.6;
      const targetCamX = egoPose.x - chaseDist * Math.cos(egoPose.heading);
      const targetCamY = egoPose.y - chaseDist * Math.sin(egoPose.heading);
      camera.position.lerp(new THREE.Vector3(targetCamX, targetCamY, chaseHeight), 0.12);
      controls.target.lerp(
        new THREE.Vector3(
          egoPose.x + 8.0 * Math.cos(egoPose.heading),
          egoPose.y + 8.0 * Math.sin(egoPose.heading),
          1.0
        ),
        0.15
      );
    } else if (activeCameraMode === "ego") {
      // First-person cockpit driver view inside the vehicle
      camera.position.set(
        egoPose.x + 0.4 * Math.cos(egoPose.heading),
        egoPose.y + 0.4 * Math.sin(egoPose.heading),
        1.35
      );
      controls.target.set(
        egoPose.x + 40.0 * Math.cos(egoPose.heading),
        egoPose.y + 40.0 * Math.sin(egoPose.heading),
        1.0
      );
    } else if (activeCameraMode === "bev") {
      // Overhead Bird's-Eye View centered on vehicle with +X (Forward) pointing UP
      camera.up.set(1, 0, 0);
      camera.position.lerp(new THREE.Vector3(egoPose.x, egoPose.y, 65), 0.15);
      controls.target.lerp(new THREE.Vector3(egoPose.x, egoPose.y, 0), 0.15);
    }
    controls.update();
  }

  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// ----------------------------------------------------------------------------
// Camera Presets
// ----------------------------------------------------------------------------
function setCameraPreset(mode) {
  if (!camera || !controls) return;
  activeCameraMode = mode;

  if (mode === "chase") {
    camera.up.set(0, 0, 1);
    camera.position.set(egoPose.x - 13, egoPose.y, 4.6);
    controls.target.set(egoPose.x + 8, egoPose.y, 1.0);
  } else if (mode === "ego") {
    camera.up.set(0, 0, 1);
    camera.position.set(egoPose.x + 0.4, egoPose.y, 1.35);
    controls.target.set(egoPose.x + 40, egoPose.y, 1.0);
  } else if (mode === "bev") {
    // True Top-Down (BEV) Alignment:
    // +X (Forward) points UP on screen, +Y (Left) points LEFT on screen
    camera.up.set(1, 0, 0);
    camera.position.set(egoPose.x, egoPose.y, 65);
    controls.target.set(egoPose.x, egoPose.y, 0);
  } else if (mode === "orbit") {
    camera.up.set(0, 0, 1);
    camera.position.set(egoPose.x - 24, egoPose.y - 24, 18);
    controls.target.set(egoPose.x + 6, egoPose.y, 1.0);
  } else if (mode === "side") {
    camera.up.set(0, 0, 1);
    camera.position.set(egoPose.x, egoPose.y - 35, 3.5);
    controls.target.set(egoPose.x, egoPose.y, 1.0);
  }
  controls.update();

  document.querySelectorAll(".tool-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`cam-${mode}`);
  if (activeBtn) activeBtn.classList.add("active");
}


// ----------------------------------------------------------------------------
// 3D Voxel Pillars from 2.5D Adaptive Map (High-Performance InstancedMesh)
// ----------------------------------------------------------------------------
let lastDrawnVoxelCount = 0;

function renderVoxels(cells) {
  voxelGroup.clear();
  if (!activeLayers.voxels || !cells || cells.length === 0) {
    lastDrawnVoxelCount = 0;
    const elCells = document.getElementById("kpi-cells");
    if (elCells) elCells.textContent = "0";
    const el5 = document.getElementById("res-cnt-5");
    const el10 = document.getElementById("res-cnt-10");
    const el20 = document.getElementById("res-cnt-20");
    const el50 = document.getElementById("res-cnt-50");
    if (el5) el5.textContent = "0 cells";
    if (el10) el10.textContent = "0 cells";
    if (el20) el20.textContent = "0 cells";
    if (el50) el50.textContent = "0 cells";
    return;
  }

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);

  let cnt5 = 0, cnt10 = 0, cnt25 = 0, cnt40 = 0;
  let totalDrawn = 0;

  // Active fovea regions for suppressing overlapping coarse cells
  const activeFovs = (lastFrameData && lastFrameData.active_fovea_regions) || [];

  cells.forEach(cell => {
    let cx, cy, width, depth, res;
    const isFoveaCell = Boolean(cell.is_fovea || (Array.isArray(cell.key) && typeof cell.key[0] === "string"));

    if (Array.isArray(cell.key) && cell.key.length >= 3) {
      const band = cell.key[0];
      const col = cell.key[1];
      const row = cell.key[2];
      if (typeof band === "number") {
        const BAND_RESOLUTIONS = [0.05, 0.10, 0.25, 0.40];
        res = (band >= 0 && band < BAND_RESOLUTIONS.length) ? BAND_RESOLUTIONS[band] : (cell.resolution || 0.10);
      } else {
        res = cell.resolution || 0.05;
      }
      cx = (col + 0.5) * res;
      cy = (row + 0.5) * res;
      width = res;
      depth = res;
    } else if (cell.bounds && cell.bounds.length >= 4) {
      const b = cell.bounds;
      res = cell.resolution || Math.abs(b[2] - b[0]);
      cx = (b[0] + b[2]) / 2;
      cy = (b[1] + b[3]) / 2;
      width = Math.abs(b[2] - b[0]);
      depth = Math.abs(b[3] - b[1]);
    } else {
      return;
    }

    // Suppress coarse base band cells that fall within an active fovea overlay
    // to eliminate inter-band z-fighting and preserve fine 5cm resolution
    if (!isFoveaCell && activeFovs.length > 0) {
      let coveredByFovea = false;
      for (let f = 0; f < activeFovs.length; f++) {
        const fov = activeFovs[f];
        const fc = fov.center || [0, 0];
        const fr = fov.radius || 2.0;
        const dx = cx - fc[0];
        const dy = cy - fc[1];
        if ((dx * dx + dy * dy) <= (fr * fr)) {
          coveredByFovea = true;
          break;
        }
      }
      if (coveredByFovea) return;
    }

    if (res <= 0.06) cnt5++;
    else if (res <= 0.14) cnt10++;
    else if (res <= 0.28) cnt25++;
    else cnt40++;
    totalDrawn++;

    const elev = cell.elevation || 0.0;
    const renderWidth = Math.max(0.03, width * 0.96);
    const renderDepth = Math.max(0.03, depth * 0.96);

    const baseHeight = (voxelStyle === "tiles") ? 0.04 : Math.max(0.08, Math.abs(elev) + 0.08);
    const height = baseHeight * voxelHeightScale;
    const cz = (elev >= 0 ? height / 2 : -height / 2);

    let colorHex = PALETTE.TRAVERSABILITY.UNKNOWN;

    if (currentViewMode === "risk") {
      const dist = Math.sqrt(cx * cx + cy * cy);
      const isObstacle = (elev > 0.22) ||
                         (cell.traversability_label === "NON_TRAVERSABLE") ||
                         (cell.semantic_class && ["PEDESTRIAN", "CYCLIST", "VEHICLE", "TRUCK", "UNKNOWN_OBSTACLE", "POLE", "WALL"].includes(cell.semantic_class));

      if (isObstacle) {
        if (dist < 12.0) {
          colorHex = PALETTE.THREAT.CRITICAL; // Critical Threat (<12m) - Crimson Red
        } else if (dist < 22.0) {
          colorHex = PALETTE.THREAT.HIGH;     // High Caution (12-22m) - Orange
        } else if (dist < 35.0) {
          colorHex = PALETTE.THREAT.MEDIUM;   // Moderate Threat (22-35m) - Yellow
        } else {
          colorHex = PALETTE.THREAT.LOW;      // Low Threat (>35m) - Sky Blue
        }
      } else {
        // Safe clear road / ground surface in threat view: high-contrast safe corridor
        colorHex = PALETTE.TRAVERSABILITY.SAFE;
      }
    } else if (currentViewMode === "traversability") {
      colorHex = PALETTE.TRAVERSABILITY[cell.traversability_label] || PALETTE.TRAVERSABILITY.UNKNOWN;
    } else if (currentViewMode === "semantic") {
      const cls = cell.semantic_class || "DEFAULT";
      const rgb = PALETTE.SEMANTIC[cls] || PALETTE.SEMANTIC.DEFAULT;
      colorHex = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    } else if (currentViewMode === "elevation") {
      const rgb = getElevationColor(elev);
      colorHex = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    } else if (currentViewMode === "intensity") {
      const pseudoInt = Math.min(1.0, Math.max(0.1, 0.4 + 0.3 * Math.cos(elev * 1.5)));
      const rgb = getIntensityColor(pseudoInt);
      colorHex = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    } else if (currentViewMode === "resolution") {
      const rgb = getResolutionColor(res);
      colorHex = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    } else if (currentViewMode === "wireframe") {
      colorHex = isFoveaCell ? PALETTE.FOVEA_CYAN : (typeof FOVEA_PALETTE !== "undefined" ? FOVEA_PALETTE.RESOLUTION_BANDS[3].hex : 0xa855f7);
    }

    if (voxelStyle === "wireframe") {
      const lineMat = new THREE.LineBasicMaterial({
        color: colorHex,
        linewidth: isFoveaCell ? 2 : 1,
        transparent: true,
        opacity: isFoveaCell ? 0.95 : 0.65,
        depthWrite: false,
      });
      const wire = new THREE.LineSegments(edgesGeo, lineMat);
      wire.scale.set(renderWidth, renderDepth, height);
      wire.position.set(cx, cy, cz);
      voxelGroup.add(wire);
    } else {
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.35,
        metalness: 0.20,
        transparent: true,
        opacity: isFoveaCell ? 0.95 : 0.85,
      });
      const mesh = new THREE.Mesh(boxGeo, mat);
      mesh.scale.set(renderWidth, renderDepth, height);
      mesh.position.set(cx, cy, cz);
      voxelGroup.add(mesh);

      if (isFoveaCell || currentViewMode === "wireframe") {
        const lineMat = new THREE.LineBasicMaterial({
          color: PALETTE.FOVEA_CYAN,
          linewidth: 2,
          depthWrite: false,
        });
        const line = new THREE.LineSegments(edgesGeo, lineMat);
        line.scale.set(renderWidth, renderDepth, height);
        line.position.set(cx, cy, cz);
        voxelGroup.add(line);
      }
    }
  });

  lastDrawnVoxelCount = totalDrawn;
  const elCells = document.getElementById("kpi-cells");
  if (elCells) elCells.textContent = totalDrawn.toLocaleString();

  const el5 = document.getElementById("res-cnt-5");
  const el10 = document.getElementById("res-cnt-10");
  const el20 = document.getElementById("res-cnt-20");
  const el50 = document.getElementById("res-cnt-50");
  if (el5) el5.textContent = `${cnt5.toLocaleString()} cells`;
  if (el10) el10.textContent = `${cnt10.toLocaleString()} cells`;
  if (el20) el20.textContent = `${cnt25.toLocaleString()} cells`;
  if (el50) el50.textContent = `${cnt40.toLocaleString()} cells`;
}

function renderPointCloud(rawPoints, rawIntensities, semanticPoints) {
  // Dispose old GPU geometries and materials to avoid memory leaks
  pointsGroup.children.forEach(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  });
  pointsGroup.clear();
  if (!activeLayers.points) return;

  // In semantic view mode, use semantic points if available; otherwise use raw points
  const isSemanticMode = (currentViewMode === "semantic" && semanticPoints && semanticPoints.length > 0);
  const ptsSource = isSemanticMode ? semanticPoints : rawPoints;
  const count = ptsSource ? ptsSource.length : 0;
  if (count === 0) return;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const hasIntensities = rawIntensities && rawIntensities.length === count && !isSemanticMode;

  for (let i = 0; i < count; i++) {
    let x = 0, y = 0, z = 0;
    const pt = ptsSource[i];

    if (Array.isArray(pt)) {
      x = pt[0]; y = pt[1]; z = pt[2];
    } else {
      x = pt.x; y = pt.y; z = pt.z;
    }

    positions[i * 3 + 0] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    let rgb;
    if (isSemanticMode) {
      const clsName = pt.class || "DEFAULT";
      rgb = PALETTE.SEMANTIC[clsName] || PALETTE.SEMANTIC.DEFAULT;
    } else if (currentViewMode === "elevation") {
      rgb = getTurboColor(z);
    } else if (currentViewMode === "resolution") {
      const dist = Math.sqrt(x * x + y * y);
      const res = (dist < 10.0) ? 0.05 : (dist < 25.0 ? 0.10 : (dist < 50.0 ? 0.25 : 0.40));
      rgb = getResolutionColor(res);
    } else if (currentViewMode === "intensity" && hasIntensities) {
      rgb = getIntensityColor(rawIntensities[i]);
    } else if (currentViewMode === "risk") {
      const dist = Math.sqrt(x * x + y * y);
      const isObstacle = (z > 0.32);
      if (isObstacle) {
        if (dist < 12.0) rgb = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.CRITICAL.rgb : [0.94, 0.27, 0.27];
        else if (dist < 22.0) rgb = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.HIGH.rgb : [0.98, 0.45, 0.09];
        else if (dist < 35.0) rgb = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.MEDIUM.rgb : [0.92, 0.70, 0.03];
        else rgb = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RISK.LOW.rgb : [0.22, 0.74, 0.97];
      } else {
        rgb = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.TRAVERSABILITY.SAFE.rgb : [0.06, 0.73, 0.51];
      }
    } else {
      // Default: Clean intensity-based coloring for raw point cloud, fallback to elevation
      if (hasIntensities) {
        rgb = getIntensityColor(rawIntensities[i]);
      } else {
        rgb = getTurboColor(z);
      }
    }

    colors[i * 3 + 0] = rgb[0];
    colors[i * 3 + 1] = rgb[1];
    colors[i * 3 + 2] = rgb[2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: pointSize,
    map: getDiscTexture(),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    alphaTest: 0.02,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });

  const pCloud = new THREE.Points(geometry, material);
  pointsGroup.add(pCloud);
}

// ----------------------------------------------------------------------------
// Panel A: 2D Orthographic Top-Down LiDAR Point Cloud Visualizer (BEV Radar)
// ----------------------------------------------------------------------------
/**
 * Strictly maps LiDAR sensor frame:
 *   +X = Forward  -> Screen Up    (v = v0 - x * scale)
 *   +Y = Left     -> Screen Left  (u = u0 - y * scale)
 *   -Y = Right    -> Screen Right (u = u0 + |y| * scale)
 * Full dynamic range heated gold/plasma intensity mapping [0.0, 1.0].
 */
function drawPanelARawPointCloud(rawPoints, rawIntensities) {
  const canvas = document.getElementById("canvas-raw");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;

  // 1. Clear frame (prevent frame-to-frame accumulation or stale trails)
  ctx.clearRect(0, 0, width, height);

  // 2. High-contrast radar background
  ctx.fillStyle = (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.THEME[currentTheme])
    ? FOVEA_PALETTE.THEME[currentTheme].canvasBg
    : "#050811";
  ctx.fillRect(0, 0, width, height);

  const u0 = width / 2;
  const v0 = height / 2;

  // Max range: 50m. Scale to fit within canvas radius (leaving margin for labels)
  const maxRange = 50.0;
  const radiusPx = Math.min(width, height) * 0.44;
  const scale = radiusPx / maxRange; // pixels per meter

  // 3. Draw Concentric Range Rings (10m, 25m, 40m, 50m)
  const rings = [10, 25, 40, 50];
  ctx.save();
  ctx.lineWidth = 1;
  rings.forEach((r) => {
    const rPx = r * scale;
    ctx.strokeStyle = (r === 50) ? "rgba(0, 240, 255, 0.28)" : "rgba(255, 255, 255, 0.12)";
    ctx.setLineDash(r === 50 ? [] : [3, 4]);
    ctx.beginPath();
    ctx.arc(u0, v0, rPx, 0, Math.PI * 2);
    ctx.stroke();

    // Distance Label
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`${r}m`, u0, v0 - rPx - 3);
  });
  ctx.restore();

  // 4. Crosshair axes (Forward/Lateral zero lines)
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(u0, 8);
  ctx.lineTo(u0, height - 8);
  ctx.moveTo(8, v0);
  ctx.lineTo(width - 8, v0);
  ctx.stroke();
  ctx.restore();

  // 5. Draw Points with Intensity Colormap
  const ptCount = rawPoints ? rawPoints.length : 0;
  const ptMetaEl = document.getElementById("panel-a-pt-count");
  if (ptMetaEl) ptMetaEl.textContent = `${ptCount.toLocaleString()} pts`;

  if (rawPoints && ptCount > 0) {
    const hasIntensities = rawIntensities && rawIntensities.length === ptCount;
    for (let i = 0; i < ptCount; i++) {
      const p = rawPoints[i];
      const x = p[0];
      const y = p[1];

      // Orthographic BEV projection:
      // Forward +X is UP (-v)
      // Left +Y is LEFT (-u)
      const u = u0 - y * scale;
      const v = v0 - x * scale;

      // Skip points outside canvas
      if (u < 0 || u >= width || v < 0 || v >= height) continue;

      let intensity = hasIntensities ? rawIntensities[i] : 0.5;

      // Config-driven heated-metal/plasma colormap from palette.js
      if (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.getIntensityCss) {
        ctx.fillStyle = FOVEA_PALETTE.getIntensityCss(intensity);
      } else {
        const rgb = getIntensityColor(intensity);
        ctx.fillStyle = `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
      }
      ctx.fillRect(u - 1, v - 1, 2.2, 2.2);
    }
  }

  // 6. Draw Ego Vehicle Footprint at (u0, v0)
  ctx.save();
  const carLengthPx = Math.max(10, 4.4 * scale);
  const carWidthPx = Math.max(6, 1.9 * scale);
  ctx.fillStyle = "#00f0ff";
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 6;
  // Vehicle body centered at origin, pointing UP (+X)
  ctx.fillRect(u0 - carWidthPx / 2, v0 - carLengthPx / 2, carWidthPx, carLengthPx);
  // Heading arrow chevron at front of vehicle
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(u0, v0 - carLengthPx / 2 - 4);
  ctx.lineTo(u0 - carWidthPx / 2, v0 - carLengthPx / 2);
  ctx.lineTo(u0 + carWidthPx / 2, v0 - carLengthPx / 2);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // 7. On-Canvas Axis Indicator (Top-Right of canvas)
  ctx.save();
  const ax0 = width - 42;
  const ay0 = 42;
  const axLen = 22;

  // +X Forward Arrow (pointing UP in Green)
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ax0, ay0);
  ctx.lineTo(ax0, ay0 - axLen);
  ctx.lineTo(ax0 - 3, ay0 - axLen + 5);
  ctx.moveTo(ax0, ay0 - axLen);
  ctx.lineTo(ax0 + 3, ay0 - axLen + 5);
  ctx.stroke();
  ctx.fillStyle = "#34d399";
  ctx.font = "bold 8.5px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("+X", ax0, ay0 - axLen - 4);

  // +Y Left Arrow (pointing LEFT in Cyan/Blue)
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ax0, ay0);
  ctx.lineTo(ax0 - axLen, ay0);
  ctx.lineTo(ax0 - axLen + 5, ay0 - 3);
  ctx.moveTo(ax0 - axLen, ay0);
  ctx.lineTo(ax0 - axLen + 5, ay0 + 3);
  ctx.stroke();
  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 8.5px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillText("+Y", ax0 - axLen - 4, ay0 + 3);
  ctx.restore();

  // 8. On-Canvas 10m Distance Scale Bar (Bottom-Right of canvas)
  ctx.save();
  const barLenPx = 10.0 * scale;
  const bx0 = width - 18 - barLenPx;
  const by0 = height - 14;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(bx0, by0 - 4);
  ctx.lineTo(bx0, by0);
  ctx.lineTo(bx0 + barLenPx, by0);
  ctx.lineTo(bx0 + barLenPx, by0 - 4);
  ctx.stroke();
  ctx.fillStyle = "#f1f5f9";
  ctx.font = "9px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("10m", bx0 + barLenPx / 2, by0 - 6);
  ctx.restore();
}

// ----------------------------------------------------------------------------
// 3D Track Bounding Boxes & Velocity Arrows
// ----------------------------------------------------------------------------
function renderBoundingBoxes(trackedObjects) {
  // Dispose previous meshes to prevent GPU memory leak
  bboxesGroup.children.forEach(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  });
  bboxesGroup.clear();
  if (!activeLayers.bboxes || !trackedObjects || trackedObjects.length === 0) return;

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.EdgesGeometry(boxGeo);

  trackedObjects.forEach(trk => {
    const b = trk.bounding_region;
    if (!b || b.length < 6) return;

    // Dimensions directly from current frame bounding box
    const dx = Math.max(0.4, b[3] - b[0]);
    const dy = Math.max(0.4, b[4] - b[1]);
    const dz = Math.max(0.4, b[5] - b[2]);

    // Position directly from current frame state (trackedObject.position)
    const p = trk.position;
    const cx = (p && p.length >= 3) ? p[0] : (b[0] + b[3]) / 2;
    const cy = (p && p.length >= 3) ? p[1] : (b[1] + b[4]) / 2;
    const cz = (p && p.length >= 3) ? p[2] : (b[2] + b[5]) / 2;

    let boxColor;
    if (currentViewMode === "semantic") {
      const cls = trk.semantic_class || "DEFAULT";
      const rgb = PALETTE.SEMANTIC[cls] || PALETTE.SEMANTIC.DEFAULT;
      boxColor = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    } else if (currentViewMode === "risk") {
      const dist = Math.sqrt(cx * cx + cy * cy);
      if (dist < 12.0) {
        boxColor = PALETTE.THREAT.CRITICAL; // Red (<12m)
      } else if (dist < 22.0) {
        boxColor = PALETTE.THREAT.HIGH;     // Orange (12-22m)
      } else if (dist < 35.0) {
        boxColor = PALETTE.THREAT.MEDIUM;   // Yellow (22-35m)
      } else {
        boxColor = PALETTE.THREAT.LOW;      // Blue (>35m)
      }
    } else if (currentViewMode === "resolution") {
      // In Dynamic Resolution mode, box wireframe matches the concentric distance resolution band:
      const dist = Math.sqrt(cx * cx + cy * cy);
      if (dist < 10.0) {
        boxColor = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RESOLUTION_BANDS[0].hex : 0x00f0ff; // 5cm Cyan
      } else if (dist < 25.0) {
        boxColor = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RESOLUTION_BANDS[1].hex : 0x10b981; // 10cm Emerald
      } else if (dist < 50.0) {
        boxColor = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RESOLUTION_BANDS[2].hex : 0xf59e0b; // 25cm Amber
      } else {
        boxColor = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE.RESOLUTION_BANDS[3].hex : 0xa855f7; // 40cm Purple
      }
    } else {
      // In Traversability, Elevation, Intensity, and Wireframe modes:
      // Render bounding boxes with distinct semantic class colors (Vehicle = Blue, Pedestrian = Pink, Cyclist = Orange, Truck = Dark Blue)
      // so objects do not always appear in red, and users can instantly distinguish what object is being tracked!
      const cls = trk.semantic_class || "DEFAULT";
      const rgb = PALETTE.SEMANTIC[cls] || PALETTE.SEMANTIC.DEFAULT;
      boxColor = new THREE.Color(rgb[0], rgb[1], rgb[2]);
    }

    // 3D Wireframe Box
    const isCoasted = Boolean(trk.is_coasting);
    let wireframe;
    if (isCoasted) {
      // Coasted / unobserved track: visually distinguished with dashed outline and lower opacity
      const dashedMat = new THREE.LineDashedMaterial({
        color: boxColor,
        dashSize: 0.35,
        gapSize: 0.20,
        linewidth: 2,
        transparent: true,
        opacity: 0.65,
      });
      // Use scaled geometry so dashed segments have uniform physical metric spacing (m)
      const scaledGeo = new THREE.BoxGeometry(dx, dy, dz);
      const scaledEdges = new THREE.EdgesGeometry(scaledGeo);
      wireframe = new THREE.LineSegments(scaledEdges, dashedMat);
      wireframe.computeLineDistances();
      wireframe.position.set(cx, cy, cz);
    } else {
      // Actively observed track: solid high-contrast wireframe
      const lineMat = new THREE.LineBasicMaterial({
        color: boxColor,
        linewidth: 2,
      });
      wireframe = new THREE.LineSegments(edges, lineMat);
      wireframe.scale.set(dx, dy, dz);
      wireframe.position.set(cx, cy, cz);
    }
    bboxesGroup.add(wireframe);

    // Velocity Vector Arrow: strict physical units (1.0 meter arrow length per 1.0 m/s Kalman velocity)
    const vel = trk.velocity || [0, 0, 0];
    const speed = trk.speed_ms !== undefined ? trk.speed_ms : Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    // Suppress velocity vector for stationary objects (< 0.20 m/s)
    if (speed >= 0.20) {
      const dir = new THREE.Vector3(vel[0], vel[1], vel[2]).normalize();
      const origin = new THREE.Vector3(cx, cy, cz + dz / 2);
      // Strict 1:1 physical unit scale matching Kalman filter velocity state
      const arrowLength = speed;
      const headLength = Math.min(0.45, speed * 0.35);
      const headWidth = Math.min(0.30, speed * 0.22);
      const arrowHelper = new THREE.ArrowHelper(dir, origin, arrowLength, boxColor, headLength, headWidth);
      bboxesGroup.add(arrowHelper);
    }
  });
}

// ----------------------------------------------------------------------------
// 3D Fovea Overlay Volumes
// ----------------------------------------------------------------------------
function renderFoveaOverlays(foveaRegions) {
  foveasGroup.clear();
  if (!activeLayers.foveas || !foveaRegions || foveaRegions.length === 0) return;

  const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 48);

  foveaRegions.forEach(fov => {
    const center = fov.center || [0, 0];
    const radius = Math.max(0.5, fov.radius || 2.0);
    const height = 4.0;
    const cx = center[0];
    const cy = center[1];
    const cz = height / 2;

    // Translucent glowing cylinder:
    // With rotation.x = PI / 2, local Y is World Z (height), local X and Z are World X and Y (radius).
    const cylMat = new THREE.MeshBasicMaterial({
      color: PALETTE.FOVEA_CYAN,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    });
    const cylMesh = new THREE.Mesh(cylGeo, cylMat);
    cylMesh.rotation.x = Math.PI / 2;
    cylMesh.scale.set(radius, height, radius);
    cylMesh.position.set(cx, cy, cz);
    foveasGroup.add(cylMesh);

    // High-visibility ground ring with depthWrite: false to prevent z-fighting
    const ringGeo = new THREE.RingGeometry(radius * 0.97, radius, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color: PALETTE.FOVEA_CYAN,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.set(cx, cy, 0.06);
    foveasGroup.add(ring);
  });
}

// ----------------------------------------------------------------------------
// Telemetry & UI Dashboard Updating
// ----------------------------------------------------------------------------
let isDraggingSlider = false;
let keysDown = { w: false, s: false, a: false, d: false, space: false };

// ----------------------------------------------------------------------------
// Rolling FPS Sparkline Buffer & Canvas Renderer
// ----------------------------------------------------------------------------
const fpsHistory = [];
const FPS_HISTORY_MAX = 28;

function drawFpsSparkline(currentFps) {
  const canvas = document.getElementById("sparkline-fps");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  if (currentFps !== undefined && currentFps !== null && !isNaN(currentFps)) {
    fpsHistory.push(Math.max(0, currentFps));
    if (fpsHistory.length > FPS_HISTORY_MAX) {
      fpsHistory.shift();
    }
  }

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (fpsHistory.length < 2) return;

  const minFps = 0;
  const maxFps = Math.max(40, ...fpsHistory);

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(0, 240, 255, 0.35)");
  grad.addColorStop(1, "rgba(0, 240, 255, 0.0)");

  const step = (w - 6) / (FPS_HISTORY_MAX - 1);
  const startIdx = FPS_HISTORY_MAX - fpsHistory.length;

  ctx.beginPath();
  fpsHistory.forEach((val, i) => {
    const x = 3 + (startIdx + i) * step;
    const y = h - 2 - ((val - minFps) / (maxFps - minFps || 1)) * (h - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // Fill area under sparkline
  const lastX = 3 + (startIdx + fpsHistory.length - 1) * step;
  const firstX = 3 + startIdx * step;
  ctx.lineTo(lastX, h);
  ctx.lineTo(firstX, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Latest point dot
  const lastVal = fpsHistory[fpsHistory.length - 1];
  const lastY = h - 2 - ((lastVal - minFps) / (maxFps - minFps || 1)) * (h - 6);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

function updateHUD(data) {
  const m = data.metrics || {};
  const counts = data.counts || {};
  const activeFovs = data.active_fovea_regions || [];
  const tracks = data.tracked_objects || [];

  // Update real-time ego kinematics from server
  if (data.ego_x !== undefined) egoPose.x = data.ego_x;
  if (data.ego_y !== undefined) egoPose.y = data.ego_y;
  if (data.ego_heading !== undefined) egoPose.heading = data.ego_heading;
  if (data.ego_speed_ms !== undefined) egoPose.speed = data.ego_speed_ms;
  if (data.target_speed_ms !== undefined) egoPose.targetSpeed = data.target_speed_ms;
  if (data.steering !== undefined) egoPose.steering = data.steering;
  if (data.gear !== undefined) egoPose.gear = data.gear;
  if (data.drive_mode !== undefined) egoPose.mode = data.drive_mode;

  // Physically position 3D Ego Vehicle & Sensor Payload in world space
  if (vehicleGroup) {
    vehicleGroup.position.set(egoPose.x, egoPose.y, 0);
    vehicleGroup.rotation.z = egoPose.heading;
  }
  if (sensorGroup) {
    sensorGroup.position.set(egoPose.x, egoPose.y, 0);
    sensorGroup.rotation.z = egoPose.heading;
  }
  if (frontWheelsGroup) {
    frontWheelsGroup.rotation.z = (egoPose.steering || 0) * 0.75;
  }

  // Interactive tail / brake lights glow on braking
  if (brakeLightsMesh) {
    const isBraking = keysDown.s || (egoPose.targetSpeed < egoPose.speed - 0.4);
    brakeLightsMesh.material.emissiveIntensity = isBraking ? 2.5 : 0.4;
    brakeLightsMesh.material.emissive.setHex(isBraking ? 0xff0000 : 0x7f1d1d);
  }

  // KPIs
  const currentFps = m.fps || 0;
  document.getElementById("kpi-fps").textContent = m.fps ? m.fps.toFixed(1) : "--";
  drawFpsSparkline(currentFps);
  document.getElementById("kpi-latency").textContent = m.total_ms ? m.total_ms.toFixed(1) : "--";
  document.getElementById("kpi-cells").textContent = (lastDrawnVoxelCount !== undefined && lastDrawnVoxelCount > 0)
    ? lastDrawnVoxelCount.toLocaleString()
    : (counts.total_cells ? counts.total_cells.toLocaleString() : "--");
  document.getElementById("kpi-foveas").textContent = activeFovs.length;

  // Ego speed & odometer KPIs in right HUD
  const egoSpeedEl = document.getElementById("kpi-ego-speed");
  const odometerEl = document.getElementById("kpi-odometer");
  if (egoSpeedEl) egoSpeedEl.textContent = egoPose.speed.toFixed(1);
  if (odometerEl) odometerEl.textContent = egoPose.x.toFixed(1);

  // Cockpit Drive Telemetry cluster updates
  const spdKmhEl = document.getElementById("cockpit-speed-kmh");
  const spdMsEl = document.getElementById("cockpit-speed-ms");
  const steerDegEl = document.getElementById("cockpit-steer-deg");
  if (spdKmhEl) spdKmhEl.textContent = (egoPose.speed * 3.6).toFixed(1);
  if (spdMsEl) spdMsEl.textContent = `(${egoPose.speed.toFixed(1)} m/s)`;
  if (steerDegEl) steerDegEl.textContent = `${((egoPose.steering || 0) * (180 / Math.PI)).toFixed(1)}°`;

  ["d", "n", "r", "p"].forEach(g => {
    const el = document.getElementById(`gear-${g}`);
    if (el) el.classList.toggle("active", (egoPose.gear || "D").toLowerCase() === g);
  });

  const btnManual = document.getElementById("btn-mode-manual");
  const btnCruise = document.getElementById("btn-mode-cruise");
  if (btnManual) btnManual.classList.toggle("active", egoPose.mode === "manual");
  if (btnCruise) btnCruise.classList.toggle("active", egoPose.mode === "cruise");

  const valTargetSpeed = document.getElementById("val-target-speed");
  const sliderTargetSpeed = document.getElementById("slider-target-speed");
  if (valTargetSpeed && !isDraggingSlider) {
    valTargetSpeed.textContent = `${egoPose.targetSpeed.toFixed(1)} m/s (${(egoPose.targetSpeed * 3.6).toFixed(0)} km/h)`;
  }
  if (sliderTargetSpeed && !isDraggingSlider) {
    sliderTargetSpeed.value = egoPose.targetSpeed;
  }

  // Update speed preset active button
  document.querySelectorAll(".preset-btn").forEach(b => {
    const pSpd = parseFloat(b.dataset.speed);
    b.classList.toggle("active", Math.abs(pSpd - egoPose.targetSpeed) < 0.8);
  });

  // Latency Waterfall
  const totalMs = m.total_ms || 1.0;
  const stageEl = document.getElementById("stage-total-ms");
  if (stageEl) stageEl.textContent = `${totalMs.toFixed(1)} ms`;

  const stages = [
    { id: "pre", ms: m.preprocessing_ms || 0 },
    { id: "perc", ms: m.perception_ms || 0 },
    { id: "trk", ms: m.tracking_ms || 0 },
    { id: "risk", ms: m.risk_ms || 0 },
    { id: "map", ms: m.mapping_ms || 0 },
    { id: "trav", ms: m.traversability_ms || 0 },
  ];

  stages.forEach(st => {
    const elText = document.getElementById(`t-${st.id}`);
    const elBar = document.getElementById(`b-${st.id}`);
    if (elText) elText.textContent = `${st.ms.toFixed(1)}ms`;
    if (elBar) {
      const pct = Math.min(100, Math.max(3, (st.ms / totalMs) * 100));
      elBar.style.width = `${pct}%`;
    }
  });

  // Dynamic Tracks Table
  const tbody = document.getElementById("tracks-table-body");
  const trackCountBadge = document.getElementById("track-count-badge");
  if (trackCountBadge) trackCountBadge.textContent = `${tracks.length} Tracks`;

  if (tbody) {
    if (tracks.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-cell">No active dynamic tracks</td></tr>`;
    } else {
      tbody.innerHTML = tracks.map(trk => {
        const vel = trk.velocity || [0, 0, 0];
        const spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).toFixed(1);
        const rk = trk.risk_level || "LOW";
        return `
          <tr>
            <td>#${trk.track_id}</td>
            <td>${trk.semantic_class}</td>
            <td>${spd}</td>
            <td><span class="threat-tag threat-${rk}">${rk}</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  // Fovea Status Banner
  const banner = document.getElementById("fovea-banner");
  const bannerText = document.getElementById("fovea-banner-text");
  if (banner && bannerText) {
    if (activeFovs.length > 0) {
      banner.className = "fovea-status-banner active-tracking";
      bannerText.textContent = `ACTIVE FOVEA: ${activeFovs.length} HIGH-RES 5CM TARGETS TRACKED`;
    } else {
      banner.className = "fovea-status-banner";
      bannerText.textContent = "FOVEA CONTROLLER IDLE (PERIPHERAL COARSE BASELINE)";
    }
  }

  // Timeline Scrubber
  const slider = document.getElementById("seek-slider");
  if (slider && !isPaused && data.frame_index !== undefined) {
    slider.value = data.frame_index % (data.total_frames || 100);
    if (data.total_frames) slider.max = Math.max(1, data.total_frames - 1);
  }
  const fCounter = document.getElementById("frame-counter");
  if (fCounter) {
    fCounter.textContent = `FRAME ${String(data.frame_index || 0).padStart(3, "0")}/${String(data.total_frames || 100).padStart(3, "0")}`;
  }
  const tCounter = document.getElementById("timestamp-counter");
  if (tCounter && data.timestamp !== undefined) {
    tCounter.textContent = `(t = ${data.timestamp.toFixed(2)}s)`;
  }
}

// ----------------------------------------------------------------------------
// Legend View Synchronizer
// ----------------------------------------------------------------------------
function updateLegendDisplay() {
  const risk = document.getElementById("legend-risk");
  const trav = document.getElementById("legend-traversability");
  const sem = document.getElementById("legend-semantic");
  const elev = document.getElementById("legend-elevation");
  const inten = document.getElementById("legend-intensity");
  const res = document.getElementById("legend-resolution");

  if (risk) risk.style.display = (currentViewMode === "risk") ? "flex" : "none";
  if (trav) trav.style.display = (currentViewMode === "traversability" || currentViewMode === "wireframe") ? "flex" : "none";
  if (sem) sem.style.display = (currentViewMode === "semantic") ? "flex" : "none";
  if (elev) elev.style.display = (currentViewMode === "elevation") ? "flex" : "none";
  if (inten) inten.style.display = (currentViewMode === "intensity") ? "flex" : "none";
  if (res) res.style.display = (currentViewMode === "resolution") ? "flex" : "none";

  // Update on-canvas overlay badge in Panel C (#panel-c-mode-overlay)
  const titleEl = document.getElementById("panel-c-mode-title");
  const itemsEl = document.getElementById("panel-c-mode-items");

  const P = (typeof FOVEA_PALETTE !== "undefined") ? FOVEA_PALETTE : null;
  if (P && titleEl && itemsEl) {
    let titleText = "🟢 3D TRAVERSABILITY GRID";
    let items = [
      { color: P.TRAVERSABILITY.SAFE.css, label: "Safe" },
      { color: P.TRAVERSABILITY.CAUTION.css, label: "Caution" },
      { color: P.TRAVERSABILITY.NON_TRAVERSABLE.css, label: "Hazard" },
      { color: P.TRAVERSABILITY.UNKNOWN.css, label: "Occluded" },
    ];

    if (currentViewMode === "risk") {
      titleText = "⚠️ COLLISION THREAT SEVERITY";
      items = [
        { color: P.RISK.LOW.css, label: "Low (>35m)" },
        { color: P.RISK.MEDIUM.css, label: "Medium (22-35m)" },
        { color: P.RISK.HIGH.css, label: "High (12-22m)" },
        { color: P.RISK.CRITICAL.css, label: "Critical (<12m)" },
      ];
    } else if (currentViewMode === "resolution") {
      titleText = "🎯 MULTI-RESOLUTION TIERS";
      items = P.RESOLUTION_BANDS.map(b => ({ color: b.css, label: `${b.res.toFixed(2)}m (${b.name})` }));
    } else if (currentViewMode === "semantic") {
      titleText = "🏷️ SEMANTIC CLASSIFICATION";
      items = [
        { color: P.SEMANTIC.PEDESTRIAN.css, label: "Pedestrian" },
        { color: P.SEMANTIC.CYCLIST.css, label: "Cyclist" },
        { color: P.SEMANTIC.VEHICLE.css, label: "Vehicle" },
        { color: P.SEMANTIC.TRUCK.css, label: "Truck" },
        { color: P.SEMANTIC.CURB.css, label: "Curb" },
        { color: P.SEMANTIC.POTHOLE.css, label: "Pothole" },
        { color: P.SEMANTIC.ROAD.css, label: "Road" },
      ];
    } else if (currentViewMode === "elevation") {
      titleText = "⛰️ HYPSOMETRIC ELEVATION";
      items = [
        { color: "#25215d", label: "-2.0m Ditch" },
        { color: "#10766e", label: "-0.2m Road" },
        { color: "#2dd4bf", label: "+0.3m Curb" },
        { color: "#eab308", label: "+1.6m Obstacle" },
        { color: "#f8fafc", label: "+3.5m Canopy" },
      ];
    } else if (currentViewMode === "intensity") {
      titleText = "✨ REFLECTANCE (HEATED GOLD)";
      items = P.INTENSITY_STOPS.map(s => ({ color: s.hex, label: s.label.split(" ")[0] }));
    } else if (currentViewMode === "wireframe") {
      titleText = "📐 ADAPTIVE GRID CELLS";
      items = [
        { color: P.FOVEA.CSS, label: "0.05m Fovea Cell" },
        { color: P.RESOLUTION_BANDS[3].css, label: "Coarse Cell" },
      ];
    }

    titleEl.innerHTML = `<span>${titleText}</span>`;
    itemsEl.innerHTML = items.map(it => `
      <div class="panel-c-item">
        <span class="panel-c-dot" style="background: ${it.color};"></span>
        <span>${it.label}</span>
      </div>
    `).join("");
  }
}

// ----------------------------------------------------------------------------
// WebSocket Connection & Event Handlers
// ----------------------------------------------------------------------------
function connectWebSocket(scenarioName) {
  if (ws) {
    ws.close();
    ws = null;
  }

  const statusPill = document.getElementById("status-pill");
  const statusLabel = document.getElementById("status-text");

  statusPill.className = "connection-pill";
  statusLabel.textContent = "CONNECTING...";

  const backendHost = urlParams.get("backend") || window.location.host;
  const protocol = (window.location.protocol === "https:" || backendHost.includes(".onrender.com") || backendHost.includes(".hf.space") || backendHost.includes(".up.railway.app")) ? "wss:" : (window.location.protocol === "https:" ? "wss:" : "ws:");
  const wsUrl = `${protocol}//${backendHost}/ws/stream?scenario=${encodeURIComponent(scenarioName)}`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusPill.className = "connection-pill";
    statusLabel.textContent = "STREAMING (3D LIVE)";
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "frame") {
        lastFrameData = data;
        renderVoxels(data.cells);
        renderPointCloud(data.raw_points, data.raw_intensities, data.semantic_points);
        renderBoundingBoxes(data.tracked_objects);
        renderFoveaOverlays(data.active_fovea_regions);
        drawPanelARawPointCloud(data.raw_points, data.raw_intensities);
        updateHUD(data);
      }
    } catch (err) {
      console.error("Error handling frame payload:", err);
    }
  };

  ws.onclose = () => {
    statusPill.className = "connection-pill disconnected";
    statusLabel.textContent = "DISCONNECTED";
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => connectWebSocket(currentScenario), 2500);
    }
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

function sendControl(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

// ----------------------------------------------------------------------------
// Re-render cached frame on UI adjustments
// ----------------------------------------------------------------------------
function reRenderCurrentFrame() {
  if (!lastFrameData) return;
  renderVoxels(lastFrameData.cells);
  renderPointCloud(lastFrameData.raw_points, lastFrameData.raw_intensities, lastFrameData.semantic_points);
  renderBoundingBoxes(lastFrameData.tracked_objects);
  renderFoveaOverlays(lastFrameData.active_fovea_regions);
  drawPanelARawPointCloud(lastFrameData.raw_points, lastFrameData.raw_intensities);
}

// ----------------------------------------------------------------------------
// Theme Handler
// ----------------------------------------------------------------------------
function applyTheme(themeName) {
  currentTheme = themeName;
  document.body.className = `${themeName}-theme`;

  if (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.applyCssVariables) {
    FOVEA_PALETTE.applyCssVariables(themeName);
  }

  const themeConfig = (typeof FOVEA_PALETTE !== "undefined" && FOVEA_PALETTE.THEME[themeName])
    ? FOVEA_PALETTE.THEME[themeName]
    : null;

  const bgHex = themeConfig ? parseInt(themeConfig.canvasBg.replace("#", "0x"), 16) : 0x050811;
  const fogHex = themeConfig ? (themeConfig.canvasFog || bgHex) : 0x050811;

  if (scene) {
    scene.background = new THREE.Color(bgHex);
    scene.fog = new THREE.FogExp2(
      fogHex,
      themeName === "cyber" ? 0.009 : (themeName === "light" ? 0.004 : 0.006)
    );
  }

  // Redraw 2D canvas with updated theme background if last frame exists
  if (lastFrameData) {
    drawPanelARawPointCloud(lastFrameData.raw_points, lastFrameData.raw_intensities);
  }
}

// ----------------------------------------------------------------------------
// UI Event Bindings & Interactive Driving Teleop Loop
// ----------------------------------------------------------------------------
function setupEventListeners() {
  // Scenario Dropdown
  const scenSelect = document.getElementById("scenario-select");
  if (scenSelect) {
    scenSelect.value = currentScenario;
    scenSelect.addEventListener("change", (e) => {
      currentScenario = e.target.value;
      sendControl({ type: "change_scenario", scenario: currentScenario });
      connectWebSocket(currentScenario);
    });
  }

  // Visual Mode Selector
  const modeSelect = document.getElementById("view-mode-select");
  if (modeSelect) {
    modeSelect.addEventListener("change", (e) => {
      currentViewMode = e.target.value;
      updateLegendDisplay();
      reRenderCurrentFrame();
    });
  }

  // Point Size Slider
  const ptSlider = document.getElementById("slider-point-size");
  const ptVal = document.getElementById("val-point-size");
  if (ptSlider) {
    ptSlider.addEventListener("input", (e) => {
      pointSize = parseFloat(e.target.value);
      if (ptVal) ptVal.textContent = `${pointSize.toFixed(2)}m`;
      reRenderCurrentFrame();
    });
  }

  // Voxel Height Slider
  const vhSlider = document.getElementById("slider-voxel-height");
  const vhVal = document.getElementById("val-voxel-height");
  if (vhSlider) {
    vhSlider.addEventListener("input", (e) => {
      voxelHeightScale = parseFloat(e.target.value);
      if (vhVal) vhVal.textContent = `${voxelHeightScale.toFixed(1)}x`;
      reRenderCurrentFrame();
    });
  }

  // Voxel Style Selector
  const vStyle = document.getElementById("select-voxel-style");
  if (vStyle) {
    vStyle.addEventListener("change", (e) => {
      voxelStyle = e.target.value;
      reRenderCurrentFrame();
    });
  }

  // Theme Selector
  const themeSelect = document.getElementById("select-theme");
  if (themeSelect) {
    themeSelect.value = currentTheme;
    themeSelect.addEventListener("change", (e) => {
      applyTheme(e.target.value);
    });
  }

  // Play / Pause
  const btnPlay = document.getElementById("btn-play-pause");
  if (btnPlay) {
    btnPlay.addEventListener("click", () => {
      isPaused = !isPaused;
      document.getElementById("btn-icon").textContent = isPaused ? "▶" : "⏸";
      document.getElementById("btn-label").textContent = isPaused ? "RESUME" : "PAUSE";
      sendControl({ type: isPaused ? "pause" : "resume" });
    });
  }

  // Playback Speeds
  document.querySelectorAll(".speed-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
      e.target.classList.add("active");
      currentSpeed = parseFloat(e.target.dataset.speed || "1.0");
      sendControl({ type: "speed", value: currentSpeed });
    });
  });

  // Timeline Slider
  const slider = document.getElementById("seek-slider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      sendControl({ type: "seek", frame: parseInt(e.target.value) });
    });
  }

  // Camera Presets
  const bindCam = (id, mode) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", () => setCameraPreset(mode));
  };
  bindCam("cam-chase", "chase");
  bindCam("cam-ego", "ego");
  bindCam("cam-bev", "bev");
  bindCam("cam-orbit", "orbit");
  bindCam("cam-side", "side");
  bindCam("cam-reset", "chase");

  // Layer Toggles
  const bindToggle = (id, key, group) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", (e) => {
        activeLayers[key] = e.target.checked;
        if (group) group.visible = e.target.checked;
      });
    }
  };
  bindToggle("layer-voxels", "voxels", voxelGroup);
  bindToggle("layer-points", "points", pointsGroup);
  bindToggle("layer-bboxes", "bboxes", bboxesGroup);
  bindToggle("layer-foveas", "foveas", foveasGroup);
  bindToggle("layer-rings", "rings", rangeRingsGroup);
  bindToggle("layer-grid", "grid", gridGroup);

  // Panel A Toggle and Size Controls
  const panelAContainer = document.getElementById("panel-a-container");
  const btnTogglePanelA = document.getElementById("btn-toggle-panel-a");
  const btnPanelAClose = document.getElementById("btn-panel-a-close");
  const btnPanelASize = document.getElementById("btn-panel-a-size");
  const canvasRaw = document.getElementById("canvas-raw");

  if (btnTogglePanelA && panelAContainer) {
    btnTogglePanelA.addEventListener("click", () => {
      const isHidden = panelAContainer.classList.toggle("hidden");
      btnTogglePanelA.classList.toggle("active", !isHidden);
      if (!isHidden && lastFrameData) {
        drawPanelARawPointCloud(lastFrameData.raw_points, lastFrameData.raw_intensities);
      }
    });
  }

  if (btnPanelAClose && panelAContainer && btnTogglePanelA) {
    btnPanelAClose.addEventListener("click", () => {
      panelAContainer.classList.add("hidden");
      btnTogglePanelA.classList.remove("active");
    });
  }

  if (btnPanelASize && panelAContainer && canvasRaw) {
    btnPanelASize.addEventListener("click", () => {
      const isEnlarged = panelAContainer.classList.toggle("enlarged");
      const dim = isEnlarged ? 460 : 340;
      canvasRaw.width = dim;
      canvasRaw.height = dim;
      if (lastFrameData) {
        drawPanelARawPointCloud(lastFrameData.raw_points, lastFrameData.raw_intensities);
      } else {
        drawPanelARawPointCloud([], []);
      }
    });
  }

  // --------------------------------------------------------------------------
  // Interactive Drive Controls & Keyboard Teleoperation
  // --------------------------------------------------------------------------

  // System Mode Buttons
  const btnInspect = document.getElementById('btn-op-inspect');
  const btnDrive = document.getElementById('btn-op-drive');

  function switchToInspect() {
    // UI updates
    if (btnInspect) btnInspect.classList.add('active');
    if (btnDrive) btnDrive.classList.remove('active');
    const panel = document.getElementById('cockpit-drive-panel');
    if (panel) panel.classList.add('hidden');

    // Hide world road mesh on inspection mode for stable view
    if (roadwayGroup) roadwayGroup.visible = false;
    setCameraPreset("orbit");

    // View mode reset to Traversability
    currentViewMode = 'traversability';
    const modeSelect = document.getElementById('view-mode-select');
    if (modeSelect) modeSelect.value = 'traversability';
    updateLegendDisplay();
    reRenderCurrentFrame();

    // Ego pose defaults for inspection
    egoPose.speed = 0.0;
    egoPose.targetSpeed = 0.0;
    egoPose.gear = 'P';
    egoPose.mode = 'inspect';

    // Notify server
    sendControl({ type: 'drive_mode', mode: 'inspect' });
  }

  function switchToDriveDemo() {
    // UI updates
    if (btnDrive) btnDrive.classList.add('active');
    if (btnInspect) btnInspect.classList.remove('active');
    const panel = document.getElementById('cockpit-drive-panel');
    if (panel) panel.classList.remove('hidden');

    // Show world road mesh on drive demo and switch to chase camera
    if (roadwayGroup) roadwayGroup.visible = true;
    setCameraPreset("chase");

    // Ego pose for manual driving
    egoPose.gear = 'D';
    egoPose.mode = 'manual';

    // Notify server
    sendControl({ type: 'drive_mode', mode: 'manual' });
  }

  // Bind button events
  if (btnInspect) {
    btnInspect.addEventListener('click', switchToInspect);
  }
  if (btnDrive) {
    btnDrive.addEventListener('click', switchToDriveDemo);
  }

  // Initialize default mode on page load
  switchToInspect();

  // Target speed slider & label elements
  const sliderTargetSpeed = document.getElementById("slider-target-speed");
  const valTargetSpeed = document.getElementById("val-target-speed");

  if (sliderTargetSpeed) {
    sliderTargetSpeed.addEventListener("mousedown", () => { isDraggingSlider = true; });
    sliderTargetSpeed.addEventListener("touchstart", () => { isDraggingSlider = true; });
    window.addEventListener("mouseup", () => { isDraggingSlider = false; });
    window.addEventListener("touchend", () => { isDraggingSlider = false; });

    sliderTargetSpeed.addEventListener("input", (e) => {
      const spd = parseFloat(e.target.value);
      egoPose.targetSpeed = spd;
      egoPose.gear = spd > 0 ? "D" : "P";
      if (valTargetSpeed) {
        valTargetSpeed.textContent = `${spd.toFixed(1)} m/s (${(spd * 3.6).toFixed(0)} km/h)`;
      }
      // Only send drive command if in manual mode
      if (egoPose.mode === "manual") {
        sendControl({ type: "drive", speed: spd, steer: egoPose.steering, gear: egoPose.gear });
      }
    });
  }

  // Quick Speed Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const spd = parseFloat(btn.dataset.speed || "0");
      egoPose.targetSpeed = spd;
      egoPose.gear = spd > 0 ? "D" : "P";
      if (sliderTargetSpeed) sliderTargetSpeed.value = spd;
      if (valTargetSpeed) {
        valTargetSpeed.textContent = `${spd.toFixed(1)} m/s (${(spd * 3.6).toFixed(0)} km/h)`;
      }
      sendControl({ type: "drive", speed: spd, steer: egoPose.steering, gear: egoPose.gear });
    });
  });

  // Mode Selector: Manual Drive vs Auto Cruise
  const btnManual = document.getElementById("btn-mode-manual");
  const btnCruise = document.getElementById("btn-mode-cruise");
  if (btnManual) {
    btnManual.addEventListener("click", () => {
      egoPose.mode = "manual";
      sendControl({ type: "drive_mode", mode: "manual" });
      btnManual.classList.add("active");
      if (btnCruise) btnCruise.classList.remove("active");
    });
  }
  if (btnCruise) {
    btnCruise.addEventListener("click", () => {
      egoPose.mode = "cruise";
      sendControl({ type: "drive_mode", mode: "cruise" });
      btnCruise.classList.add("active");
      if (btnManual) btnManual.classList.remove("active");
    });
  }

  // Reset Car Button
  const btnReset = document.getElementById("dpad-reset");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      egoPose.x = 0;
      egoPose.y = 0;
      egoPose.heading = 0;
      egoPose.speed = 8.0;
      egoPose.targetSpeed = 8.0;
      egoPose.steering = 0.0;
      sendControl({ type: "reset_car" });
    });
  }

  // Touch / Mouse D-Pad Buttons
  const bindDpad = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    const press = () => { keysDown[key] = true; el.classList.add("pressed"); };
    const release = () => { keysDown[key] = false; el.classList.remove("pressed"); };
    el.addEventListener("mousedown", press);
    el.addEventListener("touchstart", (e) => { e.preventDefault(); press(); });
    window.addEventListener("mouseup", release);
    window.addEventListener("touchend", release);
  };
  bindDpad("dpad-up", "w");
  bindDpad("dpad-down", "s");
  bindDpad("dpad-left", "a");
  bindDpad("dpad-right", "d");

  // Keyboard Driving Handlers (W/S/A/D and Arrow Keys)
  window.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

    if (e.code === "KeyW" || e.code === "ArrowUp") {
      keysDown.w = true;
      e.preventDefault();
    } else if (e.code === "KeyS" || e.code === "ArrowDown") {
      keysDown.s = true;
      e.preventDefault();
    } else if (e.code === "KeyA" || e.code === "ArrowLeft") {
      keysDown.a = true;
      e.preventDefault();
    } else if (e.code === "KeyD" || e.code === "ArrowRight") {
      keysDown.d = true;
      e.preventDefault();
    } else if (e.code === "Space") {
      e.preventDefault();
      // Quick brake to 0 km/h
      keysDown.space = true;
      egoPose.targetSpeed = 0.0;
      egoPose.gear = "P";
      sendControl({ type: "drive", speed: 0.0, steer: egoPose.steering, gear: "P" });
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "KeyW" || e.code === "ArrowUp") keysDown.w = false;
    else if (e.code === "KeyS" || e.code === "ArrowDown") keysDown.s = false;
    else if (e.code === "KeyA" || e.code === "ArrowLeft") keysDown.a = false;
    else if (e.code === "KeyD" || e.code === "ArrowRight") keysDown.d = false;
    else if (e.code === "Space") keysDown.space = false;
  });

  // Teleoperation Driving Interval (20 Hz)
  setInterval(() => {
    let driveChanged = false;

    if (keysDown.w) {
      egoPose.targetSpeed = Math.min(32.0, egoPose.targetSpeed + 1.2);
      egoPose.gear = "D";
      driveChanged = true;
    }
    if (keysDown.s) {
      if (egoPose.targetSpeed > 0.5) {
        egoPose.targetSpeed = Math.max(0.0, egoPose.targetSpeed - 2.2);
        egoPose.gear = egoPose.targetSpeed > 0.2 ? "D" : "P";
      } else {
        // Reverse
        egoPose.targetSpeed = Math.max(-5.0, egoPose.targetSpeed - 0.8);
        egoPose.gear = "R";
      }
      driveChanged = true;
    }
    if (keysDown.a) {
      egoPose.steering = Math.min(0.65, egoPose.steering + 0.08); // Steer Left
      driveChanged = true;
    }
    if (keysDown.d) {
      egoPose.steering = Math.max(-0.65, egoPose.steering - 0.08); // Steer Right
      driveChanged = true;
    }
    if (!keysDown.a && !keysDown.d && Math.abs(egoPose.steering) > 0.01) {
      egoPose.steering *= 0.65; // Self-centering steering
      driveChanged = true;
    }

    if (driveChanged) {
      sendControl({
        type: "drive",
        speed: egoPose.targetSpeed,
        steer: egoPose.steering,
        gear: egoPose.gear,
      });
    }
  }, 50);

  updateLegendDisplay();
}

// ----------------------------------------------------------------------------
// Dynamically query scenarios from backend
// ----------------------------------------------------------------------------
async function fetchScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    if (res.ok) {
      const list = await res.json();
      const select = document.getElementById("scenario-select");
      if (select && Array.isArray(list) && list.length > 0) {
        select.innerHTML = list.map(name => {
          let label = name;
          if (name === "flagship_demo") label = "flagship_demo (Jury Walkthrough)";
          if (name === "surround_360_city") label = "surround_360_city (360° Omnidirectional Urban)";
          if (name === "real_world_highway") label = "real_world_highway (Real Velodyne 64-Beam)";
          return `<option value="${name}" ${name === currentScenario ? "selected" : ""}>${label}</option>`;
        }).join("");
      }
    }
  } catch (e) {
    console.warn("Could not query /api/scenarios, using defaults:", e);
  }
}

// ----------------------------------------------------------------------------
// Bootstrap Initialization
// ----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", async () => {
  initThreeScene();
  applyTheme("dark");
  setupEventListeners();
  updateLegendDisplay();
  drawPanelARawPointCloud([], []);
  await fetchScenarios();
  connectWebSocket(currentScenario);
});

