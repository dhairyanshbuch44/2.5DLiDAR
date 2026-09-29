/**
 * Fovea-LiDAR Global Shared Color Palette Configuration
 * Central source of truth for all UI panels, WebGL Three.js shaders, 2D radar canvas, and legends.
 * 
 * Design Principles:
 * 1. Dark Neutral Theme: Obsidian & Slate backgrounds (#070b14, #0d1322) so foreground data pops.
 * 2. High-Contrast Voxel Bands: Distinct luminance & hue at every resolution band (0.05m to 0.40m).
 * 3. Categorical Semantic Palette: Colorblind-friendly (Wong/Okabe-Ito inspired), high saturation.
 * 4. Sequential Collision Risk Scale: Intuitive cool -> hot (Sky Blue -> Gold -> Orange -> Red),
 *    carefully separated from semantic classes.
 * 5. Distinct Intensity Scale: Heated gold/plasma (Navy -> Cobalt -> Amber -> White Gold).
 */

const FOVEA_PALETTE = {
  // --------------------------------------------------------------------------
  // Theme Backgrounds & Surfaces (Dark / Near-Black Neutral Theme)
  // --------------------------------------------------------------------------
  THEME: {
    dark: {
      bgCore: "#070b14",
      bgSurface: "#0d1322",
      bgSurfaceElevated: "#151e33",
      bgSurfaceHover: "#1d2945",
      bgGlass: "rgba(13, 19, 34, 0.92)",
      borderSubtle: "#1e293b",
      borderActive: "#334155",
      borderFocus: "#00f0ff",
      textPrimary: "#f8fafc",
      textSecondary: "#94a3b8",
      textMuted: "#64748b",
      canvasBg: "#050811",
      canvasFog: 0x050811,
    },
    cyber: {
      bgCore: "#020409",
      bgSurface: "#0a0f1d",
      bgSurfaceElevated: "#151f36",
      bgSurfaceHover: "#1f2d4d",
      bgGlass: "rgba(10, 15, 29, 0.92)",
      borderSubtle: "#1e293b",
      borderActive: "#00f0ff",
      borderFocus: "#00f0ff",
      textPrimary: "#ffffff",
      textSecondary: "#00f0ff",
      textMuted: "#64748b",
      canvasBg: "#020409",
      canvasFog: 0x020409,
    },
    light: {
      bgCore: "#f8fafc",
      bgSurface: "#ffffff",
      bgSurfaceElevated: "#f1f5f9",
      bgSurfaceHover: "#e2e8f0",
      bgGlass: "rgba(255, 255, 255, 0.94)",
      borderSubtle: "#e2e8f0",
      borderActive: "#cbd5e1",
      borderFocus: "#0284c7",
      textPrimary: "#0f172a",
      textSecondary: "#475569",
      textMuted: "#64748b",
      canvasBg: "#eef2f7",
      canvasFog: 0xeef2f7,
    },
  },

  // --------------------------------------------------------------------------
  // 1. Dynamic Traversability & Obstacle Hazards (Panel C)
  // --------------------------------------------------------------------------
  TRAVERSABILITY: {
    SAFE: {
      hex: 0x10b981,
      css: "#10b981",
      rgb: [0.06, 0.73, 0.51],
      label: "Safe Corridor",
      desc: "Slope < 8°, roughness < 0.05m, obstacle-free",
    },
    CAUTION: {
      hex: 0xf59e0b,
      css: "#f59e0b",
      rgb: [0.96, 0.62, 0.04],
      label: "Caution Zone",
      desc: "Slope 8-15° or adjacent to moving obstacles",
    },
    NON_TRAVERSABLE: {
      hex: 0xef4444,
      css: "#ef4444",
      rgb: [0.94, 0.27, 0.27],
      label: "Non-Traversable Hazard",
      desc: "Step height > 0.18m or active collision zone",
    },
    UNKNOWN: {
      hex: 0x64748b,
      css: "#64748b",
      rgb: [0.39, 0.45, 0.55],
      label: "Unobserved / Occluded",
      desc: "Unvisited periphery or LiDAR shadow",
    },
  },

  // --------------------------------------------------------------------------
  // 2. Multi-Resolution Concentric Grid Bands (Panel C)
  // High-contrast scale against dark background: Cyan -> Mint -> Amber -> Bright Violet
  // --------------------------------------------------------------------------
  RESOLUTION_BANDS: [
    {
      band: 0,
      res: 0.05,
      hex: 0x00f0ff,
      css: "#00f0ff",
      rgb: [0.0, 0.94, 1.0],
      name: "Fovea Overlay",
      range: "0-10m @ 0.05m",
      label: "0.05m Ultra-Fine Fovea",
    },
    {
      band: 1,
      res: 0.10,
      hex: 0x10b981,
      css: "#10b981",
      rgb: [0.06, 0.73, 0.51],
      name: "Near Focus",
      range: "10-25m @ 0.10m",
      label: "0.10m Fine Grid",
    },
    {
      band: 2,
      res: 0.25,
      hex: 0xf59e0b,
      css: "#f59e0b",
      rgb: [0.96, 0.62, 0.04],
      name: "Mid Field",
      range: "25-50m @ 0.25m",
      label: "0.25m Mid-Field",
    },
    {
      band: 3,
      res: 0.40,
      hex: 0xa855f7,
      css: "#a855f7",
      rgb: [0.66, 0.33, 0.97],
      name: "Periphery",
      range: "50-100m @ 0.40m",
      label: "0.40m Periphery",
    },
  ],

  // --------------------------------------------------------------------------
  // 3. Categorical Semantic Palette (Colorblind-friendly / Wong-inspired)
  // High saturation, distinct hue for each traffic & road participant
  // --------------------------------------------------------------------------
  SEMANTIC: {
    PEDESTRIAN: {
      css: "#ec4899",
      hex: 0xec4899,
      rgb: [0.93, 0.28, 0.60],
      label: "Pedestrian",
      dot: "#ec4899",
    },
    CYCLIST: {
      css: "#fb923c",
      hex: 0xfb923c,
      rgb: [0.98, 0.57, 0.24],
      label: "Cyclist",
      dot: "#fb923c",
    },
    VEHICLE: {
      css: "#3b82f6",
      hex: 0x3b82f6,
      rgb: [0.23, 0.51, 0.96],
      label: "Vehicle",
      dot: "#3b82f6",
    },
    TRUCK: {
      css: "#1d4ed8",
      hex: 0x1d4ed8,
      rgb: [0.11, 0.31, 0.85],
      label: "Truck / Bus",
      dot: "#1d4ed8",
    },
    CURB: {
      css: "#06b6d4",
      hex: 0x06b6d4,
      rgb: [0.02, 0.71, 0.83],
      label: "Curb / Edge",
      dot: "#06b6d4",
    },
    POTHOLE: {
      css: "#f43f5e",
      hex: 0xf43f5e,
      rgb: [0.96, 0.25, 0.37],
      label: "Pothole / Drop",
      dot: "#f43f5e",
    },
    ROAD: {
      css: "#475569",
      hex: 0x475569,
      rgb: [0.28, 0.33, 0.41],
      label: "Road Surface",
      dot: "#475569",
    },
    SIDEWALK: {
      css: "#0ea5e9",
      hex: 0x0ea5e9,
      rgb: [0.05, 0.65, 0.91],
      label: "Sidewalk",
      dot: "#0ea5e9",
    },
    WALL: {
      css: "#8b5cf6",
      hex: 0x8b5cf6,
      rgb: [0.55, 0.36, 0.96],
      label: "Wall / Barrier",
      dot: "#8b5cf6",
    },
    POLE: {
      css: "#c084fc",
      hex: 0xc084fc,
      rgb: [0.75, 0.52, 0.99],
      label: "Pole / Sign",
      dot: "#c084fc",
    },
    VEGETATION: {
      css: "#22c55e",
      hex: 0x22c55e,
      rgb: [0.13, 0.77, 0.37],
      label: "Vegetation",
      dot: "#22c55e",
    },
    GRASS: {
      css: "#16a34a",
      hex: 0x16a34a,
      rgb: [0.09, 0.64, 0.29],
      label: "Grass",
      dot: "#16a34a",
    },
    OTHER_TERRAIN: {
      css: "#94a3b8",
      hex: 0x94a3b8,
      rgb: [0.58, 0.64, 0.72],
      label: "Terrain",
      dot: "#94a3b8",
    },
    UNKNOWN_OBSTACLE: {
      css: "#a855f7",
      hex: 0xa855f7,
      rgb: [0.66, 0.33, 0.97],
      label: "Obstacle",
      dot: "#a855f7",
    },
    DEFAULT: {
      css: "#64748b",
      hex: 0x64748b,
      rgb: [0.39, 0.45, 0.55],
      label: "Unclassified",
      dot: "#64748b",
    },
  },

  // --------------------------------------------------------------------------
  // 4. Sequential Collision Risk Severity Scale (Cool -> Hot)
  // Unambiguous, intuitive progression: Blue -> Yellow -> Orange -> Crimson Red
  // --------------------------------------------------------------------------
  RISK: {
    LOW: {
      css: "#38bdf8",
      hex: 0x38bdf8,
      rgb: [0.22, 0.74, 0.97],
      bg: "rgba(56, 189, 248, 0.15)",
      border: "rgba(56, 189, 248, 0.45)",
      label: "Low Risk (>35m / Clear)",
    },
    MEDIUM: {
      css: "#eab308",
      hex: 0xeab308,
      rgb: [0.92, 0.70, 0.03],
      bg: "rgba(234, 179, 8, 0.15)",
      border: "rgba(234, 179, 8, 0.45)",
      label: "Medium Risk (22-35m)",
    },
    HIGH: {
      css: "#f97316",
      hex: 0xf97316,
      rgb: [0.98, 0.45, 0.09],
      bg: "rgba(249, 115, 22, 0.15)",
      border: "rgba(249, 115, 22, 0.45)",
      label: "High Risk (12-22m)",
    },
    CRITICAL: {
      css: "#ef4444",
      hex: 0xef4444,
      rgb: [0.94, 0.27, 0.27],
      bg: "rgba(239, 68, 68, 0.22)",
      border: "rgba(239, 68, 68, 0.55)",
      label: "Critical Risk (<12m Collision)",
    },
  },

  // --------------------------------------------------------------------------
  // 5. Point Cloud Intensity / Reflectance Colormap (Heated Metal / Plasma)
  // Separate and distinct from categorical semantics and sequential risk
  // --------------------------------------------------------------------------
  INTENSITY_STOPS: [
    { stop: 0.00, r: 10,  g: 22,  b: 48,  hex: "#0a1630", label: "0.0 (Asphalt / Low Return)" },
    { stop: 0.25, r: 30,  g: 64,  b: 175, hex: "#1e40af", label: "0.25 (Diffusive Surface)" },
    { stop: 0.60, r: 217, g: 119, b: 6,   hex: "#d97706", label: "0.60 (Obstacle / Metal)" },
    { stop: 0.85, r: 251, g: 191, b: 36,  hex: "#fbbf24", label: "0.85 (High Reflectance)" },
    { stop: 1.00, r: 255, g: 255, b: 255, hex: "#ffffff", label: "1.0 (Retro-Reflector)" },
  ],

  // --------------------------------------------------------------------------
  // 6. Dynamic Foveation & Visualizer Accents
  // --------------------------------------------------------------------------
  FOVEA: {
    CYAN: 0x00f0ff,
    CSS: "#00f0ff",
    GLOW: "rgba(0, 240, 255, 0.35)",
  },

  // --------------------------------------------------------------------------
  // Colormap Interpolation Helpers
  // --------------------------------------------------------------------------
  /**
   * Evaluates RGB array [0..1, 0..1, 0..1] for laser intensity [0..1].
   */
  getIntensityRgb(intensity) {
    let val = Math.min(1.0, Math.max(0.0, Number(intensity) || 0.0));
    if (val > 1.0) val = val / 255.0;

    if (val < 0.25) {
      const t = val / 0.25;
      return [
        (10 + (30 - 10) * t) / 255.0,
        (22 + (64 - 22) * t) / 255.0,
        (48 + (175 - 48) * t) / 255.0,
      ];
    } else if (val < 0.60) {
      const t = (val - 0.25) / 0.35;
      return [
        (30 + (217 - 30) * t) / 255.0,
        (64 + (119 - 64) * t) / 255.0,
        (175 + (6 - 175) * t) / 255.0,
      ];
    } else if (val < 0.85) {
      const t = (val - 0.60) / 0.25;
      return [
        (217 + (251 - 217) * t) / 255.0,
        (119 + (191 - 119) * t) / 255.0,
        (6 + (36 - 6) * t) / 255.0,
      ];
    } else {
      const t = (val - 0.85) / 0.15;
      return [
        (251 + (255 - 251) * t) / 255.0,
        (191 + (255 - 191) * t) / 255.0,
        (36 + (255 - 36) * t) / 255.0,
      ];
    }
  },

  /**
   * Returns CSS RGB string "rgb(r, g, b)" for laser intensity [0..1].
   */
  getIntensityCss(intensity) {
    const rgb = this.getIntensityRgb(intensity);
    return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
  },

  /**
   * Returns resolution band color RGB array for a given cell resolution in meters.
   */
  getResolutionRgb(resolution) {
    const res = Number(resolution) || 0.10;
    if (res <= 0.06) return this.RESOLUTION_BANDS[0].rgb;
    if (res <= 0.14) return this.RESOLUTION_BANDS[1].rgb;
    if (res <= 0.28) return this.RESOLUTION_BANDS[2].rgb;
    return this.RESOLUTION_BANDS[3].rgb;
  },

  /**
   * Returns resolution band hex for a given cell resolution in meters.
   */
  getResolutionHex(resolution) {
    const res = Number(resolution) || 0.10;
    if (res <= 0.06) return this.RESOLUTION_BANDS[0].hex;
    if (res <= 0.14) return this.RESOLUTION_BANDS[1].hex;
    if (res <= 0.28) return this.RESOLUTION_BANDS[2].hex;
    return this.RESOLUTION_BANDS[3].hex;
  },

  /**
   * Injects CSS variables onto :root to synchronize CSS with JavaScript palette.
   */
  applyCssVariables(theme = "dark") {
    const root = document.documentElement;
    const t = this.THEME[theme] || this.THEME.dark;

    root.style.setProperty("--bg-core", t.bgCore);
    root.style.setProperty("--bg-surface", t.bgSurface);
    root.style.setProperty("--bg-surface-elevated", t.bgSurfaceElevated);
    root.style.setProperty("--bg-surface-hover", t.bgSurfaceHover);
    root.style.setProperty("--bg-glass", t.bgGlass);
    root.style.setProperty("--border-subtle", t.borderSubtle);
    root.style.setProperty("--border-active", t.borderActive);
    root.style.setProperty("--border-focus", t.borderFocus);
    root.style.setProperty("--text-primary", t.textPrimary);
    root.style.setProperty("--text-secondary", t.textSecondary);
    root.style.setProperty("--text-muted", t.textMuted);

    // Accents
    root.style.setProperty("--cyan", this.FOVEA.CSS);
    root.style.setProperty("--emerald", this.TRAVERSABILITY.SAFE.css);
    root.style.setProperty("--amber", this.TRAVERSABILITY.CAUTION.css);
    root.style.setProperty("--crimson", this.TRAVERSABILITY.NON_TRAVERSABLE.css);
    root.style.setProperty("--threat-low", this.RISK.LOW.css);
    root.style.setProperty("--threat-medium", this.RISK.MEDIUM.css);
    root.style.setProperty("--threat-high", this.RISK.HIGH.css);
    root.style.setProperty("--threat-critical", this.RISK.CRITICAL.css);
  },
};

// Auto-expose to global window for browser use
if (typeof window !== "undefined") {
  window.FOVEA_PALETTE = FOVEA_PALETTE;
}
