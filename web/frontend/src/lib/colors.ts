import type { Label } from "./api";

/** AlphaFold pLDDT convention — recognised by structural biologists without a legend lookup. */
export const PLDDT_BANDS = [
  { min: 90, color: "#0053d6", label: "Very high", range: "> 90" },
  { min: 70, color: "#65cbf3", label: "Confident", range: "70–90" },
  { min: 50, color: "#ffdb13", label: "Low", range: "50–70" },
  { min: -Infinity, color: "#ff7d45", label: "Very low", range: "< 50" },
] as const;

export function plddtColor(v: number): string {
  return PLDDT_BANDS.find((b) => v > b.min)?.color ?? PLDDT_BANDS[3].color;
}

export const LABEL_NAMES: Record<Label, string> = {
  wild_type: "Wild type",
  pathogenic: "Pathogenic / LP",
  benign: "Benign / LB",
  vus: "VUS",
  conflicting: "Conflicting",
  cancer: "Cancer context",
  population: "gnomAD carriers",
  unlabeled: "Unlabelled",
};

/** CSS variable per clinical class (values switch with the theme). */
export function labelVar(label: Label): string {
  switch (label) {
    case "pathogenic":
      return "var(--pathogenic)";
    case "benign":
      return "var(--benign)";
    case "vus":
      return "var(--vus)";
    default:
      return "var(--other)";
  }
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * t).toString(16).padStart(2, "0");
  return `#${c(r1, r2)}${c(g1, g2)}${c(b1, b2)}`;
}

/** Diverging ΔpLDDT scale for the 3D viewer: red (loss) ↔ neutral gray ↔ blue (gain). */
export function deltaColor(v: number, scale = 15): string {
  const t = Math.max(-1, Math.min(1, v / scale));
  return t < 0 ? mix("#d9d8d3", "#d03b3b", -t) : mix("#d9d8d3", "#256abf", t);
}

/** Read a CSS custom property's current value (for canvas/WebGL consumers). */
export function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
