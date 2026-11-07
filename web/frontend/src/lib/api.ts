import { useEffect, useState } from "react";

export type Label = "wild_type" | "pathogenic" | "benign" | "vus" | "conflicting" | "cancer" | "population" | "unlabeled";

export interface VariantSummary {
  variant_id: string;
  position: number | null;
  wt: string | null;
  mt: string | null;
  label: Label;
  in_scan: number | null;
  review_stars: number | null;
  mean_plddt: number;
  ptm: number;
  delta_mean_plddt: number | null;
  delta_site_plddt: number | null;
  delta_local_plddt: number | null;
  rmsd_ca: number | null;
  lddt_local: number | null;
  llr: number | null;
  local_q: number | null;
  breaking_point: number | null;
}

export interface Sibling {
  variant_id: string;
  mt: string;
  label: Label;
  delta_local_plddt: number | null;
  breaking_point: number | null;
}

export interface VariantDetail extends VariantSummary {
  uniprot_curated: number | null;
  clinvar_ids: string | null;
  dbsnp_ids: string | null;
  diseases: string | null;
  assertions: string | null;
  max_gnomad_af: number | null;
  delta_ptm: number | null;
  site_plddt: number | null;
  min_delta_plddt: number | null;
  min_delta_position: number | null;
  n_residues_drop5: number | null;
  lddt_ca: number | null;
  wt_site_plddt: number | null;
  wt_site_burial: number | null;
  blosum62: number | null;
  local_p: number | null;
  sequence: string;
  plddt: number[];
  siblings?: Sibling[];
}

export interface Discrimination {
  metric: string;
  label: string;
  kind: "structural" | "baseline" | "positional";
  n_pos: number;
  n_neg: number;
  auc: number | null;
  auc_ci95?: [number, number];
  cliffs_delta?: number;
  p?: number;
  q_bh?: number | null;
}

export interface PairedComparison {
  metric: string;
  baseline: string;
  difference: number | null;
  ci95: [number, number] | null;
  /** Paired permutation p (primary). */
  p: number | null;
  p_delong: number | null;
}

export interface Annotation {
  type: string;
  start: number;
  end: number;
  description: string;
}

export interface Meta {
  schema_version: number;
  exported_at: string;
  target: { name: string; title: string; gene: string; uniprot: string; protein_name?: string };
  construct: { start: number; end: number; length: number; sequence: string };
  model: {
    fingerprint: string; folding: string; num_recycles: number; llr: string;
    device?: string; precision?: string; versions?: Record<string, string>;
  };
  provenance: {
    retrieved_at: string;
    uniprot_entry_version?: number;
    sources: Record<string, { url: string; file: string; sha256: string }>;
    filters: Record<string, unknown>;
    dropped: Record<string, number>;
  };
  annotations: Annotation[];
  stats: {
    generated_at: string;
    dataset: { counts: Record<string, number>; n_folded: number; n_missing: number };
    wild_type: {
      mean_plddt: number; ptm: number;
      confidence_bands: Record<string, number>;
      low_confidence_segments: [number, number][];
    };
    noise_floor: { definition: string; n: number; p01: number; p05: number; median: number };
    breaking_points: {
      fdr: number; local_window: number; n_variants: number; n_positions: number;
      by_class: Record<string, { n: number; n_breaking_points: number }>;
      pathogenic_enrichment: {
        odds_ratio: number | null; odds_ratio_kind: "finite" | "infinite" | "undefined";
        p_one_sided: number; table: number[][];
      };
    };
    discrimination: Discrimination[];
    discrimination_high_confidence_labels: Discrimination[];
    versus_llr: PairedComparison[];
    versus_blosum62?: PairedComparison[];
    versus_wt_site_plddt: PairedComparison[];
    correlations: { x: string; y: string; n: number; spearman_rho: number; p: number }[];
  };
  validation: null | {
    pdb_id: string; chain: string; n_aligned: number; coverage: number; ca_rmsd: number; lddt_ca: number;
    plddt_vs_lddt_spearman: { rho: number; p: number };
    per_residue?: { position: number; lddt: number | null; deviation: number; plddt: number }[];
  };
  wild_type: { sequence: string; plddt: number[] };
  live: { enabled: boolean; max_length: number };
}

export interface PositionRow {
  position: number;
  wt: string;
  wt_plddt: number;
  wt_burial: number;
  n_variants: number;
  n_pathogenic: number;
  n_benign: number;
  n_vus: number;
  mean_delta_local_plddt: number | null;
  min_delta_local_plddt: number | null;
  max_rmsd_ca: number | null;
  mean_llr: number | null;
  n_breaking_points: number;
}

export interface LiveStatus {
  available: boolean;
  configured: boolean;
  busy?: boolean;
  queue?: number;
  max_length?: number;
  device?: string;
}

export type LiveResult =
  | { source: "dataset"; variant_id: string; sequence: string; mutations: string[] | null }
  | {
      source: "live"; sequence: string; mutations: string[] | null; pdb: string; plddt: number[];
      mean_plddt: number; ptm: number; runtime_s: number; cached: boolean; device: string | null; comparable: boolean;
    };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (url.endsWith(".pdb") ? res.text() : res.json()) as Promise<T>;
}

const cache = new Map<string, Promise<unknown>>();

/** GET with an in-memory cache: the dataset is immutable between deploys. */
export function getCached<T>(url: string): Promise<T> {
  let p = cache.get(url) as Promise<T> | undefined;
  if (!p) {
    p = request<T>(url);
    cache.set(url, p);
    p.catch(() => cache.delete(url));
  }
  return p;
}

export const api = {
  meta: () => getCached<Meta>("/api/meta"),
  positions: () => getCached<PositionRow[]>("/api/positions"),
  variant: (id: string) => getCached<VariantDetail>(`/api/variants/${encodeURIComponent(id)}`),
  structure: (id: string) => getCached<string>(`/api/variants/${encodeURIComponent(id)}/structure.pdb`),
  variants: (params: URLSearchParams) =>
    getCached<{ total: number; items: VariantSummary[] }>(`/api/variants?${params.toString()}`),
  liveStatus: () => request<LiveStatus>("/api/live/status"),
  liveFold: (sequence: string) =>
    request<LiveResult>("/api/live/fold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    }),
};

export interface AsyncState<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
}

/**
 * Resolve a promise factory whenever ``key`` changes. With ``keepPrevious`` the old data
 * stays visible while loading (tables); otherwise it is cleared so a new page never shows
 * the previous item's data (e.g. another variant's structure).
 */
export function useAsync<T>(factory: () => Promise<T>, key: string, keepPrevious = false): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: undefined, error: undefined, loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ data: keepPrevious ? s.data : undefined, loading: true, error: undefined }));
    factory().then(
      (data) => alive && setState({ data, error: undefined, loading: false }),
      (error: Error) => alive && setState((s) => ({ data: s.data, error, loading: false })),
    );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
