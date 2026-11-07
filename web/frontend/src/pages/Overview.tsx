import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, useAsync, type Discrimination, type Meta, type PositionRow } from "../lib/api";
import { PLDDT_BANDS } from "../lib/colors";
import { num, pct, pval } from "../lib/format";
import { MolViewer } from "../components/MolViewer";
import { ResidueChart } from "../components/ResidueChart";
import { ErrorNote, LineLegend, Loading, PlddtLegend, StatTile } from "../components/ui";

const BANDS = [
  { from: 90, to: 100, color: PLDDT_BANDS[0].color },
  { from: 70, to: 90, color: PLDDT_BANDS[1].color },
  { from: 50, to: 70, color: PLDDT_BANDS[2].color },
  { from: 0, to: 50, color: PLDDT_BANDS[3].color },
];

function auc(d: Discrimination | undefined) {
  if (!d?.auc || !d.auc_ci95) return "—";
  return `${d.auc.toFixed(2)}`;
}

export function Overview({ meta }: { meta: Meta }) {
  const positions = useAsync(api.positions, "positions");
  const wtPdb = useAsync(() => api.structure("WT"), "WT");
  const navigate = useNavigate();
  const [hover, setHover] = useState<number | null>(null);
  const s = meta.stats;
  const { start, end } = meta.construct;
  const disc = s.discrimination;
  const primary = disc.find((d) => d.metric === "delta_local_plddt");
  const llr = disc.find((d) => d.metric === "llr");
  const bp = s.breaking_points;
  const nVariants = s.dataset.n_folded;

  return (
    <div className="stack">
      <section className="stack-sm">
        <div className="eyebrow">Variant structure atlas · {meta.target.gene} {start}–{end}</div>
        <h1>{meta.target.title}</h1>
        <p className="lede">
          {nVariants.toLocaleString()} clinically classified missense variants, each folded with ESMFold and compared
          with the wild type. Where does the model lose confidence, and does that track pathogenicity?
        </p>
        {s.dataset.n_missing > 0 && (
          <p className="callout warn" role="status">
            <b>Preliminary.</b> Only {nVariants.toLocaleString()} of{" "}
            {(nVariants + s.dataset.n_missing).toLocaleString()} variants have been folded so far, so every statistic
            on this site will change. Please don't cite these numbers.
          </p>
        )}
      </section>

      <section className="grid-tiles" aria-label="Headline numbers">
        <StatTile label="Structures folded" value={(nVariants + 1).toLocaleString()}
                  sub={`${s.dataset.counts.pathogenic ?? 0} pathogenic · ${s.dataset.counts.benign ?? 0} benign · ${s.dataset.counts.vus ?? 0} VUS`} />
        <StatTile label="Wild-type mean pLDDT" value={num(s.wild_type.mean_plddt)} sub={`pTM ${num(s.wild_type.ptm, 2)}`} />
        <StatTile label="Breaking points" value={bp.n_variants} sub={`variants at ${bp.n_positions} positions (FDR < ${bp.fdr})`} />
        <StatTile label="AUC · Δ local pLDDT" value={auc(primary)}
                  sub={primary?.auc_ci95 ? `95% CI ${num(primary.auc_ci95[0], 2)}–${num(primary.auc_ci95[1], 2)}` : "pathogenic vs benign"} />
        <StatTile label="AUC · ESM-2 LLR baseline" value={auc(llr)}
                  sub={llr?.auc_ci95 ? `95% CI ${num(llr.auc_ci95[0], 2)}–${num(llr.auc_ci95[1], 2)}` : "sequence-only"} />
      </section>

      <section className="grid-2">
        <div className="card">
          <div className="card-head">
            <h2>Wild-type prediction</h2>
            <Link className="small" to="/variants">Browse variants →</Link>
          </div>
          <MolViewer pdb={wtPdb.data} numberOffset={start - 1} sequence={meta.construct.sequence}
                     label={`ESMFold structure of the wild-type ${meta.target.gene} construct, coloured by pLDDT`} />
          <div style={{ marginTop: 12 }}><PlddtLegend /></div>
        </div>
        <Findings meta={meta} />
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Structural sensitivity along the domain</h2>
          <span className="small muted">Hover to inspect · click a residue to list its variants</span>
        </div>
        {positions.error && <ErrorNote error={positions.error} />}
        {!positions.data ? <Loading what="positions" /> : (
          <PositionCharts rows={positions.data} meta={meta} hover={hover} setHover={setHover}
                          onSelect={(p) => navigate(`/variants?q=${p}`)} />
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Pathogenic vs benign</h2>
          <span className="small muted">AUC &gt; 0.5: metric ranks pathogenic variants as more damaging</span>
        </div>
        <DiscriminationTable rows={disc} />
        <details style={{ marginTop: 14 }}>
          <summary>Sensitivity analysis: high-confidence labels only (ClinVar ≥1★ or UniProt-curated)</summary>
          <DiscriminationTable rows={s.discrimination_high_confidence_labels} />
        </details>
        {(s.dataset.counts.benign ?? 0) < 40 && (
          <p className="callout" style={{ marginTop: 14 }}>
            Only {s.dataset.counts.benign ?? 0} benign variants exist for this domain, so AUC confidence intervals are
            wide. Treat differences between metrics as significant only when the paired comparison says so.
          </p>
        )}
      </section>
    </div>
  );
}

function Findings({ meta }: { meta: Meta }) {
  const s = meta.stats;
  const v = meta.validation;
  const bands = s.wild_type.confidence_bands;
  const confident = (bands.very_high_gt90 ?? 0) + (bands.confident_70_90 ?? 0);
  const segs = s.wild_type.low_confidence_segments;
  const bc = s.breaking_points.by_class;
  const en = s.breaking_points.pathogenic_enrichment;
  const vsLlr = s.versus_llr.find((c) => c.metric === "delta_local_plddt" && c.difference !== null);
  const vsBlosum = s.versus_blosum62?.find((c) => c.metric === "delta_local_plddt" && c.difference !== null);
  const share = (k: string) => (bc[k] ? `${bc[k].n_breaking_points} of ${bc[k].n}` : "—");
  return (
    <div className="card">
      <div className="card-head"><h2>Key results</h2><span className="small muted">computed, not curated</span></div>
      <ul className="prose" style={{ paddingLeft: 18, margin: 0 }}>
        <li>
          The wild type folds with mean pLDDT <b>{num(s.wild_type.mean_plddt)}</b>; {pct(confident)} of residues are
          above 70{segs.length ? <> and confidence dips at {segs.map(([a, b]) => `${a}–${b}`).join(", ")}</> : null}.
        </li>
        {v && (
          <li>
            Against the crystal structure {v.pdb_id} ({v.n_aligned} residues) the prediction reaches Cα RMSD{" "}
            <b>{num(v.ca_rmsd, 2)} Å</b> and lDDT-Cα <b>{num(v.lddt_ca, 2)}</b>.
          </li>
        )}
        <li>
          Breaking points (local confidence loss beyond the distant-residue noise floor): pathogenic {share("pathogenic")},
          benign {share("benign")}, VUS {share("vus")}.
        </li>
        {en.odds_ratio_kind !== "undefined" && (
          <li>
            Pathogenic vs benign variants becoming breaking points: odds ratio{" "}
            <b>{en.odds_ratio_kind === "infinite" ? "∞" : num(en.odds_ratio, 2)}</b> (one-sided Fisher p = {pval(en.p_one_sided)}).
          </li>
        )}
        {vsLlr && (
          <li>
            Against the sequence-only ESM-2 baseline, Δ local pLDDT changes AUC by{" "}
            <b>{num(vsLlr.difference, 2, true)}</b> (95% CI {num(vsLlr.ci95![0], 2, true)} to {num(vsLlr.ci95![1], 2, true)},
            permutation p = {pval(vsLlr.p)}).
          </li>
        )}
        {vsBlosum && (
          <li>
            Against BLOSUM62 (substitution chemistry alone): ΔAUC <b>{num(vsBlosum.difference, 2, true)}</b> (95% CI{" "}
            {num(vsBlosum.ci95![0], 2, true)} to {num(vsBlosum.ci95![1], 2, true)}, permutation p = {pval(vsBlosum.p)}).
          </li>
        )}
      </ul>
    </div>
  );
}

function PositionCharts({ rows, meta, hover, setHover, onSelect }: {
  rows: PositionRow[]; meta: Meta; hover: number | null; setHover: (p: number | null) => void; onSelect: (p: number) => void;
}) {
  const start = meta.construct.start;
  const byPos = new Map(rows.map((r) => [r.position, r]));
  const minDelta = rows.map((r) => r.min_delta_local_plddt);
  const lo = Math.min(-2, ...minDelta.filter((v): v is number => v !== null));
  const extra = (p: number) => {
    const r = byPos.get(p);
    if (!r) return [];
    return [
      { label: "Variants (P / B / VUS)", value: `${r.n_pathogenic} / ${r.n_benign} / ${r.n_vus}` },
      { label: "Breaking-point variants", value: String(r.n_breaking_points) },
      { label: "Burial (Cα ≤ 10 Å)", value: String(r.wt_burial) },
    ];
  };
  return (
    <div className="stack-sm">
      <LineLegend items={[
        { label: "WT pLDDT", color: "var(--ink)" },
        { label: "Breaking point", color: "var(--neg)", kind: "swatch" },
        { label: "Other positions", color: "var(--axis)", kind: "swatch" },
        { label: "Pathogenic variant at residue", color: "var(--pathogenic)", kind: "swatch" },
        { label: "Benign variant at residue", color: "var(--benign)", kind: "swatch" },
      ]} />
      <ResidueChart
        title="Wild-type pLDDT per residue" start={start} sequence={meta.construct.sequence}
        lines={[{ key: "wt", label: "WT pLDDT", values: meta.wild_type.plddt, color: "var(--ink)", width: 1.6 }]}
        yDomain={[0, 100]} yTicks={[0, 50, 70, 90, 100]} yLabel="pLDDT" height={190} bands={BANDS}
        rug={[
          { label: "pathogenic", positions: rows.filter((r) => r.n_pathogenic).map((r) => r.position), color: "var(--pathogenic)" },
          { label: "benign", positions: rows.filter((r) => r.n_benign).map((r) => r.position), color: "var(--benign)" },
        ]}
        hover={hover} onHover={setHover} onSelect={onSelect} extraRows={extra}
      />
      <ResidueChart
        title="Most negative local pLDDT change per residue" start={start} sequence={meta.construct.sequence}
        stems={{
          label: "Min Δ local pLDDT", values: minDelta,
          color: (_v, i) => (rows[i]?.n_breaking_points ? "var(--neg)" : "var(--axis)"),
        }}
        yDomain={[Math.floor(lo), 2]} yLabel="min Δ pLDDT" height={170}
        hover={hover} onHover={setHover} onSelect={onSelect}
      />
      <details>
        <summary>Show as table</summary>
        <div className="table-wrap" style={{ maxHeight: 360 }}>
          <table>
            <thead><tr>
              <th>Residue</th><th className="num">WT pLDDT</th><th className="num">Variants</th><th className="num">Pathogenic</th>
              <th className="num">Benign</th><th className="num">Min Δ local pLDDT</th><th className="num">Breaking points</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.position}>
                  <td className="mono">{r.wt}{r.position}</td><td className="num">{num(r.wt_plddt)}</td>
                  <td className="num">{r.n_variants}</td><td className="num">{r.n_pathogenic}</td><td className="num">{r.n_benign}</td>
                  <td className="num">{num(r.min_delta_local_plddt, 2)}</td><td className="num">{r.n_breaking_points}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

export function DiscriminationTable({ rows }: { rows: Discrimination[] }) {
  const kindName = { structural: "Structure (mutant vs WT)", baseline: "Sequence baseline", positional: "Site prior" };
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>
          <th>Metric</th><th>Type</th><th className="num">n (P / B)</th><th className="num">AUC</th><th className="num">95% CI</th>
          <th className="num">Cliff's δ</th><th className="num">p</th><th className="num">q (BH)</th>
        </tr></thead>
        <tbody>
          {rows.filter((d) => d.auc !== null).map((d) => (
            <tr key={d.metric}>
              <td>{d.label}</td><td className="muted">{kindName[d.kind]}</td>
              <td className="num">{d.n_pos} / {d.n_neg}</td><td className="num"><b>{num(d.auc, 3)}</b></td>
              <td className="num">{d.auc_ci95 ? `${num(d.auc_ci95[0], 2)}–${num(d.auc_ci95[1], 2)}` : "—"}</td>
              <td className="num">{num(d.cliffs_delta, 2)}</td><td className="num">{pval(d.p)}</td><td className="num">{pval(d.q_bh)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
