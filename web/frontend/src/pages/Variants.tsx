import { Link, useNavigate, useSearchParams } from "react-router";
import { api, useAsync, type Label, type Meta, type VariantSummary } from "../lib/api";
import { LABEL_NAMES, labelVar } from "../lib/colors";
import { num } from "../lib/format";
import { BreakingBadge, ClassBadge, ErrorNote, Loading } from "../components/ui";

const PAGE = 50;
const CLASSES: Label[] = ["pathogenic", "benign", "vus", "conflicting", "cancer", "population", "unlabeled"];
const COLUMNS: { key: keyof VariantSummary; label: string; digits?: number; signed?: boolean; title: string }[] = [
  { key: "delta_local_plddt", label: "Δ local pLDDT", digits: 2, signed: true, title: "Mean pLDDT change within ±5 residues of the site" },
  { key: "delta_mean_plddt", label: "Δ mean pLDDT", digits: 2, signed: true, title: "Whole-domain mean pLDDT change" },
  { key: "rmsd_ca", label: "Cα RMSD (Å)", digits: 2, title: "Cα RMSD to the wild-type prediction after superposition" },
  { key: "lddt_local", label: "Local lDDT", digits: 3, title: "Superposition-free lDDT-Cα vs WT within ±5 residues" },
  { key: "llr", label: "ESM-2 LLR", digits: 2, signed: true, title: "Sequence-only log-likelihood ratio (negative = disfavoured)" },
];

export function Variants({ meta }: { meta: Meta }) {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const labels = (params.get("label") ?? "").split(",").filter(Boolean) as Label[];
  const q = params.get("q") ?? "";
  const breaking = params.get("breaking") === "1";
  const sort = params.get("sort") ?? "position";
  const order = params.get("order") ?? "asc";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const apiParams = new URLSearchParams({ sort, order, limit: String(PAGE), offset: String(page * PAGE) });
  if (labels.length) apiParams.set("label", labels.join(","));
  if (q) apiParams.set("q", q);
  if (breaking) apiParams.set("breaking", "true");
  const res = useAsync(() => api.variants(apiParams), apiParams.toString(), true);

  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    }
    if (!("page" in patch)) next.delete("page");
    setParams(next, { replace: true });
  };
  const toggleLabel = (l: Label) => {
    const set = new Set(labels);
    if (set.has(l)) set.delete(l);
    else set.add(l);
    update({ label: [...set].join(",") || null });
  };
  const sortBy = (key: string) => {
    if (sort === key) update({ order: order === "asc" ? "desc" : "asc" });
    else update({ sort: key, order: key.startsWith("delta") || key === "llr" || key === "lddt_local" ? "asc" : "desc" });
  };
  const available = CLASSES.filter((c) => (meta.stats.dataset.counts[c] ?? 0) > 0);
  const total = res.data?.total ?? 0;
  const pages = Math.ceil(total / PAGE);

  return (
    <div className="stack">
      <section className="stack-sm">
        <h1>Variants</h1>
        <p className="lede">
          Every folded substitution with its structural change relative to the wild type. Residue numbers are UniProt
          {" "}{meta.target.uniprot} canonical numbering.
        </p>
      </section>

      <section className="row" aria-label="Filters">
        {available.map((c) => (
          <button key={c} className="chip" aria-pressed={labels.includes(c)} onClick={() => toggleLabel(c)}>
            <span className="dot" style={{ background: labelVar(c) }} />{LABEL_NAMES[c]}
            <span className="muted">{meta.stats.dataset.counts[c]}</span>
          </button>
        ))}
        <label className="toggle">
          <input type="checkbox" checked={breaking} onChange={(e) => update({ breaking: e.target.checked ? "1" : null })} />
          Breaking points only
        </label>
        <input type="search" placeholder="Search R856H or 856" aria-label="Search variants by id or residue number"
               defaultValue={q} maxLength={20} style={{ marginLeft: "auto", minWidth: 200 }}
               onChange={(e) => update({ q: e.target.value.trim() || null })} />
      </section>

      {res.error && <ErrorNote error={res.error} />}
      {!res.data ? <Loading what="variants" /> : (
        <>
          <div className="table-wrap" style={{ opacity: res.loading ? 0.6 : 1, transition: "opacity .15s" }}>
            <table>
              <thead>
                <tr>
                  <SortHeader k="position" label="Variant" sort={sort} order={order} onSort={sortBy} />
                  <th>Class</th>
                  {COLUMNS.map((c) => <SortHeader key={c.key} k={c.key} label={c.label} title={c.title} sort={sort} order={order} onSort={sortBy} num />)}
                  <th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {res.data.items.map((v) => (
                  <tr key={v.variant_id} className="clickable" onClick={() => navigate(`/variants/${v.variant_id}`)}>
                    <td className="mono"><Link to={`/variants/${v.variant_id}`} onClick={(e) => e.stopPropagation()}>{v.variant_id}</Link></td>
                    <td><ClassBadge label={v.label} /></td>
                    {COLUMNS.map((c) => <td key={c.key} className="num">{num(v[c.key] as number | null, c.digits, c.signed)}</td>)}
                    <td>{v.breaking_point ? <BreakingBadge q={v.local_q} /> : null}</td>
                  </tr>
                ))}
                {res.data.items.length === 0 && (
                  <tr><td colSpan={COLUMNS.length + 3} className="muted" style={{ textAlign: "center", padding: 28 }}>No variants match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="small muted">{total.toLocaleString()} variants{pages > 1 ? ` · page ${page + 1} of ${pages}` : ""}</span>
            {pages > 1 && (
              <div className="row">
                <button className="btn" disabled={page === 0} onClick={() => update({ page: String(page - 1) })}>Previous</button>
                <button className="btn" disabled={page + 1 >= pages} onClick={() => update({ page: String(page + 1) })}>Next</button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SortHeader({ k, label, title, sort, order, onSort, num: isNum }: {
  k: string; label: string; title?: string; sort: string; order: string; onSort: (k: string) => void; num?: boolean;
}) {
  const active = sort === k;
  return (
    <th className={isNum ? "num" : undefined} title={title}
        aria-sort={active ? (order === "asc" ? "ascending" : "descending") : "none"}>
      <button className="sort" onClick={() => onSort(k)}>
        {label}<span aria-hidden="true" style={{ opacity: active ? 1 : 0.3 }}>{active && order === "desc" ? "↓" : "↑"}</span>
      </button>
    </th>
  );
}
