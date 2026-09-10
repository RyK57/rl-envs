import Link from "next/link";
import AutoRefresh from "@/components/AutoRefresh";
import { ago, fmt, listRuns, money } from "@/lib/runs";

export const dynamic = "force-dynamic";

export default function RunsPage() {
  const runs = listRuns();
  const metricNames = Array.from(new Set(runs.flatMap((r) => Object.keys(r.metricMeans)))).sort();
  return (
    <>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        <h1>Runs</h1>
        <span className="muted">{runs.length} with traces</span>
        <span className="spacer" style={{ flex: 1 }} />
        <AutoRefresh />
      </div>
      {runs.length === 0 ? (
        <div className="empty">No traces.jsonl found under outputs/. Run an eval, then come back.</div>
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>run</th><th>when</th><th>taskset</th><th>model</th><th>harness</th><th>runtime</th>
                <th className="num">rollouts</th><th className="num">errors</th><th className="num">reward</th>
                {metricNames.map((m) => <th key={m} className="num">{m}</th>)}
                <th className="num">tokens in / out</th><th className="num">cost</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="row">
                  <td><Link href={`/runs/${r.id}`} className="mono">{r.rel}</Link></td>
                  <td className="muted" style={{ whiteSpace: "nowrap" }}>{ago(r.mtime)}</td>
                  <td>{r.taskset}</td><td className="mono">{r.model}</td><td>{r.harness}</td><td>{r.runtime}</td>
                  <td className="num">{r.rollouts}</td>
                  <td className="num">{r.errored ? <span className="chip err">{r.errored}</span> : "0"}</td>
                  <td className="num">{r.meanReward == null ? "–" : <><span className="bar" style={{ width: `${Math.round(r.meanReward * 60)}px` }} />{fmt(r.meanReward)}</>}</td>
                  {metricNames.map((m) => <td key={m} className="num">{fmt(r.metricMeans[m])}</td>)}
                  <td className="num">{r.usage.prompt} / {r.usage.completion}</td>
                  <td className="num">{money(r.usage.cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
