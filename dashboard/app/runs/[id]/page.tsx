import Link from "next/link";
import { notFound } from "next/navigation";
import AutoRefresh from "@/components/AutoRefresh";
import Log from "@/components/Log";
import { assistantTurns, durationOf, fmt, getRun, lastReply, money, rewardClass, rewardOf, taskIndex, usageOf, type Trace } from "@/lib/runs";

export const dynamic = "force-dynamic";

function groups(traces: Trace[]): [string, Trace[]][] {
  const g = new Map<string, Trace[]>();
  for (const t of traces) {
    const k = String(taskIndex(t));
    if (!g.has(k)) g.set(k, []);
    g.get(k)!.push(t);
  }
  return Array.from(g.entries()).sort((a, b) => Number(a[0]) - Number(b[0]) || a[0].localeCompare(b[0]));
}

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const run = getRun(id);
  if (!run) notFound();
  const { summary: s, traces, config, log } = run;
  const ok = traces.filter((t) => t.ok);
  const gs = groups(traces);
  const width = Math.max(0, ...gs.map(([, ts]) => ts.length));
  const mixed = gs.filter(([, ts]) => new Set(ts.map((t) => (t.ok ? rewardOf(t) : -1))).size > 1).length;
  const rewardNames = Array.from(new Set(traces.flatMap((t) => Object.keys(t.rewards ?? {})))).sort();
  const metricNames = Array.from(new Set(traces.flatMap((t) => Object.keys(t.metrics ?? {})))).sort();
  const sampling = config?.sampling ?? traces[0]?.agent?.config?.sampling ?? {};
  const tasksetCfg = config?.env?.taskset ?? {};
  const knobs = Object.entries(tasksetCfg).filter(([k]) => !["id", "task", "system_prompt"].includes(k));
  const wall = traces.map(durationOf).filter((d): d is number => d != null);

  return (
    <>
      <div className="crumbs"><Link href="/">runs</Link> / {s.rel}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        <h1 className="mono" style={{ fontSize: 16 }}>{s.rel}</h1>
        <span className="spacer" style={{ flex: 1 }} />
        <AutoRefresh />
      </div>
      <div className="chips">
        <span className="chip">taskset <b>{s.taskset}</b></span>
        <span className="chip">model <b>{s.model}</b></span>
        <span className="chip">harness <b>{s.harness}</b></span>
        <span className="chip">runtime <b>{s.runtime}</b></span>
        <span className="chip">sampling <b>{Object.keys(sampling).length ? JSON.stringify(sampling) : "provider defaults"}</b></span>
        {config?.num_tasks != null && <span className="chip">-n <b>{String(config.num_tasks)}</b></span>}
        {config?.num_rollouts != null && <span className="chip">-r <b>{String(config.num_rollouts)}</b></span>}
        {knobs.map(([k, v]) => <span key={k} className="chip">{k} <b>{JSON.stringify(v)}</b></span>)}
      </div>

      <div className="tiles">
        <div className="tile"><div className="lab">rollouts</div><div className="val">{s.ok}<span className="muted" style={{ fontSize: 13 }}> / {s.rollouts} ok</span></div><div className="sub">{s.errored ? `${s.errored} errored` : "no errors"}</div></div>
        <div className="tile"><div className="lab">mean reward</div><div className="val">{fmt(s.meanReward)}</div><div className="sub">Σ score × weight over ok rollouts</div></div>
        {metricNames.map((m) => <div key={m} className="tile"><div className="lab">{m}</div><div className="val">{fmt(s.metricMeans[m])}</div><div className="sub">mean metric</div></div>)}
        <div className="tile"><div className="lab">groups with signal</div><div className="val">{mixed}<span className="muted" style={{ fontSize: 13 }}> / {gs.length}</span></div><div className="sub">tasks whose rollouts disagree</div></div>
        <div className="tile"><div className="lab">cost</div><div className="val">{money(s.usage.cost)}</div><div className="sub">{s.usage.calls} model calls</div></div>
        <div className="tile"><div className="lab">tokens</div><div className="val" style={{ fontSize: 16 }}>{s.usage.prompt} / {s.usage.completion}</div><div className="sub">prompt / completion{s.usage.reasoning ? ` · ${s.usage.reasoning} reasoning` : ""}</div></div>
        <div className="tile"><div className="lab">rollout time</div><div className="val" style={{ fontSize: 16 }}>{wall.length ? `${fmt(wall.reduce((a, b) => a + b, 0) / wall.length, 1)}s` : "–"}</div><div className="sub">mean, boot to scored</div></div>
      </div>

      <div className="cols" style={{ marginTop: 14 }}>
        <div>
          <h2>Task × rollout</h2>
          <div className="panel" style={{ overflowX: "auto" }}>
            {gs.length === 0 ? <div className="muted">No rollouts yet.</div> : (
              <table style={{ width: "auto" }}>
                <thead><tr><th>task</th>{Array.from({ length: width }, (_, i) => <th key={i}>r{i + 1}</th>)}<th>group</th></tr></thead>
                <tbody>
                  {gs.map(([k, ts]) => {
                    const kinds = new Set(ts.map((t) => (t.ok ? rewardOf(t) : -1)));
                    const allErr = ts.every((t) => !t.ok);
                    return (
                      <tr key={k}>
                        <td className="taskcol">{k}{(ts[0].task?.data?.answer ?? ts[0].task?.data?.secret) != null && <span className="muted"> → {String(ts[0].task.data.answer ?? ts[0].task.data.secret)}</span>}</td>
                        {ts.map((t) => (
                          <td key={t.id} style={{ padding: "3px 4px" }}>
                            <Link href={`/runs/${id}/traces/${t.id}`} className={`cell ${rewardClass(t)}`} title={t.ok ? `reward ${fmt(rewardOf(t), 2)} · ${t.stop_condition}` : `error: ${t.errors?.[0]?.type ?? "?"}`}>
                              {t.ok ? (Number.isInteger(rewardOf(t)) ? rewardOf(t) : fmt(rewardOf(t), 2)) : "✕"}
                            </Link>
                          </td>
                        ))}
                        {Array.from({ length: width - ts.length }, (_, i) => <td key={i} style={{ padding: "3px 4px" }}><span className="cell empty" /></td>)}
                        <td>{allErr ? <span className="chip err">errored</span> : kinds.size > 1 ? <span className="chip accent">mixed</span> : rewardOf(ts[0]) >= 1 ? <span className="chip good">all pass</span> : <span className="chip bad">all fail</span>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
        <div>
          <h2>eval.log · last {log.length} lines</h2>
          <Log lines={log} />
        </div>
      </div>

      <h2>Rollouts</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>trace</th><th>task</th><th className="num">reward</th>
              {rewardNames.map((r) => <th key={r} className="num">{r}</th>)}
              {metricNames.map((m) => <th key={m} className="num">{m}</th>)}
              <th className="num">turns</th><th className="num">tokens in / out</th><th className="num">cost</th><th className="num">time</th><th>stop</th><th>last reply</th>
            </tr>
          </thead>
          <tbody>
            {traces.map((t) => {
              const u = usageOf(t);
              const d = durationOf(t);
              return (
                <tr key={t.id} className="row">
                  <td><Link href={`/runs/${id}/traces/${t.id}`} className="mono">{t.id.slice(0, 8)}</Link></td>
                  <td className="mono">{String(taskIndex(t))}</td>
                  <td className="num">{t.ok ? <span className={`chip ${rewardOf(t) >= 1 ? "good" : rewardOf(t) <= 0 ? "bad" : "warn"}`}>{fmt(rewardOf(t), 2)}</span> : <span className="chip err">error</span>}</td>
                  {rewardNames.map((r) => <td key={r} className="num">{t.rewards?.[r] ? `${fmt(t.rewards[r].score, 2)}×${t.rewards[r].weight}` : "–"}</td>)}
                  {metricNames.map((m) => <td key={m} className="num">{t.metrics?.[m] != null ? fmt(t.metrics[m], 2) : "–"}</td>)}
                  <td className="num">{assistantTurns(t)}</td>
                  <td className="num">{u.prompt} / {u.completion}</td>
                  <td className="num">{money(u.cost)}</td>
                  <td className="num">{d == null ? "–" : `${fmt(d, 1)}s`}</td>
                  <td className="mono">{t.stop_condition}</td>
                  <td className="mono" style={{ maxWidth: 360, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={t.ok ? lastReply(t) : t.errors?.[0]?.message}>{t.ok ? String(lastReply(t) ?? "").slice(-120) : <span style={{ color: "var(--bad)" }}>{t.errors?.[0]?.type}: {t.errors?.[0]?.message?.slice(0, 120)}</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {ok.length > 0 && (
        <details className="raw"><summary>resolved config</summary><pre>{JSON.stringify(config ?? traces[0]?.agent?.config, null, 2)}</pre></details>
      )}
    </>
  );
}
