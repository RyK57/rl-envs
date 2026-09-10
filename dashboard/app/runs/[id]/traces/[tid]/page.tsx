import Link from "next/link";
import { notFound } from "next/navigation";
import { durationOf, fmt, getTrace, money, rewardOf, taskIndex, usageOf, type Json } from "@/lib/runs";

export const dynamic = "force-dynamic";

function pretty(v: Json): string {
  if (typeof v === "string") {
    try { return JSON.stringify(JSON.parse(v), null, 2); } catch { return v; }
  }
  return JSON.stringify(v, null, 2);
}
const text = (c: Json): string => (c == null ? "" : typeof c === "string" ? c : JSON.stringify(c, null, 2));

export default async function TracePage({ params }: { params: Promise<{ id: string; tid: string }> }) {
  const { id, tid } = await params;
  const found = getTrace(id, tid);
  if (!found) notFound();
  const { summary: s, trace: t, index, total } = found;
  const u = usageOf(t);
  const data = t.task?.data ?? {};
  const longKeys = new Set(["prompt", "system_prompt"]);
  const tm = t.timing ?? {};
  const seg = [
    ["setup", tm.setup ? tm.setup.end - tm.setup.start : 0, "var(--muted)"],
    ["model", tm.agent?.model?.duration ?? 0, "var(--accent)"],
    ["harness", tm.agent?.harness?.duration ?? 0, "var(--warn)"],
    ["finalize", tm.finalize ? tm.finalize.end - tm.finalize.start : 0, "var(--ink-2)"],
    ["scoring", tm.scoring ? tm.scoring.end - tm.scoring.start : 0, "var(--good)"],
  ] as [string, number, string][];
  const segTotal = seg.reduce((a, [, v]) => a + v, 0) || 1;
  const d = durationOf(t);

  return (
    <>
      <div className="crumbs"><Link href="/">runs</Link> / <Link href={`/runs/${id}`}>{s.rel}</Link> / {t.id.slice(0, 8)}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
        <h1 className="mono" style={{ fontSize: 16 }}>trace {t.id}</h1>
        <span className="muted">rollout {index + 1} of {total}</span>
      </div>
      <div className="chips">
        <span className="chip">task <b>{String(taskIndex(t))}</b></span>
        {t.ok ? <span className={`chip ${rewardOf(t) >= 1 ? "good" : rewardOf(t) <= 0 ? "bad" : "warn"}`}>reward {fmt(rewardOf(t), 3)}</span> : <span className="chip err">errored</span>}
        <span className="chip">stop <b>{t.stop_condition}</b></span>
        <span className="chip">harness <b>{t.agent?.config?.harness?.id ?? s.harness}</b></span>
        <span className="chip">runtime <b>{t.agent?.config?.runtime?.type ?? s.runtime}</b></span>
        <span className="chip">model <b>{t.agent?.config?.model ?? s.model}</b></span>
        <span className="chip">{u.calls} calls · {u.prompt} / {u.completion} tokens · {money(u.cost)}</span>
        {d != null && <span className="chip">{fmt(d, 1)}s</span>}
      </div>

      {t.errors?.length > 0 && (
        <div className="errbox">{t.errors.map((e, i) => <div key={i}><b>{e.type}</b>{e.status_code ? ` (${e.status_code})` : ""}: {e.message}</div>)}</div>
      )}

      <div className="cols" style={{ marginTop: 14 }}>
        <div>
          <h2>Conversation · {t.nodes?.length ?? 0} nodes</h2>
          {(t.nodes ?? []).length === 0 && <div className="muted">No messages were recorded.</div>}
          {(t.nodes ?? []).map((n: Json, i: number) => {
            const m = n.message ?? {};
            const role = m.role ?? "?";
            return (
              <div key={i} className={`msg ${role}`}>
                <div className="head">
                  <span className="role">{role}</span>
                  <span>#{i}{n.parent != null ? ` ← #${n.parent}` : ""}</span>
                  {n.sampled ? <span className="chip good" style={{ padding: "0 6px" }}>sampled</span> : <span className="chip" style={{ padding: "0 6px" }}>given</span>}
                  {m.tool_call_id && <span>tool_call_id {m.tool_call_id}</span>}
                </div>
                <div className="body">
                  {m.reasoning_content && (
                    <details><summary>reasoning · {String(m.reasoning_content).length} chars</summary><pre className="muted">{m.reasoning_content}</pre></details>
                  )}
                  {m.content != null && m.content !== "" && <pre>{text(m.content)}</pre>}
                  {(m.tool_calls ?? []).map((tc: Json, j: number) => {
                    const fn = tc.function ?? tc;
                    return (
                      <div key={j} className="toolcall">
                        <div className="mono" style={{ color: "var(--warn)" }}>{fn.name}({tc.id ? <span className="muted"> {tc.id}</span> : null})</div>
                        <pre>{pretty(fn.arguments)}</pre>
                      </div>
                    );
                  })}
                  {!m.content && !m.reasoning_content && !(m.tool_calls?.length) && <span className="muted">(empty)</span>}
                </div>
              </div>
            );
          })}
        </div>
        <div>
          <h2>Rewards</h2>
          <div className="panel">
            {Object.keys(t.rewards ?? {}).length === 0 ? <span className="muted">none recorded{t.ok ? "" : " (rollout errored before scoring)"}</span> : (
              <table><thead><tr><th>name</th><th className="num">score</th><th className="num">weight</th><th className="num">contribution</th></tr></thead>
                <tbody>{Object.entries(t.rewards).map(([k, r]) => <tr key={k}><td className="mono">{k}</td><td className="num">{fmt(r.score, 3)}</td><td className="num">{r.weight}</td><td className="num">{fmt(r.score * r.weight, 3)}</td></tr>)}</tbody></table>
            )}
          </div>
          <h2>Metrics</h2>
          <div className="panel">
            {Object.keys(t.metrics ?? {}).length === 0 ? <span className="muted">none recorded</span> : (
              <dl className="kv">{Object.entries(t.metrics).map(([k, v]) => <><dt key={k + "k"}>{k}</dt><dd key={k + "v"} className="mono">{typeof v === "number" ? fmt(v, 3) : String(v)}</dd></>)}</dl>
            )}
          </div>
          <h2>Task data</h2>
          <div className="panel">
            <dl className="kv">
              {Object.entries(data).filter(([k]) => !longKeys.has(k)).map(([k, v]) => <><dt key={k + "k"}>{k}</dt><dd key={k + "v"} className="mono">{typeof v === "string" ? v : JSON.stringify(v)}</dd></>)}
            </dl>
            {Array.from(longKeys).filter((k) => data[k] != null).map((k) => <details key={k}><summary className="mono muted" style={{ cursor: "pointer", marginTop: 6 }}>{k}</summary><pre>{text(data[k])}</pre></details>)}
          </div>
          <h2>Model calls</h2>
          <div className="panel" style={{ overflowX: "auto" }}>
            {(t.calls ?? []).length === 0 ? <span className="muted">none</span> : (
              <table><thead><tr><th>#</th><th>node</th><th>finish</th><th className="num">in</th><th className="num">out</th><th className="num">reasoning</th><th className="num">cost</th><th className="num">time</th></tr></thead>
                <tbody>{t.calls.map((c: Json, i: number) => (
                  <tr key={i}><td>{i}</td><td className="mono">{c.node ?? "–"}</td><td className="mono">{c.error ? <span style={{ color: "var(--bad)" }}>{c.error.type}</span> : c.finish_reason}</td>
                    <td className="num">{c.usage?.prompt_tokens ?? "–"}</td><td className="num">{c.usage?.completion_tokens ?? "–"}</td><td className="num">{c.usage?.reasoning_tokens ?? "–"}</td>
                    <td className="num">{c.usage?.cost != null ? money(c.usage.cost) : "–"}</td><td className="num">{c.time ? `${fmt(c.time.end - c.time.start, 2)}s` : "–"}</td></tr>
                ))}</tbody></table>
            )}
          </div>
          <h2>Timing</h2>
          <div className="panel">
            <div className="timing">{seg.map(([k, v, c]) => <span key={k} style={{ width: `${(v / segTotal) * 100}%`, background: c }} title={`${k} ${fmt(v, 2)}s`} />)}</div>
            <div className="legend">{seg.map(([k, v, c]) => <span key={k}><i style={{ background: c }} />{k} {fmt(v, 2)}s</span>)}</div>
          </div>
          {(t.tools ?? []).length > 0 && (
            <>
              <h2>Tools the harness exposed</h2>
              <div className="panel"><dl className="kv">{t.tools!.map((tool: Json) => <><dt key={tool.name + "k"}>{tool.name}</dt><dd key={tool.name + "v"}>{tool.description}</dd></>)}</dl></div>
            </>
          )}
          <h2>Agent config</h2>
          <div className="panel"><pre className="mono" style={{ fontSize: 11.5 }}>{JSON.stringify(t.agent?.config ?? {}, null, 2)}</pre></div>
        </div>
      </div>
      <details className="raw"><summary>raw trace JSON</summary><pre>{JSON.stringify(t, null, 2)}</pre></details>
    </>
  );
}
