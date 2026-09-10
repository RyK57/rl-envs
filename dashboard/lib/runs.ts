import fs from "node:fs";
import path from "node:path";

export const OUTPUTS_DIR = path.resolve(process.env.OUTPUTS_DIR ?? path.join(process.cwd(), "..", "outputs"));

// Traces are framework JSON; keep them loosely typed and read defensively.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any;

export interface Trace {
  id: string;
  ok: boolean;
  stop_condition: string;
  rewards: Record<string, { score: number; weight: number }>;
  metrics: Record<string, number>;
  errors: { type: string; message: string; status_code?: number }[];
  nodes: Json[];
  calls: Json[];
  tools?: Json[];
  task: { type: string; data: Record<string, Json>; key?: string; hash?: string };
  agent: Json;
  timing?: Json;
  info?: Json;
}

export interface Usage {
  prompt: number;
  completion: number;
  reasoning: number;
  cost: number;
  calls: number;
}

export interface RunSummary {
  id: string;
  rel: string;
  dir: string;
  mtime: number;
  taskset: string;
  model: string;
  harness: string;
  runtime: string;
  rollouts: number;
  ok: number;
  errored: number;
  meanReward: number | null;
  metricMeans: Record<string, number>;
  usage: Usage;
  hasLog: boolean;
}

export const encodeId = (rel: string) => rel.split("/").join("__");
export const decodeId = (id: string) => id.split("__").join("/");

function findTraceFiles(dir: string, depth = 0): string[] {
  if (depth > 4 || !fs.existsSync(dir)) return [];
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name === "node_modules" || entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (fs.existsSync(path.join(full, "traces.jsonl"))) out.push(full);
    else out.push(...findTraceFiles(full, depth + 1));
  }
  return out;
}

export function readTraces(runDir: string): Trace[] {
  const text = fs.readFileSync(path.join(runDir, "traces.jsonl"), "utf8");
  const traces: Trace[] = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try {
      const episode = JSON.parse(s);
      for (const t of episode.traces ?? []) traces.push(t);
    } catch {
      // A torn last line is a live run still writing; skip it.
    }
  }
  return traces;
}

export function readConfig(runDir: string): Json | null {
  for (const rel of ["configs/resolved/eval.json", "configs/eval.json"]) {
    const p = path.join(runDir, rel);
    if (fs.existsSync(p)) {
      try {
        return JSON.parse(fs.readFileSync(p, "utf8"));
      } catch {
        return null;
      }
    }
  }
  return null;
}

function logPath(runDir: string): string | null {
  const latest = path.join(runDir, "logs", "latest", "eval.log");
  if (fs.existsSync(latest)) return latest;
  const logs = path.join(runDir, "logs");
  if (!fs.existsSync(logs)) return null;
  const attempts = fs
    .readdirSync(logs)
    .filter((n) => n.startsWith("attempt_"))
    .sort()
    .reverse();
  for (const a of attempts) {
    const p = path.join(logs, a, "eval.log");
    if (fs.existsSync(p)) return p;
  }
  const flat = path.join(logs, "eval.log");
  return fs.existsSync(flat) ? flat : null;
}

export function readLog(runDir: string, maxLines = 400): string[] {
  const p = logPath(runDir);
  if (!p) return [];
  const lines = fs.readFileSync(p, "utf8").split("\n");
  return lines.slice(Math.max(0, lines.length - maxLines));
}

export const rewardOf = (t: Trace): number =>
  Object.values(t.rewards ?? {}).reduce((s, r) => s + (r.score ?? 0) * (r.weight ?? 1), 0);

export function usageOf(t: Trace): Usage {
  const u: Usage = { prompt: 0, completion: 0, reasoning: 0, cost: 0, calls: 0 };
  for (const c of t.calls ?? []) {
    if (!c.usage) continue;
    u.calls += 1;
    u.prompt += c.usage.prompt_tokens ?? 0;
    u.completion += c.usage.completion_tokens ?? 0;
    u.reasoning += c.usage.reasoning_tokens ?? 0;
    u.cost += c.usage.cost ?? 0;
  }
  return u;
}

export function addUsage(a: Usage, b: Usage): Usage {
  return { prompt: a.prompt + b.prompt, completion: a.completion + b.completion, reasoning: a.reasoning + b.reasoning, cost: a.cost + b.cost, calls: a.calls + b.calls };
}

export function durationOf(t: Trace): number | null {
  const tm = t.timing;
  if (!tm?.start) return null;
  const end = tm.scoring?.end || tm.finalize?.end || tm.agent?.end;
  return end ? end - tm.start : null;
}

export const assistantTurns = (t: Trace): number =>
  (t.nodes ?? []).filter((n) => n.sampled && n.message?.role === "assistant").length;

export function lastReply(t: Trace): string {
  const msgs = (t.nodes ?? []).filter((n) => n.sampled && n.message?.role === "assistant");
  if (!msgs.length) return "";
  const c = msgs[msgs.length - 1].message.content;
  return typeof c === "string" ? c : JSON.stringify(c);
}

export const taskIndex = (t: Trace): number | string => t.task?.data?.idx ?? t.task?.key ?? "?";

export function metricMeans(traces: Trace[]): Record<string, number> {
  const acc: Record<string, { sum: number; n: number }> = {};
  for (const t of traces) {
    if (!t.ok) continue;
    for (const [k, v] of Object.entries(t.metrics ?? {})) {
      if (typeof v !== "number") continue;
      acc[k] ??= { sum: 0, n: 0 };
      acc[k].sum += v;
      acc[k].n += 1;
    }
  }
  return Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, v.sum / v.n]));
}

export function summarize(runDir: string): RunSummary {
  const rel = path.relative(OUTPUTS_DIR, runDir).split(path.sep).join("/");
  const traces = readTraces(runDir);
  const config = readConfig(runDir);
  const first = traces[0];
  const ok = traces.filter((t) => t.ok);
  const usage = traces.map(usageOf).reduce(addUsage, { prompt: 0, completion: 0, reasoning: 0, cost: 0, calls: 0 });
  const agentCfg = config?.env?.agent ?? first?.agent?.config ?? {};
  return {
    id: encodeId(rel),
    rel,
    dir: runDir,
    mtime: fs.statSync(path.join(runDir, "traces.jsonl")).mtimeMs,
    taskset: config?.env?.taskset?.id ?? first?.task?.type ?? "?",
    model: config?.model ?? first?.agent?.config?.model ?? "?",
    harness: agentCfg?.harness?.id ?? "?",
    runtime: agentCfg?.runtime?.type ?? "?",
    rollouts: traces.length,
    ok: ok.length,
    errored: traces.length - ok.length,
    meanReward: ok.length ? ok.reduce((s, t) => s + rewardOf(t), 0) / ok.length : null,
    metricMeans: metricMeans(traces),
    usage,
    hasLog: logPath(runDir) !== null,
  };
}

export function listRuns(): RunSummary[] {
  return findTraceFiles(OUTPUTS_DIR)
    .map(summarize)
    .sort((a, b) => b.mtime - a.mtime);
}

function safeRunDir(id: string): string | null {
  const p = path.resolve(OUTPUTS_DIR, decodeId(id));
  if (!p.startsWith(OUTPUTS_DIR + path.sep)) return null;
  return fs.existsSync(path.join(p, "traces.jsonl")) ? p : null;
}

export function getRun(id: string): { summary: RunSummary; traces: Trace[]; config: Json | null; log: string[] } | null {
  const dir = safeRunDir(id);
  if (!dir) return null;
  return { summary: summarize(dir), traces: readTraces(dir), config: readConfig(dir), log: readLog(dir) };
}

export function getTrace(id: string, traceId: string): { summary: RunSummary; trace: Trace; index: number; total: number } | null {
  const dir = safeRunDir(id);
  if (!dir) return null;
  const traces = readTraces(dir);
  const index = traces.findIndex((t) => t.id === traceId);
  if (index < 0) return null;
  return { summary: summarize(dir), trace: traces[index], index, total: traces.length };
}

export const fmt = (n: number | null | undefined, d = 3): string => (n == null || Number.isNaN(n) ? "–" : n.toFixed(d));
export const money = (n: number): string => (n ? `$${n.toFixed(4)}` : "$0");
export function ago(ms: number): string {
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
export const rewardClass = (t: Trace): string => (!t.ok ? "err" : rewardOf(t) >= 1 ? "r1" : rewardOf(t) <= 0 ? "r0" : "rp");
