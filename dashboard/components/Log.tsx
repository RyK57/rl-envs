export default function Log({ lines }: { lines: string[] }) {
  if (!lines.length) return <div className="muted">No log file in this run directory.</div>;
  return (
    <pre className="log">
      {lines.map((l, i) => {
        const level = /\b(ERROR|CRITICAL)\b/.test(l) ? "error" : /\bWARNING\b/.test(l) ? "warn" : /\bDEBUG\b/.test(l) ? "debug" : "info";
        return (
          <span key={i} className={`ll ${level}`}>
            {l}
            {"\n"}
          </span>
        );
      })}
    </pre>
  );
}
