"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function AutoRefresh({ seconds = 5 }: { seconds?: number }) {
  const router = useRouter();
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => {
      router.refresh();
      setTick((t) => t + 1);
    }, seconds * 1000);
    return () => clearInterval(id);
  }, [router, seconds]);
  return (
    <span className="live" title={`Re-reads outputs/ every ${seconds}s`}>
      <i />live · {tick === 0 ? "just loaded" : `refreshed ×${tick}`}
    </span>
  );
}
