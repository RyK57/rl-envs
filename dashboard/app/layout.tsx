import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { OUTPUTS_DIR } from "@/lib/runs";

export const metadata: Metadata = { title: "rl-envs observer", description: "Live view over eval runs in outputs/" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="topbar">
          <Link href="/" className="brand">rl-envs observer</Link>
          <span className="path">{OUTPUTS_DIR}</span>
          <span className="spacer" />
        </div>
        <main>{children}</main>
      </body>
    </html>
  );
}
