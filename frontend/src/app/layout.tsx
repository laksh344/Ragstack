import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAGStack — Agentic RAG Platform",
  description:
    "Production RAG platform with hybrid search, LangGraph agent, guardrails, and LangSmith observability.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-900 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
