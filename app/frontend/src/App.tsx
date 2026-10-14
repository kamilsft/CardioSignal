import { useEffect, useState } from "react";
import { LiveMonitor } from "./components/LiveMonitor";
import { TranslatePage } from "./pages/TranslatePage";

type Page = "translate" | "live";

const PAGES: { id: Page; label: string }[] = [
  { id: "translate", label: "Translate" },
  { id: "live", label: "Live monitor" },
];

function pageFromHash(): Page {
  return window.location.hash === "#live" ? "live" : "translate";
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const onHashChange = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return (
    <div className="page">
      <nav className="top-nav" aria-label="Main">
        {PAGES.map((p) => (
          <a key={p.id} href={`#${p.id}`} className={page === p.id ? "nav-link active" : "nav-link"}
            aria-current={page === p.id ? "page" : undefined}>
            {p.label}
          </a>
        ))}
      </nav>
      {page === "translate" ? <TranslatePage /> : <LiveMonitor />}
    </div>
  );
}
