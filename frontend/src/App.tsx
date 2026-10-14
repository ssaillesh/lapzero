import { useEffect } from "react";

import { Link } from "./components/common/Link";
import { PageHeader } from "./components/common/Controls";
import { EmptyState } from "./components/common/States";
import { AppShell } from "./components/layout/AppShell";
import { HealthProvider } from "./components/layout/HealthProvider";
import { rememberHref, sectionOf } from "./lib/lastVisited";
import type { Route } from "./lib/router";
import { navigate, useRoute } from "./lib/router";
import { ChampionView } from "./views/ChampionView";
import { LeaderboardView } from "./views/LeaderboardView";
import { TrackIndexView, TrackViewerView } from "./views/TrackViewerView";

const TITLES: Record<Route["name"], string> = {
  home: "Leaderboard",
  leaderboard: "Leaderboard",
  tracks: "Track Explorer",
  track: "Track Explorer",
  governance: "Model governance",
  "not-found": "Not found",
};

function View({ route }: { route: Route }) {
  switch (route.name) {
    case "home":
      return null;
    case "leaderboard":
      return <LeaderboardView params={route.params} />;
    case "tracks":
      return <TrackIndexView />;
    case "track":
      return (
        <TrackViewerView
          trackId={route.trackId}
          lapId={route.lapId}
          compareLapId={route.compareLapId ?? null}
        />
      );
    case "governance":
      return <ChampionView />;
    case "not-found":
      return (
        <div className="grid gap-6">
          <PageHeader title="Page not found" />
          <EmptyState
            title={`Nothing lives at ${route.path}`}
            action={
              <Link href="/leaderboard" className="press btn-secondary">
                Go to the leaderboard
              </Link>
            }
          />
        </div>
      );
  }
}

export function App() {
  const route = useRoute();

  useEffect(() => {
    if (route.name === "home") navigate("/leaderboard", { replace: true });
    const section = sectionOf(route);
    if (section !== null && route.name !== "home") {
      rememberHref(section, `${window.location.pathname}${window.location.search}`);
    }
    document.title = `${TITLES[route.name]} | LapZero`;
  }, [route]);

  return (
    <HealthProvider>
      <AppShell route={route}>
        <View route={route} />
      </AppShell>
    </HealthProvider>
  );
}
