import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PlaybackClock } from "../../lib/playback";
import { prepareReplay } from "../../lib/replay";
import { straightReplay } from "../../test/fixtures";
import { TelemetryCharts } from "./TelemetryCharts";

describe("TelemetryCharts", () => {
  it("plots speed and steering on a shared time axis", () => {
    const replay = prepareReplay(straightReplay().points);
    const { container } = render(
      <TelemetryCharts clock={new PlaybackClock(replay.duration)} replay={replay} />,
    );
    expect(screen.getByRole("heading", { name: "Speed" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Steering" })).toBeInTheDocument();
    expect(container.querySelectorAll(".recharts-line")).toHaveLength(2);
    expect(screen.getByText("21 samples at 10 Hz.", { exact: false })).toBeInTheDocument();
  });

  it("says which channels the replay contract does not carry instead of plotting zeros", () => {
    const replay = prepareReplay(straightReplay().points);
    render(<TelemetryCharts clock={new PlaybackClock(replay.duration)} replay={replay} />);
    expect(
      screen.getByText(
        /Lateral velocity, lateral acceleration and tyre friction \(μ\) are not plotted/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /friction/i })).not.toBeInTheDocument();
  });

  it("explains a replay too short to plot", () => {
    const replay = prepareReplay({ t: [0], x: [0], y: [0], theta: [0], v: [0], steer: [0] });
    render(<TelemetryCharts clock={new PlaybackClock(0)} replay={replay} />);
    expect(screen.getByText(/single sample/)).toBeInTheDocument();
  });
});
