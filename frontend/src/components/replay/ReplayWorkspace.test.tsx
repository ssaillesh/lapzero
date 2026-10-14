import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Replay } from "../../api/types";
import { LAP_P2, squareTrack, straightReplay } from "../../test/fixtures";
import { ReplayWorkspace } from "./ReplayWorkspace";

function renderReplay(replay = straightReplay()) {
  return render(
    <ReplayWorkspace replay={replay} geometry={squareTrack().geometry} trackName="spielberg" />,
  );
}

function output(name: string): HTMLElement {
  const term = screen.getByText(name, { selector: "dt span" });
  const value = term.closest("div")?.querySelector<HTMLElement>("[data-readout]");
  if (!value) throw new Error(`no output for ${name}`);
  return value;
}

describe("ReplayWorkspace", () => {
  it("starts paused at 0 and does not autoplay", () => {
    renderReplay();
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    expect(output("Time")).toHaveTextContent("0.00");
  });

  it("toggles play and pause from the button and from the Space key", async () => {
    const user = userEvent.setup();
    renderReplay();
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Playing at 1 times speed");
    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: " " });
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
  });

  it("scrubbing moves the readout to the interpolated sample", () => {
    renderReplay();
    fireEvent.change(screen.getByRole("slider", { name: "Replay position" }), {
      target: { value: "1.05" },
    });
    expect(output("Time")).toHaveTextContent("1.05");
    expect(output("Position x, y")).toHaveTextContent("5.3, 0.0"); // 5 m/s x 1.05 s
    expect(output("Speed")).toHaveTextContent("5.00");
    expect(output("Steering")).toHaveTextContent("0.53 L");
  });

  it("pauses while the scrubber is dragged and resumes on release", async () => {
    const user = userEvent.setup();
    renderReplay();
    await user.click(screen.getByRole("button", { name: "Play" }));
    const slider = screen.getByRole("slider", { name: "Replay position" });
    fireEvent.pointerDown(slider);
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    fireEvent.change(slider, { target: { value: "0.5" } });
    fireEvent.pointerUp(slider);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
  });

  it("seeks with the arrow keys and restarts with Home", () => {
    renderReplay();
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    expect(output("Time")).toHaveTextContent("1.00");
    fireEvent.keyDown(document.body, { key: "ArrowRight", shiftKey: true });
    expect(output("Time")).toHaveTextContent("2.00"); // clamped to the 2 s replay
    fireEvent.keyDown(document.body, { key: "Home" });
    expect(output("Time")).toHaveTextContent("0.00");
  });

  it("changes the playback speed", async () => {
    const user = userEvent.setup();
    renderReplay();
    await user.click(screen.getByLabelText("2×"));
    expect(screen.getByLabelText("2×")).toBeChecked();
    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("status")).toHaveTextContent("Playing at 2 times speed");
  });

  it("disables playback for a single-sample replay", () => {
    const one = straightReplay();
    const points = { t: [0], x: [1], y: [2], theta: [0], v: [0], steer: [0] };
    renderReplay({ ...one, points });
    expect(screen.getByRole("button", { name: "Play" })).toBeDisabled();
    expect(screen.getByText(/single sample/)).toBeInTheDocument();
  });

  it("describes the map for assistive technology", () => {
    renderReplay();
    expect(
      screen.getByRole("img", {
        name: /Map of spielberg with the driven line of AI v8, lap time 45.98/,
      }),
    ).toBeInTheDocument();
  });

  it("says when the track has no centreline", () => {
    render(<ReplayWorkspace replay={straightReplay()} geometry={null} trackName="spielberg" />);
    expect(screen.getByText(/no centreline for this track/)).toBeInTheDocument();
  });
});

/** A 4 s human lap (twice the open lap's 2 s), 1:17.20 on the leaderboard. */
function humanReplay(): Replay {
  const t = Array.from({ length: 41 }, (_, i) => i / 10);
  return {
    ...straightReplay(LAP_P2),
    driver_label: "human",
    lap_time_s: 77.2,
    points: {
      t,
      x: t.map((s) => 2.5 * s),
      y: t.map(() => 1),
      theta: t.map(() => 0),
      v: t.map(() => 2.5),
      steer: t.map(() => 0),
    },
  };
}

describe("ReplayWorkspace with a compared lap", () => {
  it("runs the clock to the end of the longer lap and names the compared lap", () => {
    render(
      <ReplayWorkspace
        replay={straightReplay()}
        geometry={squareTrack().geometry}
        trackName="spielberg"
        compare={humanReplay()}
      />,
    );
    expect(screen.getByRole("slider", { name: "Replay position" })).toHaveAttribute("max", "4");
    expect(screen.getByText("human, 1:17.20 (+31.22 s)")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /compared with human, 1:17.20/ })).toBeInTheDocument();
  });

  it("ignores a compared lap without usable samples", () => {
    const empty = {
      ...humanReplay(),
      points: { t: [], x: [], y: [], theta: [], v: [], steer: [] },
    };
    render(
      <ReplayWorkspace
        replay={straightReplay()}
        geometry={squareTrack().geometry}
        trackName="spielberg"
        compare={empty}
      />,
    );
    expect(screen.getByRole("slider", { name: "Replay position" })).toHaveAttribute("max", "2");
    expect(screen.queryByText(/human, 1:17.20/)).not.toBeInTheDocument();
  });
});
