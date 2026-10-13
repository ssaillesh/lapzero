import { useMemo, useRef } from "react";
import type { MouseEvent } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import type { TooltipContentProps } from "recharts";

import { useElementSize } from "../../hooks/useElementSize";
import { useClockFrame } from "../../hooks/useReplayPlayback";
import { formatLapTime, formatSpeed, formatSteer } from "../../lib/formatters";
import type { PlaybackClock } from "../../lib/playback";
import type { ChartRow, PreparedReplay } from "../../lib/replay";
import { chartRows } from "../../lib/replay";
import { niceTicks } from "../../lib/ticks";
import { Notice } from "../common/States";

/** Plot geometry shared by both charts, so the playhead and click-to-seek map time to pixels. */
const MARGIN = { top: 8, right: 12, bottom: 6, left: 0 } as const;
const Y_AXIS_WIDTH = 48;
const X_AXIS_HEIGHT = 22;
const CHART_HEIGHT = 132;

const AXIS_TICK = {
  fill: "#767c84",
  fontSize: 10.5,
  fontFamily: "ui-monospace, SF Mono, Menlo, monospace",
};

interface Channel {
  key: "v" | "steer";
  title: string;
  unit: string;
  colour: string;
  domain: [number, number];
  format: (value: number) => string;
  ticks: number[];
}

/**
 * Speed and steering against replay time, on one shared time axis. A playhead tracks the clock
 * (moved by transform, not by re-rendering the chart) and clicking a chart seeks to that time.
 */
export function TelemetryCharts({
  clock,
  replay,
}: {
  clock: PlaybackClock;
  replay: PreparedReplay;
}) {
  const rows = useMemo(() => chartRows(replay), [replay]);
  const channels = useMemo<Channel[]>(() => {
    const speedTicks = niceTicks(Math.max(replay.maxSpeed, 1), 4, true);
    return [
      {
        key: "v",
        title: "Speed",
        unit: "m/s",
        colour: "#d2f34c",
        domain: [0, speedTicks.at(-1) ?? 1],
        ticks: speedTicks,
        format: formatSpeed,
      },
      {
        key: "steer",
        title: "Steering",
        unit: "normalized, +1 full left",
        colour: "#6cc6e3",
        domain: [-1, 1],
        ticks: [-1, 0, 1],
        format: formatSteer,
      },
    ];
  }, [replay.maxSpeed]);

  return (
    <section aria-labelledby="telemetry-heading" className="grid gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 id="telemetry-heading" className="text-[15px] font-semibold text-ink">
          Telemetry
        </h2>
        <p className="text-xs text-faint">
          Click a chart to jump to that moment.{" "}
          {rows.length < replay.length
            ? `${rows.length} of ${replay.length} samples plotted, peaks kept.`
            : `${replay.length} samples at 10 Hz.`}
        </p>
      </div>
      {replay.length < 2 ? (
        <Notice>This replay has a single sample, so there is nothing to plot over time.</Notice>
      ) : (
        <div className="grid gap-px overflow-hidden rounded-[var(--radius-ctl)] border border-line bg-line">
          {channels.map((channel, index) => (
            <ChannelChart
              key={channel.key}
              channel={channel}
              rows={rows}
              clock={clock}
              duration={replay.duration}
              showTimeAxis={index === channels.length - 1}
            />
          ))}
        </div>
      )}
      <Notice>
        Lateral velocity, lateral acceleration and tyre friction (μ) are not plotted: the replay
        contract (Section 6) carries only time, position, heading, longitudinal speed and the
        steering command.
      </Notice>
    </section>
  );
}

interface ChannelChartProps {
  channel: Channel;
  rows: ChartRow[];
  clock: PlaybackClock;
  duration: number;
  showTimeAxis: boolean;
}

function ChannelChart({ channel, rows, clock, duration, showTimeAxis }: ChannelChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(containerRef);
  const height = CHART_HEIGHT + (showTimeAxis ? X_AXIS_HEIGHT : 0);
  const plotLeft = MARGIN.left + Y_AXIS_WIDTH;
  const plotWidth = Math.max(0, width - plotLeft - MARGIN.right);
  const timeTicks = useMemo(
    () => niceTicks(duration, Math.max(2, Math.floor(plotWidth / 120)), false),
    [duration, plotWidth],
  );

  useClockFrame(clock, (time) => {
    if (playheadRef.current === null || duration <= 0) return;
    playheadRef.current.style.transform = `translateX(${((time / duration) * plotWidth).toFixed(2)}px)`;
  });

  const seekFromPointer = (event: MouseEvent<HTMLDivElement>) => {
    if (plotWidth <= 0 || duration <= 0) return;
    const box = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - box.left - plotLeft) / plotWidth;
    if (fraction < 0 || fraction > 1) return;
    clock.seek(fraction * duration);
  };

  return (
    <div className="bg-panel px-1 pt-2">
      <div className="flex items-baseline justify-between px-3">
        <h3 className="flex items-center gap-2 text-[13px] font-medium text-ink">
          <span aria-hidden="true" className="h-0.5 w-3" style={{ background: channel.colour }} />
          {channel.title}
        </h3>
        <span className="font-mono text-[10.5px] text-faint">{channel.unit}</span>
      </div>
      <div
        ref={containerRef}
        className="relative cursor-crosshair"
        style={{ height }}
        onClick={seekFromPointer}
      >
        {width > 0 && (
          <LineChart
            width={width}
            height={height}
            data={rows}
            margin={MARGIN}
            syncId="lapzero-replay"
            syncMethod="value"
            accessibilityLayer={false}
          >
            <CartesianGrid stroke="#22262b" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, duration]}
              allowDataOverflow
              hide={!showTimeAxis}
              height={X_AXIS_HEIGHT}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={{ stroke: "#30353b" }}
              ticks={timeTicks}
              interval={0}
              tickFormatter={(value: number) => `${value} s`}
            />
            <YAxis
              width={Y_AXIS_WIDTH}
              domain={channel.domain}
              allowDataOverflow
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              ticks={channel.ticks}
              interval={0}
            />
            {channel.key === "steer" && <ReferenceLine y={0} stroke="#30353b" />}
            <Tooltip
              isAnimationActive={false}
              cursor={{ stroke: "#9399a1", strokeDasharray: "2 3" }}
              content={(props: TooltipContentProps) => (
                <ChartTooltip {...props} channel={channel} />
              )}
            />
            <Line
              type="linear"
              dataKey={channel.key}
              stroke={channel.colour}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: channel.colour, stroke: "#0b0c0e" }}
              isAnimationActive={false}
            />
          </LineChart>
        )}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute w-px"
          style={{
            left: plotLeft,
            top: MARGIN.top,
            height: CHART_HEIGHT - MARGIN.top - MARGIN.bottom,
          }}
        >
          <div ref={playheadRef} className="h-full w-px bg-ink" />
        </div>
      </div>
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
  channel,
}: TooltipContentProps & { channel: Channel }) {
  const value = payload[0]?.value;
  if (!active || typeof value !== "number" || typeof label !== "number") return null;
  return (
    <div className="rounded-[var(--radius-ctl)] border border-line-strong bg-ground px-2.5 py-1.5 font-mono text-[11px] shadow-[0_8px_24px_rgba(0,0,0,0.45)]">
      <div className="text-faint">t {formatLapTime(label)} s</div>
      <div className="text-ink">
        {channel.title} {channel.format(value)}
        {channel.key === "v" ? " m/s" : ""}
      </div>
    </div>
  );
}
