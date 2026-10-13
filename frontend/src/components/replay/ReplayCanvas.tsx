import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { TrackGeometry } from "../../api/types";
import { useElementSize } from "../../hooks/useElementSize";
import type { PlaybackClock } from "../../lib/playback";
import type { PreparedReplay } from "../../lib/replay";
import { emptySample, sampleAt } from "../../lib/replay";
import type { Layout } from "../../lib/trackRenderer";
import {
  PALETTE,
  SPEED_COLOURS,
  drawCar,
  drawStaticLayer,
  drawTrace,
  layoutScene,
  speedBins,
} from "../../lib/trackRenderer";
import { formatSpeed } from "../../lib/formatters";

interface ReplayCanvasProps {
  geometry: TrackGeometry | null;
  replay: PreparedReplay | null;
  /** Drives the car; null draws the track and line without a car. */
  clock: PlaybackClock | null;
  /** Text alternative for the drawing. */
  description: string;
}

function useDevicePixelRatio(): number {
  const [ratio, setRatio] = useState(() => window.devicePixelRatio || 1);
  useEffect(() => {
    // Re-evaluated whenever the ratio changes (window moved to another display, zoom).
    const query = window.matchMedia(`(resolution: ${ratio}dppx)`);
    const update = () => {
      setRatio(window.devicePixelRatio || 1);
    };
    query.addEventListener("change", update);
    return () => {
      query.removeEventListener("change", update);
    };
  }, [ratio]);
  return ratio;
}

/**
 * The track map. World metres map to CSS pixels through one proportional transform; the backing
 * store is scaled by devicePixelRatio for sharp lines. The static layer is rendered offscreen
 * once per size, geometry or replay change; each frame blits it and draws the trace and car.
 */
export function ReplayCanvas({ geometry, replay, clock, description }: ReplayCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width, height } = useElementSize(containerRef);
  const dpr = useDevicePixelRatio();

  const layout = useMemo<Layout | null>(
    () => layoutScene({ geometry, replay }, width, height),
    [geometry, replay, width, height],
  );
  const bins = useMemo(() => (replay === null ? new Uint8Array(0) : speedBins(replay)), [replay]);
  const staticLayer = useRef<HTMLCanvasElement | null>(null);
  const sample = useRef(emptySample());

  const drawFrame = useCallback(
    (time: number) => {
      const canvas = canvasRef.current;
      const context = canvas?.getContext("2d");
      if (!canvas || !context) return;
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (layout === null) return;
      if (staticLayer.current !== null) context.drawImage(staticLayer.current, 0, 0);
      if (replay === null || replay.length === 0 || clock === null) return;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      sampleAt(replay, time, sample.current);
      drawTrace(context, layout.view, replay, bins, sample.current);
      drawCar(context, layout.view, sample.current);
    },
    [layout, replay, clock, bins, dpr],
  );

  // Resize the backing store and rebuild the static layer.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pixelWidth = Math.max(1, Math.round(width * dpr));
    const pixelHeight = Math.max(1, Math.round(height * dpr));
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    if (layout === null) {
      staticLayer.current = null;
    } else {
      // A fresh layer per size or data change (rare); the previous one is garbage collected.
      const offscreen = document.createElement("canvas");
      offscreen.width = pixelWidth;
      offscreen.height = pixelHeight;
      const context = offscreen.getContext("2d");
      if (context !== null) {
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawStaticLayer(context, { geometry, replay }, layout);
        staticLayer.current = offscreen;
      } else {
        staticLayer.current = null;
      }
    }
    drawFrame(clock?.getTime() ?? 0);
  }, [width, height, dpr, layout, geometry, replay, clock, drawFrame]);

  // Redraw on every clock frame. Subscribing calls drawFrame once immediately.
  useEffect(() => {
    if (clock === null) return;
    return clock.subscribeFrame(drawFrame);
  }, [clock, drawFrame]);

  // Drop the offscreen layer with the canvas.
  useEffect(
    () => () => {
      staticLayer.current = null;
    },
    [],
  );

  return (
    <div className="absolute inset-0 flex flex-col">
      <div ref={containerRef} className="relative min-h-0 flex-1">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={description}
          className="absolute inset-0 block h-full w-full"
        />
      </div>
      <MapLegend
        layout={layout}
        replay={replay}
        hasCentreline={geometry !== null}
        hasEdges={geometry?.left != null}
      />
    </div>
  );
}

/** Legend and scale bar in a strip under the map, so they never cover the track. */
function MapLegend({
  layout,
  replay,
  hasCentreline,
  hasEdges,
}: {
  layout: Layout | null;
  replay: PreparedReplay | null;
  hasCentreline: boolean;
  hasEdges: boolean;
}) {
  const barPx = layout === null ? 0 : layout.gridStep * layout.view.scale;
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-line px-3 py-2 font-mono text-[10.5px] text-muted">
      <ul aria-label="Map legend" className="flex flex-wrap items-center gap-x-5 gap-y-1.5">
        {hasEdges && (
          <li className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="h-2.5 w-4 border-y"
              style={{ borderColor: PALETTE.edge, background: PALETTE.surface }}
            />
            Track surface
          </li>
        )}
        {hasCentreline && (
          <li className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={`w-4 border-t ${hasEdges ? "border-dashed" : ""}`}
              style={{ borderColor: PALETTE.edge }}
            />
            Centreline
          </li>
        )}
        {replay !== null && (
          <li className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="h-[3px] w-6 rounded-full"
              style={{
                background: `linear-gradient(90deg, ${SPEED_COLOURS[0]}, ${SPEED_COLOURS[SPEED_COLOURS.length - 1] ?? ""})`,
              }}
            />
            Driven line by speed, 0 to {formatSpeed(replay.maxSpeed)} m/s
          </li>
        )}
      </ul>
      {layout !== null && barPx >= 12 && (
        <div className="ml-auto flex items-center gap-2">
          <span
            aria-hidden="true"
            className="h-1.5 border-x border-b border-muted"
            style={{ width: `${barPx}px` }}
          />
          <span>
            {layout.gridStep >= 1 ? `${layout.gridStep} m` : `${layout.gridStep * 100} cm`} grid
          </span>
        </div>
      )}
    </div>
  );
}
