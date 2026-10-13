import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach } from "vitest";

import { clearApiCaches } from "../hooks/useApi";

/** Size every observed element reports, CSS pixels. Tests may change it before rendering. */
export const observedSize = { width: 800, height: 400 };

// jsdom has no layout engine: report a fixed size as soon as an element is observed.
class FixedSizeResizeObserver implements ResizeObserver {
  private readonly callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }
  observe(target: Element): void {
    const rect = new DOMRectReadOnly(0, 0, observedSize.width, observedSize.height);
    const entry: ResizeObserverEntry = {
      target,
      contentRect: rect,
      borderBoxSize: [],
      contentBoxSize: [],
      devicePixelContentBoxSize: [],
    };
    this.callback([entry], this);
  }
  unobserve(): void {}
  disconnect(): void {}
}

Object.defineProperty(window, "ResizeObserver", { value: FixedSizeResizeObserver, writable: true });

// jsdom implements no canvas; the renderer treats a null context as "nothing to draw".
Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: () => null,
  writable: true,
});

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string): MediaQueryList => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  }),
});

Object.defineProperty(window, "scrollTo", { value: () => undefined, writable: true });

beforeEach(() => {
  clearApiCaches();
  window.history.replaceState(null, "", "/");
});

afterEach(() => {
  cleanup();
});
