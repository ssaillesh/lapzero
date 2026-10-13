import { useEffect, useState } from "react";
import type { RefObject } from "react";

export interface ElementSize {
  /** Content-box width, CSS pixels. */
  width: number;
  height: number;
}

/** Track an element's content-box size with a ResizeObserver; 0 x 0 until first measured. */
export function useElementSize(ref: RefObject<HTMLElement | null>): ElementSize {
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (element === null) return;
    // ResizeObserver reports the initial size right after observe().
    const observer = new ResizeObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry === undefined) return;
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      setSize((previous) =>
        previous.width === width && previous.height === height ? previous : { width, height },
      );
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, [ref]);

  return size;
}
