import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// jsdom has no canvas; charts only need a context that accepts drawing calls.
HTMLCanvasElement.prototype.getContext = function getContext() {
  return new Proxy({}, { get: () => () => undefined, set: () => true });
} as unknown as typeof HTMLCanvasElement.prototype.getContext;
