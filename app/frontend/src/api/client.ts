import { z } from "zod";
import {
  envelopeSchema,
  sampleSchema,
  sampleSummarySchema,
  translationSchema,
  type Direction,
  type Sample,
  type SampleSummary,
  type Translation,
} from "./schemas";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {}

const NETWORK_MESSAGE = "Could not reach the CardioSignal server. Is the backend running?";

async function request<T extends z.ZodTypeAny>(path: string, schema: T, init?: RequestInit): Promise<z.infer<T>> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(NETWORK_MESSAGE);
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    if (response.status === 413) throw new ApiError("File is too large (limit 5 MB).");
    throw new ApiError(`The server returned an unexpected response (HTTP ${response.status}).`);
  }
  const parsed = envelopeSchema(schema).safeParse(body);
  if (!parsed.success) {
    throw new ApiError("The server response did not have the expected format.");
  }
  if (!parsed.data.success || parsed.data.data == null) {
    throw new ApiError(parsed.data.error ?? `Request failed (HTTP ${response.status}).`);
  }
  return parsed.data.data;
}

/** WebSocket URL for the live monitor, on the same server as the REST API. */
export function liveStreamUrl(): string {
  const base = API_BASE
    ? API_BASE.replace(/^http/, "ws")
    : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
  return `${base}/api/live/stream`;
}

export function listSamples(): Promise<SampleSummary[]> {
  return request("/api/samples", z.array(sampleSummarySchema));
}

export function getSample(id: string): Promise<Sample> {
  return request(`/api/samples/${encodeURIComponent(id)}`, sampleSchema);
}

export function translateSignal(signal: number[], sampleRate: number, direction: Direction): Promise<Translation> {
  return request("/api/translate", translationSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, sample_rate: sampleRate, signal }),
  });
}

export function translateFile(file: File, direction: Direction, sampleRate: number | null): Promise<Translation> {
  const form = new FormData();
  form.append("file", file);
  form.append("direction", direction);
  if (sampleRate !== null) form.append("sample_rate", String(sampleRate));
  return request("/api/translate/upload", translationSchema, { method: "POST", body: form });
}
