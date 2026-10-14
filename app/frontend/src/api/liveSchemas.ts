import { z } from "zod";

export const patientStatusSchema = z.enum(["warming_up", "ok", "no_signal"]);
export type PatientStatus = z.infer<typeof patientStatusSchema>;

export const patientVitalsSchema = z.object({
  id: z.string(),
  heart_rate_bpm: z.number().nullable(),
  status: patientStatusSchema,
});
export type PatientVitals = z.infer<typeof patientVitalsSchema>;

export const liveSummarySchema = z.object({
  patients: z.number().int().nonnegative(),
  devices_connected: z.number().int().nonnegative(),
  ok: z.number().int().nonnegative(),
  warming_up: z.number().int().nonnegative(),
  no_signal: z.number().int().nonnegative(),
  analysis_ms: z.number().nonnegative(),
});
export type LiveSummary = z.infer<typeof liveSummarySchema>;

export const liveSnapshotSchema = z.object({
  type: z.literal("snapshot"),
  time: z.number(),
  summary: liveSummarySchema,
  patients: z.array(patientVitalsSchema),
  waveforms: z.record(z.object({ fs: z.number().positive(), samples: z.array(z.number()) })),
});
export type LiveSnapshot = z.infer<typeof liveSnapshotSchema>;
