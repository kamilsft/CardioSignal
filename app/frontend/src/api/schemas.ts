import { z } from "zod";

export const directionSchema = z.enum(["ecg_to_pcg", "pcg_to_ecg"]);
export type Direction = z.infer<typeof directionSchema>;

export const analysisSchema = z.object({
  heart_rate_bpm: z.number().nullable(),
  beats: z.array(z.number().int()),
  quality: z.number().nullable().optional(),
  quality_ok: z.boolean().nullable().optional(),
});
export type Analysis = z.infer<typeof analysisSchema>;

export const translationSchema = z.object({
  direction: directionSchema,
  sample_rate: z.number().int().positive(),
  input_signal: z.array(z.number()),
  output_signal: z.array(z.number()),
  input_analysis: analysisSchema,
  output_analysis: analysisSchema,
  ecg_was_flipped: z.boolean(),
  warnings: z.array(z.string()),
});
export type Translation = z.infer<typeof translationSchema>;

export const sampleSummarySchema = z.object({
  id: z.string(),
  label: z.enum(["healthy", "cvd"]),
  description: z.string(),
});
export type SampleSummary = z.infer<typeof sampleSummarySchema>;

export const sampleSchema = sampleSummarySchema.extend({
  sample_rate: z.number().int().positive(),
  ecg: z.array(z.number()),
  pcg: z.array(z.number()),
});
export type Sample = z.infer<typeof sampleSchema>;

export function envelopeSchema<T extends z.ZodTypeAny>(data: T) {
  return z.object({
    success: z.boolean(),
    data: data.nullable().optional(),
    error: z.string().nullable().optional(),
  });
}
