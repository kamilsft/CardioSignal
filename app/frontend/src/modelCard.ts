/**
 * Evaluation results shown in the app: test set of 10 people never seen in training
 * (person-level split, training/splits.py). Source: comparison/v4/beats_v4.json and
 * comparison/v4/eval/test_metrics.json (training/beat_accuracy.py, training/evaluate.py).
 * Update together with the deployed model file.
 */
export const MODEL_CARD = {
  version: "v4 (person-level split)",
  testSegments: 4458,
  testPeople: 10,
  metrics: [
    { value: "74%", label: "of heartbeats placed within ±50 ms", detail: "Heart sound → ECG, beat-level F1 (chance: 15%)" },
    { value: "82%", label: "median across the 10 test people", detail: "Heart sound → ECG, beat-level F1 per person" },
    { value: "88%", label: "heart rate within 5 bpm", detail: "ECG → heart sound, rhythm of the generated sound (chance: 12%)" },
  ],
  limitations: [
    "Accuracy drops on noisy heart-sound recordings (for example during exercise); the app flags low-quality input.",
    "Only 2 heart-disease patients in the test set (98% and 45% beat F1), too few to draw conclusions about diseased hearts.",
    "The exact ECG shape varies from person to person and is only loosely reproduced; heartbeat timing and rhythm are.",
    "Research demo. Not a medical device and not for diagnosis.",
  ],
} as const;
