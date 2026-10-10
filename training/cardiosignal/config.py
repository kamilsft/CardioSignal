"""Shared constants. Keep the app and training in sync by importing from here."""

SOURCE_FS = 8000          # sample rate of the original lab recordings (Hz)
TARGET_FS = 2000          # sample rate used for training and inference (Hz)
DECIMATION = SOURCE_FS // TARGET_FS
SEGMENT_SECONDS = 6
SEGMENT_LEN = TARGET_FS * SEGMENT_SECONDS   # 12000 samples

ENVELOPE_WINDOW = TARGET_FS // 20           # 50 ms window for loudness envelope

ECG = "ecg"
PCG = "pcg"
