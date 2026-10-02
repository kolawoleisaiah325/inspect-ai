# Recorded bottle evaluation

Normal-fit images: 167. Normal-calibration images: 42. Official test images: 83 (20 normal, 63 anomalous). Seed: 42.

| Method | Image AUROC | Pixel AUROC (40x40) | Missed defects | False alarms | Median local CPU ms |
| --- | --- | --- | --- | --- | --- |
| pixel_baseline | 0.9698 | 0.8971 | 12/63 | 0/20 | 23.66 |
| cnn_patch_memory | 0.9976 | 0.9848 | 2/63 | 0/20 | 46.21 |

The CNN + memory method was the primary method before the test run. These test results did not set thresholds or fit memory. The image AUROC is a ranking measure, not percent accuracy.

## Observed misses

- `bottle/test/contamination/003.png`: score 10.9386, below the 11.2513 review threshold.
- `bottle/test/contamination/019.png`: score 10.7611, below the 11.2513 review threshold.

Both misses are contamination examples. All 20 normal test images were below threshold; that small denominator does not establish a zero false-alarm rate for future images.

Pixel AUROC is computed on downsampled masks and maps; it is not the official full-resolution segmentation evaluation. Fixed example derivatives are resized JPEGs and are scored separately. Cold-start time, network latency, camera shifts and other product categories are not covered by the benchmark.

See `artifacts/evaluation.json` for every score, timing, split file name and runtime environment.
