# Phase 0 Kill-Switch Report (v5: corrected flip-aug, fresh two-arm ablation)

**Date:** 2026-07-02
**Fix:** horizontal-flip augmentation now uses the correct L/R `flip_idx`
([1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13]); earlier runs mirrored images with an identity flip_idx,
corrupting keypoints on ~half the augmented samples. Both arms re-run fresh from
the v1 checkpoint with identical settings; the only difference is the synthetic data.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp (0.5238).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
| v1 baseline (full real train) | 0.5038 | 0.7036 | 0.7606 | +0.00pp |
| arm B (v1 + 100 real x8, no synth) | 0.3808 | 0.6254 | 0.6659 | -12.30pp |
| arm A (v1 + synth_v4 + 100 real x8) | 0.3640 | 0.6253 | 0.6186 | -13.98pp |

**Synth contribution (arm A - arm B): -1.68pp OKS-mAP**

## Verdict

**FAIL (arm A 0.3640 < v1 0.5038)**
