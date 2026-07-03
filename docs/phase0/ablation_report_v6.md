# Phase 0 Ablation v6 (init from the STRONG production checkpoint)

**Date:** 2026-07-02
**Init:** flipfix production checkpoint (OKS-mAP 0.5038). Both arms
fine-tuned identically (corrected flip_idx, 25 ep, 480px, batch 16, lr0 2e-4,
100 real x8); only the synthetic data differs.

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs prod baseline |
|---|---|---|---|---|
| production baseline (full real train) | 0.5038 | 0.7036 | 0.7606 | +0.00pp |
| arm B (prod + 100 real x8, no synth) | 0.4410 | 0.6585 | 0.7157 | -6.28pp |
| arm A (prod + synth_v4 + 100 real x8) | 0.3460 | 0.6171 | 0.6100 | -15.79pp |

**Synth contribution (arm A - arm B): -9.51pp OKS-mAP**
