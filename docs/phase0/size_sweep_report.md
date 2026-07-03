# Size-sweep matched ablation (CarFusion vs CarFusion+synthetic)

**Date:** 2026-06-28
**Init:** COCO-pretrained yolo26{n,s,m,l}-pose; both arms per size trained identically
(30 ep, 480px) on the full CarFusion train; only the synthetic data differs.
Eval on the real CarFusion val set (3474 images).

Reference: nano production baseline (flipfix, full real) OKS-mAP 0.5038.

| size | real OKS-mAP | real mAP50 | real PCK | +synth OKS-mAP | +synth mAP50 | +synth PCK | synth contribution |
|---|---|---|---|---|---|---|---|
| n | 0.4956 | 0.6949 | 0.7587 | 0.4869 | 0.6906 | 0.7537 | -0.87pp |
| s | 0.4446 | 0.6035 | 0.7790 | 0.4781 | 0.6555 | 0.7757 | +3.35pp |
| m | 0.4720 | 0.6255 | 0.7976 | 0.4886 | 0.6467 | 0.7925 | +1.66pp |
| l | 0.5109 | 0.6684 | 0.7990 | 0.4876 | 0.6425 | 0.7938 | -2.33pp |

**Strongest model: l_real (OKS-mAP 0.5109).**
