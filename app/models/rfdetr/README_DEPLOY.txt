LCT FINAL80 DEPLOYMENT MODEL

MODEL
RF-DETR Medium @640
File: final_model.pth
Weights: final EMA weights after fixed 30-epoch FINAL80 training.

TRAINING DATA
65 former organizer TRAIN + 15 former organizer VAL = 80 development images.
20 TEST images were not used by this notebook.

METRIC NOTE
FINAL80 has no held-out validation set.
Do NOT claim FINAL80 mAP50-95 = 0.2398.
0.2398 belongs to the previous 65-train / 15-held-out-val development model.

INFERENCE
Recommended initial deployment:
- full-frame only
- resolution 640
- confidence threshold 0.25

Experimental crop/ROI fusion is not baked into this checkpoint.
