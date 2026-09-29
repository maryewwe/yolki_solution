# RF-DETR FINAL80 deployment model

Default model file:

```text
models/rfdetr/final_model.pth
```

This is the `last_ema.pth` artifact after the fixed 30-epoch FINAL80 training run,
exported as `final_model.pth` by `lct-final80-rfdetr-medium640.ipynb`.

Model contract:

- RF-DETR Medium;
- resolution 640;
- `rfdetr==1.10.1`;
- 13 canonical classes in `taxonomy.yaml`;
- full-frame inference at the project threshold `0.25`.

Artifact verification:

```text
size:    134409987 bytes
SHA-256: 94a68be921b4e24cf12f362278c97bc2ae4ac47b1872a306ad9189e8e3702058
```

`deployment_meta.json` and `README_DEPLOY.txt` are copied from the Kaggle
deployment bundle. The two in-sample `checkpoint_best_*` files are not deployment
weights and are intentionally not included here.

The default can still be overridden without changing the API:

```bash
RFDETR_CHECKPOINT=/path/to/final_model.pth
```
