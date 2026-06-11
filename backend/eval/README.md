# Offline Evaluation

Scripts and notebooks for validating fraud detection models against the datasets in `backend/datasets/`.

## Metrics

- Precision, Recall, F1
- AUROC
- PR-AUC

## Run

```bash
cd backend
uv run python eval/offline_validation.py --data-dir datasets
```

Or open `offline_validation.ipynb` in Jupyter.
