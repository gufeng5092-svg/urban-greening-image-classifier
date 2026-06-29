# Model Artifacts

The repository currently keeps trained PyTorch checkpoints under `final_app/checkpoints/` so the demo and API can run immediately after cloning.

For a public or enterprise release, prefer one of these strategies:

1. Keep small demo checkpoints in Git and publish full production checkpoints as GitHub Release assets.
2. Use Git LFS for `*.pth` files when checkpoint history matters.
3. Store private production weights in an internal artifact registry and mount or download them during deployment.

When replacing model artifacts, keep these files aligned:

- `final_app/class_names.json`: class order used by the classifier head.
- `configs/inference.yaml`: checkpoint directory, image size and ensemble weights.
- `final_app/checkpoints/<model_name>/fold_*.pth`: fold checkpoint names expected by the ensemble loader.

Do not change class order unless the checkpoints are retrained with the same order and the change is documented.
