# Web Demo

This directory contains the Flask inference demo and trained checkpoints for the urban greening maintenance image classifier.

## Model

The demo uses a weighted ensemble:

- EfficientNet-B0: 0.25
- MobileNetV3-Small: 0.35
- ResNet18: 0.40

For each architecture, `fold_1.pth` through `fold_5.pth` are averaged first. The three model-level probability vectors are then combined with the weights above.

## Run

Install dependencies from the repository root or from this directory:

```bash
python3 -m pip install -r requirements_app.txt
```

Start the app:

```bash
cd final_app
python3 app.py
```

Open:

```text
http://127.0.0.1:7860
```

## Notes

The first prediction loads all checkpoint files and may take a few seconds. Later predictions reuse the loaded models.

The trained checkpoints are stored in `checkpoints/`. If you retrain the models, replace the corresponding `fold_1.pth` through `fold_5.pth` files while keeping the same class order in `class_names.json`.
