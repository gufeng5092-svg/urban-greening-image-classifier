# Urban Greening Image Classifier

An engineering-ready image classification repository for urban greening maintenance inspection. The project keeps the original trained ensemble and experiment behavior intact, while providing maintainable configuration, reusable Python modules, tests, Docker deployment and CI.

The classifier predicts three maintenance issue categories:

- `存在明显暴露垃圾`: obvious exposed trash
- `存在单处面积大于1m²小于5m²的泥土裸露`: one bare-soil area larger than 1 square meter and smaller than 5 square meters
- `存在单处面积大于5m²的泥土裸露`: one bare-soil area larger than 5 square meters

## Features

- K-fold training pipeline for `small_cnn`, `resnet18`, `mobilenet_v3_small` and `efficientnet_b0`.
- Weighted trained ensemble for production inference.
- YAML-based configuration for data processing, training and inference.
- Reusable `src/urban_greening_classifier` package for paths, logging, datasets, transforms, models and inference.
- Flask browser demo kept under `final_app/`.
- FastAPI service with `POST /predict` for deployment.
- Dockerfile, docker-compose and GitHub Actions CI.
- pytest coverage for config loading, data loading, inference and utility functions.
- black, isort and flake8 configuration for consistent code style.

## Project Structure

```text
.
├── configs/                         # YAML configuration files
│   ├── data.yaml
│   ├── inference.yaml
│   └── train.yaml
├── data/                            # Dataset instructions; local data is ignored by Git
├── docs/
│   └── ARCHITECTURE.md              # Architecture and data-flow notes
├── final_app/                       # Existing Flask demo and trained checkpoints
│   ├── app.py
│   ├── inference.py
│   ├── checkpoints/
│   └── examples/
├── src/urban_greening_classifier/   # Reusable package
├── tests/                           # pytest tests
├── data_cleaning.py                 # Raw image checks and cleaned dataset export
├── data_augmentation.py             # Offline class-balancing augmentation
├── train_cross_validation.py        # K-fold training
├── export_cv_predictions.py         # Export fold validation probabilities
├── ensemble_predictions.py          # Offline weighted ensembling
├── Dockerfile
├── docker-compose.yml
└── README.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture overview.
Model artifact handling is documented in [docs/MODEL_ARTIFACTS.md](docs/MODEL_ARTIFACTS.md).
Deployment device policy is documented in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data Preparation

The original dataset is not included. Put local raw images under `data/raw/` with the label prefix before the final underscore:

```text
data/raw/
├── 存在明显暴露垃圾_001.png
├── 存在单处面积大于1m²小于5m²的泥土裸露_002.png
└── 存在单处面积大于5m²的泥土裸露_003.png
```

Clean the dataset:

```bash
python data_cleaning.py --source data/raw --clean data/cleaned --report cleaning_report
```

Optional offline augmentation:

```bash
python data_augmentation.py --source data/cleaned --output data/augmented --report augmentation_report
```

Defaults are stored in [configs/data.yaml](configs/data.yaml).

## Model Training

Training defaults live in [configs/train.yaml](configs/train.yaml). Command-line arguments override the YAML defaults.

```bash
python train_cross_validation.py --config configs/train.yaml
```

Example EfficientNet-B0 run matching the original style:

```bash
python train_cross_validation.py \
  --data data/cleaned \
  --model efficientnet_b0 \
  --pretrained \
  --class-weights \
  --no-augment
```

Outputs are written to `cv_results/`, including fold checkpoints, metrics CSV files, confusion matrices and a markdown report.

## Model Inference

### Flask Demo

```bash
python final_app/app.py
```

Open:

```text
http://127.0.0.1:7860
```

### FastAPI

```bash
uvicorn urban_greening_classifier.api:app --host 0.0.0.0 --port 7860
```

Prediction request:

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -F "image=@final_app/examples/存在明显暴露垃圾_586.png"
```

Response:

```json
{
  "class": "存在明显暴露垃圾",
  "confidence": 0.98
}
```

## Example Images

Sample images are provided in `final_app/examples/` for local smoke testing:

```text
final_app/examples/存在明显暴露垃圾_586.png
final_app/examples/存在明显暴露垃圾_753.png
final_app/examples/存在明显暴露垃圾_790.png
```

## Performance Metrics

The included inference system uses a weighted ensemble of three backbones:

| Model | Ensemble Weight | Checkpoints |
|---|---:|---:|
| EfficientNet-B0 | 0.25 | 5 folds |
| MobileNetV3-Small | 0.35 | 5 folds |
| ResNet18 | 0.40 | 5 folds |

Reproducible training runs write `fold_metrics.csv`, `summary_metrics.csv`, confusion matrices and `cross_validation_report.md` under `cv_results/`. Because the source competition data is not committed, aggregate metrics should be generated from the local dataset used for training.

## Deployment

Build and run with Docker:

```bash
docker build -t urban-greening-classifier .
docker run --rm -p 7860:7860 urban-greening-classifier
```

Or use docker-compose:

```bash
docker compose up --build
```

The container starts the FastAPI service and loads checkpoints from `final_app/checkpoints/`. Docker installs CPU-only PyTorch by default to avoid shipping unnecessary CUDA dependencies in the API image. CUDA is an optional Linux GPU deployment path, not the default.

## Development

Run quality checks:

```bash
black .
isort .
flake8 .
pytest
```

CI runs dependency installation, flake8 and pytest on every push and pull request.

## FAQ

**Does this upgrade change the algorithm or trained weights?**  
No. Model architectures, inference weighting and checkpoint files are preserved.

**Where should paths and parameters be changed?**  
Use YAML files under `configs/`. CLI arguments can override training defaults.

**Why is the dataset not included?**  
Raw, cleaned and augmented datasets can be large and may be governed by competition or privacy constraints, so they are ignored by Git.

**Can I change class names?**  
Only if you retrain checkpoints and update `final_app/class_names.json` in the exact output order used during training.

**Which API should I deploy?**  
Use FastAPI for service deployment. The Flask app is kept as a browser demo.

**Will this use Apple MPS?**  
Yes. With `device: auto`, PyTorch uses CUDA first when available, then Apple MPS, then CPU. On your Apple Silicon machine, MPS is the expected accelerator.

**Why is Docker CPU-only by default?**  
The API image should be portable. CUDA dependencies are large and only useful on Linux NVIDIA deployments, so CUDA is documented as an optional deployment variant.

**Should checkpoints stay in Git?**  
Small demo checkpoints can stay in Git for reproducibility. For larger production models, use GitHub Releases, Git LFS or an internal artifact registry.

## License

This project is released under the [MIT License](LICENSE).
