# Architecture

## Overview

The repository is organized around stable command-line entry points and a reusable Python package:

- Root scripts remain convenient operational entry points for data cleaning, augmentation, training and prediction export.
- `src/urban_greening_classifier/` contains shared code for configuration, logging, paths, datasets, transforms, model creation and inference.
- `configs/` owns runtime defaults so paths and training parameters are not scattered across scripts.
- `final_app/` keeps the original browser demo and trained checkpoints.
- `urban_greening_classifier.api` exposes a production-friendly FastAPI service.

## Data Flow

1. Raw images are placed in `data/raw/`.
2. `data_cleaning.py` validates images, removes exact duplicates from the cleaned copy and writes reports.
3. `data_augmentation.py` optionally balances classes with conservative image augmentations.
4. `train_cross_validation.py` trains K-fold models using the YAML defaults plus CLI overrides.
5. `export_cv_predictions.py` exports fold probabilities.
6. `ensemble_predictions.py` fuses exported probabilities for offline evaluation.
7. `final_app/app.py` or `urban_greening_classifier.api` loads the trained ensemble for inference.

## Configuration

- `configs/train.yaml`: model, optimizer, data, fold and reproducibility settings.
- `configs/data.yaml`: cleaning and augmentation paths plus thresholds.
- `configs/inference.yaml`: checkpoint paths, class names, model weights and service settings.

## Deployment

The Docker image runs the FastAPI service with Uvicorn and loads checkpoints from `final_app/checkpoints/`.
