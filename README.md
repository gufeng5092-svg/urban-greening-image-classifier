# Trained Urban Greening Maintenance Image Classifier

This repository contains a trained image classification system for urban greening maintenance inspection. It can classify uploaded street-scene images into three maintenance issue categories:

- `存在明显暴露垃圾`: obvious exposed trash
- `存在单处面积大于1m²小于5m²的泥土裸露`: a single bare-soil area larger than 1 square meter and smaller than 5 square meters
- `存在单处面积大于5m²的泥土裸露`: a single bare-soil area larger than 5 square meters

The trained model weights are included under `final_app/checkpoints/`. The web demo loads these checkpoints and runs a weighted three-model ensemble for inference.

## Repository Layout

```text
.
├── data_cleaning.py              # Raw image checks, duplicate detection, cleaned dataset export
├── data_augmentation.py          # Offline augmentation for minority classes
├── train_cross_validation.py     # K-fold training for CNN backbones
├── export_cv_predictions.py      # Export validation probabilities from saved folds
├── ensemble_predictions.py       # Weighted probability ensembling
├── final_app/                    # Flask inference demo and trained checkpoints
├── requirements.txt              # Training and analysis dependencies
└── README.md
```

## Trained System

The included inference system uses a weighted ensemble of three CNN backbones:

- EfficientNet-B0: 0.25
- MobileNetV3-Small: 0.35
- ResNet18: 0.40

Each backbone uses five cross-validation checkpoints. During inference, the five folds are averaged for each backbone first, and then the three backbone probability vectors are combined with the weights above.

Run the trained web demo:

```bash
python3 -m pip install -r final_app/requirements_app.txt
cd final_app
python3 app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Data Policy

The original competition dataset is not included in this repository. Put local training images under `data/raw/` when reproducing or extending the pipeline. Generated datasets and training outputs are ignored by Git because they are large and may contain competition data.

Expected raw-data format:

```text
data/raw/
├── 存在单处面积大于1m²小于5m²的泥土裸露_001.png
├── 存在单处面积大于5m²的泥土裸露_002.png
└── 存在明显暴露垃圾_003.png
```

The scripts infer labels from the filename prefix before the final underscore.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Training Pipeline

Clean the raw dataset:

```bash
python3 data_cleaning.py --source data/raw --clean data/cleaned --report cleaning_report
```

Generate offline augmented data if needed:

```bash
python3 data_augmentation.py --source data/cleaned --output data/augmented --report augmentation_report
```

Train a 5-fold EfficientNet-B0 baseline:

```bash
python3 train_cross_validation.py \
  --data data/cleaned \
  --model efficientnet_b0 \
  --pretrained \
  --class-weights \
  --no-augment
```

Export fold predictions:

```bash
python3 export_cv_predictions.py \
  --result-dir cv_results/efficientnet_b0_k5_seed20260617_noaug_cw_nosampler_pretrained \
  --model efficientnet_b0
```

Combine multiple model prediction files:

```bash
python3 ensemble_predictions.py \
  --predictions path/to/model_a/overall_predictions.csv path/to/model_b/overall_predictions.csv \
  --weights 0.6 0.4 \
  --output cv_results/ensemble
```

## Adding Images and Training Again

To add more training images, place the new files in `data/raw/` with the same filename convention:

```text
<class_name>_<unique_id>.png
```

Examples:

```text
data/raw/存在明显暴露垃圾_1001.png
data/raw/存在单处面积大于1m²小于5m²的泥土裸露_1002.png
data/raw/存在单处面积大于5m²的泥土裸露_1003.png
```

After adding images, rerun the pipeline:

```bash
python3 data_cleaning.py --source data/raw --clean data/cleaned --report cleaning_report

python3 train_cross_validation.py \
  --data data/cleaned \
  --model efficientnet_b0 \
  --pretrained \
  --class-weights \
  --no-augment \
  --run-name retrained
```

Repeat training for any other backbone you want to include in the ensemble, such as `mobilenet_v3_small` or `resnet18`. After training, copy the selected fold checkpoints into the corresponding `final_app/checkpoints/<model_name>/` directory using the names expected by the demo:

```text
fold_1.pth
fold_2.pth
fold_3.pth
fold_4.pth
fold_5.pth
```

Keep `final_app/class_names.json` in the same class order used during training. If class order changes, predictions will be mapped to the wrong labels.

## Web Demo

See [final_app/README.md](final_app/README.md) for the inference app details.

## Notes for GitHub Publishing

- Do not commit raw data, cleaned data, augmented data, cache folders, or large CV output directories.
- The trained checkpoints under `final_app/checkpoints/` are kept so the repository can run as a ready-to-use classifier.
- The Chinese class names are kept because they are the model labels and define checkpoint output order.
