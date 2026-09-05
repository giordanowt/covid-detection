# covid-detection

Chest X-ray classifier for three classes (COVID-19, bacterial pneumonia, normal) built on VGG19 transfer learning. Undergraduate thesis in Computer Science, Unisinos, 2020.

Published as: Trombetta GBW, Fröhlich WR, Rigo SJ, Rodrigues CA. *Aplicação de Deep Learning para Diagnóstico de Pneumonia Causada por COVID-19 a partir de Imagens de Raio X*. J. Health Inform. 2020; Número Especial SBIS: 286-91. [Article](https://jhi.sbis.org.br/index.php/jhi-sbis/article/view/828)

## Status

Historical project, archived. In September 2026 the repository received a cleanup only: dataset images and trained weights were removed from the git history, this README was added, and the code was rewritten to fix the issues listed below. **No new experiments were run.** The original 2020 code is preserved at tag [`tcc-2020`](https://github.com/giordanowt/covid-detection/tree/tcc-2020) and the original weights are attached to the [release of the same name](https://github.com/giordanowt/covid-detection/releases/tag/tcc-2020).

## Errata (2026 retrospective)

Written by the author while revisiting the work. The qualitative conclusion of the paper (a VGG19 transfer-learning model separates the three classes on this dataset) still holds, but the reported numbers and part of the method do not survive review.

- **The ~96% accuracy is the wrong metric.** The model was compiled with `binary_crossentropy` and `acc` on a 3-class softmax, so Keras silently reported `binary_accuracy`: three comparisons per image, where a wrong prediction still scores one. The confusion matrices in the thesis and the paper were computed separately with scikit-learn on the argmax and are not affected. Summing their diagonals gives the real 3-class accuracy; applying the binary formula (3 × correct + wrong) / (3 × N) to the same matrices reproduces the published numbers, which confirms the source of the error.

  | Experiment | Reported | 3-class accuracy from confusion matrix |
  |---|---|---|
  | Thesis, dataset I (160 test images) | 0.9608 | 148/160 = 0.925 |
  | Thesis, dataset II (173 test images) | 0.9686 | 165/173 = 0.954 |
  | Paper (160 test images) | 0.9657 | 152/160 = 0.950 |

- **COVID-19 sensitivity was not reported.** In the thesis experiments 5/53 and 6/58 COVID-19 images were classified as normal (sensitivity ≈ 0.89). Accuracy alone hid the clinically worst error.
- **The split leaked patients.** Images were split at random. Both sources contain several images per patient (`personN_bacteria_*` in Kermany, follow-up series in Cohen), so the same patient appears in train and test.
- **No held-out test set.** The same 20% was used to pick the best checkpoint, to draw the curves and to report the final matrix. The matrix also came from the last-epoch model, not from the saved checkpoint.
- **Training configuration bugs.** `ReduceLROnPlateau` monitored `val_loss` with `mode='max'`, cutting the learning rate every two epochs from the start (the flat test curve after epoch ~25 is that). The unfreeze loop never unfroze `block5`, so no fine-tuning happened. The `alpha = 1e-5` listed as learning rate in the paper was never used; Adam ran at its default.
- **Related work was misreported.** COVID-Net (Wang & Wong, 2020) is a 3-class model (normal, non-COVID pneumonia, COVID-19), not binary, so the multi-class framing was not a gap in the literature. The "95% accuracy" attributed to CheXNet does not appear in that paper, which reports F1 and AUROC.
- **Source shortcut.** COVID-19 images came from an adult, multi-hospital collection and the other two classes from a pediatric collection. A model can separate these by population or acquisition source rather than pathology. This was established for the field in 2021 (Roberts et al., *Nature Machine Intelligence*; DeGrave et al., *Nature Machine Intelligence*) and applies to this work.

## Dataset

Images are not redistributed here. The 2020 experiments used 266 or 288 images per class, selected by hand from two public sources:

- Kermany DS, Goldbaum M, et al. *Chest X-Ray Images (Pneumonia)*, Kaggle. Normal and bacterial pneumonia classes (pediatric).
- Cohen JP, Morrison P, Dao L. *covid-chestxray-dataset*, GitHub / Kaggle mirror. COVID-19 class.

`prepare_dataset.py` rebuilds a manifest from the raw downloads: it keeps frontal X-rays whose finding is COVID-19 from Cohen (filtering out CT, lateral views and other viral findings via `metadata.csv`), normal and bacterial images from Kermany, and records a patient id for every image so the split can be done by patient.

## Usage

```bash
pip install -r requirements.txt

python prepare_dataset.py --kermany-dir data/chest_xray --cohen-dir data/covid-chestxray-dataset --per-label 288
python train.py --manifest manifest.csv --output-dir artifacts
python predict.py --model artifacts/model.keras --labels artifacts/labels.json --images some/folder
python -m pytest
```

`train.py` writes `split.json` (class counts per split), `history.json`, `test_metrics.json` (accuracy, confusion matrix, per-class sensitivity, specificity, precision and AUROC) and the best checkpoint `model.keras`.

## What the current code does differently

- Patient-level, stratified train/validation/test split. The test set is touched once, with the best checkpoint.
- `categorical_crossentropy` and categorical accuracy.
- VGG19 `preprocess_input` instead of dividing by 255.
- Two phases: head training with a frozen backbone, then fine-tuning from `block5_conv1` at a low learning rate.
- `ReduceLROnPlateau` and `EarlyStopping` on `val_loss` with the correct mode.
- Augmentation as Keras layers (rotation and zoom, no horizontal flip), 224 px input.
- Per-class metrics, seeds fixed, paths passed as arguments, no dataset or weights in git.

None of this has been run on the original data. Treat the code as the corrected recipe, not as a new result.
