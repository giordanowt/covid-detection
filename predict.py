import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import keras
import numpy as np

from covid_detection.data import make_dataset
from covid_detection.manifest import IMAGE_SUFFIXES, ManifestRow


def parse_args():
    parser = argparse.ArgumentParser(description="Classify chest X-ray images with a trained model and write one row per image.")
    parser.add_argument("--model", default="artifacts/model.keras")
    parser.add_argument("--labels", default="artifacts/labels.json")
    parser.add_argument("--images", nargs="+", required=True, help="Image files and/or directories (searched recursively)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="predictions.csv")
    return parser.parse_args()


def collect_images(inputs):
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return sorted(set(files))


def main():
    args = parse_args()
    labels = json.loads(Path(args.labels).read_text())
    files = collect_images(args.images)
    rows = [ManifestRow(str(f), labels[0], f.stem, "unlabeled") for f in files]
    model = keras.models.load_model(args.model)
    probabilities = model.predict(make_dataset(rows, args.image_size, args.batch_size), verbose=0)
    predicted = [labels[i] for i in np.argmax(probabilities, axis=1)]

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "predicted", *labels])
        for path, label, probs in zip(files, predicted, probabilities):
            writer.writerow([str(path), label, *[f"{p:.4f}" for p in probs]])

    for label, count in sorted(Counter(predicted).items()):
        print(f"{label}: {count}")
    print(f"wrote {len(files)} predictions to {args.output}")


if __name__ == "__main__":
    main()
