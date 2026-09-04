import argparse
import json
import random
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from covid_detection.data import make_dataset, patients_overlap, split_by_patient, true_indices
from covid_detection.manifest import LABELS, read_manifest
from covid_detection.metrics import evaluate_predictions
from covid_detection.model import DEFAULT_FINE_TUNE_FROM, build_model, compile_model, unfreeze_from


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VGG19 chest X-ray classifier with a patient-level train/validation/test split.")
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-from", default=DEFAULT_FINE_TUNE_FROM)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="imagenet", help="imagenet or none")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def callbacks_for(checkpoint_path, patience):
    return [
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.3, patience=2, min_lr=1e-7),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True),
    ]


def fit_phase(model, train_ds, val_ds, epochs, checkpoint_path, patience):
    if epochs <= 0:
        return None
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_for(checkpoint_path, patience))


def history_rows(phase, history):
    if history is None:
        return []
    return [{"phase": phase, "epoch": i + 1, **{k: float(v[i]) for k, v in history.history.items()}} for i in range(len(history.epoch))]


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = split_by_patient(read_manifest(args.manifest), args.test_fraction, args.validation_fraction, args.seed)
    assert not patients_overlap(split.train, split.test)
    assert not patients_overlap(split.train, split.validation)
    assert not patients_overlap(split.validation, split.test)
    (output_dir / "split.json").write_text(json.dumps(split.summary(), indent=2))
    (output_dir / "labels.json").write_text(json.dumps(list(LABELS)))

    train_ds = make_dataset(split.train, args.image_size, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(split.validation, args.image_size, args.batch_size)
    test_ds = make_dataset(split.test, args.image_size, args.batch_size)

    checkpoint_path = str(output_dir / "model.keras")
    weights = None if args.weights.lower() == "none" else args.weights
    model = compile_model(build_model(image_size=args.image_size, weights=weights), args.head_lr)
    head_history = fit_phase(model, train_ds, val_ds, args.head_epochs, checkpoint_path, args.patience)

    unfreeze_from(model, args.fine_tune_from)
    compile_model(model, args.fine_tune_lr)
    fine_tune_history = fit_phase(model, train_ds, val_ds, args.fine_tune_epochs, checkpoint_path, args.patience)

    best = keras.models.load_model(checkpoint_path)
    report = evaluate_predictions(true_indices(split.test), best.predict(test_ds, verbose=0), LABELS)
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    history = history_rows("head", head_history) + history_rows("fine_tune", fine_tune_history)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
