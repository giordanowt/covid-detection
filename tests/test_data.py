import numpy as np

from covid_detection.data import make_dataset, patients_overlap, split_by_patient, true_indices
from covid_detection.manifest import LABELS, ManifestRow
from tests.conftest import write_image


def synthetic_rows(tmp_path, patients=30, images_per_patient=3):
    rows = []
    for p in range(patients):
        label = LABELS[p % len(LABELS)]
        for i in range(images_per_patient):
            path = write_image(tmp_path / label / f"patient{p}_{i}.png")
            rows.append(ManifestRow(str(path), label, f"patient-{p}", "synthetic"))
    return rows


def test_split_by_patient_has_no_patient_in_two_splits(tmp_path):
    split = split_by_patient(synthetic_rows(tmp_path), seed=7)
    assert not patients_overlap(split.train, split.test)
    assert not patients_overlap(split.train, split.validation)
    assert not patients_overlap(split.validation, split.test)
    assert len(split.train) + len(split.validation) + len(split.test) == 90
    assert len(split.test) == 18


def test_split_summary_counts_labels(tmp_path):
    summary = split_by_patient(synthetic_rows(tmp_path), seed=7).summary()
    assert set(summary) == {"train", "validation", "test"}
    assert sum(summary["test"].values()) == 18


def test_make_dataset_yields_preprocessed_batches(tmp_path):
    rows = synthetic_rows(tmp_path, patients=3, images_per_patient=2)
    images, labels = next(iter(make_dataset(rows, image_size=32, batch_size=4)))
    assert images.shape == (4, 32, 32, 3)
    assert labels.shape == (4, 3)
    assert np.allclose(images.numpy()[0, 0, 0], [128 - 103.939, 128 - 116.779, 128 - 123.68], atol=1e-3)
    assert np.array_equal(true_indices(rows), [0, 0, 1, 1, 2, 2])
