from collections import Counter
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input
from sklearn.model_selection import StratifiedGroupKFold

from covid_detection.manifest import LABELS


@dataclass(frozen=True)
class Split:
    train: list
    validation: list
    test: list

    def summary(self):
        return {
            name: dict(Counter(row.label for row in rows))
            for name, rows in (("train", self.train), ("validation", self.validation), ("test", self.test))
        }


def patients_overlap(a, b):
    return {row.patient_id for row in a} & {row.patient_id for row in b}


def _group_split(rows, fraction, seed):
    groups = [row.patient_id for row in rows]
    labels = [row.label for row in rows]
    folds = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    keep_idx, held_idx = next(splitter.split(rows, labels, groups=groups))
    return [rows[i] for i in keep_idx], [rows[i] for i in held_idx]


def split_by_patient(rows, test_fraction=0.2, validation_fraction=0.2, seed=42):
    remaining, test = _group_split(rows, test_fraction, seed)
    train, validation = _group_split(remaining, validation_fraction, seed)
    return Split(train, validation, test)


def label_index(label):
    return LABELS.index(label)


def _decode(path, image_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size))
    return preprocess_input(tf.cast(image, tf.float32))


def make_dataset(rows, image_size, batch_size, shuffle=False, seed=42):
    paths = [row.path for row in rows]
    labels = tf.one_hot([label_index(row.label) for row in rows], len(LABELS))
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda p, y: (_decode(p, image_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def true_indices(rows):
    return np.array([label_index(row.label) for row in rows])
