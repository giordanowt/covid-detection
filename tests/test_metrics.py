import numpy as np

from covid_detection.metrics import evaluate_predictions

LABELS = ("bacterial_pneumonia", "covid19", "normal")
TCC_EXPERIMENT_2 = [[56, 2, 0], [0, 52, 6], [0, 0, 57]]


def probabilities_from_matrix(matrix):
    y_true, y_prob = [], []
    for true_idx, row in enumerate(matrix):
        for pred_idx, count in enumerate(row):
            for _ in range(count):
                probs = np.full(3, 0.05)
                probs[pred_idx] = 0.9
                y_true.append(true_idx)
                y_prob.append(probs)
    return np.array(y_true), np.array(y_prob)


def test_three_class_accuracy_matches_confusion_matrix_diagonal():
    report = evaluate_predictions(*probabilities_from_matrix(TCC_EXPERIMENT_2), LABELS)
    assert report["n"] == 173
    assert report["confusion_matrix"] == TCC_EXPERIMENT_2
    assert abs(report["accuracy"] - 165 / 173) < 1e-9


def test_covid_sensitivity_is_reported_separately():
    report = evaluate_predictions(*probabilities_from_matrix(TCC_EXPERIMENT_2), LABELS)
    covid = report["per_class"]["covid19"]
    assert covid["support"] == 58
    assert abs(covid["sensitivity"] - 52 / 58) < 1e-9
    assert abs(covid["specificity"] - 113 / 115) < 1e-9
    assert covid["auroc"] is not None
