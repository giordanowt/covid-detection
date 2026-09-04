import csv
from pathlib import Path

import pytest
from PIL import Image


def write_image(path, size=(40, 40), mode="L"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=128).save(path)
    return path


@pytest.fixture
def kermany_root(tmp_path):
    root = tmp_path / "kermany"
    for name in ("train/NORMAL/IM-0001-0001.jpeg", "test/NORMAL/NORMAL2-IM-0002-0001.jpeg",
                 "train/PNEUMONIA/person1_bacteria_1.jpeg", "train/PNEUMONIA/person1_bacteria_2.jpeg",
                 "train/PNEUMONIA/person2_virus_3.jpeg", "train/PNEUMONIA/person3_bacteria_4.jpeg"):
        write_image(root / name)
    return root


@pytest.fixture
def cohen_root(tmp_path):
    root = tmp_path / "cohen"
    records = [
        ("p1", "Pneumonia/Viral/COVID-19", "PA", "X-ray", "p1-day1.jpg"),
        ("p1", "Pneumonia/Viral/COVID-19", "AP", "X-ray", "p1-day3.jpg"),
        ("p2", "Pneumonia/Viral/COVID-19", "L", "X-ray", "p2-lateral.jpg"),
        ("p3", "Pneumonia/Viral/SARS", "PA", "X-ray", "p3-sars.jpg"),
        ("p4", "Pneumonia/Viral/COVID-19", "Axial", "CT", "p4-ct.png"),
        ("p5", "Pneumonia/Viral/COVID-19", "PA", "X-ray", "p5-missing.jpg"),
        ("p6", "Pneumonia/Viral/COVID-19", "AP Supine", "X-ray", "p6.png"),
    ]
    (root / "images").mkdir(parents=True)
    with open(root / "metadata.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patientid", "finding", "view", "modality", "filename"])
        writer.writerows(records)
    for record in records:
        if "missing" not in record[4]:
            write_image(root / "images" / record[4])
    return root
