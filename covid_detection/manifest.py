import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

LABEL_BACTERIAL = "bacterial_pneumonia"
LABEL_COVID = "covid19"
LABEL_NORMAL = "normal"
LABELS = (LABEL_BACTERIAL, LABEL_COVID, LABEL_NORMAL)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FRONTAL_VIEWS = {"PA", "AP", "AP Supine", "AP Erect"}
MANIFEST_COLUMNS = ("path", "label", "patient_id", "source")

KERMANY_PERSON = re.compile(r"person(\d+)_", re.IGNORECASE)
KERMANY_NORMAL_STUDY = re.compile(r"IM-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: str
    patient_id: str
    source: str

    def as_dict(self):
        return {"path": self.path, "label": self.label, "patient_id": self.patient_id, "source": self.source}


def kermany_patient_id(filename):
    person = KERMANY_PERSON.search(filename)
    if person:
        return f"kermany-person-{person.group(1)}"
    study = KERMANY_NORMAL_STUDY.search(filename)
    if study:
        return f"kermany-study-{study.group(1)}"
    return f"kermany-file-{Path(filename).stem}"


def _image_files(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def rows_from_kermany(root):
    rows = []
    for path in _image_files(root):
        parent = path.parent.name.upper()
        name = path.name.lower()
        if parent == "NORMAL":
            label = LABEL_NORMAL
        elif parent == "PNEUMONIA" and "_bacteria_" in name:
            label = LABEL_BACTERIAL
        else:
            continue
        rows.append(ManifestRow(str(path), label, kermany_patient_id(path.name), "kermany"))
    return rows


def rows_from_cohen(root, views=FRONTAL_VIEWS):
    root = Path(root)
    rows = []
    with open(root / "metadata.csv", newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("modality", "X-ray") != "X-ray":
                continue
            if "COVID-19" not in record.get("finding", ""):
                continue
            if record.get("view", "") not in views:
                continue
            path = root / "images" / record["filename"]
            if not path.exists():
                continue
            rows.append(ManifestRow(str(path), LABEL_COVID, f"cohen-{record['patientid']}", "cohen"))
    return rows


def cap_per_label(rows, per_label, seed):
    if per_label is None:
        return list(rows)
    rng = random.Random(seed)
    capped = []
    for label in LABELS:
        candidates = [row for row in rows if row.label == label]
        rng.shuffle(candidates)
        capped.extend(sorted(candidates[:per_label], key=lambda row: row.path))
    return capped


def write_manifest(rows, output):
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [ManifestRow(**{key: record[key] for key in MANIFEST_COLUMNS}) for record in csv.DictReader(handle)]
