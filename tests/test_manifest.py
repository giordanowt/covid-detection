from covid_detection.manifest import (
    LABEL_BACTERIAL, LABEL_COVID, LABEL_NORMAL, cap_per_label, kermany_patient_id,
    read_manifest, rows_from_cohen, rows_from_kermany, write_manifest,
)


def test_kermany_patient_id_groups_images_of_the_same_person():
    assert kermany_patient_id("person1_bacteria_1.jpeg") == kermany_patient_id("person1_bacteria_2.jpeg")
    assert kermany_patient_id("person1_bacteria_1.jpeg") != kermany_patient_id("person2_bacteria_1.jpeg")
    assert kermany_patient_id("NORMAL2-IM-0027-0001.jpeg") == "kermany-study-0027"
    assert kermany_patient_id("IM-0115-0001.jpeg") == "kermany-study-0115"


def test_kermany_rows_keep_normal_and_bacterial_only(kermany_root):
    rows = rows_from_kermany(kermany_root)
    labels = sorted(row.label for row in rows)
    assert labels == [LABEL_BACTERIAL] * 3 + [LABEL_NORMAL] * 2
    assert not any("virus" in row.path for row in rows)
    assert all(row.source == "kermany" for row in rows)


def test_cohen_rows_keep_frontal_covid_xrays_with_existing_files(cohen_root):
    rows = rows_from_cohen(cohen_root)
    filenames = sorted(row.path.rsplit("/", 1)[-1] for row in rows)
    assert filenames == ["p1-day1.jpg", "p1-day3.jpg", "p6.png"]
    assert {row.label for row in rows} == {LABEL_COVID}
    assert {row.patient_id for row in rows} == {"cohen-p1", "cohen-p6"}


def test_cap_per_label_is_deterministic_and_bounded(kermany_root, cohen_root):
    rows = rows_from_kermany(kermany_root) + rows_from_cohen(cohen_root)
    first = cap_per_label(rows, 2, seed=1)
    second = cap_per_label(rows, 2, seed=1)
    assert first == second
    assert sorted(row.label for row in first) == [LABEL_BACTERIAL] * 2 + [LABEL_COVID] * 2 + [LABEL_NORMAL] * 2
    assert cap_per_label(rows, None, seed=1) == rows


def test_manifest_roundtrip(tmp_path, kermany_root):
    rows = rows_from_kermany(kermany_root)
    output = tmp_path / "manifest.csv"
    write_manifest(rows, output)
    assert read_manifest(output) == rows
