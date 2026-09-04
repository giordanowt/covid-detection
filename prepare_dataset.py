import argparse
from collections import Counter

from covid_detection.manifest import cap_per_label, rows_from_cohen, rows_from_kermany, write_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build a manifest CSV (path,label,patient_id,source) from the raw Kaggle downloads.")
    parser.add_argument("--kermany-dir", required=True, help="Root of 'Chest X-Ray Images (Pneumonia)' (contains NORMAL/ and PNEUMONIA/ folders at any depth)")
    parser.add_argument("--cohen-dir", required=True, help="Root of covid-chestxray-dataset (contains metadata.csv and images/)")
    parser.add_argument("--per-label", type=int, default=None, help="Cap the number of images per label (sampled with --seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="manifest.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = rows_from_kermany(args.kermany_dir) + rows_from_cohen(args.cohen_dir)
    rows = cap_per_label(rows, args.per_label, args.seed)
    write_manifest(rows, args.output)
    counts = Counter(row.label for row in rows)
    patients = Counter(row.label for row in {row.patient_id: row for row in rows}.values())
    for label in sorted(counts):
        print(f"{label}: {counts[label]} images, {patients[label]} patients")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
