import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"D:\lct\datasets\construction_v1")
META = ROOT / "metadata"
CLASSES = [
    "excavator", "dump_truck", "truck", "loader", "bulldozer",
    "motor_grader", "roller", "concrete_mixer", "telehandler",
    "piling_machine", "crane_manipulator", "mobile_crane", "tower_crane",
]


def read_csv(name):
    with (META / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def roboflow_base(path_text):
    stem = Path(path_text).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)


manifest = read_csv("manifest.csv")
stats = read_csv("stats.csv")
errors = []
hashes = defaultdict(set)
groups = defaultdict(set)
base_ids = defaultdict(lambda: defaultdict(set))
actual_stats = Counter()
seen_outputs = set()

for row in manifest:
    split = row["split"]
    image = ROOT / row["output_image"]
    label = ROOT / row["output_label"]
    key = (row["output_image"], row["output_label"])
    if key in seen_outputs:
        errors.append(f"duplicate output paths: {key}")
    seen_outputs.add(key)
    if not image.is_file():
        errors.append(f"missing image: {image}")
    if not label.is_file():
        errors.append(f"missing label: {label}")
        continue
    hashes[split].add(row["sha1"])
    if row["group_id"]:
        groups[split].add(row["group_id"])
    base_ids[row["source"]][split].add(roboflow_base(row["original_image"]))
    box_count = 0
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label}:{line_no}: field_count={len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label}:{line_no}: parse_error")
            continue
        if not 0 <= class_id < len(CLASSES):
            errors.append(f"{label}:{line_no}: class_id={class_id}")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label}:{line_no}: invalid_bbox")
        if x - width / 2 < -1e-8 or x + width / 2 > 1 + 1e-8 or y - height / 2 < -1e-8 or y + height / 2 > 1 + 1e-8:
            errors.append(f"{label}:{line_no}: bbox_outside")
        actual_stats[(row["source"], split, CLASSES[class_id])] += 1
        box_count += 1
    if box_count != int(row["n_boxes"]):
        errors.append(f"{label}: manifest_n_boxes={row['n_boxes']} actual={box_count}")

reported_stats = Counter({
    (row["source"], row["split"], row["class"]): int(row["bbox_count"])
    for row in stats
})
if reported_stats != actual_stats:
    errors.append("stats.csv does not equal independently aggregated labels")

split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
exact_leakage = {}
group_leakage = {}
base_id_leakage = {}
for left, right in split_pairs:
    exact = hashes[left] & hashes[right]
    if exact:
        exact_leakage[f"{left}_{right}"] = len(exact)
        errors.append(f"exact leakage {left}<->{right}: {len(exact)}")
    overlap = groups[left] & groups[right]
    if overlap:
        group_leakage[f"{left}_{right}"] = sorted(overlap)
        errors.append(f"group leakage {left}<->{right}: {len(overlap)}")
    for source in sorted(base_ids):
        base_overlap = base_ids[source][left] & base_ids[source][right]
        if base_overlap:
            base_id_leakage[f"{source}:{left}_{right}"] = sorted(base_overlap)
            errors.append(f"base-id leakage {source} {left}<->{right}: {len(base_overlap)}")

filename_patterns = []
for source in ("d1", "d6", "d7", "d8"):
    rows = [row for row in manifest if row["source"] == source]
    grouped = sum(bool(row["group_id"]) for row in rows)
    methods = Counter(row["group_method"] or "unrecovered" for row in rows)
    for method, count in sorted(methods.items()):
        filename_patterns.append({
            "source": source,
            "pattern_or_method": method,
            "image_count": count,
            "reliable_grouping": "yes" if method != "unrecovered" else "no",
            "note": (
                "camera/scene/sequence encoded explicitly in filename"
                if method != "unrecovered"
                else "source split preserved; filename alone is insufficient for reliable scene grouping"
            ),
        })

with (META / "filename_structure_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["source", "pattern_or_method", "image_count", "reliable_grouping", "note"])
    writer.writeheader()
    writer.writerows(filename_patterns)

with (META / "stats_source_split_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["source", "split", "class", "class_id", "bbox_count"])
    writer.writeheader()
    for source in ("d1", "d6", "d7", "d8"):
        for split in ("train", "val", "test"):
            for class_id, class_name in enumerate(CLASSES):
                writer.writerow({
                    "source": source,
                    "split": split,
                    "class": class_name,
                    "class_id": class_id,
                    "bbox_count": actual_stats[(source, split, class_name)],
                })

report = {
    "errors": errors,
    "manifest_rows": len(manifest),
    "image_counts": dict(Counter(row["split"] for row in manifest)),
    "label_files_checked": len(manifest),
    "image_label_correspondence_ok": not any("missing image" in error or "missing label" in error or "duplicate output paths" in error for error in errors),
    "bbox_count": sum(actual_stats.values()),
    "class_id_range": [0, len(CLASSES) - 1],
    "bbox_validity_ok": not any("bbox" in error or "field_count" in error or "parse_error" in error or "class_id=" in error for error in errors),
    "exact_hash_leakage": exact_leakage,
    "recovered_group_leakage": group_leakage,
    "roboflow_base_id_leakage": base_id_leakage,
    "stats_csv_matches_labels": reported_stats == actual_stats,
    "grouping_limitation": "D7 numeric-only IDs, most D6 heterogeneous names, and D8 bare numeric names lack reliable camera/scene boundaries; source-provided splits are retained for those records.",
}
(META / "independent_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(META / "sanity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
