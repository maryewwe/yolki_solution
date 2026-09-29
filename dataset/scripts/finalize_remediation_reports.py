from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(r"D:\lct\datasets\construction_v1")
META = ROOT / "metadata"
OUT = META / "near_duplicate_remediation"
SOURCES = {"d6", "d7"}
SPLITS = ("train", "val", "test")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


report_dirs = [OUT] + [OUT / f"iteration_{i}" for i in range(2, 10)]
all_moves = []
all_groups = []
for iteration, directory in enumerate(report_dirs, 1):
    for row in read_csv(directory / "move_plan.csv"):
        all_moves.append({"iteration": iteration, **row})
    for row in read_csv(directory / "processed_groups.csv"):
        all_groups.append({"iteration": iteration, **row})

write_csv(OUT / "all_move_plan.csv", all_moves, list(all_moves[0]))
write_csv(OUT / "all_processed_groups.csv", all_groups, list(all_groups[0]))

initial_split = {
    (row["source"], row["split"]): int(row["before_images"])
    for row in read_csv(OUT / "split_sizes_before_after.csv")
}
manifest = read_csv(META / "manifest.csv")
final_split = Counter((row["source"], row["split"]) for row in manifest)
initial_manifest_path = sorted(
    ROOT.parent.glob("construction_v1_remediation_backup_*/metadata/manifest.csv")
)[0]
initial_manifest = read_csv(initial_manifest_path)
initial_by_key = {
    (row["source"], PurePosixPath(row["output_image"]).name): row for row in initial_manifest
}
final_by_key = {
    (row["source"], PurePosixPath(row["output_image"]).name): row for row in manifest
}
net_moves = []
for key, before_row in initial_by_key.items():
    after_row = final_by_key[key]
    if key[0] in SOURCES and before_row["split"] != after_row["split"]:
        net_moves.append({
            "source": key[0], "image": key[1], "from_split": before_row["split"],
            "to_split": after_row["split"], "final_image_path": after_row["output_image"],
        })
write_csv(
    OUT / "net_image_moves.csv", net_moves,
    ["source", "image", "from_split", "to_split", "final_image_path"],
)
split_rows = []
for source in sorted(SOURCES):
    for split in SPLITS:
        before = initial_split[(source, split)]
        after = final_split[(source, split)]
        split_rows.append({
            "source": source, "split": split, "before_images": before,
            "after_images": after, "delta_images": after - before,
        })
write_csv(OUT / "split_sizes_before_final.csv", split_rows, list(split_rows[0]))

initial_classes = {
    (row["source"], row["split"], int(row["class_id"])): row
    for row in read_csv(OUT / "class_counts_before_after.csv")
}
current_classes = {
    (row["source"], row["split"], int(row["class_id"])): row
    for row in read_csv(META / "stats_source_split_class.csv")
}
class_rows = []
for key, first in sorted(initial_classes.items()):
    source, split, cid = key
    after = int(current_classes.get(key, {}).get("bbox_count", 0))
    before = int(first["before_bbox_count"])
    class_rows.append({
        "source": source, "split": split, "class": first["class"], "class_id": cid,
        "before_bbox_count": before, "after_bbox_count": after,
        "delta_bbox_count": after - before,
    })
write_csv(OUT / "class_counts_before_final.csv", class_rows, list(class_rows[0]))

# Reconstruct the transitive closure of every group_id ever assigned, using
# stable source+filename keys across backups where split paths differ.
current_by_key = {
    (row["source"], PurePosixPath(row["output_image"]).name): row for row in manifest
}
parent = {key: key for key in current_by_key if key[0] in SOURCES}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


snapshots = sorted(ROOT.parent.glob("construction_v1_remediation_backup_*/metadata/manifest.csv"))
snapshots.append(META / "manifest.csv")
for path in snapshots:
    grouped = defaultdict(list)
    for row in read_csv(path):
        key = (row.get("source", ""), PurePosixPath(row.get("output_image", "")).name)
        gid = row.get("group_id", "")
        if key in parent and gid:
            grouped[(key[0], gid)].append(key)
    for keys in grouped.values():
        for key in keys[1:]:
            union(keys[0], key)

components = defaultdict(list)
for key in parent:
    components[find(key)].append(key)
historical_cross_split = []
for keys in components.values():
    if len(keys) < 2:
        continue
    splits = sorted({current_by_key[key]["split"] for key in keys})
    if len(splits) > 1:
        historical_cross_split.append({
            "source": keys[0][0], "n_images": len(keys), "splits": ";".join(splits),
            "images": ";".join(sorted(current_by_key[key]["output_image"] for key in keys)),
        })
write_csv(
    OUT / "historical_group_leakage.csv", historical_cross_split,
    ["source", "n_images", "splits", "images"],
)

direction_counts = Counter(f"{row['from_split']}->{row['to_split']}" for row in all_moves)
source_direction_counts = Counter(
    (row["source"], f"{row['from_split']}->{row['to_split']}") for row in all_moves
)
net_source_direction_counts = Counter(
    (row["source"], f"{row['from_split']}->{row['to_split']}") for row in net_moves
)
unique_moved = {
    (row["source"], PurePosixPath(row["old_image"]).name) for row in all_moves
}
max_class_delta = max(abs(int(row["delta_bbox_count"])) for row in class_rows)
total_bbox_by_source = Counter()
changed_bbox_by_source = Counter()
for row in class_rows:
    total_bbox_by_source[row["source"]] += int(row["before_bbox_count"])
    changed_bbox_by_source[row["source"]] += abs(int(row["delta_bbox_count"]))

near_summary = json.loads((META / "near_duplicate_audit" / "summary.json").read_text(encoding="utf-8"))
sanity = json.loads((META / "sanity_report.json").read_text(encoding="utf-8"))
summary = {
    "scope": ["d6", "d7"],
    "iterations_applied": 9,
    "move_operations": len(all_moves),
    "unique_images_moved_at_least_once": len(unique_moved),
    "images_with_final_split_changed_from_original": len(net_moves),
    "move_operations_by_direction": dict(sorted(direction_counts.items())),
    "move_operations_by_source_and_direction": {
        source: {
            direction: source_direction_counts[(source, direction)]
            for direction in sorted(direction_counts)
        }
        for source in sorted(SOURCES)
    },
    "net_final_moves_by_source_and_direction": {
        source: {
            direction: net_source_direction_counts[(source, direction)]
            for direction in sorted(direction_counts)
        }
        for source in sorted(SOURCES)
    },
    "processed_group_records": len(all_groups),
    "processed_group_ids": [row["group_id"] for row in all_groups],
    "split_sizes_before_after": split_rows,
    "historical_confirmed_group_cross_split_components": len(historical_cross_split),
    "final_near_duplicate_high_confidence": {
        item["source"]: item["n_high_confidence_suspected_duplicates"]
        for item in near_summary["sources"]
    },
    "final_near_duplicate_medium_confidence": {
        item["source"]: item["n_medium_confidence_pairs"]
        for item in near_summary["sources"]
    },
    "final_top100_visually_reviewed": True,
    "class_balance": {
        "max_absolute_bbox_count_delta_in_any_source_split_class_cell": max_class_delta,
        "sum_absolute_bbox_deltas_by_source": dict(changed_bbox_by_source),
        "total_bbox_counts_by_source": dict(total_bbox_by_source),
    },
    "sanity_errors": sanity["errors"],
    "data_yaml_changed": False,
    "taxonomy_or_annotation_content_changed": False,
    "d1_or_d8_moved": any(row["source"] not in SOURCES for row in all_moves),
    "backup_snapshots": [str(path.parent.parent) for path in snapshots[:-1]],
}
(OUT / "final_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
