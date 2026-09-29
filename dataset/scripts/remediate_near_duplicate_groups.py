from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath


ROOT = Path(r"D:\lct\datasets\construction_v1")
META = ROOT / "metadata"
GROUPS_CSV = META / "near_duplicate_audit" / "suspected_sequence_groups.csv"
REPORT_DIR = META / "near_duplicate_remediation"
SPLITS = ("train", "val", "test")
SOURCES = ("d6", "d7")
CLASSES = (
    "excavator", "dump_truck", "truck", "loader", "bulldozer",
    "motor_grader", "roller", "concrete_mixer", "telehandler",
    "piling_machine", "crane_manipulator", "mobile_crane", "tower_crane",
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_from_rel(path_text: str) -> str:
    parts = PurePosixPath(path_text).parts
    if len(parts) < 3 or parts[0] != "images" or parts[1] not in SPLITS:
        raise ValueError(f"unexpected image path: {path_text}")
    return parts[1]


def moved_rel(path_text: str, destination: str, kind: str) -> str:
    parts = list(PurePosixPath(path_text).parts)
    if parts[0] != kind or parts[1] not in SPLITS:
        raise ValueError(f"unexpected {kind} path: {path_text}")
    parts[1] = destination
    return PurePosixPath(*parts).as_posix()


def label_counts(path: Path) -> Counter:
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            counts[int(parts[0])] += 1
    return counts


def snapshot_counts(manifest: list[dict], use_existing_stats: bool = False):
    images = Counter()
    boxes = Counter()
    for row in manifest:
        source, split = row["source"], row["split"]
        images[(source, split)] += 1
    if use_existing_stats:
        for row in read_csv(META / "stats_source_split_class.csv"):
            boxes[(row["source"], row["split"], int(row["class_id"]))] = int(row["bbox_count"])
    else:
        for row in manifest:
            source, split = row["source"], row["split"]
            for class_id, count in label_counts(ROOT / row["output_label"]).items():
                boxes[(source, split, class_id)] += count
    return images, boxes


def build_group_data(manifest: list[dict], group_rows: list[dict]) -> list[dict]:
    by_image = {row["output_image"]: row for row in manifest}
    by_stable_name = {
        (row["source"], PurePosixPath(row["output_image"]).name): row["output_image"]
        for row in manifest
    }
    parent = {image: image for image in by_image}

    def find(image: str) -> str:
        while parent[image] != image:
            parent[image] = parent[parent[image]]
            image = parent[image]
        return image

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Preserve the transitive closure of every confirmed group ever applied.
    # Older group_id values can be overwritten by a later merge, so recover
    # their memberships from the pre-apply metadata snapshots in each backup.
    historical_manifests = [manifest]
    historical_seed_images = set()
    for path in sorted(ROOT.parent.glob("construction_v1_remediation_backup_*/metadata/manifest.csv")):
        historical_manifests.append(read_csv(path))
    for snapshot in historical_manifests:
        grouped = defaultdict(list)
        for row in snapshot:
            image = by_stable_name.get(
                (row.get("source", ""), PurePosixPath(row.get("output_image", "")).name), ""
            )
            group_id = row.get("group_id", "")
            if image in parent and row.get("source") in SOURCES and group_id:
                grouped[(row["source"], group_id)].append(image)
        for images in grouped.values():
            if len(images) > 1:
                historical_seed_images.update(images)
            for image in images[1:]:
                union(images[0], image)

    selected_rows = []
    new_group_ids = defaultdict(list)
    new_group_types = defaultdict(set)
    seed_images = set()
    for group in group_rows:
        if group["status"] != "high_confidence_cross_split" or group["source"] not in SOURCES:
            continue
        images = [item for item in group["images"].split(";") if item]
        if len(images) != len(set(images)):
            raise RuntimeError(f"duplicate member inside {group['group_id']}")
        for image in images:
            if image not in by_image:
                raise RuntimeError(f"group member absent from manifest: {image}")
            if by_image[image]["source"] != group["source"]:
                raise RuntimeError(f"source mismatch for {image}")
        for image in images[1:]:
            union(images[0], image)
        selected_rows.append((group, images))
        seed_images.update(images)

    # Re-evaluate roots after all new and historical edges have been unioned.
    for group, images in selected_rows:
        root = find(images[0])
        new_group_ids[root].append(group["group_id"])
        new_group_types[root].add(group.get("group_type", ""))

    component_members = defaultdict(list)
    for image, row in by_image.items():
        if row["source"] in SOURCES:
            component_members[find(image)].append(image)

    groups = []
    relevant_images = seed_images | historical_seed_images
    for root in sorted({find(image) for image in relevant_images}):
        images = component_members[root]
        members = []
        split_counts = Counter()
        group_boxes = Counter()
        for image in images:
            row = by_image[image]
            split_counts[row["split"]] += 1
            counts = label_counts(ROOT / row["output_label"])
            group_boxes.update(counts)
            members.append((row, counts))
        if len(split_counts) < 2:
            continue
        max_count = max(split_counts.values())
        destinations = tuple(split for split in SPLITS if split_counts[split] == max_count)
        source = members[0][0]["source"]
        merged_ids = sorted(new_group_ids[root])
        if not merged_ids:
            digest = hashlib.sha256("|".join(sorted(images)).encode("utf-8")).hexdigest()[:12]
            merged_ids = [f"historical_closure_{digest}"]
            new_group_types[root].add("historical_confirmed_group_closure")
        groups.append({
            "source": source,
            "group_id": "+".join(merged_ids),
            "group_type": "+".join(sorted(new_group_types[root])),
            "members": members,
            "split_counts": split_counts,
            "boxes": group_boxes,
            "destinations": destinations,
        })
    return groups


def score_source(source: str, groups: list[dict], assignments: dict, before_images: Counter, before_boxes: Counter) -> float:
    after_images = Counter({split: before_images[(source, split)] for split in SPLITS})
    after_boxes = Counter({(split, cid): before_boxes[(source, split, cid)] for split in SPLITS for cid in range(len(CLASSES))})
    for group in groups:
        destination = assignments[group["group_id"]]
        for split, count in group["split_counts"].items():
            if split != destination:
                after_images[split] -= count
                after_images[destination] += count
        for row, counts in group["members"]:
            original = row["split"]
            if original == destination:
                continue
            for cid, count in counts.items():
                after_boxes[(original, cid)] -= count
                after_boxes[(destination, cid)] += count
    total_images = sum(before_images[(source, split)] for split in SPLITS)
    image_score = sum(abs(after_images[split] - before_images[(source, split)]) for split in SPLITS) / max(total_images, 1)
    class_terms = []
    for cid in range(len(CLASSES)):
        total = sum(before_boxes[(source, split, cid)] for split in SPLITS)
        if total:
            class_terms.append(sum(abs(after_boxes[(split, cid)] - before_boxes[(source, split, cid)]) for split in SPLITS) / total)
    class_score = sum(class_terms) / max(len(class_terms), 1)
    return 2.0 * image_score + class_score


def optimize(groups: list[dict], before_images: Counter, before_boxes: Counter) -> dict:
    assignments = {}
    rng = random.Random(20260922)
    for source in SOURCES:
        source_groups = [group for group in groups if group["source"] == source]
        best_assignment = None
        best_score = float("inf")
        for restart in range(80):
            current = {
                group["group_id"]: (group["destinations"][0] if restart == 0 else rng.choice(group["destinations"]))
                for group in source_groups
            }
            improved = True
            while improved:
                improved = False
                for group in source_groups:
                    old = current[group["group_id"]]
                    local_best = old
                    local_score = score_source(source, source_groups, current, before_images, before_boxes)
                    for candidate in group["destinations"]:
                        current[group["group_id"]] = candidate
                        candidate_score = score_source(source, source_groups, current, before_images, before_boxes)
                        if candidate_score < local_score - 1e-12:
                            local_best, local_score = candidate, candidate_score
                    current[group["group_id"]] = local_best
                    improved |= local_best != old
            final_score = score_source(source, source_groups, current, before_images, before_boxes)
            if final_score < best_score - 1e-12:
                best_score, best_assignment = final_score, dict(current)
        assignments.update(best_assignment or {})
    return assignments


def create_backup() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = ROOT.parent / f"construction_v1_remediation_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copytree(META, backup / "metadata")
    for yaml_path in ROOT.glob("*.yaml"):
        shutil.copy2(yaml_path, backup / yaml_path.name)
    (backup / "RESTORE_INSTRUCTIONS.txt").write_text(
        "Reverse every row in move_plan.csv (new_* -> old_*), then restore metadata/ and *.yaml from this directory.\n",
        encoding="utf-8",
    )
    return backup


def aggregate_rows(images: Counter, boxes: Counter, full: bool) -> list[dict]:
    rows = []
    sources = ("d1", "d6", "d7", "d8")
    for source in sources:
        for split in SPLITS:
            for cid, name in enumerate(CLASSES):
                count = boxes[(source, split, cid)]
                if full or count:
                    rows.append({"source": source, "split": split, "class": name, "class_id": cid, "bbox_count": count})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tag", default="iteration_1")
    args = parser.parse_args()
    manifest_path = META / "manifest.csv"
    manifest = read_csv(manifest_path)
    group_rows = read_csv(GROUPS_CSV)
    before_images, before_boxes = snapshot_counts(manifest, use_existing_stats=True)
    groups = build_group_data(manifest, group_rows)
    assignments = optimize(groups, before_images, before_boxes)

    plan = []
    processed = []
    for group in groups:
        destination = assignments[group["group_id"]]
        moved = 0
        for row, _ in group["members"]:
            if row["split"] == destination:
                continue
            old_image, old_label = row["output_image"], row["output_label"]
            new_image = moved_rel(old_image, destination, "images")
            new_label = moved_rel(old_label, destination, "labels")
            plan.append({
                "source": group["source"], "group_id": group["group_id"],
                "from_split": row["split"], "to_split": destination,
                "old_image": old_image, "new_image": new_image,
                "old_label": old_label, "new_label": new_label,
                "image_sha256": sha256(ROOT / old_image),
                "label_sha256": sha256(ROOT / old_label),
            })
            moved += 1
        processed.append({
            "source": group["source"], "group_id": group["group_id"],
            "group_type": group["group_type"], "n_images": len(group["members"]),
            "before_train": group["split_counts"]["train"],
            "before_val": group["split_counts"]["val"],
            "before_test": group["split_counts"]["test"],
            "destination_split": destination, "n_images_moved": moved,
            "status": "planned_high_confidence_group_remediation",
        })

    direction_counts = Counter(f"{row['from_split']}->{row['to_split']}" for row in plan)
    split_deltas = Counter()
    for row in plan:
        split_deltas[(row["source"], row["from_split"])] -= 1
        split_deltas[(row["source"], row["to_split"])] += 1
    print(json.dumps({
        "groups": len(groups), "moves": len(plan),
        "theoretical_minimum_moves": sum(len(group["members"]) - max(group["split_counts"].values()) for group in groups),
        "directions": dict(direction_counts),
        "split_deltas": {f"{source}:{split}": split_deltas[(source, split)] for source in SOURCES for split in SPLITS},
    }, indent=2))
    if not args.apply:
        return

    backup = create_backup()
    write_csv(backup / "move_plan.csv", plan)
    write_csv(backup / "processed_groups.csv", processed)
    moved_pairs = []
    try:
        manifest_by_image = {row["output_image"]: row for row in manifest}
        for move in plan:
            old_image, new_image = ROOT / move["old_image"], ROOT / move["new_image"]
            old_label, new_label = ROOT / move["old_label"], ROOT / move["new_label"]
            if new_image.exists() or new_label.exists():
                raise RuntimeError(f"destination exists: {new_image} or {new_label}")
            new_image.parent.mkdir(parents=True, exist_ok=True)
            new_label.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_image, new_image)
            os.replace(old_label, new_label)
            moved_pairs.append((new_image, old_image, new_label, old_label))
            row = manifest_by_image.pop(move["old_image"])
            row["split"] = move["to_split"]
            row["output_image"] = move["new_image"]
            row["output_label"] = move["new_label"]
            row["group_id"] = move["group_id"]
            row["group_method"] = "near_duplicate_visual_audit"
            manifest_by_image[row["output_image"]] = row
        group_by_id = {group["group_id"]: group for group in groups}
        for group in groups:
            for row, _ in group["members"]:
                current = manifest_by_image.get(row["output_image"])
                if current is None:
                    # This member moved, so find it by its unchanged sha1 and group destination/name.
                    candidates = [r for r in manifest if r["sha1"] == row["sha1"] and Path(r["output_image"]).name == Path(row["output_image"]).name]
                    current = candidates[0] if candidates else None
                if current:
                    current["group_id"] = group["group_id"]
                    current["group_method"] = "near_duplicate_visual_audit"

        manifest.sort(key=lambda row: (SPLITS.index(row["split"]), row["source"], row["output_image"]))
        write_csv(manifest_path, manifest)
        after_images, after_boxes = snapshot_counts(manifest)
        write_csv(META / "stats.csv", aggregate_rows(after_images, after_boxes, full=False))
        write_csv(META / "stats_source_split_class.csv", aggregate_rows(after_images, after_boxes, full=True))

        report_dir = REPORT_DIR / args.tag
        report_dir.mkdir(parents=True, exist_ok=True)
        write_csv(report_dir / "move_plan.csv", plan)
        write_csv(report_dir / "processed_groups.csv", processed)
        size_rows = []
        class_rows = []
        for source in SOURCES:
            for split in SPLITS:
                size_rows.append({
                    "source": source, "split": split,
                    "before_images": before_images[(source, split)],
                    "after_images": after_images[(source, split)],
                    "delta_images": after_images[(source, split)] - before_images[(source, split)],
                })
                for cid, name in enumerate(CLASSES):
                    class_rows.append({
                        "source": source, "split": split, "class": name, "class_id": cid,
                        "before_bbox_count": before_boxes[(source, split, cid)],
                        "after_bbox_count": after_boxes[(source, split, cid)],
                        "delta_bbox_count": after_boxes[(source, split, cid)] - before_boxes[(source, split, cid)],
                    })
        write_csv(report_dir / "split_sizes_before_after.csv", size_rows)
        write_csv(report_dir / "class_counts_before_after.csv", class_rows)
        summary = {
            "backup_path": str(backup),
            "processed_high_confidence_groups": len(groups),
            "moved_images": len(plan),
            "move_directions": dict(Counter(f"{row['from_split']}->{row['to_split']}" for row in plan)),
            "selection_policy": "minimum moves first (destination restricted to an existing group majority/tie), then joint image-split and per-class bbox balance optimization",
            "taxonomy_or_annotation_content_changed": False,
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception:
        for new_image, old_image, new_label, old_label in reversed(moved_pairs):
            if new_image.exists():
                old_image.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new_image, old_image)
            if new_label.exists():
                old_label.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new_label, old_label)
        raise


if __name__ == "__main__":
    main()
