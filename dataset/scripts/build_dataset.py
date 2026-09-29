#!/usr/bin/env python3
"""
Build a unified YOLO detection dataset from D1/D6/D7/D8.

Expected input layout:
    D:\\lct\\datasets\\
        d1\\   # contains one downloaded archive OR one extracted dataset
        d6\\
        d7\\
        d8\\

The script:
  1) auto-extracts one archive per dN folder when needed;
  2) discovers YOLO data.yaml and train/valid/test folders;
  3) remaps source labels to 13 canonical classes;
  4) drops non-target classes;
  5) excludes WHOLE images containing ambiguous generic "crane";
  6) preserves source train/val/test splits;
  7) deduplicates exact identical image bytes across sources/splits;
  8) writes source-separated folders for per-dataset validation;
  9) creates manifests, stats, YAML configs, and sanity checks.

Important:
  - Organizer 100 images are NOT included yet.
  - This does NOT claim true scene-aware splitting if a source dataset does not
    provide camera/sequence IDs. It preserves the source split and checks exact
    duplicate leakage. See README_BUILD.txt in the output.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    class _MinimalYaml:
        @staticmethod
        def safe_load(stream):
            text = stream.read() if hasattr(stream, "read") else str(stream)
            data = {}
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("roboflow:"):
                    continue
                if raw_line[:1].isspace():
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if not value:
                    data[key] = {}
                elif value.startswith("[") or value.startswith("{"):
                    data[key] = ast.literal_eval(value)
                elif value.isdigit():
                    data[key] = int(value)
                else:
                    data[key] = value.strip("'\"")
            return data

        @staticmethod
        def safe_dump(obj, stream, **_kwargs):
            json.dump(obj, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    yaml = _MinimalYaml()

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Missing dependency: pillow. Install: pip install pillow pyyaml") from e


CANONICAL_CLASSES = [
    "excavator",
    "dump_truck",
    "truck",
    "loader",
    "bulldozer",
    "motor_grader",
    "roller",
    "concrete_mixer",
    "telehandler",
    "piling_machine",
    "crane_manipulator",
    "mobile_crane",
    "tower_crane",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

# None = intentionally drop from target taxonomy.
ALIASES = {
    "excavator": "excavator",

    "dump truck": "dump_truck",
    "dumptruck": "dump_truck",
    "dumb truck": "dump_truck",  # typo seen in D6
    "dumbtruck": "dump_truck",

    "truck": "truck",

    "loader": "loader",
    "bucket loader": "loader",
    "bucket loader big": "loader",
    "bucket loader standard": "loader",
    "bucket loader standart": "loader",
    "wheel loader": "loader",
    "front loader": "loader",
    "backhoe loader": "loader",

    "bulldozer": "bulldozer",
    "bull dozer": "bulldozer",
    "bull_dozer": "bulldozer",

    "motor grader": "motor_grader",
    "grader": "motor_grader",

    "roller": "roller",
    "compactor": "roller",

    "mixer": "concrete_mixer",
    "concrete mixer": "concrete_mixer",
    "concretemixer": "concrete_mixer",
    "concrete mixer truck": "concrete_mixer",
    "concrete transport mixer": "concrete_mixer",
    "mixed": "concrete_mixer",  # D6: visually verify after build

    "forklift giraffe": "telehandler",
    "forklift_giraffe": "telehandler",
    "telehandler": "telehandler",
    "telescopic handler": "telehandler",

    "pile driving": "piling_machine",
    "piledriving": "piling_machine",
    "pile driver": "piling_machine",
    "piling machine": "piling_machine",
    "piling-machine": "piling_machine",
    "pilingmachine": "piling_machine",
    "drill rig": "piling_machine",

    "crane manipulator": "crane_manipulator",
    "crane_manipulator": "crane_manipulator",

    "autocran": "mobile_crane",
    "autocrane": "mobile_crane",
    "mobile crane": "mobile_crane",
    "mobile_crane": "mobile_crane",
    "vehicle crane": "mobile_crane",
    "truck crane": "mobile_crane",
    "crawler crane": "mobile_crane",
    "rough terrain crane": "mobile_crane",

    "staticcrane": "tower_crane",
    "static crane": "tower_crane",
    "tower crane": "tower_crane",
    "tower_crane": "tower_crane",

    # Explicit drops
    "gazelle": None,
    "tanker": None,
    "trailer": None,
    "tractor": None,
    "forklift standard": None,
    "forklift standart": None,
    "forklift_standard": None,
    "cleaning equipment": None,
    "cleaning_equipment": None,
    "forklift": None,
    "lifting equipment": None,
    "lifting-equipment": None,
    "concrete pump": None,
    "concrete-pump": None,
    "pumptruck": None,
    "pump truck": None,
    "pump-truck": None,
}

# A generic crane is target-like but subtype-ambiguous, so the WHOLE image is excluded.
AMBIGUOUS_LABELS = {"crane"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_NAMES = ("d1", "d6", "d7", "d8")

# D1 is a YOLO export with no data.yaml. These names were recovered by
# inspecting representative GT crops for every source class ID. Keeping this
# ordered mapping here makes the otherwise implicit source schema auditable.
D1_ID_TO_NAME = {
    0: "dump truck",
    1: "excavator",
    2: "bucket loader big",
    3: "roller",
    4: "crane manipulator",
    5: "gazelle",
    6: "forklift standard",
    7: "backhoe loader",
    8: "concrete mixer",
    9: "tanker",
    10: "cleaning equipment",
    11: "bucket loader standard",
    12: "trailer",
    13: "truck",
    14: "forklift giraffe",
    15: "loader",
    16: "autocran",
}


def norm_label(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class Box:
    cls_name: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class Record:
    source: str
    split: str
    original_split: str
    image_path: Path
    label_path: Path | None
    raw_names: list[str]
    boxes: list[Box]
    sha1: str
    width: int
    height: int
    group_id: str = ""
    group_method: str = ""
    duplicate_sources: set[str] = field(default_factory=set)


def is_archive(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(s) for s in (
        ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"
    ))


def choose_dataset_root(yaml_files: list[Path]) -> Path:
    def score(y: Path):
        parent = y.parent
        split_bonus = sum((parent / x).exists() for x in ("train", "valid", "val", "test"))
        return (-split_bonus, len(parent.parts), str(parent))

    chosen = sorted(yaml_files, key=score)[0]
    if len(yaml_files) > 1:
        print("[warn] multiple data.yaml files found; using:", chosen)
        for y in yaml_files:
            if y != chosen:
                print("       other:", y)
    return chosen.parent


def prepare_source_dir(source_dir: Path, cache_root: Path) -> Path:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    yaml_files = list(source_dir.rglob("data.yaml"))
    if yaml_files:
        return choose_dataset_root(yaml_files)

    entries = [p for p in source_dir.iterdir() if p.name not in {".DS_Store", "Thumbs.db"}]
    archives = [p for p in entries if p.is_file() and is_archive(p)]

    if len(archives) == 1:
        dst = cache_root / source_dir.name
        marker = dst / ".extracted_ok"
        if not marker.exists():
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            print(f"[extract] {archives[0]} -> {dst}")
            shutil.unpack_archive(str(archives[0]), str(dst))
            marker.touch()

        yaml_files = list(dst.rglob("data.yaml"))
        if not yaml_files:
            # D1 is a valid YOLO tree but its downloaded archive omitted YAML.
            if source_dir.name == "d1" and (dst / "train" / "images").is_dir() and (dst / "valid" / "images").is_dir():
                return dst
            raise RuntimeError(f"No data.yaml found after extracting {archives[0]}")
        return choose_dataset_root(yaml_files)

    dirs = [p for p in entries if p.is_dir()]
    if len(dirs) == 1:
        yaml_files = list(dirs[0].rglob("data.yaml"))
        if yaml_files:
            return choose_dataset_root(yaml_files)

    raise RuntimeError(
        f"Cannot identify YOLO dataset inside {source_dir}.\n"
        f"Expected an extracted dataset with data.yaml OR exactly one supported archive.\n"
        f"Found entries: {[p.name for p in entries]}"
    )


def load_yaml(source: str, root: Path) -> tuple[Path, dict]:
    y = root / "data.yaml"
    if not y.exists():
        candidates = list(root.rglob("data.yaml"))
        if not candidates:
            if source == "d1":
                return root / "data.yaml", {
                    "train": "train/images",
                    "val": "valid/images",
                    "names": D1_ID_TO_NAME,
                    "schema_note": "data.yaml absent in source archive; class names recovered by GT visual audit",
                }
            raise RuntimeError(f"No data.yaml under {root}")
        y = candidates[0]
    with y.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return y, data


def parse_names(data: dict) -> dict[int, str]:
    names = data.get("names")
    if names is None:
        raise RuntimeError("data.yaml has no 'names' field")
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise RuntimeError(f"Unsupported names format: {type(names)}")


def find_split_image_dir(root: Path, yaml_path: Path, data: dict, split: str) -> Path | None:
    aliases = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}[split]

    for name in aliases:
        p = root / name / "images"
        if p.is_dir():
            return p

    candidates = []
    for p in root.rglob("images"):
        if p.parent.name.lower() in aliases:
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[0]

    yaml_key = "val" if split == "val" else split
    raw = data.get(yaml_key)
    if isinstance(raw, str):
        for p in ((yaml_path.parent / raw).resolve(), (root / raw).resolve()):
            if p.is_dir():
                return p
    return None


def label_dir_for_image_dir(image_dir: Path) -> Path:
    if image_dir.name != "images":
        raise RuntimeError(f"Expected .../images directory, got {image_dir}")
    return image_dir.parent / "labels"


def read_yolo_label(label_path: Path | None, id_to_name: dict[int, str]):
    if label_path is None or not label_path.exists():
        return [], [], False, []

    raw_names: list[str] = []
    boxes: list[Box] = []
    ambiguous = False
    warnings: list[str] = []

    with label_path.open("r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                warnings.append(
                    f"{label_path}:{ln}: expected 5 YOLO detection fields, got {len(parts)}; line dropped"
                )
                continue
            try:
                raw_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                warnings.append(f"{label_path}:{ln}: parse error; line dropped")
                continue

            if raw_id not in id_to_name:
                warnings.append(f"{label_path}:{ln}: class id {raw_id} absent from data.yaml; line dropped")
                continue

            raw_name = id_to_name[raw_id]
            raw_names.append(raw_name)
            key = norm_label(raw_name)

            if key in AMBIGUOUS_LABELS:
                ambiguous = True
                continue

            if key not in ALIASES:
                warnings.append(f"{label_path}:{ln}: UNKNOWN LABEL '{raw_name}' -> dropped")
                continue

            target = ALIASES[key]
            if target is None:
                continue
            if target not in CLASS_TO_ID:
                raise RuntimeError(f"Mapping error: {raw_name!r} -> {target!r}")

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                warnings.append(f"{label_path}:{ln}: invalid bbox {(x, y, w, h)}; line dropped")
                continue

            if x - w/2 < -1e-3 or x + w/2 > 1+1e-3 or y - h/2 < -1e-3 or y + h/2 > 1+1e-3:
                warnings.append(f"{label_path}:{ln}: bbox extends outside image {(x, y, w, h)}; line dropped")
                continue

            left, top = max(0.0, x - w / 2), max(0.0, y - h / 2)
            right, bottom = min(1.0, x + w / 2), min(1.0, y + h / 2)
            clipped = (left, top, right, bottom) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if clipped:
                warnings.append(f"{label_path}:{ln}: bbox clipped to image boundary from {(x, y, w, h)}")
                x, y = (left + right) / 2, (top + bottom) / 2
                w, h = right - left, bottom - top

            boxes.append(Box(target, x, y, w, h))

    return raw_names, boxes, ambiguous, warnings


def sha1_file(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_source_records(source: str, root: Path, keep_negatives: bool):
    yaml_path, data = load_yaml(source, root)
    id_to_name = parse_names(data)

    print(f"\n[{source}] root={root}")
    print(f"[{source}] classes:")
    for i in sorted(id_to_name):
        raw = id_to_name[i]
        key = norm_label(raw)
        if key in AMBIGUOUS_LABELS:
            target = "AMBIGUOUS -> WHOLE IMAGE EXCLUDED"
        elif key in ALIASES:
            target = ALIASES[key] if ALIASES[key] is not None else "DROP"
        else:
            target = "UNKNOWN -> DROP"
        print(f"  {i:>2}: {raw!r} -> {target}")

    records = []
    excluded = []
    warnings_counter = Counter()
    raw_class_counter = Counter()

    split_dirs = {s: find_split_image_dir(root, yaml_path, data, s) for s in ("train", "val", "test")}
    if split_dirs["train"] is None:
        raise RuntimeError(f"{source}: no train/images found")
    if split_dirs["val"] is None:
        raise RuntimeError(
            f"{source}: no validation split found. Do not silently random-split; "
            "inspect this source and add a scene/source-aware split."
        )

    for split, image_dir in split_dirs.items():
        if image_dir is None:
            continue
        label_dir = label_dir_for_image_dir(image_dir)
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        print(f"[{source}] {split}: {len(images)} images @ {image_dir}")

        for image_path in images:
            rel = image_path.relative_to(image_dir)
            label_path = (label_dir / rel).with_suffix(".txt")
            raw_names, boxes, ambiguous, warns = read_yolo_label(
                label_path if label_path.exists() else None, id_to_name
            )

            for n in raw_names:
                raw_class_counter[n] += 1
            for wmsg in warns:
                warnings_counter[wmsg] += 1

            if ambiguous:
                excluded.append({
                    "source": source,
                    "split": split,
                    "image": str(image_path),
                    "reason": "contains_ambiguous_generic_crane",
                })
                continue

            if not boxes and not keep_negatives:
                excluded.append({
                    "source": source,
                    "split": split,
                    "image": str(image_path),
                    "reason": "no_target_boxes_after_mapping",
                })
                continue

            try:
                with Image.open(image_path) as im:
                    width, height = im.size
                    if width <= 0 or height <= 0:
                        raise ValueError("non-positive dimensions")
                    im.verify()
            except Exception as e:
                excluded.append({
                    "source": source,
                    "split": split,
                    "image": str(image_path),
                    "reason": f"invalid_image:{type(e).__name__}:{e}",
                })
                continue

            records.append(Record(
                source=source,
                split=split,
                original_split=split,
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                raw_names=raw_names,
                boxes=boxes,
                sha1=sha1_file(image_path),
                width=width,
                height=height,
                duplicate_sources={source},
            ))

    if warnings_counter:
        print(f"[{source}] label warnings: {sum(warnings_counter.values())}; full list saved later")
    return records, excluded, warnings_counter, raw_class_counter


def roboflow_original_stem(path: Path) -> str:
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)


def recover_reliable_group(source: str, image_path: Path) -> tuple[str, str]:
    """Return only sequence groups supported directly by filename semantics."""
    stem = roboflow_original_stem(image_path)
    if source == "d1":
        iso_camera = re.match(r"camera_([^_]+)_(\d{4}-\d{2}-\d{2})T", stem, flags=re.IGNORECASE)
        if iso_camera:
            return f"d1:camera={iso_camera.group(1).lower()}:date={iso_camera.group(2)}", "d1_camera_and_date"
        camera = stem.split("_", 1)[0]
        date_match = re.search(r"-(\d{4}-\d{2}-\d{2})$", stem)
        if re.fullmatch(r"\d+", camera) and date_match:
            return f"d1:camera={camera}:date={date_match.group(1)}", "d1_camera_and_date"

    if source == "d8":
        cc = re.match(r"(cc-\d+_\d+)_(\d{8})T\d+", stem, flags=re.IGNORECASE)
        if cc:
            return f"d8:{cc.group(1).lower()}:date={cc.group(2)}", "d8_cc_camera_and_date"
        flight = re.match(r"(.+_Flight[_-]?\d+)_\d+$", stem, flags=re.IGNORECASE)
        if flight:
            return f"d8:{flight.group(1).lower()}", "d8_explicit_flight"
        if re.fullmatch(r"DJI_\d+", stem, flags=re.IGNORECASE):
            return "d8:dji_numbered_series", "d8_dji_numbered_series"

    if source == "d6" and re.fullmatch(r"ezgif-frame-\d+", stem, flags=re.IGNORECASE):
        return "d6:ezgif-frame", "d6_explicit_frame_series"

    return "", ""


def apply_group_aware_splits(records: list[Record]) -> list[dict]:
    """Keep reliable filename groups intact while approximating source ratios."""
    for record in records:
        record.group_id, record.group_method = recover_reliable_group(record.source, record.image_path)

    audit_rows: list[dict] = []
    for source in SOURCE_NAMES:
        source_records = [r for r in records if r.source == source and r.group_id]
        groups = defaultdict(list)
        for record in source_records:
            groups[record.group_id].append(record)

        split_groups = {
            gid: members
            for gid, members in groups.items()
            if len({r.original_split for r in members}) > 1
        }
        if split_groups:
            split_order = [s for s in ("train", "val", "test") if any(r.original_split == s for r in source_records)]
            target_images = Counter(r.original_split for r in source_records)
            target_boxes = defaultdict(Counter)
            for record in source_records:
                target_boxes[record.original_split].update(b.cls_name for b in record.boxes)

            fixed = [r for gid, members in groups.items() if gid not in split_groups for r in members]
            current_images = Counter(r.split for r in fixed)
            current_boxes = defaultdict(Counter)
            for record in fixed:
                current_boxes[record.split].update(b.cls_name for b in record.boxes)

            def assignment_score(candidate_split: str, members: list[Record]) -> tuple[float, str]:
                projected_images = current_images.copy()
                projected_images[candidate_split] += len(members)
                group_boxes = Counter(b.cls_name for r in members for b in r.boxes)
                score = 0.0
                for split in split_order:
                    score += ((projected_images[split] - target_images[split]) ** 2) / max(target_images[split], 1)
                    classes = set(target_boxes[split]) | set(group_boxes)
                    for cls in classes:
                        value = current_boxes[split][cls] + (group_boxes[cls] if split == candidate_split else 0)
                        score += 0.15 * ((value - target_boxes[split][cls]) ** 2) / max(target_boxes[split][cls], 1)
                tie = hashlib.sha1(f"{members[0].group_id}:{candidate_split}".encode()).hexdigest()
                return score, tie

            for gid, members in sorted(split_groups.items(), key=lambda item: (-len(item[1]), item[0])):
                chosen_split = min(split_order, key=lambda split: assignment_score(split, members))
                group_boxes = Counter(b.cls_name for r in members for b in r.boxes)
                for record in members:
                    record.split = chosen_split
                current_images[chosen_split] += len(members)
                current_boxes[chosen_split].update(group_boxes)

        for gid, members in sorted(groups.items()):
            audit_rows.append({
                "source": source,
                "group_id": gid,
                "group_method": members[0].group_method,
                "image_count": len(members),
                "original_splits": ";".join(sorted({r.original_split for r in members})),
                "final_splits": ";".join(sorted({r.split for r in members})),
            })
    return audit_rows


SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def box_key(b: Box, precision=6):
    return (b.cls_name, round(b.x, precision), round(b.y, precision), round(b.w, precision), round(b.h, precision))


def dedupe_records(records: list[Record]):
    groups = defaultdict(list)
    for r in records:
        groups[r.sha1].append(r)

    out = []
    report = []
    for sha1, group in groups.items():
        if len(group) == 1:
            out.append(group[0])
            continue

        chosen = sorted(
            group,
            key=lambda r: (-SPLIT_PRIORITY[r.split], SOURCE_NAMES.index(r.source), str(r.image_path)),
        )[0]
        merged = {box_key(b): b for r in group for b in r.boxes}
        chosen.boxes = list(merged.values())
        chosen.duplicate_sources = {r.source for r in group}

        report.append({
            "sha1": sha1,
            "kept_source": chosen.source,
            "kept_split": chosen.split,
            "kept_image": str(chosen.image_path),
            "copies": len(group),
            "all_sources": ";".join(sorted({r.source for r in group})),
            "all_splits": ";".join(sorted({r.split for r in group})),
            "all_images": ";".join(str(r.image_path) for r in group),
        })
        out.append(chosen)
    return out, report


def safe_name(source: str, r: Record) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", r.image_path.stem).strip("_")
    stem = stem[:100] if stem else "image"
    return f"{source}_{stem}_{r.sha1[:10]}{r.image_path.suffix.lower()}"


def write_label(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for b in sorted(boxes, key=lambda z: (CLASS_TO_ID[z.cls_name], z.x, z.y, z.w, z.h)):
            f.write(
                f"{CLASS_TO_ID[b.cls_name]} {b.x:.8f} {b.y:.8f} {b.w:.8f} {b.h:.8f}\n"
            )


def write_yaml(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def build_output(records: list[Record], output: Path):
    manifest = []
    stats = Counter()
    for r in sorted(records, key=lambda z: (z.split, z.source, str(z.image_path))):
        filename = safe_name(r.source, r)
        out_img = output / "images" / r.split / r.source / filename
        out_lbl = output / "labels" / r.split / r.source / (Path(filename).stem + ".txt")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(r.image_path, out_img)
        except OSError:
            shutil.copy2(r.image_path, out_img)
        write_label(out_lbl, r.boxes)

        class_counts = Counter(b.cls_name for b in r.boxes)
        for c, n in class_counts.items():
            stats[(r.source, r.split, c)] += n

        manifest.append({
            "source": r.source,
            "split": r.split,
            "original_split": r.original_split,
            "original_image": str(r.image_path),
            "output_image": str(out_img.relative_to(output)).replace("\\", "/"),
            "output_label": str(out_lbl.relative_to(output)).replace("\\", "/"),
            "sha1": r.sha1,
            "width": r.width,
            "height": r.height,
            "n_boxes": len(r.boxes),
            "classes": ";".join(sorted(class_counts)),
            "raw_classes": ";".join(sorted(set(r.raw_names))),
            "group_id": r.group_id,
            "group_method": r.group_method,
            "duplicate_sources": ";".join(sorted(r.duplicate_sources)),
        })
    return manifest, stats


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def create_data_yamls(output: Path, present_splits: set[str]) -> None:
    names_dict = {i: n for i, n in enumerate(CANONICAL_CLASSES)}
    main = {"path": ".", "train": "images/train", "val": "images/val", "names": names_dict}
    if "test" in present_splits:
        main["test"] = "images/test"
    write_yaml(output / "data.yaml", main)

    for source in SOURCE_NAMES:
        y = {
            "path": ".",
            "train": "images/train",
            "val": f"images/val/{source}",
            "names": names_dict,
        }
        if (output / "images" / "test" / source).exists():
            y["test"] = f"images/test/{source}"
        write_yaml(output / f"data_val_{source}.yaml", y)


def validate_built_dataset(output: Path, manifest: list[dict]) -> dict:
    errors = []
    counts_by_split = Counter()
    hashes_by_split = defaultdict(set)
    groups_by_split = defaultdict(set)

    for row in manifest:
        split = row["split"]
        counts_by_split[split] += 1
        hashes_by_split[split].add(row["sha1"])
        if row.get("group_id"):
            groups_by_split[split].add(row["group_id"])
        img = output / row["output_image"]
        lbl = output / row["output_label"]
        if not img.exists():
            errors.append(f"missing image: {img}")
        if not lbl.exists():
            errors.append(f"missing label: {lbl}")
            continue

        with lbl.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                p = line.split()
                if len(p) != 5:
                    errors.append(f"{lbl}:{ln}: expected 5 fields")
                    continue
                try:
                    cid = int(p[0])
                    x, y, w, h = map(float, p[1:])
                except ValueError:
                    errors.append(f"{lbl}:{ln}: parse error")
                    continue
                if cid not in range(len(CANONICAL_CLASSES)):
                    errors.append(f"{lbl}:{ln}: class id out of range {cid}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"{lbl}:{ln}: invalid bbox {(x, y, w, h)}")
                elif x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                    errors.append(f"{lbl}:{ln}: bbox extends outside image {(x, y, w, h)}")

    leakage = {}
    group_leakage = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = hashes_by_split[a] & hashes_by_split[b]
        if inter:
            leakage[f"{a}_{b}"] = len(inter)
            errors.append(f"EXACT IMAGE LEAKAGE {a}<->{b}: {len(inter)} images")
        group_inter = groups_by_split[a] & groups_by_split[b]
        if group_inter:
            group_leakage[f"{a}_{b}"] = sorted(group_inter)
            errors.append(f"RECOVERED GROUP LEAKAGE {a}<->{b}: {len(group_inter)} groups")

    return {
        "errors": errors,
        "image_counts": dict(counts_by_split),
        "exact_hash_leakage": leakage,
        "recovered_group_leakage": group_leakage,
    }


def write_readme(output: Path) -> None:
    classes = "\n".join(f"{i:2d} {c}" for i, c in enumerate(CANONICAL_CLASSES))
    txt = f"""construction_v1 build notes
===========================

Final classes ({len(CANONICAL_CLASSES)}):
{classes}

Policy and provenance:
- Organizer 100 images are NOT included.
- D1 source archive omitted data.yaml. Its 17 source class IDs were recovered
  by a visual GT audit; see metadata/raw_class_mapping_audit.csv.
- Reliable filename sequences are kept wholly in one split: D1 camera+date,
  D8 explicit cc camera/date and flight series, D8 DJI numbered series, and
  D6 ezgif-frame. Their assignments approximate the source split ratios.
- Other records preserve source-provided train/val/test. D7 numeric-only names
  do not expose reliable camera/scene boundaries; D6 is mostly heterogeneous
  scraped naming; D8 bare numeric names are also not grouped.
- Images containing ambiguous generic class 'crane' are excluded entirely.
- Non-target classes (Gazelle, trailer, tanker, forklift, concrete pump, etc.)
  are ignored; images with zero remaining target boxes are dropped by default.
- Exact duplicate image bytes across datasets/splits are deduplicated. If an
  exact duplicate occurs across train and val/test, one copy is retained in the
  stronger holdout split.
- D6 class 'mixed' is currently mapped to concrete_mixer based on the team's
  audit. Visually inspect this class before final training.
- 'loader' intentionally merges wheel/front/backhoe/bucket-loader variants.
- 'mobile_crane' includes truck/mobile/crawler/rough-terrain cranes where the
  source annotation is specific enough.
- Exact duplicate leakage and recovered-group leakage are checked automatically.
- Scene leakage still cannot be ruled out for records whose filenames expose no
  reliable scene/camera identity. No synthetic grouping is invented for them.

Recommended next steps:
1. Open metadata/stats.csv and metadata/excluded.csv.
2. Visually inspect 50-100 GT images per source, especially D6 'mixed', cranes,
   telehandler, and piling_machine.
3. Review metadata/group_split_audit.csv and the stated grouping limitation.
4. Upload construction_v1 as ONE private Kaggle Dataset.
5. Use the same immutable dataset version for all overnight experiments.
"""
    (output / "README_BUILD.txt").write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path(r"D:\lct\datasets"),
        help=r"Folder containing d1, d6, d7, d8. Default: D:\lct\datasets"
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output folder. Default: <root>/construction_v1"
    )
    ap.add_argument(
        "--keep-negatives", action="store_true",
        help="Keep images with zero target boxes after mapping (default: drop them)."
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing output before rebuilding."
    )
    args = ap.parse_args()

    root = args.root.resolve()
    output = (args.output or (root / "construction_v1")).resolve()
    cache_root = root / ".build_cache"

    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}\nUse --overwrite to rebuild.")
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_excluded = []
    all_warnings = []
    raw_class_rows = []

    for source in SOURCE_NAMES:
        src_dir = root / source
        ds_root = prepare_source_dir(src_dir, cache_root)
        records, excluded, warning_counts, raw_counts = collect_source_records(
            source, ds_root, args.keep_negatives
        )
        all_records.extend(records)
        all_excluded.extend(excluded)

        for msg, count in warning_counts.items():
            all_warnings.append({"source": source, "count": count, "warning": msg})
        for raw_name, count in raw_counts.items():
            key = norm_label(raw_name)
            raw_class_rows.append({
                "source": source,
                "raw_class": raw_name,
                "raw_class_normalized": key,
                "bbox_count": count,
                "mapped_to": (
                    "AMBIGUOUS" if key in AMBIGUOUS_LABELS
                    else ALIASES.get(key, "UNKNOWN")
                ),
            })

    print(f"\nCollected usable records before exact dedup: {len(all_records)}")
    group_audit = apply_group_aware_splits(all_records)
    moved_by_source = Counter(r.source for r in all_records if r.split != r.original_split)
    print("Group-aware split moves:", dict(moved_by_source))
    deduped, dedup_report = dedupe_records(all_records)
    print(f"After exact dedup: {len(deduped)}")
    print(f"Exact duplicate groups merged: {len(dedup_report)}")

    manifest, stats = build_output(deduped, output)
    present_splits = {r["split"] for r in manifest}
    create_data_yamls(output, present_splits)

    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)

    write_csv(metadata / "manifest.csv", manifest)
    write_csv(
        metadata / "group_split_audit.csv", group_audit,
        ["source", "group_id", "group_method", "image_count", "original_splits", "final_splits"]
    )
    write_csv(metadata / "excluded.csv", all_excluded, ["source", "split", "image", "reason"])
    write_csv(metadata / "warnings.csv", all_warnings, ["source", "count", "warning"])
    write_csv(
        metadata / "raw_class_mapping_audit.csv", raw_class_rows,
        ["source", "raw_class", "raw_class_normalized", "bbox_count", "mapped_to"]
    )
    write_csv(
        metadata / "exact_duplicates.csv", dedup_report,
        ["sha1", "kept_source", "kept_split", "kept_image", "copies", "all_sources", "all_splits", "all_images"]
    )

    stat_rows = []
    for (source, split, cls), n in sorted(stats.items()):
        stat_rows.append({
            "source": source,
            "split": split,
            "class": cls,
            "class_id": CLASS_TO_ID[cls],
            "bbox_count": n,
        })
    write_csv(metadata / "stats.csv", stat_rows, ["source", "split", "class", "class_id", "bbox_count"])

    write_yaml(metadata / "class_mapping.yaml", {
        "canonical_classes": {i: c for i, c in enumerate(CANONICAL_CLASSES)},
        "aliases": {k: v for k, v in sorted(ALIASES.items())},
        "ambiguous_labels_exclude_whole_image": sorted(AMBIGUOUS_LABELS),
    })

    validation = validate_built_dataset(output, manifest)
    (metadata / "sanity_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output)

    print("\n" + "=" * 72)
    print("BUILD SUMMARY")
    print("=" * 72)
    print("Output:", output)
    print("Images by split:", validation["image_counts"])
    print("Excluded images:", len(all_excluded))
    print("Exact duplicate groups:", len(dedup_report))

    totals = Counter()
    for row in stat_rows:
        totals[row["class"]] += int(row["bbox_count"])

    print("\nBBox counts:")
    for cls in CANONICAL_CLASSES:
        print(f"  {cls:20s} {totals[cls]:8d}")

    unknown_rows = [r for r in raw_class_rows if r["mapped_to"] == "UNKNOWN"]
    if unknown_rows:
        print("\n[WARN] UNKNOWN SOURCE LABELS WERE DROPPED:")
        for r in unknown_rows:
            print(f"  {r['source']}: {r['raw_class']!r} ({r['bbox_count']} boxes)")
        print("Inspect metadata/raw_class_mapping_audit.csv")

    if validation["errors"]:
        print("\n[SANITY CHECK FAILED]")
        for e in validation["errors"][:50]:
            print(" -", e)
        print("See:", metadata / "sanity_report.json")
        raise SystemExit(2)

    print("\n[SANITY CHECK PASSED]")
    print("Main config:", output / "data.yaml")
    for s in SOURCE_NAMES:
        print(f"Per-source val config: {output / f'data_val_{s}.yaml'}")
    print("\nBefore training, visually inspect D6 'mixed' and sample GT boxes.")
    print("Also inspect whether filenames expose camera/sequence IDs; if yes,")
    print("refine the split group-wise before freezing construction_v1.")


if __name__ == "__main__":
    main()
