from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"D:\lct\datasets\construction_v1")
MANIFEST = ROOT / "metadata" / "manifest.csv"
OUT = ROOT / "metadata" / "near_duplicate_audit"
SOURCES = ("d6", "d7")
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
PHASH_RADIUS = 16
DHASH_RADIUS = 12
TOP_PAIRS = 100
FONT = ImageFont.load_default(size=14)


def dct_matrix(n: int) -> np.ndarray:
    matrix = np.empty((n, n), dtype=np.float32)
    factor = math.pi / (2 * n)
    matrix[0] = math.sqrt(1 / n)
    scale = math.sqrt(2 / n)
    for k in range(1, n):
        for i in range(n):
            matrix[k, i] = scale * math.cos((2 * i + 1) * k * factor)
    return matrix


DCT32 = dct_matrix(32)


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def fingerprint(path: Path):
    with Image.open(path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        gray32 = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        coeffs = DCT32 @ gray32 @ DCT32.T
        low = coeffs[:8, :8].copy()
        median = float(np.median(low.reshape(-1)[1:]))
        phash_bits = low > median
        phash_bits[0, 0] = False
        gray_d = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        dhash_bits = gray_d[:, 1:] > gray_d[:, :-1]
        gray64 = np.asarray(rgb.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        rgb32 = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
        hist = np.concatenate([
            np.histogram(rgb32[:, :, channel], bins=16, range=(0, 256))[0]
            for channel in range(3)
        ]).astype(np.float32)
        hist /= max(float(np.linalg.norm(hist)), 1e-12)
        return bits_to_int(phash_bits), bits_to_int(dhash_bits), gray64, hist


class BKNode:
    __slots__ = ("value", "indices", "children")

    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    def add(self, value: int, index: int):
        if self.root is None:
            self.root = BKNode(value, index)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def query(self, value: int, radius: int):
        if self.root is None:
            return []
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                result.extend(node.indices)
            low, high = distance - radius, distance + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return result


@dataclass
class Item:
    source: str
    split: str
    output_image: str
    original_image: str
    sha1: str
    phash: int = 0
    dhash: int = 0
    gray: np.ndarray | None = None
    hist: np.ndarray | None = None


def normalized_base(path_text: str) -> str:
    stem = Path(path_text).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)


def sequence_hint(path_text: str):
    base = normalized_base(path_text)
    match = re.match(r"^(.*?)(\d+)$", base)
    if not match:
        return "", None
    family = match.group(1).rstrip("_-.").lower()
    return family, int(match.group(2))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def pair_metrics(a: Item, b: Item):
    centered_a = a.gray.reshape(-1) - float(a.gray.mean())
    centered_b = b.gray.reshape(-1) - float(b.gray.mean())
    ncc = cosine(centered_a, centered_b)
    mae = float(np.mean(np.abs(a.gray - b.gray)))
    hist_cosine = cosine(a.hist, b.hist)
    phash_distance = (a.phash ^ b.phash).bit_count()
    dhash_distance = (a.dhash ^ b.dhash).bit_count()
    family_a, frame_a = sequence_hint(a.original_image)
    family_b, frame_b = sequence_hint(b.original_image)
    same_family = bool(family_a and family_a == family_b)
    frame_delta = abs(frame_a - frame_b) if same_family and frame_a is not None and frame_b is not None else None
    score = (
        0.30 * (1 - phash_distance / 64)
        + 0.15 * (1 - dhash_distance / 64)
        + 0.40 * max(ncc, 0)
        + 0.15 * hist_cosine
    )
    if same_family and frame_delta is not None:
        score += 0.05 * math.exp(-frame_delta / 10)
    if phash_distance <= 3 and dhash_distance <= 5 and ncc >= 0.985 and mae <= 0.04:
        initial = "high_confidence_leakage"
    elif same_family and frame_delta is not None and frame_delta <= 3 and ncc >= 0.94:
        initial = "high_confidence_leakage"
    elif (phash_distance <= 8 and ncc >= 0.94) or (phash_distance <= 12 and dhash_distance <= 10 and ncc >= 0.90):
        initial = "possible_leakage"
    elif same_family and frame_delta is not None and frame_delta <= 15 and ncc >= 0.84:
        initial = "possible_leakage"
    else:
        initial = "visually_not_leakage"
    return {
        "phash_distance": phash_distance,
        "dhash_distance": dhash_distance,
        "pixel_ncc": ncc,
        "pixel_mae": mae,
        "color_hist_cosine": hist_cosine,
        "same_filename_family": same_family,
        "frame_delta": frame_delta,
        "suspicion_score": score,
        "visual_review": initial,
        "review_note": "initial metric triage; requires contact-sheet review",
    }


def load_items():
    items = {source: [] for source in SOURCES}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source"] in items:
                items[row["source"]].append(Item(
                    source=row["source"],
                    split=row["split"],
                    output_image=row["output_image"],
                    original_image=row["original_image"],
                    sha1=row["sha1"],
                ))
    return items


def candidate_pairs(items: list[Item]):
    by_split = defaultdict(list)
    for index, item in enumerate(items):
        by_split[item.split].append(index)
    candidates = set()
    for split_a, split_b in SPLIT_PAIRS:
        indices_a, indices_b = by_split[split_a], by_split[split_b]
        phash_tree, dhash_tree = BKTree(), BKTree()
        for index in indices_b:
            phash_tree.add(items[index].phash, index)
            dhash_tree.add(items[index].dhash, index)
        for index_a in indices_a:
            for index_b in phash_tree.query(items[index_a].phash, PHASH_RADIUS):
                candidates.add((index_a, index_b))
            for index_b in dhash_tree.query(items[index_a].dhash, DHASH_RADIUS):
                candidates.add((index_a, index_b))
    return candidates


PAIR_FIELDS = [
    "pair_id", "source", "split_a", "image_a", "original_image_a", "split_b", "image_b", "original_image_b",
    "phash_distance", "dhash_distance", "embedding_cosine_similarity", "pixel_ncc", "pixel_mae",
    "color_hist_cosine", "same_filename_family", "frame_delta", "suspicion_score", "visual_review", "review_note",
]


def build_rows(source: str, items: list[Item], candidates):
    rows = []
    for index_a, index_b in candidates:
        a, b = items[index_a], items[index_b]
        metrics = pair_metrics(a, b)
        pair_key = "|".join(sorted((a.output_image, b.output_image)))
        rows.append({
            "pair_id": hashlib.sha1(pair_key.encode()).hexdigest()[:12],
            "source": source,
            "split_a": a.split,
            "image_a": a.output_image,
            "original_image_a": a.original_image,
            "split_b": b.split,
            "image_b": b.output_image,
            "original_image_b": b.original_image,
            "embedding_cosine_similarity": "",
            **metrics,
        })
    rows.sort(key=lambda row: (-row["suspicion_score"], row["phash_distance"], row["pair_id"]))
    return rows


def image_panel(row, side: str, width=430, height=250):
    image_path = ROOT / row[f"image_{side}"]
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "#eeeeee")
    panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return panel


def pair_cell(row, width=900, height=320):
    cell = Image.new("RGB", (width, height), "white")
    left = image_panel(row, "a", width // 2, height - 66)
    right = image_panel(row, "b", width // 2, height - 66)
    cell.paste(left, (0, 66))
    cell.paste(right, (width // 2, 66))
    draw = ImageDraw.Draw(cell)
    name_a = Path(row["original_image_a"]).name
    name_b = Path(row["original_image_b"]).name
    draw.text((6, 4), f"{row['pair_id']}  {row['split_a']} | {row['split_b']}  p={row['phash_distance']} d={row['dhash_distance']} ncc={row['pixel_ncc']:.4f} mae={row['pixel_mae']:.4f}", fill="black", font=FONT)
    draw.text((6, 24), f"A {name_a[:66]}", fill="#1c4ea0", font=FONT)
    draw.text((6, 43), f"B {name_b[:66]}", fill="#a03020", font=FONT)
    draw.line((width // 2, 66, width // 2, height), fill="#444444", width=2)
    return cell


def save_contact_sheet(source: str, rows: list[dict]):
    top = rows[:TOP_PAIRS]
    columns = 2
    cell_w, cell_h = 900, 320
    page_size = 20
    full = Image.new("RGB", (columns * cell_w, math.ceil(len(top) / columns) * cell_h), "#dddddd")
    for index, row in enumerate(top):
        full.paste(pair_cell(row, cell_w, cell_h), ((index % columns) * cell_w, (index // columns) * cell_h))
    full.save(OUT / f"{source}_top_phash_pairs.jpg", quality=91)
    for page_start in range(0, len(top), page_size):
        subset = top[page_start:page_start + page_size]
        page = Image.new("RGB", (columns * cell_w, math.ceil(len(subset) / columns) * cell_h), "#dddddd")
        for offset, row in enumerate(subset):
            page.paste(pair_cell(row, cell_w, cell_h), ((offset % columns) * cell_w, (offset // columns) * cell_h))
        page.save(OUT / f"{source}_top_phash_pairs_page_{page_start // page_size + 1:02d}.jpg", quality=92)


def write_pairs(source: str, rows: list[dict]):
    with (OUT / f"{source}_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sequence_components(all_rows: dict[str, list[dict]]):
    output = []
    for source, rows in all_rows.items():
        edges = []
        nodes = set()
        for row in rows:
            if row["visual_review"] not in {"high_confidence_leakage", "possible_leakage"}:
                continue
            if not row["same_filename_family"]:
                continue
            if row["frame_delta"] in (None, "") or int(row["frame_delta"]) > 30:
                continue
            a, b = row["image_a"], row["image_b"]
            nodes.update((a, b))
            edges.append((a, b))
        graph = defaultdict(set)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        seen = set()
        component_number = 0
        for start in sorted(nodes):
            if start in seen:
                continue
            queue = deque([start])
            component = []
            seen.add(start)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            splits = sorted({part.split("/")[1] for part in component})
            if len(component) >= 2 and len(splits) >= 2:
                component_number += 1
                output.append({
                    "source": source,
                    "group_id": f"{source}_suspected_{component_number:04d}",
                    "n_images": len(component),
                    "splits": ";".join(splits),
                    "images": ";".join(sorted(component)),
                    "status": "suspected_sequence_crosses_split",
                })
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items_by_source = load_items()
    all_rows = {}
    summaries = []
    for source, items in items_by_source.items():
        print(f"[{source}] fingerprinting {len(items)} images", flush=True)
        for index, item in enumerate(items, 1):
            item.phash, item.dhash, item.gray, item.hist = fingerprint(ROOT / item.output_image)
            if index % 500 == 0:
                print(f"[{source}] {index}/{len(items)}", flush=True)
        candidates = candidate_pairs(items)
        print(f"[{source}] candidates={len(candidates)}", flush=True)
        rows = build_rows(source, items, candidates)
        all_rows[source] = rows
        write_pairs(source, rows)
        save_contact_sheet(source, rows)
        split_counts = Counter(item.split for item in items)
        total_possible = sum(split_counts[a] * split_counts[b] for a, b in SPLIT_PAIRS)
        summaries.append({
            "source": source,
            "n_images_train": split_counts["train"],
            "n_images_val": split_counts["val"],
            "n_images_test": split_counts["test"],
            "n_cross_split_pairs_possible": total_possible,
            "n_cross_split_pairs_examined": len(rows),
            "n_high_confidence_suspected_duplicates": sum(r["visual_review"] == "high_confidence_leakage" for r in rows),
            "n_medium_confidence_pairs": sum(r["visual_review"] == "possible_leakage" for r in rows),
            "n_visually_not_leakage": sum(r["visual_review"] == "visually_not_leakage" for r in rows),
            "thresholds_used": {
                "candidate_phash_hamming_max": PHASH_RADIUS,
                "candidate_dhash_hamming_max": DHASH_RADIUS,
                "note": "pHash/dHash generate candidates only; final categories require contact-sheet review.",
            },
            "method": "64-bit pHash + 64-bit dHash BK-tree cross-split search; candidate reranking with 64x64 grayscale NCC/MAE and RGB histogram cosine",
        })

    groups = sequence_components(all_rows)
    with (OUT / "suspected_sequence_groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["source", "group_id", "n_images", "splits", "images", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups)

    summary = {
        "scope": ["d6", "d7"],
        "embeddings_used": False,
        "embedding_reason": "No local vision model weights or Torch/Transformers/FAISS runtime were available; no download was attempted.",
        "sources": summaries,
        "suspected_sequence_groups": len(groups),
        "review_status": "initial_metric_triage_pending_visual_review",
        "no_split_changes_made": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
