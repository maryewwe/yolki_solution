from __future__ import annotations

import csv
import json
import re
import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(r"D:\lct\datasets\construction_v1")
OUT = ROOT / "metadata" / "near_duplicate_audit"
VISUAL_TOP_N = 100


def frame_number(path_text: str) -> int | None:
    name = Path(path_text).name
    match = re.search(r"(?:^|_)(\d+)_jpg\.rf\.", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_rows(source: str) -> list[dict]:
    path = OUT / f"{source}_pairs.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_visual_review(
    source: str, rows: list[dict], high_count: int, high_ids: set[str] | None = None
) -> None:
    for rank, row in enumerate(rows, 1):
        prior = row["visual_review"]
        row["review_rank"] = rank
        if rank <= VISUAL_TOP_N:
            row["reviewed_visually"] = "true"
            is_high = row["pair_id"] in high_ids if high_ids is not None else rank <= high_count
            if is_high:
                row["visual_review"] = "high_confidence_leakage"
            else:
                row["visual_review"] = "visually_not_leakage"
            if not is_high:
                row["review_note"] = (
                    "visually reviewed on contact sheet: similar construction subject/site, but not the same "
                    "frame, adjacent sequence, or duplicated source asset"
                )
            elif source == "d6":
                row["review_note"] = (
                    "visually reviewed on contact sheet: same source/stock photograph across splits, "
                    "with JPEG, resize, crop, padding, or watermark variation"
                )
            else:
                row["review_note"] = (
                    "visually reviewed on contact sheet: same fixed-camera scene or adjacent/nearby "
                    "video frame across splits; object/layout continuity is visible"
                )
        else:
            row["reviewed_visually"] = "false"
            if prior in {"high_confidence_leakage", "possible_leakage"}:
                row["visual_review"] = "possible_leakage"
                row["review_note"] = (
                    "metric candidate below the visually reviewed top-100 cutoff; manual adjudication still required"
                )
            else:
                row["visual_review"] = "visually_not_leakage"
                row["review_note"] = (
                    "screened out by combined pHash/dHash and pixel/color metrics; not individually reviewed"
                )


def write_rows(source: str, rows: list[dict]) -> None:
    fields = list(rows[0])
    path = OUT / f"{source}_pairs.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen: set[str] = set()
    result: list[list[str]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        result.append(sorted(component))
    return result


def build_groups(all_rows: dict[str, list[dict]], iteration: str) -> list[dict]:
    groups: list[dict] = []
    for source, rows in all_rows.items():
        if source == "d6":
            selected = [r for r in rows if r["visual_review"] == "high_confidence_leakage"]
            group_type = "duplicate_asset_cluster"
        else:
            selected = []
            for row in rows:
                a_num, b_num = frame_number(row["image_a"]), frame_number(row["image_b"])
                delta = abs(a_num - b_num) if a_num is not None and b_num is not None else None
                visually_confirmed = row["visual_review"] == "high_confidence_leakage"
                sequence_metric = (
                    delta is not None
                    and delta <= 30
                    and float(row["pixel_ncc"]) >= 0.90
                    and int(row["phash_distance"]) <= 12
                )
                if visually_confirmed or sequence_metric:
                    selected.append(row)
            group_type = "suspected_video_sequence"

        edge_rows = {(r["image_a"], r["image_b"]): r for r in selected}
        for number, component in enumerate(components(list(edge_rows)), 1):
            split_counts = Counter(Path(image).parts[1] for image in component)
            if len(split_counts) < 2:
                continue
            nums = [frame_number(image) for image in component]
            nums = [value for value in nums if value is not None]
            reviewed_edges = sum(
                1
                for (a, b), row in edge_rows.items()
                if a in component and b in component
                and row["visual_review"] == "high_confidence_leakage"
            )
            non_train = [image for image in component if Path(image).parts[1] != "train"]
            groups.append({
                "source": source,
                "group_id": f"{source}_{iteration}_{'asset' if source == 'd6' else 'sequence'}_{number:04d}",
                "group_type": group_type,
                "n_images": len(component),
                "n_train": split_counts.get("train", 0),
                "n_val": split_counts.get("val", 0),
                "n_test": split_counts.get("test", 0),
                "splits": ";".join(sorted(split_counts)),
                "min_frame_id": min(nums) if nums else "",
                "max_frame_id": max(nums) if nums else "",
                "n_visually_confirmed_edges": reviewed_edges,
                "status": "high_confidence_cross_split" if reviewed_edges else "possible_cross_split",
                "images": ";".join(component),
                "potential_move_candidates": ";".join(non_train),
                "recommendation": (
                    "keep the whole asset/sequence in one split; inspect labels/class balance before choosing destination"
                ),
            })
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", default="r1")
    parser.add_argument("--high-d6", type=int, default=VISUAL_TOP_N)
    parser.add_argument("--high-d7", type=int, default=VISUAL_TOP_N)
    parser.add_argument("--high-d6-ids-file")
    parser.add_argument("--high-d7-ids-file")
    args = parser.parse_args()
    all_rows = {source: read_rows(source) for source in ("d6", "d7")}
    high_counts = {"d6": args.high_d6, "d7": args.high_d7}
    high_id_files = {"d6": args.high_d6_ids_file, "d7": args.high_d7_ids_file}
    for source, rows in all_rows.items():
        ids_path = high_id_files[source]
        high_ids = None
        if ids_path:
            high_ids = {
                line.strip()
                for line in Path(ids_path).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
        apply_visual_review(source, rows, high_counts[source], high_ids)
        write_rows(source, rows)

    groups = build_groups(all_rows, args.iteration)
    group_fields = list(groups[0]) if groups else [
        "source", "group_id", "group_type", "n_images", "n_train", "n_val", "n_test",
        "splits", "min_frame_id", "max_frame_id", "n_visually_confirmed_edges", "status",
        "images", "potential_move_candidates", "recommendation",
    ]
    with (OUT / "suspected_sequence_groups.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(groups)

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for source_summary in summary["sources"]:
        source = source_summary["source"]
        rows = all_rows[source]
        source_groups = [group for group in groups if group["source"] == source]
        source_summary["n_high_confidence_suspected_duplicates"] = sum(
            row["visual_review"] == "high_confidence_leakage" for row in rows
        )
        source_summary["n_medium_confidence_pairs"] = sum(
            row["visual_review"] == "possible_leakage" for row in rows
        )
        source_summary["n_visually_not_leakage"] = sum(
            row["visual_review"] == "visually_not_leakage" for row in rows
        )
        source_summary["n_pairs_visually_reviewed"] = sum(
            row["reviewed_visually"] == "true" for row in rows
        )
        source_summary["n_cross_split_groups"] = len(source_groups)
        source_summary["n_high_confidence_cross_split_groups"] = sum(
            group["status"] == "high_confidence_cross_split" for group in source_groups
        )
        source_summary["n_unique_images_in_high_confidence_pairs"] = len({
            image
            for row in rows
            if row["visual_review"] == "high_confidence_leakage"
            for image in (row["image_a"], row["image_b"])
        })
        source_summary["visual_review_policy"] = (
            "The top 100 ranked pairs were reviewed side-by-side. Counts labelled high confidence are limited "
            "to those reviewed pairs; lower-ranked metric positives remain possible leakage."
        )

    summary["suspected_sequence_groups"] = len(groups)
    summary["review_status"] = "top_100_pairs_per_source_visually_reviewed"
    summary["contact_sheet_reviewed"] = {
        "d6": "d6_top_phash_pairs.jpg (100 pairs)",
        "d7": "d7_top_phash_pairs.jpg (100 pairs)",
    }
    summary["classification_categories"] = [
        "high_confidence_leakage",
        "possible_leakage",
        "visually_not_leakage",
    ]
    summary["interpretation_warning"] = (
        "Pair counts are not independent cases: one sequence or duplicated source asset can create many pairs. "
        "Use suspected_sequence_groups.csv for group-level remediation."
    )
    summary["exact_hash_cross_split_groups_in_d6_d7"] = 0
    summary["filename_id_assessment"] = {
        "d6": (
            "Heavy_Equipment and mixer filename numbers behave like scraped-item indices, not trustworthy "
            "camera/sequence timestamps."
        ),
        "d7": (
            "The seven-digit numeric token is useful as an identifier but is not reliably chronological: "
            "visually identical fixed-camera scenes recur at widely separated numbers. Sequence groups are "
            "therefore similarity-graph clusters, not groups invented from filename proximity alone."
        ),
    }
    summary["group_id_iteration"] = args.iteration
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
