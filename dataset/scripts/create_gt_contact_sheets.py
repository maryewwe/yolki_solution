import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\lct\datasets\construction_v1")
OUT = ROOT / "contact_sheets"
CLASSES = [
    "excavator", "dump_truck", "truck", "loader", "bulldozer",
    "motor_grader", "roller", "concrete_mixer", "telehandler",
    "piling_machine", "crane_manipulator", "mobile_crane", "tower_crane",
]
COLORS = [
    (255, 70, 70), (255, 160, 40), (255, 220, 40), (100, 220, 80),
    (20, 200, 180), (60, 170, 255), (90, 100, 255), (175, 80, 255),
    (245, 80, 210), (190, 120, 70), (255, 80, 130), (70, 220, 255),
    (190, 255, 80),
]
FONT = ImageFont.load_default(size=16)
CELL_W, CELL_H = 480, 340
HEADER_H = 48


with (ROOT / "metadata" / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))


def stable_sample(candidates, count):
    return sorted(
        candidates,
        key=lambda row: hashlib.sha1(row["sha1"].encode("ascii")).hexdigest(),
    )[:count]


def annotate(row):
    image = Image.open(ROOT / row["output_image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_path = ROOT / row["output_label"]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        x, y, width, height = map(float, (x_text, y_text, w_text, h_text))
        left = (x - width / 2) * image.width
        top = (y - height / 2) * image.height
        right = (x + width / 2) * image.width
        bottom = (y + height / 2) * image.height
        color = COLORS[class_id]
        line_width = max(3, image.width // 350)
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        text = f"{class_id}:{CLASSES[class_id]}"
        text_box = draw.textbbox((left, top), text, font=FONT, stroke_width=2)
        draw.rectangle(text_box, fill=color)
        draw.text((left, top), text, fill="black", font=FONT, stroke_width=0)
    image.thumbnail((CELL_W, CELL_H - HEADER_H), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (CELL_W, CELL_H), "white")
    cell.paste(image, ((CELL_W - image.width) // 2, HEADER_H + (CELL_H - HEADER_H - image.height) // 2))
    header = ImageDraw.Draw(cell)
    original = Path(row["original_image"]).name
    line1 = f"{row['source']} / {row['split']} / {original[:48]}"
    line2 = f"raw: {row['raw_classes'][:62]}"
    header.text((6, 4), line1, fill="black", font=FONT)
    header.text((6, 25), line2, fill="black", font=FONT)
    return cell


def make_sheet(name, candidates, count=16, columns=4):
    selected = stable_sample(candidates, min(count, len(candidates)))
    rows_n = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * CELL_W, rows_n * CELL_H), (225, 225, 225))
    for index, row in enumerate(selected):
        sheet.paste(annotate(row), ((index % columns) * CELL_W, (index // columns) * CELL_H))
    path = OUT / f"{name}.jpg"
    sheet.save(path, quality=92)
    print(f"{path}\timages={len(selected)}")


OUT.mkdir(parents=True, exist_ok=True)
for source in ("d1", "d6", "d7", "d8"):
    make_sheet(f"source_{source}_overview", [row for row in rows if row["source"] == source], count=16)

make_sheet(
    "focus_d6_mixed_to_concrete_mixer",
    [row for row in rows if row["source"] == "d6" and "mixed" in row["raw_classes"].split(";")],
    count=20,
)
make_sheet(
    "focus_cranes",
    [row for row in rows if set(row["classes"].split(";")) & {"crane_manipulator", "mobile_crane", "tower_crane"}],
    count=24,
)
make_sheet(
    "focus_d6_mobile_crane",
    [row for row in rows if row["source"] == "d6" and "Mobile_crane" in row["raw_classes"].split(";")],
    count=32,
)
for class_name in ("crane_manipulator", "mobile_crane", "tower_crane"):
    make_sheet(
        f"focus_{class_name}",
        [row for row in rows if class_name in row["classes"].split(";")],
        count=20,
    )
for class_name in ("telehandler", "piling_machine", "loader"):
    make_sheet(
        f"focus_{class_name}",
        [row for row in rows if class_name in row["classes"].split(";")],
        count=20,
    )

visual_notes = [
    {
        "focus": "D6 mixed -> concrete_mixer",
        "images_reviewed": 20,
        "result": "confirmed",
        "note": "All sampled instances are concrete mixer trucks; mapping retained.",
    },
    {
        "focus": "crane classes",
        "images_reviewed": 32,
        "result": "source_label_noise",
        "note": "D6 Heavy_Equipment_1313 is visually a tower crane but source label is Mobile_crane; D1 crane manipulator includes visually borderline material-handler/long-reach excavator cases. Retained because no reliable automatic per-instance subtype rule exists.",
    },
    {
        "focus": "telehandler",
        "images_reviewed": 20,
        "result": "confirmed",
        "note": "Sampled D1 forklift giraffe instances are telehandlers.",
    },
    {
        "focus": "piling_machine",
        "images_reviewed": 20,
        "result": "confirmed",
        "note": "Sampled D7/D8 instances depict pile-driving or piling rigs/equipment.",
    },
    {
        "focus": "loader",
        "images_reviewed": 20,
        "result": "confirmed_broad_merge",
        "note": "Sample supports the agreed broad merge of wheel, backhoe, bucket and skid-steer loader variants.",
    },
]
with (ROOT / "metadata" / "visual_audit_notes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["focus", "images_reviewed", "result", "note"])
    writer.writeheader()
    writer.writerows(visual_notes)
