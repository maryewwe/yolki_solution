from pathlib import Path
from PIL import Image
from rfdetr import RFDETRMedium

MODEL_PATH = Path("final_model.pth")
THRESHOLD = 0.25

model = RFDETRMedium(
    pretrain_weights=str(MODEL_PATH),
    resolution=640,
    num_classes=13,
)

def detect(image_path: str, threshold: float = THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    det = model.predict(image, threshold=threshold)
    rows = []
    if len(det) == 0:
        return rows
    names = det.data["class_name"]
    for box, score, raw_name in zip(det.xyxy, det.confidence, names):
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        x1, y1, x2, y2 = map(float, box)
        rows.append({
            "class": name,
            "confidence": float(score),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return rows

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python inference_example.py path/to/image.jpg")
    print(json.dumps(detect(sys.argv[1]), indent=2, ensure_ascii=False))
