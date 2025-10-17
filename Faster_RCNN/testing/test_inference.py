import cv2
import torch
import yaml
from model.faster_rcnn import faster_rcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load config ---
with open("Faster_RCNN/config/voc.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset_params"]
model_config = config["model_params"]

# --- 2. Recreate model architecture ---
model = faster_rcnn(model_config, num_classes=dataset_config["num_classes"])
model.to(device)

# --- 3. Load trained weights ---
checkpoint_path = "Faster_RCNN/voc/faster_rcnn_voc2007.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded and ready for inference on", device)

# --- 4. Load video ---
cap = cv2.VideoCapture("Faster_RCNN/white_car.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process only first 10 seconds
max_frames = int(fps * 10)
# Skip every Nth frame
skip_interval = 2  # process every 2nd frame

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

frame_count = 0
processed_count = 0

# --- 5. Inference loop ---
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Skip frames for speed
    if frame_count % skip_interval != 0:
        continue

    processed_count += 1

    # (Optional) resize for speed — uncomment if needed
    # frame = cv2.resize(frame, (640, 480))

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img = img.to(device)

    with torch.no_grad():
        rpn_output, frcnn_output = model(img)

        # Visualization
        if frcnn_output and "boxes" in frcnn_output:
            boxes = frcnn_output["boxes"].cpu().numpy()
            scores = frcnn_output["scores"].cpu().numpy()

            for box, score in zip(boxes, scores):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(
    f"✅ Processed {processed_count} frames out of {frame_count} ({frame_count / fps:.2f}s)."
)
