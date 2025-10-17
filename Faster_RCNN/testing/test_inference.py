# --- 1. Imports ---
import torch
import cv2
import time
import yaml
from torchvision import transforms
from model.faster_rcnn import faster_rcnn   # class name is capitalized in your repo
from dataset.voc import VOCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Paths & settings ---
video_path   = "drone_car.mp4"
output_path  = "output.mp4"
config_path  = "config/voc.yaml"
max_duration = 100.0  # = 10 seconds

# --- 3. Load YAML config ---
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

dataset_cfg = cfg["dataset_params"]
model_cfg   = cfg["model_params"]
train_cfg   = cfg["train_params"]

# --- 4. Build model ---
model = faster_rcnn(model_cfg, num_classes=dataset_cfg["num_classes"]).to(device)
state = torch.load(
    "faster_rcnn_voc2007.pth",
    map_location=device
)

# Handle both wrapped and raw state dicts
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

model.load_state_dict(state, strict=False)
model.eval()
print("✅ Model loaded successfully on", device)

# --- 5. Video input/output setup ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

transform = transforms.ToTensor()
start_time = time.time()

# --- 6. Frame loop (10 seconds) ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if (time.time() - start_time) > max_duration:
        break

    img_tensor = transform(frame).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Your forward returns (rpn_output, frcnn_output)
        _, output = model(img_tensor, None)
        boxes  = output["boxes"]
        scores = output["scores"]

    # Draw boxesvoc/faster_rcnn_voc2007.pth
    for box, score in zip(boxes, scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    out.write(frame)

cap.release()
out.release()
print("🎥 Done! Saved annotated video to:", output_path)
