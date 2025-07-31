import cv2
from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np
import boto3
import os
import shutil
import boto3

# === Configuration ===
s3_bucket = "nextstairs-tv-data"
s3_video_key = "videos/fc_tokyo_5-10.mp4"

local_video_path = "/tmp/fc_tokyo_video.mp4"
output_json_path = "/tmp/logo_text_results.json"
output_frames_dir = Path("/tmp/output_frames_fc_tokyo")

json_s3_key = "results/logo_text_results_home.json"
frames_s3_prefix = "results/frames_home/"

# === Setup AWS S3 client ===
s3 = boto3.client("s3")

try:
    # === Step 1: Download Video from S3 ===
    print("⬇️ Downloading video from S3...")
    s3.download_file(s3_bucket, s3_video_key, local_video_path)

    # === Step 2: Load YOLO Model ===
    yolo_model = YOLO("runs/detect/train9/weights/best.pt")
    print("✅ Model loaded with classes:", yolo_model.names)

    # === Step 3: Video Setup ===
    output_frames_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    saved_idx = 0
    results_data = []
    confidence_threshold = 0.5
    def compute_area(bbox, frame_width, frame_height):
        x1, y1, x2, y2 = bbox
        return ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

    print("🚀 Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            detections = yolo_model(frame)[0]

            frame_result = {
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "logos": [],
                "texts": []
            }

            for det in detections.boxes:
                xyxy = det.xyxy[0].tolist()
                conf = float(det.conf[0])
                cls = int(det.cls[0])
                label = yolo_model.names[cls]
                # if conf < confidence_threshold:
                #     continue
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                percent_area = compute_area(xyxy, width, height) * 100

                frame_result["logos"].append({
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,
                    "area": percent_area
                })

            frame_file = output_frames_dir / f"frame_{saved_idx}.jpg"
            cv2.imwrite(str(frame_file), frame)
            results_data.append(frame_result)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    # === Step 4: Save JSON ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2, default=convert_numpy)
    print(f"📄 Saved JSON results: {output_json_path}")

    # === Step 5: Upload Outputs to S3 ===
    print("⬆️ Uploading JSON to S3...")
    s3.upload_file(output_json_path, s3_bucket, json_s3_key)

    print("⬆️ Uploading frames to S3...")
    for frame_file in output_frames_dir.glob("*.jpg"):
        frame_s3_key = f"{frames_s3_prefix}{frame_file.name}"
        s3.upload_file(str(frame_file), s3_bucket, frame_s3_key)

    print("✅ Upload complete.")

finally:
    # === Step 6: Cleanup ===
    if os.path.exists(local_video_path):
        os.remove(local_video_path)
        print(f"🧹 Deleted: {local_video_path}")

    if os.path.exists(output_json_path):
        os.remove(output_json_path)
        print(f"🧹 Deleted: {output_json_path}")

    if output_frames_dir.exists():
        shutil.rmtree(output_frames_dir)
        print(f"🧹 Deleted: {output_frames_dir}")

    print("✅ All temp files cleaned.")
