from ultralytics import YOLO
import cv2

model = YOLO("./pretrained_weights/yolov8n-face.pt")

video_name = "{비디오 영상 제목}.mp4"
video_path = f"./video/{video_name}"
output_path = f"./predict_video/{video_name}"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 0.05
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1

            roi = frame[y1:y2, x1:x2]
            new_w = int(max(1, scale * w))
            new_h = int(max(1, scale * h))
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()