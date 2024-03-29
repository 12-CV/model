import cv2
import numpy as np
import os
import onnxruntime as ort
import math
import json

from utils.da_transform import load_image

def center_square(image):
    # 이미지 크기 확인
    height, width, _ = image.shape

    # 가로와 세로 중 짧은 길이 결정
    min_dim = min(height, width)

    # 정사각형으로 이미지 자르기
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    end_x = start_x + min_dim
    end_y = start_y + min_dim
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def inference_yw(image, session) -> list:
    height, width = image.shape[:2]
    if width != 640 and height == 640:
        raise Exception("이미지 크기를 640x640으로 맞춰주세요.")
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: image})

    class_ids = outputs[0][0]
    bbox = outputs[1][0]
    scores = outputs[2][0]
    additional_info = outputs[3][0]
    score_threshold = [0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01]

    metadata = []

    for i, score in enumerate(scores):
        if additional_info[i] >= 0:
            if score > score_threshold[additional_info[i]]:
                metadata.append(bbox[i].tolist() + [int(additional_info[i])])
    
    return metadata

def inference_da(image, session):
    image, (orig_h, orig_w) = load_image(image)
    depth = session.run(None, {"image": image})[0]
    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
    
    return depth

def update_figure(metadata, frame):
        is_danger = [False] * 9

        for bbox in metadata:
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                     
            mean_point = bbox[7] 
            middle_point = bbox[8]
            x = bbox[9]
            y = bbox[10]
            rad = bbox[11]
            distance = bbox[12]

            if class_id < 9 and distance < 5:
                is_danger[class_id] = True

        return is_danger

# 모델 경로에 모델이 있어야함
session_yw = ort.InferenceSession('./models/yolow-l.onnx', providers=['CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider'])
session_da = ort.InferenceSession("./models/depth_anything_vits14.onnx", providers=['CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider'])

# 비디오 혹은 비디오의 폴더 경로
video_path = 'PATH/TO/DATASET'

# 결과 값을 저장할 경로
outdir = 'PATH/TO/SAVE'

if os.path.isfile(video_path):
    if video_path.endswith('txt'):
        with open(video_path, 'r') as f:
            lines = f.read().splitlines()
    else:
        filenames = [video_path]
else:
    filenames = os.listdir(video_path)
    filenames = [os.path.join(video_path, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()

os.makedirs(outdir, exist_ok=True)

output = outdir + '/result.json'

file_json = dict()

for k, filename in enumerate(filenames):
    print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
    raw_video = cv2.VideoCapture(filename)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    
    filename = os.path.basename(filename)
    
    frame_number = 0
    danger_json = [None] # 0번 프레임은 없으므로 인덱스에 None을 채워 놓음

    while raw_video.isOpened():

        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        frame_number += 1

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        frame = center_square(frame)
        frame = cv2.resize(frame, (640, 640))
        
        bboxes = inference_yw(frame, session_yw)
        depth = inference_da(frame, session_da)

        infos = dict()
        metadata = []
        min_dist = 100
        for bbox in bboxes:
            bbox_depth_region = depth[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if bbox_depth_region.size == 0:
                continue

            median_point = np.median(bbox_depth_region)
            max_point = np.max(bbox_depth_region)                        
            mean_point = np.mean(bbox_depth_region)      
            middle_point = depth[int((bbox[1] + bbox[3]) / 2)][int((bbox[0] + bbox[2]) / 2)]
            bbox += [float(median_point), float(max_point), float(mean_point), float(middle_point)]

            # 클라이언트 렌더링을 서버에서 하도록 설정
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            corr = 15

            # 객체 x축의 중간값을 각도로 수정 (-45 ~ 45도)
            r = ((x1 + x2) / 2 - 320) * (45 / 320)
            r = math.radians(r)

            # median 값을 실제 거리로 근사
            median_depth = 21 - (median_point * 4 / 3) + corr
            
            # depth와 각도에 따른 x, y값
            x = median_depth * math.sin(r)
            y = median_depth * math.cos(r) - corr

            # x축 너비의 따른 원 크기 조절
            rad = (x2 - x1) / 160

            distance = (x ** 2 + y ** 2) ** 0.5 - rad
            
            if distance < min_dist:
                min_dist = distance

            if y < 0 and distance < 10:
                y = -np.log(-y + 1) + 0.7

            bbox.extend([x, y, rad, distance])

            metadata.append(bbox)

        is_danger = update_figure(metadata, frame)

        infos['metadata'] = metadata
        infos['is_danger'] = is_danger
        infos['min_dist'] = min_dist
        danger_json.append(infos)

    file_json[filename] = danger_json

with open(output, "w") as outfile:
    json.dump(file_json, outfile)