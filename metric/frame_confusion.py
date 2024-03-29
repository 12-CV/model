import cv2
import numpy as np
import os
import cv2
import numpy as np
from utils.da_transform import load_image

import json

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

# 파일 경로 지정
input_file_path = "PATH/TO/JSON"

# JSON 파일 불러오기
with open(input_file_path, "r") as infile:
    YC_data = json.load(infile)

# Ground Truth 파일 경로 (txt)
txt_path = 'PATH/TO/GT'

sumary = [0, 0, 0, 0] # 총 [TP, FP, FN, TN]
for filename in YC_data.keys():
    print(filename)

    YC = YC_data[filename]

    # txt에서 GT값을 bool list로 불러옴
    GT = [False] * len(YC)
    # 텍스트 파일 불러오기
    with open(txt_path + '/' + filename[:-4] + '.txt', "r") as file:
        for line in file:
            # 쉼표로 분리하여 데이터 추출
            data = line.strip().split(',')
            if len(data) < 3:
                print('잘못된 GT: ', data)
                continue
            for i in range(int(data[1]), int(data[2])):
                GT[i] = True

    # 현재 영상의 (TP, FP, FN, TN) 계산
    check = [0, 0, 0, 0]
    flow = ['s']
    for i in range(1, len(YC)):
        # CASE TP
        if GT[i] and any(YC[i]['is_danger']):
            check[0] += 1
            flow.append(0)
        if not GT[i] and any(YC[i]['is_danger']):
            check[1] += 1
            flow.append(1)
        if GT[i] and not any(YC[i]['is_danger']):
            check[2] += 1
            flow.append(2)
        if not GT[i] and not any(YC[i]['is_danger']):
            check[3] += 1
            flow.append(3)
    flow.append('e')

    # flow는 's'부터 'e'까지 프레임 별 예측 결과 상태
    print('총 프레임 수: ', len(flow) - 2)

    TP = check[0]
    FP = check[1]
    FN = check[2]
    TN = check[3]
    sumary[0] += TP
    sumary[1] += FP
    sumary[2] += FN
    sumary[3] += TN

    if TP == 0:
        print('NO TP')
        print('-' * 50)
        continue

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    print('Precision: ', Precision)
    print('Recall: ', Recall)
    print('Accuracy: ', (TP + TN) / sum(check))
    print('F1: ', 2 * Precision * Recall / (Precision + Recall))
    print('-' * 50)

print(sumary)
TP = sumary[0]
FP = sumary[1]
FN = sumary[2]
TN = sumary[3]
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

print('Precision: ', Precision)
print('Recall: ', Recall)
print('Accuracy: ', (TP + TN) / sum(sumary))
print('F1: ', 2 * Precision * Recall / (Precision + Recall))