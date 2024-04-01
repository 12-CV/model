# 단안 카메라 충돌 방지를 위한 거리 출력 시스템

## [Main Project Repository](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-12)
<img src="https://github.com/12-CV/model/assets/90448406/ac8bb055-fcfb-481b-ad21-4314c14affd7" alt="img" style="width:400px;">
<img src="https://github.com/12-CV/model/assets/90448406/e385be39-ed12-45ba-b011-af73849c72ef" alt="img" style="width:500px;">

## Used Model 
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
    - https://github.com/fabio-sim/Depth-Anything-ONNX
    - 해당 사이트에서 Depth-Anything 모델을 ONNX로 Export할 수 있음
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
    - https://huggingface.co/spaces/stevengrove/YOLO-World
    - 해당 사이트를 통해 Prompt를 직접 조정하여 YOLO-World 모델을 ONNX로 Export 할 수 있음

## Dependency
- torch
- torchvision
- opencv-python
- ultralytics
- depth_anything

## Directory
```
├── model
│   ├── distance
│   ├── metric
│   ├── utils
│   └── blur
```

## Usage
### 실제 거리 출력 방법(distance folder)

```
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
```
Depth-Anything 폴더에 distance 폴더 내부에 있는 calc_depth.py와 calc_func.py 파일을 위치시킨다.

1. 실제 거리를 측정한 이미지들을 하나의 폴더에 저장한다. (이미지 최소 10장 이상)
2. 이미지 명을 {실제거리(cm)_이름}으로 저장한다. 
3. calc_depth.py 에서 이미지 폴더 경로를 지정하고 실행한다.
4. 저장된 이미지를 확인하여 정상적으로 bbox와 depth가 출력되었는지 확인한다.
5. 저장된 output.txt를 calc_func.py를 실행하여 depth-distance 근사식을 출력한다.
6. 저장된 plot을 확인하여 정상 출력을 확인한다.

    <img src="https://github.com/12-CV/model/assets/90448406/86e7a607-542b-4c17-9d81-fc25edcb1a18" alt="img" style="width:400px;">


#### Absolute Distance MSE
- 추론된 거리 값이 실제 거리 값과 얼마나 유시한지를 확인하는 성능 지표

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2  = 0.3071m^2 $$

---
### 단안 카메라 충돌 방지 관련 Metric 계산(metric folder)
- 실제 충돌 방지 상황을 제대로 감지했는 지에 대한 성능 지표
- 영상의 모든 프레임에 대해 ‘충돌 위험 프레임’과 ‘충돌 위험이 없는 프레임’으로 나누어 실제 충돌 방지 상황과 예측 상황을 비교하여 성능을 측정

#### 코드 설명
- inference.py
   - 모델을 돌려 각종 모델 예측 값들을 json 파일로 저장하는 코드
- youtube_result.json
   - 위 코드를 13개 유튜브 영상에 대해 돌린 결과값
- frame_confusion.py
   - 프레임 단위로 confusion matrix 관련 결과 추출
- plot_dist.ipynb
   - 결과를 Box Plot, Violin Plot로 추출
- 주의
   - model 추가되지 않았으므로 back 레포에서 두개의 onnx 파일 추가 필요

#### 일반 영상(1542개의 데이터)에 대한 결과
| Precision | Recall | Accuracy | F1 |
| ------- | ------- | ------- | ------- |
| 0.8903 | 0.8160 | 0.8917 | 0.8516 |

#### 13개의 영상에 대한 결과
<img src="https://github.com/12-CV/model/assets/56228633/f35d473e-7244-47e3-a26b-e38cf3c881e5" alt="img" style="width:400px;">

---
### 얼굴 모자이크(blur folder)
해당 코드는 Depth_Anything 모델 테스트에서 사용했던 얼굴 모자이크 코드

<img src="https://github.com/12-CV/model/assets/56228633/0d5b1378-60d0-4906-9555-886007440768" alt="img" style="width:350px;">

#### Settings
기본 디렉토리 세팅
```
cd blur
mkdir video
mkdir predict_video
```

모델 압축 해제
```
cd ./pretrained_weights
./decompress_model.sh yolov8n-face.pt.7z
```

#### Blur 코드 사용
1. video 폴더에 Blur 처리할 비디오 영상 삽입
2. blur.py의 video_name 변수에 Blur 처리할 비디오 영상 제목으로 변경하고 사용
```
cd blur
python blur.py
```