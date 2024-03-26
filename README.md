# 단안 카메라 충돌 방지를 위한 거리 출력 시스템

### [Main Project Repository](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-12)



![거리출력](https://github.com/12-CV/model/assets/90448406/e385be39-ed12-45ba-b011-af73849c72ef)

<img src="https://github.com/12-CV/model/assets/90448406/ac8bb055-fcfb-481b-ad21-4314c14affd7" alt="img" style="width:530px;">


### Used Model 
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)


### Dependency
- torch
- torchvision
- opencv-python
- ultralytics
- depth_anything


## Usage
### settings
```
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
```
Depth-Anything 폴더에 calc_depth.py와 calc_func.py 파일을 위치시킨다.

### 실제 거리 출력 방법

1. 실제 거리를 측정한 이미지들을 하나의 폴더에 저장한다. (이미지 최소 10장 이상)
2. 이미지 명을 {실제거리(cm)_이름}으로 저장한다. 
3. calc_depth.py 에서 이미지 폴더 경로를 지정하고 실행한다.
4. 저장된 이미지를 확인하여 정상적으로 bbox와 depth가 출력되었는지 확인한다.
5. 저장된 output.txt를 calc_func.py를 실행하여 depth-distance 근사식을 출력한다.
6. 저장된 plot을 확인하여 정상 출력을 확인한다.

   ![final_graph](https://github.com/12-CV/model/assets/90448406/86e7a607-542b-4c17-9d81-fc25edcb1a18)


## Metric
$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = 0.3071$$

## ETC
https://huggingface.co/spaces/stevengrove/YOLO-World

해당 사이트를 통해 Prompt를 직접 조정하여 모델을 export 할 수 있음

