# 얼굴 모자이크
해당 코드는 Depth_Anything모델 테스트에서 사용했던 얼굴 모자이크 코드입니다.

<img src="https://github.com/12-CV/model/assets/56228633/0d5b1378-60d0-4906-9555-886007440768" alt="img" style="width:530px;">

## Usage
### Settings
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

### Blur 코드 사용
1. video 폴더에 Blur 처리할 비디오 영상 삽입
2. blur.py의 video_name 변수에 Blur 처리할 비디오 영상 제목으로 변경하고 사용
```
cd blur
python blur.py
```
