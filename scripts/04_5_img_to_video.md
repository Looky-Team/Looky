## Image → Video (Post-Processing)

- Real-ESRGAN으로 복원된 프레임들을
- 프레임 순서대로 정렬하여
- MP4 영상으로 재구성

### Script
`external/imgtovideo.py`

### Input
- ./video_realesrgan_final/*.png

### Output
- highvideo.mp4

### Note
- 이후 얼굴 검출 및 타겟 추적의 입력 영상으로 사용
