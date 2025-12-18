import cv2
import os

# -------------------------------------------------------
# 설정
# -------------------------------------------------------
image_folder = "./video_realesrgan_final"  # 이미지들이 들어있는 폴더
output_video = "./highvideo.mp4"  # 생성될 영상 경로
fps = 30  # 초당 프레임 수
# -------------------------------------------------------

# 이미지 파일들 로드
images = sorted([img for img in os.listdir(image_folder)
                 if img.endswith((".png", ".jpg", ".jpeg"))])

# 첫 이미지 크기 확인
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# 비디오 라이터 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 이미지 하나씩 영상에 추가
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print("🎉 영상 생성 완료! →", output_video)
