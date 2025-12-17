# 03_codeformer_facesr.md — CodeFormer (FaceSR) + Face Crop + Merge

이 단계는 SwinIR x4로 해상도를 올린 프레임(`video_swin2sr_x4`)을 입력으로 받아,
**얼굴 검출 → 얼굴 crop(`cropped_faces`) → 얼굴 복원(`restored_faces`) → 원본 프레임에 재합성(`final_results`)**
까지 한 번에 수행한다.

> `inference_codeformer.py` 실행 과정 내부에서 자동으로 얼굴을 crop한다.

---

## 0) 입력/출력 경로

### Input (SwinIR x4 결과 프레임)
- `C:\Users\hcwon\KEEP\video_swin2sr_x4`

### Output (CodeFormer 결과 폴더)
- `C:\Users\hcwon\KEEP\video_swin2sr_x4_faceSR`

출력 폴더 내부에 아래 하위 폴더가 생성된다(자동 생성):
- `cropped_faces/` : 얼굴 crop 이미지
- `restored_faces/` : 복원된 얼굴 이미지
- `final_results/` : 얼굴이 원본 프레임에 합성된 최종 프레임(후속 Real-ESRGAN 입력)

---

## 1) 환경 준비

Anaconda Prompt에서:

```bat
conda activate keep_py310
cd /d C:\Users\hcwon\CodeFormer

---

## 2) 실행 명령 (Face Crop + Face Restoration)

아래 명령을 실행하면 얼굴 검출 → crop → 복원 → 재합성이 자동으로 수행된다.

```bat
python inference_codeformer.py ^
-i "C:\Users\hcwon\KEEP\video_swin2sr_x4" ^
-o "C:\Users\hcwon\KEEP\video_swin2sr_x4_faceSR" ^
-w 0.95 ^
--bg_upsampler none ^
--face_upsample
