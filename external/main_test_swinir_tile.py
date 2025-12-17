import os
import argparse
import glob

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from models.network_swinir import SwinIR


def load_swinir_model(model_path, scale, device):
    """
    SwinIR-M x4 RealSR (BSRGAN) 설정
    checkpoint: 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
    """
    # 공식 config 기준 (SwinIR-M x4)
    model = SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,                  # Medium 모델이라 180
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2.0,
        upsampler='nearest+conv',
        resi_connection='1conv',
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # checkpoint 구조에 따라 실제 state_dict 꺼내기
    if isinstance(checkpoint, dict):
        if 'params_ema' in checkpoint:
            # EMA 가중치가 있으면 이걸 우선 사용
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 'module.' 접두어 제거 (DDP 저장본 대비)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def img2tensor(img):
    """
    PIL RGB -> tensor [1, 3, H, W], range [0,1]
    """
    img = np.array(img).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def tensor2img(tensor):
    """
    tensor [1, 3, H, W], [0,1] -> PIL Image
    """
    tensor = tensor.clamp(0, 1)
    img = tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def infer_with_tile(model, img, scale, tile=None, tile_overlap=32, device='cuda'):
    """
    img: tensor [1,3,H,W] (0~1)
    tile: int, 타일 크기 (H,W 방향에서 한 변 길이)
    tile_overlap: 타일끼리 겹치는 픽셀 수
    """
    img = img.to(device)

    # 타일 분할 없이 한 번에 처리
    if tile is None or tile <= 0:
        out = model(img)
        return out

    b, c, h, w = img.shape
    tile = min(tile, h, w)
    stride = tile - tile_overlap
    if stride <= 0:
        stride = tile

    out_h = h * scale
    out_w = w * scale

    # 출력/가중치 버퍼
    output = torch.zeros(b, c, out_h, out_w, device=device)
    weight = torch.zeros_like(output)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)

            in_patch = img[:, :, y:y_end, x:x_end]
            out_patch = model(in_patch)

            out_y = y * scale
            out_x = x * scale
            out_y_end = out_y + (y_end - y) * scale
            out_x_end = out_x + (x_end - x) * scale

            output[:, :, out_y:out_y_end, out_x:out_x_end] += out_patch
            weight[:, :, out_y:out_y_end, out_x:out_x_end] += 1.0

    output /= weight
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--folder_lq', type=str, required=True)
    parser.add_argument('--folder_gt', type=str, required=True)  # output 폴더로 사용
    parser.add_argument('--tile', type=int, default=0)
    parser.add_argument('--tile_overlap', type=int, default=32)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(args)

    os.makedirs(args.folder_gt, exist_ok=True)

    # 모델 로드
    model = load_swinir_model(args.model_path, args.scale, device)

    # 입력 이미지 찾아오기
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_paths = []
    for e in exts:
        img_paths.extend(glob.glob(os.path.join(args.folder_lq, e)))
    img_paths = sorted(img_paths)

    print(f'Found {len(img_paths)} images in {args.folder_lq}')

    for idx, path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f'[{idx+1}/{len(img_paths)}] Processing {name}')

        img_lq = Image.open(path).convert('RGB')
        tensor_lq = img2tensor(img_lq)

        out = infer_with_tile(
            model,
            tensor_lq,
            scale=args.scale,
            tile=args.tile,
            tile_overlap=args.tile_overlap,
            device=device
        )

        img_sr = tensor2img(out)
        save_path = os.path.join(args.folder_gt, f'{name}_x{args.scale}.png')
        img_sr.save(save_path)

    print('Done.')


if __name__ == '__main__':
    main()
