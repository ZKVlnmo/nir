"""LoFTR + гомография + параллакс. Максимально просто."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
from PIL import Image, ImageOps


def load_image(path, max_side=960):
    """Читает изображение, уменьшает, переводит в серое."""
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    nw, nh = round(w * scale), round(h * scale)
    gray = img.convert('L').resize((nw, nh))
    # Добить до кратности 8
    data = np.asarray(gray, dtype=np.float32) / 255.0
    data = np.pad(data, ((0, (-nh) % 8), (0, (-nw) % 8)), mode='edge')
    return img, data, (nw / w, nh / h)


def find_matches(path0, path1, max_side=960, device='cpu'):
    """LoFTR: находит соответствия между двумя кадрами."""
    img0, a, s0 = load_image(path0, max_side)
    img1, b, s1 = load_image(path1, max_side)

    loftr = LoFTR(pretrained='outdoor').to(device).eval()

    with torch.inference_mode():
        out = loftr({
            'image0': torch.from_numpy(a)[None, None].to(device),
            'image1': torch.from_numpy(b)[None, None].to(device),
        })

    p0 = out['keypoints0'].cpu().numpy()
    p1 = out['keypoints1'].cpu().numpy()
    conf = out['confidence'].cpu().numpy()

    # Вернуть в исходное разрешение
    p0 = (p0 + 0.5) / np.array(s0) - 0.5
    p1 = (p1 + 0.5) / np.array(s1) - 0.5

    return p0, p1, conf, img0, img1


def background_homography(p0, p1, objects, threshold=3.0):
    """Гомография по фону + остатки объектов.

    Фон = всё, что вне прямоугольников объектов.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)


    obj_mask = np.zeros(len(p0), dtype=bool)
    for obj in objects:
        x0, y0, x1, y1 = obj['box0']
        m = (p0[:, 0] >= x0) & (p0[:, 0] < x1) & (p0[:, 1] >= y0) & (p0[:, 1] < y1)
        obj_mask |= m


    bg = ~obj_mask


    H, inliers = cv2.findHomography(
        p0[bg], p1[bg],
        cv2.RANSAC,
        ransacReprojThreshold=threshold,
    )


    p0_h = np.hstack([p0, np.ones((len(p0), 1))])
    predicted = (p0_h @ H.T)
    predicted = predicted[:, :2] / predicted[:, 2:3]
    residual = p1 - predicted


    results = []
    for obj in objects:
        x0, y0, x1, y1 = obj['box0']
        m = (p0[:, 0] >= x0) & (p0[:, 0] < x1) & (p0[:, 1] >= y0) & (p0[:, 1] < y1)
        n = int(m.sum())

        item = {'id': obj['id'], 'matches': n}

        if n >= 5:
            vecs = residual[m]
            lengths = np.linalg.norm(vecs, axis=1)
            median_vec = np.median(vecs, axis=0)

            item['median_residual_px'] = median_vec.tolist()
            item['median_length_px'] = float(np.linalg.norm(median_vec))
            item['median_magnitude_px'] = float(np.median(lengths))

        results.append(item)

    return H, results, residual


def run(frame0, frame1, objects_file, output='result.json'):
    """Полный запуск: LoFTR → гомография → остатки."""

    objects = json.loads(Path(objects_file).read_text(encoding='utf-8'))


    print('LoFTR: поиск соответствий...')
    p0, p1, conf, _, _ = find_matches(frame0, frame1)


    keep = conf >= 0.5
    p0, p1 = p0[keep], p1[keep]
    print(f'Соответствий после фильтра: {len(p0)}')


    H, results, residual = background_homography(p0, p1, objects)


    out = {
        'homography': H.tolist(),
        'objects': results,
    }

    Path(output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == '__main__':

    run(
        frame0='frame0.jpg',
        frame1='frame1.jpg',
        objects_file='objects.json',
        output='result.json',
    )