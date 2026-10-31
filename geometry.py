"""Planar calibration. All AI coordinates are explicitly [x,y], 0..1000."""
import json
import math
import cv2
import numpy as np


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} は有限の数値にしてください')
    return float(value)


class Calibration:
    def __init__(self, cfg):
        self.cfg = cfg
        self.src = np.asarray(cfg['image_points'], np.float32)
        dst = np.asarray(cfg['robot_points'], np.float32)
        if self.src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError('キャリブレーションには4点ずつ必要です')
        self.H = cv2.getPerspectiveTransform(self.src, dst)
        if not np.isfinite(self.H).all() or abs(np.linalg.det(self.H)) < 1e-12:
            raise ValueError('キャリブレーション点が不正です')
        self.hull = cv2.convexHull(self.src)

    def xy(self, p):
        v = self.H @ [*p, 1.0]
        if abs(v[2]) < 1e-9:
            raise ValueError('座標変換できない点です')
        return v[:2] / v[2]

    def convert(self, item):
        if not isinstance(item, dict) or not isinstance(item.get('label'), str):
            raise ValueError('label が必要です')
        label = item['label'].strip()
        if not label or len(label) > 120:
            raise ValueError('label は1〜120文字です')
        p = item.get('point')
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError('point は [x,y] の2要素です')
        x, y = [number(v, 'point') for v in p]
        a = number(item.get('closing_angle_deg'), 'closing_angle_deg')
        if not (0 <= x <= 1000 and 0 <= y <= 1000 and 0 <= a < 180):
            raise ValueError('point は0〜1000、角度は0以上180未満です')
        w, h = self.cfg['image_size']
        pixel = np.array([x*w/1000, y*h/1000])
        if cv2.pointPolygonTest(self.hull, tuple(pixel), False) < 0:
            raise ValueError(f'{label}: 4つの黒丸の内側を指定してください')
        xy = self.xy(pixel)
        # Local derivative of the homography, evaluated in original pixel aspect.
        v = self.H @ [*pixel, 1.0]
        J = (self.H[:2, :2]*v[2] - np.outer(v[:2], self.H[2, :2])) / v[2]**2
        t = math.radians(a)
        d = J @ [math.cos(t), math.sin(t)]
        horizontal = J @ [1., 0.]
        delta = math.degrees(math.atan2(d[1], d[0]) - math.atan2(horizontal[1], horizontal[0]))
        delta = (delta + 90) % 180 - 90
        rz = (self.cfg['horizontal_rz_deg'] + delta + 180) % 360 - 180
        # Equivalent Euler angle in [-180,180). Taught FIG is preserved by the driver.
        return {'label': label, 'point': [x, y], 'pixel': pixel.tolist(),
                'closing_angle_deg': a, 'angle_delta_deg': delta,
                'pose': [float(xy[0]), float(xy[1]), self.cfg['pick_z_mm'],
                         self.cfg['rx_deg'], self.cfg['ry_deg'], rz]}

    def plan(self, text):
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        raw = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if isinstance(raw, dict):
            raw = raw.get('targets', [raw] if 'point' in raw else None)
        if not isinstance(raw, list) or len(raw) > 20:
            raise ValueError('targets は最大20個の配列です')
        targets = [self.convert(p) for p in raw]
        for i, target in enumerate(targets):
            for prev in targets[:i]:
                if math.dist(target['pose'][:2], prev['pose'][:2]) < 3:
                    raise ValueError('同じ位置の重複指定があります。対象を確認してください')
        return targets
