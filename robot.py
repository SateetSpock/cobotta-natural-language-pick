"""COBOTTA b-CAP adapter. NEXT + polling allows cooperative stop."""
import json
import math
import time
from ctypes import c_float
from datetime import datetime
from pathlib import Path
from bcap import BCAPClient


class Stopped(RuntimeError):
    pass


def check_stop(stop):
    if stop.is_set():
        raise Stopped('停止要求を受け付けました')


def sequence(robot, targets, stop, log):
    """Robot is at P10 with open fingers. No AI call or retries inside a cycle."""
    for i, target in enumerate(targets, 1):
        check_stop(stop)
        log(f'{i}/{len(targets)}: {target["label"]}')
        robot.set_pick(target['pose'])
        for n, linear in [(11, False), (12, False), (13, True)]:
            check_stop(stop)
            log(f'P{n}へ移動')
            robot.move(n, linear)
        check_stop(stop)
        log('ハンド把持')
        robot.grip()
        for n, linear in [(12, True), (14, False), (15, False)]:
            check_stop(stop)
            log(f'P{n}へ移動')
            robot.move(n, linear)
        check_stop(stop)
        log('ハンド開放 30 mm')
        robot.open_hand()
        check_stop(stop)
        log('P10へ戻る')
        robot.move(10)


class Robot:
    def __init__(self, cfg, stop, log, root):
        self.cfg, self.stop, self.log, self.root = cfg, stop, log, Path(root)
        self.client = self.ctrl = self.arm = None
        self.handles = {}
        self.points = {}
        self.taken = False
        self.force = None

    def connect(self):
        self.client = BCAPClient(self.cfg['robot_ip'], self.cfg['robot_port'], 5.0)
        self.client.service_start()
        self.ctrl = self.client.controller_connect('', 'CaoProv.DENSO.VRC', 'localhost', '')
        self.arm = self.client.controller_getrobot(self.ctrl, 'Arm', '')
        self.points = {n: self.read(f'P{n}') for n in range(10, 16)}
        for n, p in self.points.items():
            if not isinstance(p, (list, tuple)) or len(p) != 7 or not all(math.isfinite(v) for v in p):
                raise ValueError(f'P{n}は7要素のポジション型ではありません')
        backup = self.root / 'data' / ('taught_points_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        backup.parent.mkdir(exist_ok=True)
        backup.write_text(json.dumps(self.points, indent=2), encoding='utf-8')
        return {'points': self.points, 'tool': self.read('@CURRENT_TOOL', True),
                'work': self.read('@CURRENT_WORK', True),
                'current_pose': self.read('@CURRENT_POSITION', True)}

    def var(self, name, robot=False):
        key = (name, robot)
        if key not in self.handles:
            fn = self.client.robot_getvariable if robot else self.client.controller_getvariable
            self.handles[key] = fn(self.arm if robot else self.ctrl, name, '')
        return self.handles[key]

    def read(self, name, robot=False):
        return self.client.variable_getvalue(self.var(name, robot))

    def healthy(self):
        if self.read('@ERROR_CODE') != 0:
            raise RuntimeError(f'コントローラーエラー: {self.read("@ERROR_CODE")}')
        if self.read('@EMERGENCY_STOP'):
            raise RuntimeError('非常停止中です')

    def prepare(self, tool, work, force):
        check_stop(self.stop)
        if not 0 < force <= 20:
            raise ValueError('把持力は0より大きく20 N以下にしてください（使用ハンドの仕様内）')
        if self.read('@MODE') != 3:
            raise RuntimeError('コントローラーを自動モードにしてください')
        if self.read('@LOCK'):
            raise RuntimeError('マシンロック中です')
        self.healthy()
        if (self.read('@CURRENT_TOOL', True), self.read('@CURRENT_WORK', True)) != (tool, work):
            raise RuntimeError('教示時のTool / Work番号と現在値が一致しません')
        self.client.robot_execute(self.arm, 'TakeArm', [0, 1])
        self.taken = True
        self.force = float(force)
        # Keep=1 preserves the calibrated Tool and Work; do not silently reset them.
        self.client.robot_speed(self.arm, -1, self.cfg['robot_speed_percent'])
        self.client.robot_speed(self.arm, 0, self.cfg['robot_speed_percent'])
        check_stop(self.stop)
        self.client.robot_execute(self.arm, 'Motor', [1, 0])

    def verify_points(self):
        for n in (10, 11, 14, 15):
            if self.read(f'P{n}') != self.points[n]:
                raise RuntimeError(f'P{n}が接続後に変更されています。再接続・再撮影してください')

    def at(self, target):
        p = self.read('@CURRENT_POSITION', True)
        return (math.dist(p[:3], target[:3]) <= 0.5 and
                all(abs((p[i]-target[i]+180) % 360-180) <= 0.5 for i in range(3, 6)))

    def move(self, n, linear=False):
        check_stop(self.stop)
        target = self.read(f'P{n}')
        self.client.robot_move(self.arm, 2 if linear else 1, f'@E P{n}', 'NEXT')
        deadline = time.monotonic() + self.cfg['move_timeout_seconds']
        while True:
            if self.stop.wait(.08):
                raise Stopped('移動停止要求')
            self.healthy()
            if not self.read('@BUSY_STATUS', True) and self.at(target):
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f'P{n}到達確認がタイムアウトしました')

    def wait_hand(self, closing=False):
        deadline = time.monotonic() + self.cfg['hand_timeout_seconds']
        while True:
            if self.stop.wait(.08):
                raise Stopped('ハンド停止要求')
            self.healthy()
            busy = self.client.controller_execute(self.ctrl, 'HandBusyState')
            holding = self.client.controller_execute(self.ctrl, 'HandHoldState')
            if closing and holding:
                return
            if not busy:
                if closing:
                    raise RuntimeError('把持を確認できません。持ち上げを中止しました')
                pos = self.client.controller_execute(self.ctrl, 'HandCurPos')
                if abs(float(pos)-self.cfg['open_mm']) > .5:
                    raise RuntimeError('ハンド開放位置に到達していません')
                return
            if time.monotonic() > deadline:
                raise TimeoutError('ハンド動作がタイムアウトしました')

    def open_hand(self):
        check_stop(self.stop)
        self.client.controller_execute(self.ctrl, 'HandMoveA',
                                       [float(self.cfg['open_mm']), c_float(self.cfg['hand_speed_percent']), 'NEXT'])
        self.wait_hand()

    def grip(self):
        check_stop(self.stop)
        self.client.controller_execute(self.ctrl, 'HandMoveH', [float(self.force), True, 'NEXT'])
        self.wait_hand(True)

    def set_pick(self, pose):
        check_stop(self.stop)
        p13 = [float(v) for v in pose] + [float(self.points[13][6])]
        p12 = p13.copy()
        p12[2] += self.cfg['lift_mm']
        for n, p in [(12, p12), (13, p13)]:
            check_stop(self.stop)
            self.client.variable_putvalue(self.var(f'P{n}'), p)
            if any(abs(a-b) > .001 for a, b in zip(p, self.read(f'P{n}'))):
                raise RuntimeError(f'P{n}の書込み確認に失敗しました')

    def capture(self):
        import cv2
        import numpy as np
        check_stop(self.stop)
        camera = variable = None
        try:
            camera = self.client.controller_connect('N10-W02', 'CaoProv.Canon.N10-W02', '',
                        f'Conn=eth:{self.cfg["camera_ip"]}, Timeout=3000')
            self.client.controller_execute(camera, 'OneShotFocus', '')
            variable = self.client.controller_getvariable(camera, 'IMAGE', '')
            data = self.client.variable_getvalue(variable)
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError('カメラIMAGEが画像バイト列ではありません')
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('カメラ画像をデコードできません')
            return img
        finally:
            for fn, arg in [(self.client.variable_release, variable), (self.client.controller_disconnect, camera)]:
                if arg is not None:
                    try:
                        fn(arg)
                    except Exception:
                        pass

    def halt(self):
        errors = []
        if self.client is not None:
            for fn, args in [(self.client.robot_halt, (self.arm, '')),
                             (self.client.controller_execute, (self.ctrl, 'HandStop'))]:
                try:
                    fn(*args)
                except Exception as exc:
                    errors.append(type(exc).__name__)
        if errors:
            self.log('停止指令の到達を確認できません。実機の停止操作をしてください: ' + ', '.join(errors))

    def release(self):
        if self.taken:
            self.client.robot_execute(self.arm, 'GiveArm')
            self.taken = False

    def close(self):
        c = self.client
        if c is None:
            return
        try:
            self.release()
            for h in self.handles.values():
                c.variable_release(h)
            if self.arm is not None:
                c.robot_release(self.arm)
            if self.ctrl is not None:
                c.controller_disconnect(self.ctrl)
            c.service_stop()
        finally:
            if c._sock is not None:
                c._sock.close()
                c._sock = None
            self.client = None


class DryRobot:
    def __init__(self, stop):
        self.stop = stop
    def step(self):
        if self.stop.wait(.2):
            raise Stopped('模擬動作を停止しました')
    def set_pick(self, pose):
        self.step()
    def move(self, n, linear=False):
        self.step()
    def grip(self):
        self.step()
    def open_hand(self):
        self.step()
