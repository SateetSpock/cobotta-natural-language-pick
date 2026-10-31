import copy
import io
import json
import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from geometry import Calibration, number
from robot import Robot, DryRobot, sequence, check_stop
from bcap import ORiNException

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
CAL = Calibration(CFG)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
TOKEN = secrets.token_urlsafe(32)
lock = threading.RLock()
stop = threading.Event()
robot = None
worker_thread = None
frame = None
frame_at = 0.
frame_live = False
frame_settings = None
plan = []
plan_id = None
state = {'busy': False, 'job': '', 'connected': False, 'status': '画像を読み込むか、実機へ接続してください',
         'error': '', 'logs': [], 'targets': [], 'image_version': 0, 'live_image': False,
         'plan_id': None, 'connection': None}


def log(message):
    with lock:
        state['status'] = message
        state['logs'].append(datetime.now().strftime('%H:%M:%S') + ' ' + message)
        state['logs'] = state['logs'][-150:]


def invalidate(keep_targets=False):
    global plan, plan_id
    plan, plan_id = [], None
    state['plan_id'] = None
    if not keep_targets:
        state['targets'] = []


def put_image(img, live=False, settings=None):
    global frame, frame_at, frame_live, frame_settings
    if [img.shape[1], img.shape[0]] != CFG['image_size']:
        raise ValueError(f'画像はキャリブレーションと同じ {CFG["image_size"]} が必要です。自動リサイズはしません')
    with lock:
        frame, frame_at, frame_live, frame_settings = img, time.monotonic(), live, settings
        invalidate()
        state.update(image_version=state['image_version']+1, live_image=live)
    if live:
        folder = ROOT / 'data' / 'captures'
        folder.mkdir(parents=True, exist_ok=True)
        ok, data = cv2.imencode('.png', img)
        if not ok:
            raise ValueError('撮影画像の保存に失敗しました')
        (folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.png')).write_bytes(data.tobytes())


def set_plan(raw):
    global plan, plan_id
    targets = CAL.plan(raw)
    with lock:
        check_stop(stop)
        plan, plan_id = targets, secrets.token_hex(16)
        state.update(targets=copy.deepcopy(plan), plan_id=plan_id, json_text=json.dumps({'targets': [{k: t[k] for k in ('label', 'point', 'closing_angle_deg')} for t in targets]}, ensure_ascii=False, indent=2))
    log(f'{len(targets)}個の候補を取得しました' if state['job'] == 'automatic' else f'{len(targets)}個の候補を表示しました。位置・角度・順番を確認してください')


def worker(name, fn):
    global robot, frame_live
    try:
        fn()
    except Exception as exc:
        if robot is not None and robot.taken:
            robot.halt()
            try:
                robot.close()
            except Exception:
                pass
            robot = None
            with lock:
                state.update(connected=False, connection=None)
        detail = (f'b-CAP HRESULT=0x{exc.hresult & 0xffffffff:08X}' if isinstance(exc, ORiNException)
                  else str(exc) or type(exc).__name__)
        with lock:
            invalidate(keep_targets=True)
            state['error'] = detail
            if name in ('capture', 'execute', 'connect', 'automatic'):
                frame_live = False
                state['live_image'] = False
        log('中断: ' + detail)
    finally:
        with lock:
            state.update(busy=False, job='')


def launch(name, fn):
    global worker_thread
    with lock:
        if state['busy']:
            return jsonify(error='処理中です'), 409
        stop.clear()
        state.update(busy=True, job=name, error='')
        worker_thread = threading.Thread(target=worker, args=(name, fn), daemon=True)
        worker_thread.start()
    return jsonify(ok=True)


@app.before_request
def local_only():
    if request.host not in ('127.0.0.1:8765', 'localhost:8765'):
        return jsonify(error='localhostからアクセスしてください'), 403
    if request.method == 'POST':
        if request.headers.get('X-App-Token') != TOKEN:
            return jsonify(error='画面を再読み込みしてください'), 403
        origin = request.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:8765', 'http://localhost:8765'):
            return jsonify(error='外部サイトからの操作は拒否しました'), 403


@app.after_request
def headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


@app.errorhandler(Exception)
def error_response(exc):
    return jsonify(error=str(exc)), exc.code if isinstance(exc, HTTPException) else 400


@app.get('/')
def index():
    return render_template('index.html', token=TOKEN, model=CFG['model'], hull=CAL.hull.reshape(-1, 2).tolist(), speed=CFG['robot_speed_percent'])


@app.get('/api/state')
def get_state():
    with lock:
        return jsonify(copy.deepcopy(state))


@app.get('/api/image')
def image():
    with lock:
        if frame is None:
            return '', 404
        ok, data = cv2.imencode('.png', frame)
    if not ok:
        raise ValueError('PNG変換に失敗しました')
    return send_file(io.BytesIO(data.tobytes()), mimetype='image/png')


@app.post('/api/upload')
def upload():
    data = request.files['image'].read()
    def job():
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('PNG / JPEG画像を選んでください')
        put_image(img)
        log('画像を読み込みました（座標確認用。実機実行にはP10での再撮影が必要です）')
    return launch('upload', job)


@app.post('/api/parse')
def parse():
    raw = request.get_json()['text']
    def job():
        if frame is None:
            raise ValueError('先に画像を読み込んでください')
        with lock:
            invalidate()
        set_plan(raw)
    return launch('parse', job)


def build_grasp_prompt(cfg):
    scale_reference = json.dumps(
        [{'image_pixel': pixel, 'plane_xy_mm': xy}
         for pixel, xy in zip(cfg['image_points'], cfg['robot_points'])],
        ensure_ascii=False)
    prompt = f'''あなたは平行2爪グリッパのピック候補を画像から選びます。
ユーザーの色・種類・個数・順番の指示に従い、実際に見える対象だけを選びます。
種類だけなら色は任意、色だけなら種類は任意です。同一物体を重複指定しません。
重なりや遮蔽で両爪の入る場所が判断できない対象は選びません。
使用ハンドと把持条件:
平行2爪グリッパで上方から接近し、左右の爪で対象を挟んで持ち上げます。
ピック前の爪の開き幅は{cfg["open_mm"]} mmです。これは今回使用する開き幅で、ハンドの最大仕様を意味しません。
閉じる方向に測った把持部分の幅が、この開き幅より小さく、両側に爪を入れる余裕がある位置を選びます。
細長い物体は原則として長い側面を挟み、長手方向に対して直角に閉じます。
短辺を挟むことと短辺方向に閉じることを混同しないでください。
USBメモリは金属端子、ストラップ穴、細い付属部分を避け、安定して挟める本体部分を選びます。
金魚模型は目玉、尾びれ、ひれを避け、胴体の安定して挟める部分を選びます。
他の対象も突起や壊れやすい部分を避け、本体の両側を挟める位置を選びます。
ユーザーが対象と順番のみ指定した場合も、以上の把持条件を自動的に適用してください。
爪の厚み・長さは未提供です。余裕のない隙間に入れられると決めつけないでください。
実寸の基準:
以下は同じ撮影姿勢P10における、作業平面上の画像点と実座標の対応です。
画像点は元画像のピクセル[x,y]、実座標はmm単位の[X,Y]です。
{scale_reference}
実座標間の距離を作業平面上の長さの基準として、対象の幅と周辺の余裕を判断してください。
遠近による縮尺の違いに注意してください。対象の高さ・厚みはこの対応だけでは求まりません。
この情報は把持可能な候補の選定に使い、返答pointは引き続き0〜1000の正規化画像座標にしてください。
pointは物体を挟む中心で、必ず[x,y]順です。[y,x]ではありません。
xは画像左端0から右端1000、yは上端0から下端1000に正規化します。
closing_angle_degは爪を閉じる直線の方向です。物体の長軸ではありません。
元画像のピクセル縦横比で、画像右方向0度、時計回り、0以上180未満です。
画像は{cfg["image_size"][0]}x{cfg["image_size"][1]}です。角度を1000x1000に変形した画像で測らないでください。
4つの黒丸で囲まれた作業領域の内側のみ選択します。黒丸・文字・紙コップは対象外です。
対象がない、または把持可能と判断できない場合はtargetsを空配列にしてください。
最大20個。並び順はピック順です。ロボット座標やZや命令コードは生成しません。
返答はJSONのみ: {{"targets":[{{"label":"対象名","point":[500,500],"closing_angle_deg":90}}]}}
'''
    return prompt


def analyze_image(body):
    instruction = str(body.get('instruction', '')).strip()
    key = str(body.get('key', '')).strip() or os.environ.get('GEMINI_API_KEY', '')
    model = str(body.get('model', CFG['model'])).strip()
    if frame is None:
        raise ValueError('先に画像を読み込むか、撮影してください')
    if not instruction or len(instruction) > 3000:
        raise ValueError('指示を1〜3000文字で入力してください')
    if not key:
        raise ValueError('Gemini APIキーを入力してください')
    with lock:
        invalidate()
    from google import genai
    from google.genai import types
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise ValueError('画像を変換できません')
    prompt = build_grasp_prompt(CFG)
    schema = {'type': 'OBJECT', 'properties': {'targets': {'type': 'ARRAY', 'items': {
        'type': 'OBJECT', 'properties': {'label': {'type': 'STRING'},
        'point': {'type': 'ARRAY', 'items': {'type': 'NUMBER'}, 'minItems': 2, 'maxItems': 2},
        'closing_angle_deg': {'type': 'NUMBER'}},
        'required': ['label', 'point', 'closing_angle_deg']}}}, 'required': ['targets']}
    log('Geminiに画像と指示を送信しています')
    try:
        with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120000)) as client:
            result = client.models.generate_content(model=model,
                contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type='image/png'), instruction],
                config=types.GenerateContentConfig(system_instruction=prompt,
                    response_mime_type='application/json', response_schema=schema))
    except Exception as exc:
        # SDK exception messages may contain request details. Never echo the key.
        code = getattr(exc, 'code', None)
        raise RuntimeError(f'Gemini API失敗: {type(exc).__name__}, code={code}。キー・モデル・利用枠を確認してください') from None
    check_stop(stop)
    if not result.text:
        raise ValueError('GeminiからJSONが返りませんでした')
    set_plan(result.text)


@app.post('/api/analyze')
def analyze():
    body = request.get_json()
    return launch('analyze', lambda: analyze_image(body))


def connect_robot():
    global robot, frame_live
    with lock:
        invalidate()
        frame_live = False
        state.update(connected=False, live_image=False)
    if robot is not None:
        robot.close()
    robot = Robot(CFG, stop, log, ROOT)
    try:
        info = robot.connect()
    except Exception:
        try:
            robot.close()
        except Exception:
            pass
        robot = None
        raise
    with lock:
        state.update(connected=True, connection=info)
    log('接続・P10〜P15読取り完了。ロボットは動かしていません')


@app.post('/api/connect')
def connect():
    return launch('connect', connect_robot)


def settings(body):
    if body.get('confirmed') is not True:
        raise ValueError('実機設定の確認欄をチェックしてください')
    tool = number(body.get('tool'), 'Tool')
    work = number(body.get('work'), 'Work')
    if tool != int(tool) or work != int(work) or min(tool, work) < 0:
        raise ValueError('Tool / Workは0以上の整数です')
    force = number(body.get('force'), '把持力')
    if not 0 < force <= 20:
        raise ValueError('把持力は0より大きく20 N以下にしてください')
    return int(tool), int(work), force


def capture_image(s):
    global frame_live
    if robot is None:
        raise ValueError('先に実機へ接続してください')
    with lock:
        invalidate()
        frame_live = False
        state['live_image'] = False
    robot.verify_points()
    robot.prepare(*s)
    log('P10へ移動')
    robot.move(10)
    log('ハンド開放 30 mm')
    robot.open_hand()
    log('P10で撮影')
    img = robot.capture()
    check_stop(stop)
    robot.release()
    put_image(img, live=True, settings=s)
    log('撮影完了。指示をGeminiへ送信できます')


@app.post('/api/capture')
def capture():
    body = request.get_json()
    return launch('capture', lambda: capture_image(settings(body)))


def execute_live(s, expected_id):
    global frame_live
    with lock:
        check_stop(stop)
        if not plan or expected_id != plan_id:
            raise ValueError("表示中の有効な候補がありません")
        targets = copy.deepcopy(plan)
    if robot is None or not frame_live or s != frame_settings:
        raise ValueError('同じ設定でP10撮影をやり直してください')
    if time.monotonic()-frame_at > CFG['max_image_age_seconds']:
        raise ValueError('撮影から5分以上経過しました。再撮影してください')
    robot.verify_points()
    if not robot.at(robot.points[10]):
        raise ValueError('現在位置がP10ではありません。再撮影してください')
    # Consume before the first physical action; never reuse after success or failure.
    with lock:
        invalidate(keep_targets=True)
        frame_live = False
        state['live_image'] = False
    robot.prepare(*s)
    sequence(robot, targets, stop, log)
    robot.release()
    log('全ピック完了・P10に戻りました。次の処理は再撮影してください')


@app.post('/api/execute')
def execute():
    body = request.get_json()
    def job():
        global frame_live
        if not plan or body.get('plan_id') != plan_id:
            raise ValueError('表示中の有効な候補がありません')
        targets = copy.deepcopy(plan)
        if body.get('live') is not True:
            log('模擬動作開始（実機への命令は送信しません）')
            sequence(DryRobot(stop), targets, stop, log)
            log('模擬動作完了')
            return
        s = settings(body)
        execute_live(s, body.get('plan_id'))
    return launch('execute', job)


@app.post('/api/start')
def start():
    body = request.get_json()
    def job():
        s = settings(body)
        mode = body.get('mode', 'gemini')
        # Validate input before any connection or motion; never fall back to old plans.
        if mode == 'json':
            raw = body.get('text', '')
            if not isinstance(raw, str) or not CAL.plan(raw):
                raise ValueError('JSONに1個以上の対象を指定してください')
        elif mode == 'gemini':
            instruction = str(body.get('instruction', '')).strip()
            if not instruction or len(instruction) > 3000:
                raise ValueError('指示を1〜3000文字で入力してください')
            if not (str(body.get('key', '')).strip() or os.environ.get('GEMINI_API_KEY', '')):
                raise ValueError('Gemini APIキーを入力してください')
        else:
            raise ValueError('GeminiまたはJSONを選んでください')
        with lock:
            invalidate()
        check_stop(stop)
        if robot is None:
            log('実機へ接続しています')
            connect_robot()
        check_stop(stop)
        capture_image(s)
        check_stop(stop)
        if mode == 'gemini':
            analyze_image(body)
        else:
            log('貼り付けたJSONを使用します（Geminiへのアクセスなし）')
            set_plan(raw)
        check_stop(stop)
        if not plan:
            log('対象が見つからなかったため、P10で終了しました')
            return
        log('認識完了。表示順でピックを開始します')
        execute_live(s, plan_id)
    return launch('automatic', job)


@app.post('/api/stop')
def stop_job():
    with lock:
        stop.set()
        invalidate(keep_targets=True)
    log('停止要求を受信しました。動作処理で停止指令を送ります')
    return jsonify(ok=True)


if __name__ == '__main__':
    from waitress import serve
    print('COBOTTA Pick: http://127.0.0.1:8765', flush=True)
    print('終了: Ctrl+C。動作中は先に画面の停止要求または実機の停止操作をしてください。', flush=True)
    try:
        serve(app, host='127.0.0.1', port=8765, threads=6)
    finally:
        stop.set()
        if worker_thread is not None:
            worker_thread.join(timeout=12)
        if robot is not None and not state['busy']:
            robot.close()
