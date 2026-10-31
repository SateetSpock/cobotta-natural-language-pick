import copy
import io
import json
import math
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import app as web
from geometry import Calibration
from robot import sequence, Stopped, Robot


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration(web.CFG)

    def test_four_correspondences(self):
        for src, dest in zip(web.CFG['image_points'], web.CFG['robot_points']):
            np.testing.assert_allclose(self.cal.xy(src), dest, atol=1e-4)

    def test_known_target_and_orientation(self):
        t = self.cal.convert({'label': '緑色の金魚', 'point': [380,645], 'closing_angle_deg': 150})
        self.assertAlmostEqual(t['pose'][0], 176.418, places=2)
        self.assertAlmostEqual(t['pose'][1], -20.434, places=2)
        self.assertAlmostEqual(t['pose'][5], -144.197, places=2)
        self.assertEqual(t['pose'][2], 4.5)
        t = self.cal.convert({'label': '水平', 'point': [380,645], 'closing_angle_deg': 0})
        self.assertAlmostEqual(t['pose'][5], -174.4)

    def test_invalid_or_duplicate_ai_output_rejected(self):
        for point, angle in [([0,0],0), ([380,float('nan')],0), ([True,645],0), ([380,645],180)]:
            with self.assertRaises(ValueError):
                self.cal.convert({'label':'a','point':point,'closing_angle_deg':angle})
        t = {'label':'a','point':[380,645],'closing_angle_deg':15}
        with self.assertRaises(ValueError):
            self.cal.plan(json.dumps([t,t]))
        with self.assertRaises(ValueError):
            self.cal.plan('{"label":"a","point":[NaN,645],"closing_angle_deg":15}')


class SequenceTests(unittest.TestCase):
    def test_exact_order_and_vertical_moves(self):
        calls = []
        class Fake:
            def set_pick(self,p): calls.append(('write',p))
            def move(self,n,linear=False): calls.append(('move',n,linear))
            def grip(self): calls.append(('grip',))
            def open_hand(self): calls.append(('open',))
        targets = [{'label':'a','pose':[1,2,4.5,179.47,.93,-174.4]}]*2
        sequence(Fake(), targets, threading.Event(), lambda x: None)
        expected = [('write',targets[0]['pose']), ('move',11,False),('move',12,False),
                    ('move',13,True),('grip',),('move',12,True),('move',14,False),
                    ('move',15,False),('open',),('move',10,False)]
        self.assertEqual(calls, expected*2)

    def test_grip_failure_never_lifts(self):
        calls = []
        class Fake:
            def set_pick(self,p): pass
            def move(self,n,linear=False): calls.append(n)
            def grip(self): raise RuntimeError('no hold')
        with self.assertRaises(RuntimeError):
            sequence(Fake(),[{'label':'a','pose':[]}],threading.Event(),lambda x:None)
        self.assertEqual(calls,[11,12,13])

    def test_stop_at_pick_prevents_grip(self):
        event = threading.Event()
        class Fake:
            def set_pick(self,p): pass
            def move(self,n,linear=False):
                if n == 13: event.set()
            def grip(self): raise AssertionError('must not grip')
        with self.assertRaises(Stopped):
            sequence(Fake(),[{'label':'a','pose':[]}],event,lambda x:None)

    def test_write_z_and_preserve_fig(self):
        r=Robot(web.CFG,threading.Event(),lambda x:None,'.')
        r.points={13:[0.,0.,0.,0.,0.,0.,261.]}
        saved={}
        class Client:
            def variable_putvalue(self,h,p): saved[h]=p
        r.client=Client()
        r.var=lambda n:n
        r.read=lambda n:saved[n]
        r.set_pick([176.,-20.,4.5,179.47,.93,-144.2])
        self.assertEqual(saved['P12'][2],44.5)
        self.assertEqual(saved['P13'][2],4.5)
        self.assertEqual(saved['P13'][6],261.)
        self.assertEqual(set(saved),{'P12','P13'})


class WebTests(unittest.TestCase):
    def setUp(self):
        web.robot=None
        web.frame=None
        web.frame_live=False
        web.state.update(busy=False,job='',connected=False,error='',targets=[],image_version=0,live_image=False,plan_id=None)
        web.invalidate()
        web.stop.clear()
        self.c=web.app.test_client()
        self.base='http://127.0.0.1:8765'
        self.headers={'X-App-Token':web.TOKEN}

    def post(self,path,**kwargs):
        return self.c.post(path,base_url=self.base,headers=self.headers,**kwargs)

    def done(self):
        deadline=time.monotonic()+5
        while web.state['busy'] and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertFalse(web.state['busy'])

    def upload(self):
        ok,png=cv2.imencode('.png',np.zeros((1080,1920,3),np.uint8))
        self.post('/api/upload',data={'image':(io.BytesIO(png.tobytes()),'test.png')})
        self.done()
        self.assertFalse(web.state['error'])

    def plan(self):
        self.post('/api/parse',json={'text':'{"label":"緑色の金魚","point":[380,645],"closing_angle_deg":150}'})
        self.done()
        self.assertFalse(web.state['error'])

    def test_ui_render_and_csrf(self):
        self.assertEqual(self.c.get('/',base_url=self.base).status_code,200)
        self.assertEqual(self.c.post('/api/stop',base_url=self.base).status_code,403)
        self.assertEqual(self.c.get('/',base_url='http://evil.example:8765').status_code,403)

    def test_offline_upload_json_and_no_live_execution(self):
        self.upload()
        self.plan()
        self.assertEqual(len(web.state['targets']),1)
        self.assertFalse(web.frame_live)
        self.post('/api/execute',json={'live':True,'plan_id':web.plan_id,'confirmed':True,'tool':0,'work':0,'force':10})
        self.done()
        self.assertIn('P10撮影',web.state['error'])

    def test_resolution_mismatch(self):
        ok,png=cv2.imencode('.png',np.zeros((540,960,3),np.uint8))
        self.post('/api/upload',data={'image':(io.BytesIO(png.tobytes()),'wrong.png')})
        self.done()
        self.assertIn('1920',web.state['error'])
        self.assertIsNone(web.frame)

    def test_second_job_rejected_while_busy(self):
        web.state['busy']=True
        self.assertEqual(self.post('/api/parse',json={'text':'[]'}).status_code,409)
        web.state['busy']=False

    def test_stale_live_frame_cannot_move(self):
        self.upload()
        self.plan()
        web.robot=object()  # Raises if any hardware attribute is accessed.
        web.frame_live=True
        web.frame_at=time.monotonic()-301
        web.frame_settings=(0,0,10.)
        # Worker cleanup reads taken only on errors: no live authority was obtained.
        class Unused:
            taken=False
        web.robot=Unused()
        self.post('/api/execute',json={'live':True,'plan_id':web.plan_id,'confirmed':True,'tool':0,'work':0,'force':10})
        self.done()
        self.assertIn('5分',web.state['error'])

    def test_stop_discards_late_result(self):
        self.upload()
        web.stop.set()
        with self.assertRaises(Stopped):
            web.set_plan('{"label":"a","point":[380,645],"closing_angle_deg":0}')
        self.assertEqual(web.plan,[])


if __name__=='__main__':
    unittest.main()
