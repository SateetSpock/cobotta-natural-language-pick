import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
import app as web
import test_app

RAW=json.dumps({'targets':[{'label':'緑','point':[380,645],'closing_angle_deg':150},{'label':'黒','point':[450,400],'closing_angle_deg':30}]})

class AutomaticTests(unittest.TestCase):
    post=test_app.WebTests.post
    done=test_app.WebTests.done
    def setUp(self):
        test_app.WebTests.setUp(self)
        self.events=[]
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=patch.object(web,'ROOT',__import__('pathlib').Path(self.temp.name));root.start();self.addCleanup(root.stop)
        owner=self
        class FakeRobot:
            def __init__(self,*args):
                self.taken=False
                self.points={10:[1,2,3,0,0,0,261]}
            def connect(self):
                owner.events.append('connect');return {'points':self.points,'tool':0,'work':0}
            def verify_points(self): owner.events.append('verify')
            def prepare(self,*args): self.taken=True;owner.events.append('prepare')
            def move(self,n,linear=False): owner.events.append(('move',n,linear))
            def open_hand(self): owner.events.append('open')
            def capture(self):
                owner.events.append('capture')
                return np.full((1080,1920,3),77,np.uint8)
            def release(self): self.taken=False;owner.events.append('release')
            def at(self,p): return True
            def set_pick(self,p): owner.events.append('write_pick')
            def grip(self): owner.events.append('grip')
            def halt(self): owner.events.append('halt')
            def close(self): self.taken=False;owner.events.append('close')
        self.fake=FakeRobot
        rp=patch.object(web,'Robot',FakeRobot);rp.start();self.addCleanup(rp.stop)
        self.body={'mode':'gemini','confirmed':True,'tool':0,'work':0,'force':10,'instruction':'緑、黒の順','key':'dummy','text':RAW}

    def analyzed(self,body):
        self.events.append('gemini')
        self.assertEqual(int(web.frame[0,0,0]),77)
        self.assertEqual(body['instruction'],'緑、黒の順')
        web.set_plan(RAW)

    def test_gemini_full_cycle_and_consumption(self):
        web.frame=np.zeros((1080,1920,3),np.uint8)
        with patch.object(web,'analyze_image',side_effect=self.analyzed) as ai:
            self.post('/api/start',json=self.body);self.done()
            ai.assert_called_once()
        self.assertFalse(web.state['error'])
        self.assertLess(self.events.index('capture'),self.events.index('gemini'))
        self.assertLess(self.events.index('gemini'),self.events.index('write_pick'))
        self.assertEqual(self.events.count('grip'),2)
        self.assertEqual([v for v in self.events if isinstance(v,tuple)],
            [('move',10,False)]+[('move',11,False),('move',12,False),('move',13,True),('move',12,True),('move',14,False),('move',15,False),('move',10,False)]*2)
        self.assertIsNone(web.plan_id)
        self.assertFalse(web.frame_live)
        self.assertEqual(len(json.loads(web.state['json_text'])['targets']),2)

    def test_json_cycle_does_not_call_gemini(self):
        self.body.update(mode='json',key='',instruction='')
        with patch.object(web,'analyze_image') as ai:
            self.post('/api/start',json=self.body);self.done();ai.assert_not_called()
        self.assertFalse(web.state['error'])
        self.assertEqual(self.events.count('capture'),1)
        self.assertEqual(self.events.count('grip'),2)

    def test_bad_json_no_motion_or_connection(self):
        self.body.update(mode='json',text='{"targets":[{"label":"a","point":[0,0],"closing_angle_deg":0}]}')
        self.post('/api/start',json=self.body);self.done()
        self.assertTrue(web.state['error']);self.assertEqual(self.events,[])

    def test_empty_gemini_no_pick(self):
        with patch.object(web,'analyze_image',side_effect=lambda body:web.set_plan('{"targets":[]}')):
            self.post('/api/start',json=self.body);self.done()
        self.assertFalse(web.state['error']);self.assertNotIn('write_pick',self.events)
        self.assertIn('対象が見つからなかった',web.state['status'])

    def test_api_failure_no_pick_and_invalidate_old_plan(self):
        web.set_plan(RAW)
        with patch.object(web,'analyze_image',side_effect=RuntimeError('API failed')):
            self.post('/api/start',json=self.body);self.done()
        self.assertIn('API failed',web.state['error'])
        self.assertNotIn('write_pick',self.events);self.assertIsNone(web.plan_id);self.assertFalse(web.frame_live)

    def test_stop_after_capture_no_gemini_no_pick(self):
        original=self.fake.capture
        def capture(r):
            img=original(r);web.stop.set();return img
        with patch.object(self.fake,'capture',capture),patch.object(web,'analyze_image') as ai:
            self.post('/api/start',json=self.body);self.done();ai.assert_not_called()
        self.assertNotIn('write_pick',self.events);self.assertIn('halt',self.events)

    def test_stop_during_api_discards_result(self):
        def analyze(body):
            web.stop.set();web.set_plan(RAW)
        with patch.object(web,'analyze_image',side_effect=analyze):
            self.post('/api/start',json=self.body);self.done()
        self.assertTrue(web.state['error']);self.assertNotIn('write_pick',self.events)

    def test_duplicate_start_rejected_during_api(self):
        arrived=threading.Event();resume=threading.Event()
        def analyze(body):
            arrived.set();resume.wait(3);web.set_plan(RAW)
        with patch.object(web,'analyze_image',side_effect=analyze):
            self.post('/api/start',json=self.body)
            self.assertTrue(arrived.wait(3))
            response=self.post('/api/start',json=self.body)
            resume.set();self.done()
        self.assertEqual(response.status_code,409);self.assertEqual(self.events.count('capture'),1)

    def test_force_validation_before_motion(self):
        self.body['force']=0
        self.post('/api/start',json=self.body);self.done()
        self.assertEqual(self.events,[]);self.assertTrue(web.state['error'])

    def test_speed_display_uses_configuration(self):
        with patch.dict(web.CFG,robot_speed_percent=50.):
            html=self.c.get('/',base_url=self.base).text
        self.assertIn('内部速度50.0%',html)
        self.assertIn('id="start"',html)

if __name__=='__main__':unittest.main()
