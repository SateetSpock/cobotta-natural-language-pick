import copy
import unittest
from unittest.mock import patch

import app as web
import test_automatic as automatic

RAW = automatic.RAW


class DisplayTests(unittest.TestCase):
    setUp = automatic.AutomaticTests.setUp
    post = automatic.AutomaticTests.post
    done = automatic.AutomaticTests.done
    analyzed = automatic.AutomaticTests.analyzed

    def snapshot(self):
        return self.c.get('/api/state', base_url=self.base).get_json()

    def test_visible_during_motion_and_after_completion_but_not_reusable(self):
        original = self.fake.set_pick
        snapshots = []
        def pick(robot, pose):
            snapshots.append(self.snapshot())
            original(robot, pose)
        with patch.object(web, 'analyze_image', side_effect=self.analyzed), patch.object(self.fake, 'set_pick', pick):
            self.post('/api/start', json=self.body)
            self.done()
        self.assertFalse(web.state['error'])
        for state in snapshots + [self.snapshot()]:
            self.assertEqual([t['label'] for t in state['targets']], ['緑', '黒'])
            self.assertIsNone(state['plan_id'])
            self.assertFalse(state['live_image'])
        self.assertEqual(web.plan, [])
        events = copy.deepcopy(self.events)
        self.post('/api/execute', json={**self.body, 'live': True, 'plan_id': None})
        self.done()
        self.assertEqual(self.events, events)
        self.assertIn('有効な候補がありません', web.state['error'])
        self.assertEqual(len(self.snapshot()['targets']), 2)

    def test_failure_retains_coordinates_and_next_capture_clears_them(self):
        with patch.object(web, 'analyze_image', side_effect=self.analyzed), patch.object(self.fake, 'grip', side_effect=RuntimeError('grip failed')):
            self.post('/api/start', json=self.body)
            self.done()
        self.assertIn('grip failed', web.state['error'])
        self.assertEqual(len(self.snapshot()['targets']), 2)
        self.assertIsNone(web.plan_id)
        with patch.object(web, 'analyze_image', side_effect=lambda body: web.set_plan('{"targets":[]}')):
            self.post('/api/start', json=self.body)
            self.done()
        self.assertEqual(self.snapshot()['targets'], [])

    def test_stop_retains_display_but_invalidates_plan(self):
        web.set_plan(RAW)
        old_id = web.plan_id
        self.post('/api/stop')
        self.assertEqual(len(self.snapshot()['targets']), 2)
        self.assertIsNone(web.plan_id)
        self.post('/api/execute', json={'live': False, 'plan_id': old_id})
        self.done()
        self.assertIn('有効な候補がありません', web.state['error'])

    def test_manual_analysis_still_produces_executable_plan(self):
        web.set_plan(RAW)
        self.assertIsNotNone(self.snapshot()['plan_id'])
        self.assertEqual(len(self.snapshot()['targets']), 2)
        web.invalidate()
        self.assertEqual(self.snapshot()['targets'], [])


if __name__ == '__main__':
    unittest.main()
