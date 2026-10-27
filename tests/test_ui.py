"""Verify the workspace starts and clearing storage resets uploader identity."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class WorkspaceTests(unittest.TestCase):
    def test_workspace_and_session_reset(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, 'Crowd occupancy & flow review')
        old_id = app.session_state['session_id']
        generation = app.session_state['upload_generation']
        next(b for b in app.button if b.label == 'Clear this session').click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state['session_id'], old_id)
        self.assertEqual(app.session_state['upload_generation'], generation + 1)
        app.radio[0].set_value('Model lab').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Model comparison' in h.value for h in app.subheader))
        app.radio[0].set_value('Saved sessions').run()
        self.assertFalse(app.exception)
