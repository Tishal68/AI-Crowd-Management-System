"""Verify failed launch recovery and cross-session admission."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from filelock import FileLock
from src.services.jobs import start_job,read_status
from src.services.storage import atomic_json


class JobTests(unittest.TestCase):
    def test_active_job_blocks_second_browser_and_crash_recovers(self):
        with tempfile.TemporaryDirectory() as folder,patch('config.OUTPUTS_ROOT',folder):
            root=Path(folder)
            existing=root/'sessions'/('a'*32)/('b'*24)
            atomic_json(existing/'status.json',{'state':'running','updated':time.time()})
            second=root/'sessions'/('c'*32)/('d'*24)
            with patch('src.services.jobs.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'Another analysis'): start_job(second,{})
                spawn.assert_not_called()
            atomic_json(existing/'status.json',{'state':'running','updated':time.time()-200})
            # A live worker's lock protects long inference calls from stale-heartbeat recovery.
            with FileLock(str(root/'.analysis.lock')):
                self.assertEqual(read_status(existing)['state'],'running')
            self.assertEqual(read_status(existing)['state'],'failed')

    def test_failed_process_launch_does_not_leave_queued_job(self):
        with tempfile.TemporaryDirectory() as folder,patch('config.OUTPUTS_ROOT',folder):
            run=Path(folder)/'sessions'/('a'*32)/('b'*24)
            with patch('src.services.jobs.subprocess.Popen',side_effect=OSError('Launch failed')):
                with self.assertRaises(OSError): start_job(run,{})
            self.assertEqual(read_status(run)['state'],'failed')
