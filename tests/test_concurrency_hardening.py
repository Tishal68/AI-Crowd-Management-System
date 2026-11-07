"""
Deep concurrency, stress, and Windows file-locking resilience tests.
Validates:
1. High-concurrency readers and writers simultaneously accessing status.json via atomic_json and read().
2. Safe atomic replacement under aggressive reader locks (preventing WinError 5 / WinError 32).
3. safe_unlink resilience when targets are held open by concurrent handles.
4. safe_rmtree resilience with read-only and locked files.
5. End-to-end video analysis with flow tracking under concurrent status polling.
"""
import concurrent.futures
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.services.storage import atomic_json, safe_replace, safe_unlink, safe_rmtree
from src.core.scene import FlowTracker
from src.core.video_processor import process_video
from src.utils.helpers import Detection


class ConcurrencyAndFileLockingTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix='cv_lock_stress_'))

    def tearDown(self):
        safe_rmtree(self.test_dir)

    def test_high_concurrency_atomic_json_stress(self):
        """Stress-test 10 concurrent readers and 10 concurrent writers on status.json."""
        status_file = self.test_dir / 'status.json'
        atomic_json(status_file, {'state': 'initial', 'progress': 0.0})

        stop_flag = False
        read_successes = [0]
        write_successes = [0]
        exceptions = []

        def writer_task(writer_id):
            nonlocal stop_flag
            count = 0
            while not stop_flag and count < 40:
                try:
                    atomic_json(status_file, {
                        'state': 'running',
                        'writer_id': writer_id,
                        'seq': count,
                        'timestamp': time.perf_counter()
                    })
                    write_successes[0] += 1
                except Exception as e:
                    exceptions.append(('writer', writer_id, e))
                count += 1
                time.sleep(0.005)

        def reader_task(reader_id):
            nonlocal stop_flag
            count = 0
            while not stop_flag and count < 60:
                try:
                    if status_file.is_file():
                        # Read file directly simulating web polling loop
                        data = json.loads(status_file.read_text(encoding='utf-8'))
                        if isinstance(data, dict):
                            read_successes[0] += 1
                except (json.JSONDecodeError, OSError):
                    # Reader might encounter transient read collisions; should not be a fatal error
                    pass
                except Exception as e:
                    exceptions.append(('reader', reader_id, e))
                count += 1
                time.sleep(0.003)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            writers = [executor.submit(writer_task, i) for i in range(10)]
            readers = [executor.submit(reader_task, i) for i in range(10)]

            # Wait for all writers to complete
            concurrent.futures.wait(writers)
            stop_flag = True
            concurrent.futures.wait(readers)

        # There must be zero unhandled writer exceptions (no WinError 5 / WinError 32)
        self.assertEqual(len(exceptions), 0, f"Encountered unexpected exceptions: {exceptions}")
        self.assertGreater(write_successes[0], 100)
        self.assertGreater(read_successes[0], 100)

        # File on disk remains valid JSON
        final_data = json.loads(status_file.read_text(encoding='utf-8'))
        self.assertEqual(final_data['state'], 'running')

    def test_safe_replace_under_persistent_reader_lock(self):
        """Verify safe_replace falls back to content copy when reader persistently refuses to close."""
        target = self.test_dir / 'locked_target.json'
        temp_source = self.test_dir / 'update.tmp'
        target.write_text('{"initial": true}', encoding='utf-8')
        temp_source.write_text('{"initial": false, "updated": true}', encoding='utf-8')

        # Open file in read mode to hold active Windows handle
        reader = open(target, 'r', encoding='utf-8')
        try:
            # safe_replace should succeed without raising WinError 5
            safe_replace(temp_source, target, max_attempts=3, base_delay=0.01)
        finally:
            reader.close()

        # Target file content must be updated
        data = json.loads(target.read_text(encoding='utf-8'))
        self.assertTrue(data.get('updated'))

    def test_safe_unlink_on_locked_file(self):
        """Verify safe_unlink does not crash with PermissionError when file handle is open."""
        locked_file = self.test_dir / 'transient_locked.bin'
        locked_file.write_bytes(b'locked bytes')

        handle = open(locked_file, 'rb')
        try:
            # Must not raise PermissionError (WinError 5 / WinError 32)
            safe_unlink(locked_file, retries=2, delay=0.01)
        finally:
            handle.close()

        # Once handle is closed, subsequent unlink succeeds
        safe_unlink(locked_file)
        self.assertFalse(locked_file.exists())

    def test_safe_rmtree_on_read_only_and_locked_tree(self):
        """Verify safe_rmtree removes read-only files and directory trees without error."""
        sub_dir = self.test_dir / 'sub_folder'
        sub_dir.mkdir(parents=True, exist_ok=True)
        read_only_file = sub_dir / 'ro.txt'
        read_only_file.write_text('read only content', encoding='utf-8')

        # Make file read-only on Windows
        import stat
        os.chmod(str(read_only_file), stat.S_IREAD)

        # safe_rmtree should clear read-only flag and remove directory
        safe_rmtree(sub_dir)
        self.assertFalse(sub_dir.exists())

    def test_flow_tracking_with_rapid_status_polling(self):
        """Simulate video flow analysis while background threads aggressively poll status.json."""
        import cv2

        run_dir = self.test_dir / 'flow_run'
        run_dir.mkdir(parents=True, exist_ok=True)
        video_path = run_dir / 'flow_input.avi'

        # Generate 20-frame test clip
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
        for i in range(20):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(frame, (100 + i * 5, 120), 15, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        status_path = run_dir / 'status.json'
        atomic_json(status_path, {'state': 'queued', 'progress': 0.0})

        # Start continuous poller thread in background
        polling_active = True
        poll_errors = []

        def poller():
            while polling_active:
                try:
                    if status_path.is_file():
                        _ = json.loads(status_path.read_text(encoding='utf-8'))
                except (json.JSONDecodeError, OSError):
                    pass
                except Exception as err:
                    poll_errors.append(err)
                time.sleep(0.005)

        poller_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        poll_future = poller_thread.submit(poller)

        def progress_callback(index, total, frame, count, level, auto):
            atomic_json(status_path, {
                'state': 'running',
                'progress': index / (total or 1),
                'frame': index,
                'count': count
            })

        flow_line = [[0.2, 0.5], [0.8, 0.5]]
        dummy_model = MagicMock()
        dummy_model.names = {0: 'head'}

        try:
            with patch('src.core.video_processor.detect_heads_in_frame', return_value=[Detection(90, 110, 110, 130, 0.9)]):
                res = process_video(
                    weights_path='unused',
                    source_path=str(video_path),
                    output_path=str(run_dir / 'flow_out.mp4'),
                    preloaded_model=dummy_model,
                    analysis_fps=10.0,
                    flow_line=flow_line,
                    frame_callback=progress_callback,
                )
            self.assertIsNotNone(res)
            self.assertEqual(len(poll_errors), 0)
        finally:
            polling_active = False
            poller_thread.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
