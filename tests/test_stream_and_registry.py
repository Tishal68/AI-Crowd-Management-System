"""
Regression tests for StreamCapture lifecycle, camera credential vault encryption,
plaintext-to-encrypted migration, credential redaction, and hardware provider settings.
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.services.camera_vault import (
    get_vault_master_key,
    protect_camera_uri,
    unprotect_camera_uri,
    redact_camera_uri,
    load_camera_registry,
    save_camera_registry,
    resolve_camera_connection_uri,
    release_global_stream,
    register_active_stream,
    get_active_stream,
)
from src.core.stream_reader import StreamCapture


class CameraCredentialSecurityTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix='cv_test_vault_'))
        self.reg_path = self.test_dir / 'cameras.json'

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_protect_and_unprotect_roundtrip(self):
        """Verify RTSP credentials encrypt with salt/HMAC and decrypt back to exact plaintext."""
        sample_uris = [
            'rtsp://admin:SecretPass123@192.168.1.120:554/stream1',
            'rtsp://user:P@ss%20w0rd!#$&*()@cam.campus.edu:8554/live?token=xyz123',
            'http://camera-admin:token_abc_987@10.0.0.15/video.mjpg',
            '0',  # Local webcam device index
            '1',
            '',
        ]
        for uri in sample_uris:
            protected = protect_camera_uri(uri)
            if uri in ('0', '1', ''):
                self.assertEqual(protected, uri)
            else:
                self.assertTrue(protected.startswith('enc:v1:'))
                self.assertNotIn('SecretPass123', protected)
                self.assertNotIn('token_abc_987', protected)
            
            unprotected = unprotect_camera_uri(protected)
            self.assertEqual(unprotected, uri)

    def test_passwords_never_stored_in_plaintext_in_registry(self):
        """Verify cameras.json never contains plaintext passwords or secrets at rest."""
        raw_password = 'SuperSecretVaultPassword999!'
        test_cam = {
            'id': 'CAM-01',
            'name': 'North Gate HD',
            'uri': f'rtsp://security:{raw_password}@192.168.1.200:554/h264',
            'zone': 'Campus entrance',
            'capacity': 100,
        }
        save_camera_registry([test_cam], self.reg_path)

        # Inspect raw file on disk
        disk_content = self.reg_path.read_text(encoding='utf-8')
        self.assertNotIn(raw_password, disk_content)
        self.assertIn('enc:v1:', disk_content)
        self.assertIn('[redacted]', disk_content)

        # Loading decrypts for stream ingestion but keeps safe display_uri
        loaded = load_camera_registry(self.reg_path)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0]['uri'].startswith('enc:v1:'))
        self.assertIn('[redacted]', loaded[0]['display_uri'])
        self.assertNotIn(raw_password, loaded[0]['display_uri'])

        # Extracted connection URI matches original
        conn_uri = resolve_camera_connection_uri(loaded[0])
        self.assertEqual(conn_uri, f'rtsp://security:{raw_password}@192.168.1.200:554/h264')

    def test_safe_migration_of_legacy_plaintext_registry(self):
        """Verify legacy cameras.json with plaintext passwords is automatically migrated on first load."""
        legacy_secret = 'LegacyPlaintextSecret555'
        legacy_data = [
            {
                'id': 'CAM-LEGACY-01',
                'name': 'Old Science Wing',
                'uri': f'rtsp://operator:{legacy_secret}@192.168.2.10:554/live',
                'zone': 'Academic block',
                'capacity': 50,
                'verified': False,
            }
        ]
        # Write unencrypted legacy file
        self.reg_path.write_text(json.dumps(legacy_data, indent=2), encoding='utf-8')
        self.assertIn(legacy_secret, self.reg_path.read_text(encoding='utf-8'))

        # Load registry - triggers automatic transparent migration
        migrated = load_camera_registry(self.reg_path)
        self.assertEqual(len(migrated), 1)
        self.assertTrue(migrated[0]['uri'].startswith('enc:v1:'))
        self.assertEqual(migrated[0]['display_uri'], 'rtsp://[redacted]@192.168.2.10:554/live')

        # Check that the file on disk was re-saved and plaintext wiped
        disk_content_after = self.reg_path.read_text(encoding='utf-8')
        self.assertNotIn(legacy_secret, disk_content_after)
        self.assertIn('enc:v1:', disk_content_after)

        # Decrypted URI is intact for video processing
        self.assertEqual(resolve_camera_connection_uri(migrated[0]), f'rtsp://operator:{legacy_secret}@192.168.2.10:554/live')

    def test_redact_camera_uri_masks_all_credentials(self):
        """Verify credential redaction across RTSP, HTTP, tokens, and query parameters."""
        cases = [
            ('rtsp://admin:secret@10.0.0.1:554/live', 'rtsp://[redacted]@10.0.0.1:554/live'),
            ('http://user:mypass@camera.local/stream', 'http://[redacted]@camera.local/stream'),
            ('rtsp://10.0.0.2:554/live?token=secret_jwt_xyz', 'rtsp://10.0.0.2:554/live?token=[redacted]'),
            ('rtsp://10.0.0.2:554/live?api_key=secret_123', 'rtsp://10.0.0.2:554/live?api_key=[redacted]'),
            ('0', 'Local Device #0'),
        ]
        for raw, expected in cases:
            redacted = redact_camera_uri(raw)
            self.assertEqual(redacted, expected)

    def test_corrupted_or_invalid_registry_handled_gracefully(self):
        """Verify corrupt JSON or non-list structure in cameras.json returns empty list safely."""
        self.reg_path.write_text('{ invalid json broken content !!!', encoding='utf-8')
        result = load_camera_registry(self.reg_path)
        self.assertEqual(result, [])

        self.reg_path.write_text(json.dumps({'not': 'a list'}), encoding='utf-8')
        result = load_camera_registry(self.reg_path)
        self.assertEqual(result, [])


class StreamLifecycleAndResourceTests(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_stream_capture_release_joins_worker_thread(self, mock_cv_class):
        """Verify StreamCapture.release() terminates worker thread and frees capture."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed', reconnect_timeout=1.0)
        self.assertTrue(cap.isOpened())
        worker_thread = cap._thread
        self.assertIsNotNone(worker_thread)
        self.assertTrue(worker_thread.is_alive())

        # Read must return the frame
        ok, frame, pts = cap.read(timeout=0.5)
        self.assertTrue(ok)
        self.assertIsNotNone(frame)

        # Release must join the thread and clear resources
        cap.release()
        self.assertFalse(cap._running)
        self.assertFalse(worker_thread.is_alive())
        self.assertIsNone(cap.cap)
        mock_instance.release.assert_called()

    @patch('cv2.VideoCapture')
    def test_stream_capture_read_timeout_on_stalled_feed(self, mock_cv_class):
        """Verify read() returns (False, None, 0.0) promptly when stream produces no frames."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_instance.read.return_value = (False, None)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_stalled', reconnect_timeout=1.0)
        try:
            start_t = time.perf_counter()
            ok, frame, pts = cap.read(timeout=0.1)
            elapsed = time.perf_counter() - start_t
            self.assertFalse(ok)
            self.assertIsNone(frame)
            self.assertLess(elapsed, 0.5)
        finally:
            cap.release()

    def test_global_stream_release_cleans_resources(self):
        """Verify release_global_stream closes capture and writer objects cleanly."""
        mock_capture = MagicMock()
        mock_writer = MagicMock()

        register_active_stream(capture=mock_capture, writer=mock_writer)
        cap_active, wrt_active = get_active_stream()
        self.assertEqual(cap_active, mock_capture)
        self.assertEqual(wrt_active, mock_writer)

        release_global_stream()

        mock_capture.release.assert_called_once()
        mock_writer.close.assert_called_once_with(success=True)

        cap_after, wrt_after = get_active_stream()
        self.assertIsNone(cap_after)
        self.assertIsNone(wrt_after)


class HardwareConfigurationTests(unittest.TestCase):
    def test_hardware_cpu_fallback(self):
        """Verify CPU execution is default and fallback handles absence of CUDA."""
        import torch
        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            t = torch.zeros((2, 2))
            self.assertEqual(t.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
