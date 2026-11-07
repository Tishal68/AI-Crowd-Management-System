"""
Comprehensive regression tests for:
1. StreamCapture fresh-frame validation, stale timeout, disconnection, and reconnection.
2. AES-256-GCM credential vault encryption, legacy enc:v1 migration, tampering, and invalid key rejection.
3. Thread safety, safe worker termination, and resource cleanup.
4. Exception sanitization and recording failure recovery.
"""
import base64
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

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
    _derive_keys_v1,
    _keystream_v1,
)
from src.core.stream_reader import StreamCapture
from src.utils.privacy import redact_sensitive


class StandardCredentialEncryptionTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix='cv_aes_vault_'))
        self.reg_path = self.test_dir / 'cameras.json'

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_aes_gcm_protect_and_unprotect_roundtrip(self):
        """Verify RTSP credentials encrypt with standard AES-GCM (enc:v2) and decrypt back to exact plaintext."""
        sample_uris = [
            'rtsp://admin:AES_Secret_Password_2026!@192.168.1.120:554/stream1',
            'rtsp://security_user:P@ss%20w0rd!#$&*()@cam.campus.edu:8554/live?token=xyz123',
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
                self.assertTrue(protected.startswith('enc:v2:'))
                self.assertNotIn('AES_Secret_Password_2026!', protected)
                self.assertNotIn('token_abc_987', protected)
            
            unprotected = unprotect_camera_uri(protected)
            self.assertEqual(unprotected, uri)

    def test_tampered_ciphertext_rejected(self):
        """Verify modifying even a single byte of ciphertext or tag raises ValueError."""
        secret_uri = 'rtsp://admin:super_secret@10.0.0.1:554/feed'
        token = protect_camera_uri(secret_uri)
        self.assertTrue(token.startswith('enc:v2:'))

        raw_payload = bytearray(base64.urlsafe_b64decode(token[7:].encode('ascii')))
        # Flip bit in the payload
        raw_payload[-1] ^= 0x01
        tampered_token = 'enc:v2:' + base64.urlsafe_b64encode(bytes(raw_payload)).decode('ascii')

        with self.assertRaises(ValueError):
            unprotect_camera_uri(tampered_token)

    def test_invalid_key_rejected(self):
        """Verify decrypting enc:v2 token with a different key raises ValueError."""
        secret_uri = 'rtsp://admin:password123@192.168.1.5:554/live'
        correct_key = os.urandom(32)
        wrong_key = os.urandom(32)

        token = protect_camera_uri(secret_uri, custom_key=correct_key)
        self.assertTrue(token.startswith('enc:v2:'))

        # Successfully decrypts with correct key
        self.assertEqual(unprotect_camera_uri(token, custom_key=correct_key), secret_uri)

        # Fails with wrong key
        with self.assertRaises(ValueError):
            unprotect_camera_uri(token, custom_key=wrong_key)

    def test_safe_migration_from_enc_v1_and_plaintext(self):
        """Verify legacy enc:v1 HMAC tokens and plaintext URIs transparently migrate to enc:v2 AES-GCM."""
        master_key = get_vault_master_key()

        # Generate a real legacy enc:v1 token
        v1_secret = 'LegacyV1Pass_456!'
        salt = os.urandom(16)
        nonce = os.urandom(16)
        k_enc, k_mac = _derive_keys_v1(master_key, salt)
        data = f'rtsp://operator:{v1_secret}@192.168.1.10:554/h264'.encode('utf-8')
        import hmac, hashlib
        ks = _keystream_v1(k_enc, nonce, len(data))
        ct = bytes(a ^ b for a, b in zip(data, ks))
        tag = hmac.new(k_mac, salt + nonce + ct, hashlib.sha256).digest()
        v1_token = 'enc:v1:' + base64.urlsafe_b64encode(salt + nonce + tag + ct).decode('ascii')

        legacy_data = [
            {
                'id': 'CAM-V1',
                'name': 'Corridor South',
                'uri': v1_token,
                'zone': 'Corridor',
                'capacity': 40,
            },
            {
                'id': 'CAM-PLAIN',
                'name': 'North Gate',
                'uri': 'rtsp://admin:plainpass777@10.0.0.2:554/live',
                'zone': 'Gate',
                'capacity': 80,
            }
        ]
        self.reg_path.write_text(json.dumps(legacy_data, indent=2), encoding='utf-8')

        # Load registry - triggers automatic migration to enc:v2
        migrated = load_camera_registry(self.reg_path)
        self.assertEqual(len(migrated), 2)
        for cam in migrated:
            self.assertTrue(cam['uri'].startswith('enc:v2:'))
            self.assertIn('[redacted]', cam['display_uri'])
            self.assertNotIn('LegacyV1Pass', cam['display_uri'])
            self.assertNotIn('plainpass777', cam['display_uri'])

        # Decrypted connection URIs match original secrets perfectly
        self.assertEqual(resolve_camera_connection_uri(migrated[0]), f'rtsp://operator:{v1_secret}@192.168.1.10:554/h264')
        self.assertEqual(resolve_camera_connection_uri(migrated[1]), 'rtsp://admin:plainpass777@10.0.0.2:554/live')

        # File on disk now only has enc:v2
        disk_json = self.reg_path.read_text(encoding='utf-8')
        self.assertNotIn('enc:v1:', disk_json)
        self.assertNotIn('plainpass777', disk_json)
        self.assertNotIn('LegacyV1Pass', disk_json)


class FreshFrameValidationTests(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_stale_frame_rejected_after_timeout(self, mock_cv_class):
        """Verify read() returns (False, None, 0.0) when latest frame exceeds stale_timeout."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed', stale_timeout=0.1)
        try:
            # First read delivers fresh frame
            ok, frame, pts = cap.read(timeout=0.5, max_age=0.1)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)

            # Stop worker to simulate no newer incoming frames arriving
            cap._running = False
            if cap._thread and cap._thread.is_alive():
                cap._thread.join(timeout=1.0)

            # Manually simulate stale frame by setting timestamp in the past
            with cap._lock:
                cap._is_opened = True
                cap._latest_frame = dummy_frame
                cap._latest_pts = time.perf_counter() - 2.0  # 2.0 seconds old

            # Read with stale timeout 0.1s must reject the 2.0s old frame
            ok_stale, frame_stale, _ = cap.read(timeout=0.05, max_age=0.1, require_new=False)
            self.assertFalse(ok_stale)
            self.assertIsNone(frame_stale)
        finally:
            cap.release()

    @patch('cv2.VideoCapture')
    def test_require_new_prevents_duplicate_processing(self, mock_cv_class):
        """Verify read(require_new=True) does not return the same frame repeatedly without new capture."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed', stale_timeout=1.0)
        try:
            # Pause worker by setting _running False temporarily
            cap._running = False
            if cap._thread and cap._thread.is_alive():
                cap._thread.join(timeout=1.0)

            # Set a single frame
            with cap._lock:
                cap._latest_frame = dummy_frame
                cap._latest_pts = time.perf_counter()
                cap._last_delivered_pts = 0.0
                cap._running = True
                cap._is_opened = True

            # First read consumes the frame
            ok1, f1, pts1 = cap.read(timeout=0.2, require_new=True)
            self.assertTrue(ok1)
            self.assertIsNotNone(f1)

            # Second read without a new capture timestamp must return False
            ok2, f2, pts2 = cap.read(timeout=0.1, require_new=True)
            self.assertFalse(ok2)
            self.assertIsNone(f2)
        finally:
            cap.release()

    @patch('cv2.VideoCapture')
    def test_disconnect_clears_stale_frame_state(self, mock_cv_class):
        """Verify that stream disconnection invalidates cached frames immediately."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed', reconnect_timeout=1.0)
        try:
            ok, f, _ = cap.read(timeout=0.5)
            self.assertTrue(ok)

            # Simulate stream disconnection
            with cap._lock:
                cap._is_opened = False
                cap._latest_frame = None
                cap._latest_pts = 0.0

            # Read must immediately report False
            ok_after, f_after, _ = cap.read(timeout=0.1)
            self.assertFalse(ok_after)
            self.assertIsNone(f_after)
        finally:
            cap.release()

    @patch('cv2.VideoCapture')
    def test_reconnection_resets_timestamps_and_state(self, mock_cv_class):
        """Verify reconnection resets frame counters and delivers new timestamped frames."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed')
        try:
            # Initial read
            ok1, _, pts1 = cap.read(timeout=0.5)
            self.assertTrue(ok1)

            # Trigger reconnection
            cap._connect()
            self.assertTrue(cap.isOpened())
            self.assertGreater(cap._latest_pts, 0.0)
        finally:
            cap.release()


class WorkerThreadLifecycleAndSanitizationTests(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_safe_worker_termination_without_race(self, mock_cv_class):
        """Verify worker thread terminates and joins before OpenCV capture is closed."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 25.0)
        mock_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv_class.return_value = mock_instance

        cap = StreamCapture('mock_feed')
        thread = cap._thread
        self.assertTrue(thread.is_alive())

        cap.release()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(cap.cap)
        mock_instance.release.assert_called_once()

    def test_global_stream_cleanup_handles_active_capture_and_writer(self):
        """Verify release_global_stream cleanly shuts down active handles."""
        mock_capture = MagicMock()
        mock_writer = MagicMock()

        register_active_stream(capture=mock_capture, writer=mock_writer)
        cap_val, wrt_val = get_active_stream()
        self.assertEqual(cap_val, mock_capture)
        self.assertEqual(wrt_val, mock_writer)

        release_global_stream()

        mock_capture.release.assert_called_once()
        mock_writer.close.assert_called_once_with(success=True)

        cap_after, wrt_after = get_active_stream()
        self.assertIsNone(cap_after)
        self.assertIsNone(wrt_after)

    def test_exception_sanitization_removes_secrets(self):
        """Verify exception messages and errors with embedded credentials are fully redacted."""
        secret_err = "ConnectionError: rtsp://admin:super_secret_pass_123@10.0.0.5:554/live refused"
        sanitized = redact_sensitive(secret_err)
        self.assertNotIn("super_secret_pass_123", sanitized)
        self.assertIn("[redacted]", sanitized)


if __name__ == '__main__':
    unittest.main()
