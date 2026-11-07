"""
Camera Credential Security Vault and Local Registry Service.

Ensures RTSP credentials (passwords, tokens) are NEVER stored in plaintext in cameras.json.
Provides:
- Standard AES-256-GCM authenticated encryption from the `cryptography` library.
- Windows DPAPI master key protection where available with secure 0o600 cross-platform fallback.
- Transparent, loss-free migration from existing `enc:v1` keystream entries to `enc:v2` AES-GCM.
- Automatic password redaction across UI listings, logs, exception traces, and reports.
"""
from __future__ import annotations
import base64
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import config
from src.utils.privacy import redact_sensitive

_VAULT_KEY_FILE = Path(config.OUTPUTS_ROOT) / '.camera_vault.key'
_REGISTRY_FILE = Path(config.OUTPUTS_ROOT) / 'cameras.json'
_MASTER_KEY_CACHE: Optional[bytes] = None


def _dpapi_available() -> bool:
    return sys.platform == 'win32'


def _dpapi_protect(data: bytes) -> bytes:
    from ctypes import wintypes
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_byte))]

    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_byte)))
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(blob_in), 'CrowdVisionVault', None, None, None, 0, ctypes.byref(blob_out)):
        raise ctypes.WinError()
    res = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return res


def _dpapi_unprotect(data: bytes) -> bytes:
    from ctypes import wintypes
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_byte))]

    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_byte)))
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise ctypes.WinError()
    res = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return res


def get_vault_master_key() -> bytes:
    """
    Retrieve or generate the local 32-byte (256-bit) AES vault master key.
    Uses Windows DPAPI on Windows; uses restricted permissions (0o600) on non-Windows.
    """
    global _MASTER_KEY_CACHE
    if _MASTER_KEY_CACHE is not None:
        return _MASTER_KEY_CACHE

    _VAULT_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _VAULT_KEY_FILE.is_file():
        try:
            raw = _VAULT_KEY_FILE.read_bytes()
            if raw.startswith(b'DPAPI:'):
                key = _dpapi_unprotect(raw[6:])
            elif raw.startswith(b'RAW:'):
                key = raw[4:]
            else:
                key = raw
            if len(key) == 32:
                _MASTER_KEY_CACHE = key
                return key
        except Exception:
            pass  # Fall through to regenerate if corrupted

    # Generate fresh 32-byte key
    key = secrets.token_bytes(32)
    saved = False
    if _dpapi_available():
        try:
            enc = _dpapi_protect(key)
            _VAULT_KEY_FILE.write_bytes(b'DPAPI:' + enc)
            saved = True
        except Exception:
            pass

    if not saved:
        _VAULT_KEY_FILE.write_bytes(b'RAW:' + key)
        try:
            os.chmod(str(_VAULT_KEY_FILE), 0o600)
        except Exception:
            pass

    _MASTER_KEY_CACHE = key
    return key


def _derive_keys_v1(master_key: bytes, salt: bytes) -> Tuple[bytes, bytes]:
    k_enc = hmac.new(master_key, salt + b':enc', hashlib.sha256).digest()
    k_mac = hmac.new(master_key, salt + b':mac', hashlib.sha256).digest()
    return k_enc, k_mac


def _keystream_v1(key: bytes, nonce: bytes, length: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = hmac.new(key, nonce + counter.to_bytes(4, 'big'), hashlib.sha256).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def _decrypt_legacy_v1(token: str, master_key: bytes) -> str:
    raw = base64.urlsafe_b64decode(token[7:].encode('ascii'))
    if len(raw) < 64:
        raise ValueError('Tampered or corrupted v1 credential token')
    salt = raw[:16]
    nonce = raw[16:32]
    tag = raw[32:64]
    ct = raw[64:]
    k_enc, k_mac = _derive_keys_v1(master_key, salt)
    expected_tag = hmac.new(k_mac, salt + nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError('Tampered or corrupted v1 credential token')
    ks = _keystream_v1(k_enc, nonce, len(ct))
    pt = bytes(a ^ b for a, b in zip(ct, ks))
    return pt.decode('utf-8')


def protect_camera_uri(uri: Union[str, int], custom_key: Optional[bytes] = None) -> str:
    """
    Encrypt a camera URI or credentials string with standard AES-256-GCM.
    Returns 'enc:v2:<base64-payload>' containing a 96-bit nonce and authenticated ciphertext+tag.
    Leaves webcam indices (e.g. '0' or 0) and empty strings untouched.
    """
    if uri is None:
        return ""
    s = str(uri).strip()
    if s.isdigit() or s == "":
        return s
    if s.startswith('enc:v2:'):
        return s

    master_key = custom_key if custom_key is not None else get_vault_master_key()
    aesgcm = AESGCM(master_key)
    nonce = secrets.token_bytes(12)  # Standard 96-bit IV for AES-GCM
    ct = aesgcm.encrypt(nonce, s.encode('utf-8'), None)
    payload = nonce + ct
    return 'enc:v2:' + base64.urlsafe_b64encode(payload).decode('ascii')


def unprotect_camera_uri(protected_token: Union[str, int], custom_key: Optional[bytes] = None) -> str:
    """
    Decrypt a camera URI from AES-GCM ('enc:v2:...') or legacy ('enc:v1:...').
    Raises ValueError on tampering or invalid key.
    Plain webcam indices and unencrypted legacy strings pass through safely.
    """
    if protected_token is None:
        return ""
    s = str(protected_token).strip()
    if not s.startswith(('enc:v1:', 'enc:v2:')):
        return s

    master_key = custom_key if custom_key is not None else get_vault_master_key()

    if s.startswith('enc:v1:'):
        return _decrypt_legacy_v1(s, master_key)

    # Standard AES-256-GCM
    try:
        raw = base64.urlsafe_b64decode(s[7:].encode('ascii'))
        if len(raw) < 28:  # 12-byte nonce + 16-byte minimum tag
            raise ValueError('Truncated or invalid ciphertext')
        nonce = raw[:12]
        ct = raw[12:]
        aesgcm = AESGCM(master_key)
        pt = aesgcm.decrypt(nonce, ct, None)
        return pt.decode('utf-8')
    except (InvalidTag, ValueError) as err:
        raise ValueError('Tampered or corrupted encrypted credential') from err


def redact_camera_uri(uri_or_token: Union[str, int]) -> str:
    """
    Produce a user-safe representation of any camera URI with secrets masked.
    If the URI is protected, decrypts in memory strictly for redaction.
    """
    if uri_or_token is None:
        return ""
    s = str(uri_or_token).strip()
    if s.isdigit():
        return f"Local Device #{s}"
    if s.startswith(('enc:v1:', 'enc:v2:')):
        try:
            plaintext = unprotect_camera_uri(s)
            return str(redact_sensitive(plaintext))
        except Exception:
            return "rtsp://[redacted]@[protected-feed]"
    return str(redact_sensitive(s))


def load_camera_registry(registry_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load registered cameras from cameras.json.
    Transparently migrates legacy plaintext passwords and `enc:v1` entries to `enc:v2` (AES-GCM),
    immediately persisting the upgraded records back to disk.
    """
    path = registry_path or _REGISTRY_FILE
    if not path.is_file():
        return []

    try:
        content = path.read_text(encoding='utf-8')
        cams = json.loads(content)
        if not isinstance(cams, list):
            return []
    except Exception:
        return []

    migrated = False
    cleaned_cams = []
    for c in cams:
        if not isinstance(c, dict):
            continue
        raw_uri = str(c.get('uri', '')).strip()

        # Migrate unencrypted or legacy enc:v1 to modern enc:v2
        if raw_uri and not raw_uri.isdigit():
            if not raw_uri.startswith('enc:v2:'):
                try:
                    decrypted = unprotect_camera_uri(raw_uri)
                    c['uri'] = protect_camera_uri(decrypted)
                    migrated = True
                except Exception:
                    # Keep as is if corrupted
                    pass

        # Ensure display_uri is always redacted
        c['display_uri'] = redact_camera_uri(c.get('uri', ''))
        cleaned_cams.append(c)

    if migrated:
        try:
            save_camera_registry(cleaned_cams, path)
        except Exception:
            pass

    return cleaned_cams


def save_camera_registry(cameras: List[Dict[str, Any]], registry_path: Optional[Path] = None):
    """
    Persist camera list to cameras.json, ensuring all credentials use AES-GCM (enc:v2).
    """
    path = registry_path or _REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = []
    for c in cameras:
        cam_copy = dict(c)
        raw_uri = str(cam_copy.get('uri', '')).strip()
        if raw_uri and not raw_uri.startswith('enc:v2:'):
            try:
                decrypted = unprotect_camera_uri(raw_uri)
                cam_copy['uri'] = protect_camera_uri(decrypted)
            except Exception:
                cam_copy['uri'] = protect_camera_uri(raw_uri)
        cam_copy['display_uri'] = redact_camera_uri(cam_copy.get('uri', ''))
        sanitized.append(cam_copy)

    temp_path = path.with_suffix('.tmp')
    temp_path.write_text(json.dumps(sanitized, indent=2), encoding='utf-8')
    temp_path.replace(path)


def resolve_camera_connection_uri(cam_or_uri: Union[Dict[str, Any], str, int]) -> Union[str, int]:
    """Extract and unprotect connection string for StreamCapture."""
    if isinstance(cam_or_uri, dict):
        token = cam_or_uri.get('uri', '')
        try:
            return unprotect_camera_uri(token)
        except Exception:
            return ""
    if isinstance(cam_or_uri, int):
        return cam_or_uri
    s = str(cam_or_uri).strip()
    if s.startswith(('enc:v1:', 'enc:v2:')):
        try:
            return unprotect_camera_uri(s)
        except Exception:
            return ""
    return s


def test_camera_connection(source_uri: Union[str, int]) -> Tuple[bool, int, int, int]:
    """
    Safely probe a stream URL or webcam index.
    Credentials are unprotected strictly in memory and never logged or leaked.
    Returns (ok, width, height, fps).
    """
    from src.core.stream_reader import StreamCapture
    clean_target = resolve_camera_connection_uri(source_uri)
    try:
        cap = StreamCapture(clean_target, target_fps=5)
        ok, frame, _ = cap.read(timeout=2.5)
        w, h, fps = cap.width, cap.height, int(cap.fps)
        cap.release()
        return ok and frame is not None, w, h, fps
    except Exception:
        return False, 0, 0, 0


# ---------------------------------------------------------
# Global Live Stream Resource Tracking and Cleanup
# ---------------------------------------------------------
_GLOBAL_CAPTURE_LOCK = threading.Lock()
_CURRENT_CAPTURE: Optional[Any] = None
_CURRENT_WRITER: Optional[Any] = None


def register_active_stream(capture: Optional[Any] = None, writer: Optional[Any] = None):
    """Register active stream capture and writer objects for tracking and cleanup."""
    global _CURRENT_CAPTURE, _CURRENT_WRITER
    with _GLOBAL_CAPTURE_LOCK:
        if capture is not None:
            _CURRENT_CAPTURE = capture
        if writer is not None:
            _CURRENT_WRITER = writer


def get_active_stream() -> Tuple[Optional[Any], Optional[Any]]:
    """Return currently active (capture, writer) handles."""
    with _GLOBAL_CAPTURE_LOCK:
        return _CURRENT_CAPTURE, _CURRENT_WRITER


def release_global_stream():
    """Immediately stop and release any active OpenCV capture and video writer resources."""
    global _CURRENT_CAPTURE, _CURRENT_WRITER
    with _GLOBAL_CAPTURE_LOCK:
        if _CURRENT_CAPTURE is not None:
            try:
                _CURRENT_CAPTURE.release()
            except Exception:
                pass
            _CURRENT_CAPTURE = None
        if _CURRENT_WRITER is not None:
            try:
                _CURRENT_WRITER.close(success=True)
            except Exception:
                pass
            _CURRENT_WRITER = None
