"""
Camera Credential Security Vault and Local Registry Service.

Ensures RTSP credentials (passwords, tokens) are NEVER stored in plaintext in cameras.json.
Provides:
- Authenticated local encryption with salt and HMAC tag verification.
- Windows DPAPI master key protection where available with cross-platform fallback.
- Redaction of credentials from UI listings, logs, exception traces, and reports.
- Safe, transparent migration of existing plaintext camera registries on first load.
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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Retrieve or generate the local 32-byte vault master key."""
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
            pass  # Re-initialize if corrupted

    # Generate fresh key
    key = secrets.token_bytes(32)
    try:
        if _dpapi_available():
            enc = _dpapi_protect(key)
            _VAULT_KEY_FILE.write_bytes(b'DPAPI:' + enc)
        else:
            _VAULT_KEY_FILE.write_bytes(b'RAW:' + key)
            try:
                os.chmod(str(_VAULT_KEY_FILE), 0o600)
            except Exception:
                pass
    except Exception:
        # Fallback to direct raw key
        _VAULT_KEY_FILE.write_bytes(b'RAW:' + key)

    _MASTER_KEY_CACHE = key
    return key


def _derive_keys(master_key: bytes, salt: bytes) -> Tuple[bytes, bytes]:
    k_enc = hmac.new(master_key, salt + b':enc', hashlib.sha256).digest()
    k_mac = hmac.new(master_key, salt + b':mac', hashlib.sha256).digest()
    return k_enc, k_mac


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = hmac.new(key, nonce + counter.to_bytes(4, 'big'), hashlib.sha256).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def protect_camera_uri(uri: Union[str, int]) -> str:
    """
    Encrypt a camera URI or credentials string for safe storage in cameras.json.
    Returns 'enc:v1:<base64-payload>'.
    Leaves webcam indices (like '0' or 0) unencrypted.
    """
    if uri is None:
        return ""
    s = str(uri).strip()
    if s.isdigit() or s == "":
        return s
    if s.startswith('enc:v1:'):
        return s

    master_key = get_vault_master_key()
    data = s.encode('utf-8')
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    k_enc, k_mac = _derive_keys(master_key, salt)
    ks = _keystream(k_enc, nonce, len(data))
    ct = bytes(a ^ b for a, b in zip(data, ks))
    tag = hmac.new(k_mac, salt + nonce + ct, hashlib.sha256).digest()
    payload = salt + nonce + tag + ct
    return 'enc:v1:' + base64.urlsafe_b64encode(payload).decode('ascii')


def unprotect_camera_uri(protected_token: Union[str, int]) -> str:
    """
    Decrypt a camera URI from its protected 'enc:v1:...' form back to plaintext
    for StreamCapture ingestion. Plain webcam indices and legacy plaintext
    strings pass through safely.
    """
    if protected_token is None:
        return ""
    s = str(protected_token).strip()
    if not s.startswith('enc:v1:'):
        return s

    try:
        master_key = get_vault_master_key()
        raw = base64.urlsafe_b64decode(s[7:].encode('ascii'))
        if len(raw) < 64:
            return ""
        salt = raw[:16]
        nonce = raw[16:32]
        tag = raw[32:64]
        ct = raw[64:]
        k_enc, k_mac = _derive_keys(master_key, salt)
        expected_tag = hmac.new(k_mac, salt + nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            return ""
        ks = _keystream(k_enc, nonce, len(ct))
        pt = bytes(a ^ b for a, b in zip(ct, ks))
        return pt.decode('utf-8')
    except Exception:
        return ""


def redact_camera_uri(uri_or_token: Union[str, int]) -> str:
    """
    Produce a user-safe representation of any camera URI with secrets masked.
    If the URI is protected, decrypts in memory only to apply redaction.
    """
    if uri_or_token is None:
        return ""
    s = str(uri_or_token).strip()
    if s.isdigit():
        return f"Local Device #{s}"
    if s.startswith('enc:v1:'):
        plaintext = unprotect_camera_uri(s)
        return str(redact_sensitive(plaintext))
    return str(redact_sensitive(s))


def load_camera_registry(registry_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load registered cameras from cameras.json.
    Transparently migrates any legacy plaintext RTSP passwords to protected tokens
    and immediately updates cameras.json to eliminate plaintext storage.
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
        # Check if legacy plaintext contains credentials
        if raw_uri and not raw_uri.startswith('enc:v1:') and not raw_uri.isdigit():
            c['uri'] = protect_camera_uri(raw_uri)
            migrated = True

        # Ensure display_uri is always present and redacted
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
    Persist camera list to cameras.json, guaranteeing all credentials are encrypted.
    """
    path = registry_path or _REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = []
    for c in cameras:
        cam_copy = dict(c)
        raw_uri = str(cam_copy.get('uri', '')).strip()
        if raw_uri and not raw_uri.startswith('enc:v1:'):
            cam_copy['uri'] = protect_camera_uri(raw_uri)
        cam_copy['display_uri'] = redact_camera_uri(cam_copy.get('uri', ''))
        sanitized.append(cam_copy)

    temp_path = path.with_suffix('.tmp')
    temp_path.write_text(json.dumps(sanitized, indent=2), encoding='utf-8')
    temp_path.replace(path)


def resolve_camera_connection_uri(cam_or_uri: Union[Dict[str, Any], str, int]) -> Union[str, int]:
    """Extract and unprotect the connection string for StreamCapture."""
    if isinstance(cam_or_uri, dict):
        token = cam_or_uri.get('uri', '')
        return unprotect_camera_uri(token)
    if isinstance(cam_or_uri, int):
        return cam_or_uri
    s = str(cam_or_uri).strip()
    if s.startswith('enc:v1:'):
        return unprotect_camera_uri(s)
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


import threading

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

