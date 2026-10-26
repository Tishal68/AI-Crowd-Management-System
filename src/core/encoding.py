"""Direct H.264 compression without large temporary video files."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def ffmpeg_executable() -> str:
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('Video export needs FFmpeg. Install requirements.txt or add FFmpeg to PATH.') from exc


class CompactVideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float, bitrate: int, max_bytes: int = 128 * 1024**2):
        self.max_bytes = max_bytes
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_name(self.path.stem + '.partial.mp4')
        self.log = tempfile.TemporaryFile()
        self.closed = False
        self.process = subprocess.Popen([
            ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-b:v', str(bitrate), '-maxrate', str(bitrate),
            '-bufsize', str(bitrate * 2), '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.partial),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
            if self.partial.exists() and self.partial.stat().st_size > self.max_bytes:
                raise RuntimeError('Video output exceeded its size budget. Choose a lower resolution or shorter clip.')
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('Video compression failed. Check disk space and FFmpeg installation.') from exc

    def close(self, success: bool = True):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError('Video encoder timed out.')
            if success:
                if code:
                    self.log.seek(0)
                    raise RuntimeError('Video compression failed: ' + self.log.read().decode(errors='replace')[-1200:])
                if self.partial.stat().st_size > self.max_bytes:
                    raise RuntimeError('Video output exceeded its size budget. Choose a lower resolution or shorter clip.')
                os.replace(self.partial, self.path)
        finally:
            self.log.close()
            self.partial.unlink(missing_ok=True)
