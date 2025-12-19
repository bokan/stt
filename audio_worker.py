#!/usr/bin/env python3
"""
Audio recording worker process for STT.

Runs PortAudio/sounddevice recording in a separate process so any rare driver
deadlocks during stop/close can't freeze the main UI.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


def _write_json(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Recorder:
    def __init__(self):
        self._recording = False
        self._stream = None
        self._chunks = []
        self._sample_rate = None
        self._channels = None

    def start(self, *, device: int | None, sample_rate: int, channels: int) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        import numpy as np
        import sounddevice as sd

        self._chunks = []
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording = True

        def callback(indata, frames, time, status):
            if status:
                _log(f"[stt:audio-worker] Status: {status}")
            if self._recording:
                self._chunks.append(indata.copy())

        stream = sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32,
            callback=callback,
        )
        stream.start()
        self._stream = stream

    def stop(self, *, wav_path: str) -> int:
        if not self._recording:
            return 0

        self._recording = False
        stream = self._stream
        self._stream = None
        chunks = self._chunks
        self._chunks = []

        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            except Exception:
                _log(traceback.format_exc())

        if not chunks:
            return 0

        import numpy as np
        from scipy.io import wavfile

        audio = np.concatenate(chunks, axis=0)
        frames = int(audio.shape[0])

        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, int(self._sample_rate or 16000), audio_int16)
        return frames

    def cancel(self) -> None:
        self._recording = False
        self._chunks = []

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            except Exception:
                _log(traceback.format_exc())

    def shutdown(self) -> None:
        self.cancel()


def main() -> int:
    recorder = Recorder()
    _write_json({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception:
            _log(f"[stt:audio-worker] Non-JSON input ignored: {line!r}")
            continue

        msg_type = message.get("type")
        req_id = message.get("id")

        try:
            if msg_type == "shutdown":
                recorder.shutdown()
                _write_json({"type": "shutdown_ack"})
                return 0

            if msg_type == "start":
                recorder.start(
                    device=message.get("device"),
                    sample_rate=int(message.get("sample_rate") or 16000),
                    channels=int(message.get("channels") or 1),
                )
                _write_json({"type": "started", "id": req_id})
                continue

            if msg_type == "stop":
                wav_path = message.get("wav_path")
                if not wav_path:
                    raise ValueError("Missing wav_path")
                frames = recorder.stop(wav_path=str(wav_path))
                _write_json({"type": "stopped", "id": req_id, "wav_path": wav_path, "frames": frames})
                continue

            if msg_type == "cancel":
                recorder.cancel()
                _write_json({"type": "canceled", "id": req_id})
                continue

            _log(f"[stt:audio-worker] Unknown message type: {msg_type!r}")
        except Exception as e:
            _log(traceback.format_exc())
            _write_json({"type": "error", "id": req_id, "error": str(e)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

