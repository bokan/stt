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


WAVEFORM_BARS = 24
WAVEFORM_INTERVAL_S = 0.033  # ~30fps


class Recorder:
    def __init__(self):
        self._recording = False
        self._stream = None
        self._chunks = []
        self._sample_rate = None
        self._channels = None
        self._last_waveform_time = 0.0
        self._waveform_buffer = []
        self._peak_level = 0.01  # Auto-normalizing peak (starts low)

    def start(self, *, device_name: str | None, sample_rate: int, channels: int) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        import time
        import numpy as np
        import sounddevice as sd

        # Resolve device name to index at recording time (handles plug/unplug)
        device_index = None
        if device_name:
            device_index = self._resolve_device(device_name, sd)
            if device_index is None:
                raise RuntimeError(f"Audio device '{device_name}' not found")

        self._chunks = []
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording = True
        self._last_waveform_time = time.time()
        self._waveform_buffer = []
        self._peak_level = 0.01  # Reset peak for new recording

        def callback(indata, frames, time_info, status):
            if status:
                _log(f"[stt:audio-worker] Status: {status}")
            if self._recording:
                self._chunks.append(indata.copy())
                self._waveform_buffer.append(indata.copy())

                # Send waveform at interval
                now = time.time()
                if now - self._last_waveform_time >= WAVEFORM_INTERVAL_S:
                    self._last_waveform_time = now
                    self._send_waveform(np)

        stream = sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32,
            callback=callback,
        )
        stream.start()
        self._stream = stream

    def _resolve_device(self, name: str, sd) -> int | None:
        """Resolve device name to current index."""
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and dev['name'] == name:
                return i
        return None

    def _send_waveform(self, np) -> None:
        """Calculate and send waveform data with auto-normalization"""
        if not self._waveform_buffer:
            return

        # Concatenate recent audio
        audio = np.concatenate(self._waveform_buffer, axis=0)
        self._waveform_buffer = []

        # Take absolute values and flatten to mono
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = np.abs(audio)

        # Downsample to WAVEFORM_BARS values
        samples_per_bar = len(audio) // WAVEFORM_BARS
        if samples_per_bar < 1:
            samples_per_bar = 1

        raw_values = []
        for i in range(WAVEFORM_BARS):
            start = i * samples_per_bar
            end = start + samples_per_bar
            if end > len(audio):
                end = len(audio)
            if start < len(audio):
                # Use RMS for smoother visualization
                chunk = audio[start:end]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                raw_values.append(rms)
            else:
                raw_values.append(0.0)

        # Auto-normalize: track peak and scale to it
        current_max = max(raw_values) if raw_values else 0
        if current_max > self._peak_level:
            # Quick attack: immediately raise peak
            self._peak_level = current_max
        else:
            # Slow decay: gradually lower peak for consistent visualization
            self._peak_level = self._peak_level * 0.995 + current_max * 0.005
            # Don't let it drop too low
            self._peak_level = max(self._peak_level, 0.005)

        # Normalize values to 0-1 range based on peak
        values = [min(1.0, v / self._peak_level * 0.85) for v in raw_values]

        _write_json({"type": "waveform", "values": values, "raw_peak": current_max})

    def stop(self, *, wav_path: str) -> tuple[int, float]:
        if not self._recording:
            return 0, 0.0

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
            return 0, 0.0

        import numpy as np
        from scipy.io import wavfile

        audio = np.concatenate(chunks, axis=0)
        frames = int(audio.shape[0])
        peak = float(np.max(np.abs(audio)))

        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, int(self._sample_rate or 16000), audio_int16)
        return frames, peak

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
                    device_name=message.get("device_name"),
                    sample_rate=int(message.get("sample_rate") or 16000),
                    channels=int(message.get("channels") or 1),
                )
                _write_json({"type": "started", "id": req_id})
                continue

            if msg_type == "stop":
                wav_path = message.get("wav_path")
                if not wav_path:
                    raise ValueError("Missing wav_path")
                frames, peak = recorder.stop(wav_path=str(wav_path))
                _write_json({"type": "stopped", "id": req_id, "wav_path": wav_path, "frames": frames, "peak": peak})
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

