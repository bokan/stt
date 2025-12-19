#!/usr/bin/env python3
"""
MLX Whisper worker process for STT.

Runs MLX transcription in a separate process so the main app stays responsive and
can recover by restarting the worker if MLX hangs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any


def _write_json(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="STT MLX Whisper worker")
    parser.add_argument("--model", required=True, help="HF repo or local path for MLX Whisper model")
    args = parser.parse_args()

    model = args.model

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx

        _log(f"[stt:mlx-worker] Loading model: {model}")
        ModelHolder.get_model(model, mx.float16)
        _write_json({"type": "ready"})
        _log("[stt:mlx-worker] Ready")
    except Exception as e:
        _write_json({"type": "error", "error": f"Failed to initialize MLX Whisper: {e}"})
        _log(traceback.format_exc())
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception:
            _log(f"[stt:mlx-worker] Non-JSON input ignored: {line!r}")
            continue

        msg_type = message.get("type")
        if msg_type == "shutdown":
            _write_json({"type": "shutdown_ack"})
            return 0

        if msg_type != "transcribe":
            _log(f"[stt:mlx-worker] Unknown message type: {msg_type!r}")
            continue

        req_id = message.get("id")
        audio_file_path = message.get("audio_file_path")
        language = message.get("language")
        prompt = message.get("prompt") or None

        try:
            result = mlx_whisper.transcribe(
                audio_file_path,
                path_or_hf_repo=model,
                language=language,
                initial_prompt=prompt,
                verbose=False,
            )
            text = (result.get("text") or "").strip()
            _write_json({"type": "result", "id": req_id, "text": text, "error": None})
        except Exception as e:
            _log(traceback.format_exc())
            _write_json({"type": "result", "id": req_id, "text": "", "error": str(e)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

