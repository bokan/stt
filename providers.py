"""
Transcription providers for STT
"""

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any


class TranscriptionProvider(ABC):
    """Base class for transcription providers"""

    @abstractmethod
    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        """Transcribe audio file to text"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name"""
        pass

    def warmup(self) -> None:
        """Initialize/preload resources. Override if needed."""
        pass


class GroqProvider(TranscriptionProvider):
    """Groq Whisper API provider (cloud)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    @property
    def name(self) -> str:
        return "Groq (cloud)"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        import requests

        print("🔄 Transcribing via Groq...")

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3",
                "response_format": "text",
                "language": language,
            }
            if prompt:
                data["prompt"] = prompt

            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=(5, 20))
                response.raise_for_status()
                return response.text.strip()
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return None


class _MLXWorkerClient:
    def __init__(self, model: str):
        self.model = model
        self._proc: subprocess.Popen[str] | None = None
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, startup_timeout_s: int) -> None:
        if self.is_running():
            return

        worker_path = os.path.join(os.path.dirname(__file__), "mlx_worker.py")
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing MLX worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit stderr to avoid pipe deadlocks
            text=True,
            bufsize=1,
            env=env,
        )

        assert self._proc.stdout is not None
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

        ready = self._wait_for(lambda m: m.get("type") in {"ready", "error"}, timeout_s=startup_timeout_s)
        if not ready:
            raise TimeoutError("MLX worker did not become ready in time")
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("error") or "MLX worker failed to start")

    def stop(self, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        try:
            if not force and proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=2)
                except Exception:
                    pass

            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=2)
            except Exception:
                pass
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

    def transcribe(
        self,
        audio_file_path: str,
        language: str,
        prompt: str | None,
        timeout_s: int,
    ) -> str:
        if not self.is_running():
            raise RuntimeError("MLX worker is not running")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(
                json.dumps(
                    {
                        "type": "transcribe",
                        "id": req_id,
                        "audio_file_path": audio_file_path,
                        "language": language,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._proc.stdin.flush()

            message = self._wait_for(
                lambda m: m.get("type") == "result" and m.get("id") == req_id,
                timeout_s=timeout_s,
            )

            if not message:
                raise TimeoutError("Timed out waiting for MLX worker response")

            error = message.get("error")
            if error:
                raise RuntimeError(str(error))

            return str(message.get("text") or "").strip()

    def _read_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                self._messages.put({"type": "stdout", "line": line})
        self._messages.put({"type": "eof"})

    def _wait_for(self, predicate, timeout_s: int) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s if timeout_s > 0 else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
            else:
                remaining = None

            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                return None

            if message.get("type") == "eof":
                return {"type": "error", "error": "MLX worker exited unexpectedly"}

            if predicate(message):
                return message


class MLXWhisperProvider(TranscriptionProvider):
    """Local Whisper provider using MLX (Apple Silicon optimized)"""

    DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
    _WORKER_STARTUP_TIMEOUT_S = 1800
    _TRANSCRIBE_TIMEOUT_S = 180

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("WHISPER_MODEL", self.DEFAULT_MODEL)
        # Normalize model name if user provides short form
        if self.model and not self.model.startswith("mlx-community/"):
            if self.model in ("large-v3", "large", "medium", "small", "base", "tiny"):
                self.model = f"mlx-community/whisper-{self.model}-mlx"
        self._mlx_whisper = None
        self._use_worker = True
        self._worker: _MLXWorkerClient | None = None
        self._worker_startup_timeout_s = self._WORKER_STARTUP_TIMEOUT_S
        self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        self._cleanup_registered = False

    @property
    def name(self) -> str:
        return f"MLX Whisper ({self.model.split('/')[-1]})"

    def is_available(self) -> bool:
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    def warmup(self) -> None:
        """Pre-load the model at startup (downloads if needed)"""
        print(f"Loading MLX Whisper model ({self.model.split('/')[-1]})...", flush=True)

        if self._use_worker:
            try:
                self._ensure_worker()
                print("Model loaded.")
            except Exception as e:
                print(f"❌ Failed to start MLX worker: {e}")
            return

        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx

        # Load into ModelHolder cache so transcribe() reuses it
        ModelHolder.get_model(self.model, mx.float16)
        self._mlx_whisper = mlx_whisper
        print("Model loaded.")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        if self._use_worker:
            print("🔄 Transcribing...")
            try:
                self._ensure_worker()
                assert self._worker is not None
                return self._worker.transcribe(
                    audio_file_path=audio_file_path,
                    language=language,
                    prompt=prompt,
                    timeout_s=self._transcribe_timeout_s,
                )
            except TimeoutError:
                print("❌ MLX transcription timed out. Restarting MLX worker...")
                self._stop_worker(force=True)
                return None
            except Exception as e:
                print(f"❌ MLX Whisper Error: {e}")
                self._stop_worker(force=True)
                return None

        if self._mlx_whisper is None:
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
            except ImportError:
                print("❌ mlx-whisper not installed. Run: pip install mlx-whisper")
                return None

        print("🔄 Transcribing...")

        try:
            result = self._mlx_whisper.transcribe(
                audio_file_path,
                path_or_hf_repo=self.model,
                language=language,
                initial_prompt=prompt,
            )
            return result["text"].strip()
        except Exception as e:
            print(f"❌ MLX Whisper Error: {e}")
            return None

    def cancel(self) -> None:
        """Best-effort cancellation of an in-flight transcription."""
        if self._use_worker:
            self._stop_worker(force=True)

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = _MLXWorkerClient(model=self.model)
        if not self._worker.is_running():
            self._worker.start(startup_timeout_s=self._worker_startup_timeout_s)
        if not self._cleanup_registered:
            atexit.register(self._stop_worker)
            self._cleanup_registered = True

    def _stop_worker(self, force: bool = False) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        worker.stop(force=force)


def get_provider(provider_name: str = None) -> TranscriptionProvider:
    """Get a transcription provider by name"""
    provider_name = provider_name or os.environ.get("PROVIDER", "mlx")
    provider_name = provider_name.lower()

    providers = {
        "groq": GroqProvider,
        "mlx": MLXWhisperProvider,
    }

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    return providers[provider_name]()
