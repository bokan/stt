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

        print("Transcribing...")

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
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit stderr to avoid pipe deadlocks
            text=True,
            bufsize=1,
            env=env,
        )

        messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        thread = threading.Thread(target=self._read_stdout, args=(proc, messages), daemon=True)
        thread.start()

        self._proc = proc
        self._messages = messages
        self._reader_thread = thread

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

        # Close stdin first to signal worker to stop and unblock any writes
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if not force and proc.poll() is None:
                # Give process a chance to exit gracefully
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.terminate()
                # Close stdout to unblock reader thread before waiting
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            # Ensure stdout is closed (may have been closed above, but that's ok)
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

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

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


class _WorkerClient:
    """Generic worker client for subprocess-based transcription."""

    def __init__(self, model: str, worker_script: str):
        self.model = model
        self.worker_script = worker_script
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

        worker_path = os.path.join(os.path.dirname(__file__), self.worker_script)
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )

        messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        thread = threading.Thread(target=self._read_stdout, args=(proc, messages), daemon=True)
        thread.start()

        self._proc = proc
        self._messages = messages
        self._reader_thread = thread

        ready = self._wait_for(lambda m: m.get("type") in {"ready", "error"}, timeout_s=startup_timeout_s)
        if not ready:
            raise TimeoutError("Worker did not become ready in time")
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("error") or "Worker failed to start")

    def stop(self, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if not force and proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.terminate()
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

    def send_request(self, request: dict[str, Any], timeout_s: int) -> dict[str, Any]:
        if not self.is_running():
            raise RuntimeError("Worker is not running")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

            request["id"] = req_id
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

            message = self._wait_for(
                lambda m: m.get("type") == "result" and m.get("id") == req_id,
                timeout_s=timeout_s,
            )

            if not message:
                raise TimeoutError("Timed out waiting for worker response")

            return message

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

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
                return {"type": "error", "error": "Worker exited unexpectedly"}

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
            if self.model in ("large-v3", "large-v3-turbo", "large", "medium", "small", "base", "tiny"):
                self.model = f"mlx-community/whisper-{self.model}-mlx"
        self._mlx_whisper = None
        self._use_worker = True
        self._worker: _MLXWorkerClient | None = None
        self._worker_startup_timeout_s = self._WORKER_STARTUP_TIMEOUT_S
        self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        self._cleanup_registered = False
        self._worker_lock = threading.Lock()
        self._restart_thread: threading.Thread | None = None
        self._ever_started = False
        self._shutting_down = False

        monitor = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor.start()

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
        from rich.console import Console
        console = Console()
        model_name = self.model.split('/')[-1]
        console.print(f"[dim]Loading model...[/dim]", end="")

        if self._use_worker:
            try:
                self._ensure_worker()
                console.print(f"\r[green]✓[/green] Model: [cyan]{model_name}[/cyan]   ")
            except Exception as e:
                console.print(f"\r[red]✗[/red] Failed to load model: {e}")
            return

        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx

        # Load into ModelHolder cache so transcribe() reuses it
        ModelHolder.get_model(self.model, mx.float16)
        self._mlx_whisper = mlx_whisper
        console.print(f"\r[green]✓[/green] Model: [cyan]{model_name}[/cyan]   ")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        if self._use_worker:
            print("Transcribing...")
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
                self._schedule_worker_warmup()
                return None
            except Exception as e:
                print(f"❌ MLX Whisper Error: {e}")
                self._stop_worker(force=True)
                self._schedule_worker_warmup()
                return None

        if self._mlx_whisper is None:
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
            except ImportError:
                print("❌ mlx-whisper not installed. Run: pip install mlx-whisper")
                return None

        print("Transcribing...")

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
            self._schedule_worker_warmup()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._shutting_down:
                return
            if self._worker is None:
                self._worker = _MLXWorkerClient(model=self.model)
            if not self._worker.is_running():
                self._worker.start(startup_timeout_s=self._worker_startup_timeout_s)
                self._ever_started = True
            if not self._cleanup_registered:
                atexit.register(self._shutdown)
                self._cleanup_registered = True

    def _stop_worker(self, force: bool = False) -> None:
        with self._worker_lock:
            worker = self._worker
        if worker is None:
            return
        worker.stop(force=force)

    def _shutdown(self) -> None:
        self._shutting_down = True
        self._stop_worker(force=True)

    def _schedule_worker_warmup(self) -> None:
        if self._shutting_down:
            return
        with self._worker_lock:
            existing = self._restart_thread
            if existing and existing.is_alive():
                return
            thread = threading.Thread(target=self._warm_worker, daemon=True)
            self._restart_thread = thread
            thread.start()

    def _warm_worker(self) -> None:
        try:
            self._ensure_worker()
        except Exception as e:
            print(f"⚠️  Failed to restart MLX worker: {e}")

    def _monitor_worker(self) -> None:
        while not self._shutting_down:
            if self._use_worker and self._ever_started:
                try:
                    with self._worker_lock:
                        worker = self._worker
                    if worker is not None and not worker.is_running():
                        self._schedule_worker_warmup()
                except Exception:
                    pass
            time.sleep(0.5)


class ParakeetProvider(TranscriptionProvider):
    """Nvidia Parakeet provider using MLX (Apple Silicon optimized, English-only)"""

    DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
    _WORKER_STARTUP_TIMEOUT_S = 1800
    _TRANSCRIBE_TIMEOUT_S = 180

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("PARAKEET_MODEL", self.DEFAULT_MODEL)
        self._worker: _WorkerClient | None = None
        self._worker_startup_timeout_s = self._WORKER_STARTUP_TIMEOUT_S
        self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        self._cleanup_registered = False
        self._worker_lock = threading.Lock()
        self._restart_thread: threading.Thread | None = None
        self._ever_started = False
        self._shutting_down = False

        monitor = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor.start()

    @property
    def name(self) -> str:
        return f"Parakeet ({self.model.split('/')[-1]})"

    def is_available(self) -> bool:
        try:
            import parakeet_mlx
            return True
        except ImportError:
            return False

    def warmup(self) -> None:
        """Pre-load the model at startup (downloads if needed)"""
        from rich.console import Console
        console = Console()
        model_name = self.model.split('/')[-1]
        console.print(f"[dim]Loading model...[/dim]", end="")
        try:
            self._ensure_worker()
            console.print(f"\r[green]✓[/green] Model: [cyan]{model_name}[/cyan]   ")
        except Exception as e:
            console.print(f"\r[red]✗[/red] Failed to load model: {e}")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        # Parakeet is English-only
        if language and language.lower() not in ("en", "english"):
            print(f"❌ Parakeet only supports English. Got: {language}")
            return None

        print("Transcribing...")
        try:
            self._ensure_worker()
            assert self._worker is not None
            result = self._worker.send_request(
                {"type": "transcribe", "audio_file_path": audio_file_path},
                timeout_s=self._transcribe_timeout_s,
            )
            error = result.get("error")
            if error:
                raise RuntimeError(str(error))
            text = str(result.get("text") or "").strip()

            # Apply phonetic correction using PROMPT vocabulary
            if prompt and text:
                from postprocess import parse_vocabulary, correct_text
                vocab = parse_vocabulary(prompt)
                if vocab:
                    text = correct_text(text, vocab)

            return text
        except TimeoutError:
            print("❌ Parakeet transcription timed out. Restarting worker...")
            self._stop_worker(force=True)
            self._schedule_worker_warmup()
            return None
        except Exception as e:
            print(f"❌ Parakeet Error: {e}")
            self._stop_worker(force=True)
            self._schedule_worker_warmup()
            return None

    def cancel(self) -> None:
        """Best-effort cancellation of an in-flight transcription."""
        self._stop_worker(force=True)
        self._schedule_worker_warmup()

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._shutting_down:
                return
            if self._worker is None:
                self._worker = _WorkerClient(model=self.model, worker_script="parakeet_worker.py")
            if not self._worker.is_running():
                self._worker.start(startup_timeout_s=self._worker_startup_timeout_s)
                self._ever_started = True
            if not self._cleanup_registered:
                atexit.register(self._shutdown)
                self._cleanup_registered = True

    def _stop_worker(self, force: bool = False) -> None:
        with self._worker_lock:
            worker = self._worker
        if worker is None:
            return
        worker.stop(force=force)

    def _shutdown(self) -> None:
        self._shutting_down = True
        self._stop_worker(force=True)

    def _schedule_worker_warmup(self) -> None:
        if self._shutting_down:
            return
        with self._worker_lock:
            existing = self._restart_thread
            if existing and existing.is_alive():
                return
            thread = threading.Thread(target=self._warm_worker, daemon=True)
            self._restart_thread = thread
            thread.start()

    def _warm_worker(self) -> None:
        try:
            self._ensure_worker()
        except Exception as e:
            print(f"⚠️  Failed to restart Parakeet worker: {e}")

    def _monitor_worker(self) -> None:
        while not self._shutting_down:
            if self._ever_started:
                try:
                    with self._worker_lock:
                        worker = self._worker
                    if worker is not None and not worker.is_running():
                        self._schedule_worker_warmup()
                except Exception:
                    pass
            time.sleep(0.5)


def get_provider(provider_name: str = None) -> TranscriptionProvider:
    """Get a transcription provider by name"""
    provider_name = provider_name or os.environ.get("PROVIDER", "mlx")
    provider_name = provider_name.lower()

    providers = {
        "groq": GroqProvider,
        "mlx": MLXWhisperProvider,
        "parakeet": ParakeetProvider,
    }

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    return providers[provider_name]()
