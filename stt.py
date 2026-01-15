#!/usr/bin/env python3
"""
Simple Speech-to-Text app for macOS
Hold Right Command key to record, release to transcribe and type
"""

import os
import re
import sys
import json
import tempfile
import threading
import queue
import time
import subprocess
import atexit
import fcntl
import concurrent.futures
from enum import Enum
from typing import Any, Callable, Optional

# Version for update checking
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("stt")
except Exception:
    __version__ = "0.0.0"  # Fallback for dev mode
RELEASES_URL = "https://api.github.com/repos/bokan/stt/releases/latest"

from dotenv import load_dotenv
import sounddevice as sd
from pynput import keyboard, mouse
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from providers import get_provider
from menubar import STTMenuBar
from overlay import get_overlay


class AppState(Enum):
    """Application state for menu bar icon"""
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


def check_for_updates():
    """Check if a newer version is available on GitHub releases"""
    try:
        from packaging.version import parse as parse_version
        response = requests.get(RELEASES_URL, timeout=5)
        if response.status_code == 200:
            latest = response.json()["tag_name"].lstrip("v")
            if parse_version(latest) > parse_version(__version__):
                print(f"\n📦 Update available: {__version__} → {latest}", flush=True)
                print(f"   Run: uv tool install --reinstall git+https://github.com/bokan/stt.git", flush=True)
    except Exception:
        pass  # Silently fail if offline or no releases


def check_accessibility_permissions():
    """Check and request accessibility permissions on macOS"""
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions
        # Setting this option to True will prompt the user if not trusted
        options = {"AXTrustedCheckOptionPrompt": True}
        trusted = AXIsProcessTrustedWithOptions(options)
        return trusted
    except ImportError:
        print("⚠️  Could not check accessibility permissions")
        return True  # Assume trusted if we can't check


# Config paths
CONFIG_DIR = os.path.expanduser("~/.config/stt")
CONFIG_FILE = os.path.join(CONFIG_DIR, ".env")
INITIALIZED_MARKER = os.path.join(CONFIG_DIR, ".initialized")
LOCK_FILE = os.path.join(tempfile.gettempdir(), "stt.lock")

# Global lock file handle
_lock_file = None


def acquire_lock():
    """Ensure only one instance is running"""
    global _lock_file
    _lock_file = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
        return True
    except (BlockingIOError, OSError):
        # Lock held by another process
        _lock_file.close()
        _lock_file = None
        return False


def release_lock():
    """Release the lock file"""
    global _lock_file
    if _lock_file:
        fcntl.flock(_lock_file, fcntl.LOCK_UN)
        _lock_file.close()
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass


def is_first_run() -> bool:
    """Check if this is the first run (no config initialized)"""
    return not os.path.exists(INITIALIZED_MARKER)


def mark_initialized():
    """Mark that first-run setup is complete"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(INITIALIZED_MARKER, "w") as f:
        f.write("")


# Load environment variables from .env file (check multiple locations)

# Try local .env first, then global config
load_dotenv()  # local .env
if os.path.exists(CONFIG_FILE):
    load_dotenv(CONFIG_FILE)

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "")
LANGUAGE = os.environ.get("LANGUAGE", "en")
HOTKEY = os.environ.get("HOTKEY", "cmd_r")  # Right Command by default
PROMPT = os.environ.get("PROMPT", "")  # Context prompt for Whisper
SOUND_ENABLED = os.environ.get("SOUND_ENABLED", "true").lower() == "true"
PROVIDER = os.environ.get("PROVIDER", "mlx")  # "mlx" (local) or "groq" (cloud)
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Skip transcription if peak below this

# Hotkey configuration
HOTKEYS = {
    "cmd_r": {"key": keyboard.Key.cmd_r, "name": "Right ⌘"},
    "alt_r": {"key": keyboard.Key.alt_r, "name": "Right ⌥"},
    "ctrl_r": {"key": keyboard.Key.ctrl_r, "name": "Right ⌃"},
    "shift_r": {"key": keyboard.Key.shift_r, "name": "Right ⇧"},
}

# macOS system sounds
SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_CANCEL = "/System/Library/Sounds/Basso.aiff"


class ConfigWatcher:
    """Watch config files for changes and trigger reload"""

    def __init__(self, on_config_change: Callable[[dict], None]):
        self._on_config_change = on_config_change
        self._observer: Optional[Observer] = None
        self._watched_files: set[str] = set()
        self._last_mtime: dict[str, float] = {}
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def start(self):
        """Start watching config files"""
        # Determine which files to watch
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            self._watched_files.add(local_env)
        if os.path.exists(CONFIG_FILE):
            self._watched_files.add(CONFIG_FILE)
        elif os.path.exists(CONFIG_DIR):
            # Watch the directory for file creation
            self._watched_files.add(CONFIG_FILE)

        if not self._watched_files:
            return

        # Get initial mtimes
        for path in self._watched_files:
            if os.path.exists(path):
                self._last_mtime[path] = os.path.getmtime(path)

        # Create observer
        self._observer = Observer()
        handler = _ConfigFileHandler(self._on_file_changed, self._watched_files)

        # Watch directories containing config files
        watched_dirs = set()
        for path in self._watched_files:
            dir_path = os.path.dirname(path)
            if dir_path and dir_path not in watched_dirs:
                watched_dirs.add(dir_path)
                self._observer.schedule(handler, dir_path, recursive=False)

        self._observer.start()
        # Config watcher started (silent)

    def stop(self):
        """Stop watching config files"""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None

    def _on_file_changed(self, path: str):
        """Handle file change event with debouncing"""
        with self._lock:
            # Check if file actually changed (debounce duplicate events)
            if os.path.exists(path):
                new_mtime = os.path.getmtime(path)
                old_mtime = self._last_mtime.get(path, 0)
                if new_mtime == old_mtime:
                    return
                self._last_mtime[path] = new_mtime

            # Debounce: wait 100ms before triggering reload
            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(0.1, self._reload_config)
            self._debounce_timer.start()

    def _reload_config(self):
        """Reload config and notify callback"""
        old_values = {
            "GROQ_API_KEY": GROQ_API_KEY,
            "AUDIO_DEVICE": AUDIO_DEVICE,
            "LANGUAGE": LANGUAGE,
            "HOTKEY": HOTKEY,
            "PROMPT": PROMPT,
            "SOUND_ENABLED": SOUND_ENABLED,
            "PROVIDER": PROVIDER,
            "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", ""),
        }

        # Reload environment from files
        for path in sorted(self._watched_files):  # Global first, local last = local wins with override
            if os.path.exists(path):
                load_dotenv(path, override=True)

        # Get new values
        new_values = {
            "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
            "AUDIO_DEVICE": os.environ.get("AUDIO_DEVICE", ""),
            "LANGUAGE": os.environ.get("LANGUAGE", "en"),
            "HOTKEY": os.environ.get("HOTKEY", "cmd_r"),
            "PROMPT": os.environ.get("PROMPT", ""),
            "SOUND_ENABLED": os.environ.get("SOUND_ENABLED", "true").lower() == "true",
            "PROVIDER": os.environ.get("PROVIDER", "mlx"),
            "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", ""),
        }

        # Find what changed
        changes = {}
        for key in old_values:
            if old_values[key] != new_values[key]:
                changes[key] = new_values[key]

        if changes:
            print(f"Config reloaded: {', '.join(changes.keys())}")
            self._on_config_change(changes)


class _ConfigFileHandler(FileSystemEventHandler):
    """Handle file system events for config files"""

    def __init__(self, callback: Callable[[str], None], watched_files: set[str]):
        self._callback = callback
        self._watched_files = watched_files

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)


def play_sound(sound_path):
    """Play a sound file asynchronously"""
    if not SOUND_ENABLED:
        return
    subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_config(key, value, force_global=False):
    """Save a config value to the config file"""
    # Prefer local .env if it exists (for dev mode), otherwise use global config
    local_env = os.path.join(os.getcwd(), ".env")
    if not force_global and os.path.exists(local_env):
        env_path = local_env
    else:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        env_path = CONFIG_FILE

    lines = []
    found = False

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    return env_path


def mask_api_key(key):
    """Mask API key for display"""
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def setup_wizard():
    """Configuration wizard - uses the onboarding module."""
    global GROQ_API_KEY, LANGUAGE, HOTKEY, PROMPT, SOUND_ENABLED, PROVIDER, AUDIO_DEVICE

    from onboarding import run_setup

    def save(key, value):
        save_config(key, value, force_global=True)
        # Update globals
        if key == "PROVIDER":
            global PROVIDER
            PROVIDER = value
        elif key == "GROQ_API_KEY":
            global GROQ_API_KEY
            GROQ_API_KEY = value
        elif key == "HOTKEY":
            global HOTKEY
            HOTKEY = value
        elif key == "AUDIO_DEVICE":
            global AUDIO_DEVICE
            AUDIO_DEVICE = value

    current_config = {
        "provider": PROVIDER,
        "model": os.environ.get("WHISPER_MODEL", ""),
        "groq_api_key": GROQ_API_KEY,
        "hotkey": HOTKEY,
        "audio_device": AUDIO_DEVICE,
    }

    run_setup(save, current_config=current_config, reconfigure=True)


def save_device_to_env(device_name):
    """Save device name to config file"""
    env_path = save_config("AUDIO_DEVICE", device_name)
    print(f"  (saved to {env_path})")

def select_audio_device():
    """List and select an audio input device. Returns device NAME (not index)."""
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev))

    # Check if device is saved in .env (by name)
    if AUDIO_DEVICE:
        for i, dev in input_devices:
            if dev['name'] == AUDIO_DEVICE:
                return AUDIO_DEVICE  # Return name, not index
        print(f"⚠️  Saved device '{AUDIO_DEVICE}' not found, please select again")

    print("\nAvailable input devices:")
    for i, dev in input_devices:
        marker = "*" if i == sd.default.device[0] else " "
        print(f"  {marker} [{i}] {dev['name']}")

    print("\n  (* = default)")

    while True:
        choice = input("\nSelect device number (or press Enter for default): ").strip()
        if choice == "":
            return None  # Use system default
        try:
            device_idx = int(choice)
            matching = [(i, d) for i, d in input_devices if i == device_idx]
            if matching:
                device_name = matching[0][1]['name']
                # Ask to save
                save = input("Save this device for future use? [y/N]: ").strip().lower()
                if save == "y":
                    save_device_to_env(device_name)
                return device_name  # Return name, not index
            print("Invalid device number")
        except ValueError:
            print("Please enter a number")


class _AudioWorkerClient:
    _WORKER_STARTUP_TIMEOUT_S = 10
    _START_TIMEOUT_S = 10
    _STOP_TIMEOUT_S = 10
    _CANCEL_TIMEOUT_S = 5

    def __init__(self):
        self._proc: subprocess.Popen[str] | None = None
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._cleanup_registered = False
        self._waveform_callback: Optional[Callable[[list[float], float], None]] = None

    def set_waveform_callback(self, callback: Optional[Callable[[list[float], float], None]]):
        """Set callback for waveform updates (values, raw_peak)"""
        self._waveform_callback = callback

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_running(self) -> None:
        with self._lock:
            self._ensure_running_locked()

    def stop(self, force: bool = False) -> None:
        with self._lock:
            self._stop_locked(force=force)

    def start_recording(self, *, device_name: str | None, sample_rate: int, channels: int) -> None:
        with self._lock:
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    self._ensure_running_locked()
                    req_id = self._next_id
                    self._next_id += 1

                    assert self._proc is not None
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(
                        json.dumps(
                            {
                                "type": "start",
                                "id": req_id,
                                "device_name": device_name,
                                "sample_rate": sample_rate,
                                "channels": channels,
                            }
                        )
                        + "\n"
                    )
                    self._proc.stdin.flush()

                    message = self._wait_for_locked(
                        lambda m: m.get("type") in {"started", "error"} and m.get("id") == req_id,
                        timeout_s=self._START_TIMEOUT_S,
                    )
                    if not message:
                        raise TimeoutError("Timed out starting audio recording")
                    if message.get("type") == "error":
                        raise RuntimeError(message.get("error") or "Failed to start recording")
                    return
                except Exception as e:
                    last_error = e
                    self._stop_locked(force=True)
                    if attempt == 0:
                        continue
                    raise
            if last_error:
                raise last_error

    def stop_recording(self, *, wav_path: str) -> tuple[int, float]:
        with self._lock:
            self._ensure_running_locked()
            req_id = self._next_id
            self._next_id += 1

            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"type": "stop", "id": req_id, "wav_path": wav_path}) + "\n")
            self._proc.stdin.flush()

            message = self._wait_for_locked(
                lambda m: m.get("type") in {"stopped", "error"} and m.get("id") == req_id,
                timeout_s=self._STOP_TIMEOUT_S,
            )
            if not message:
                raise TimeoutError("Timed out stopping audio recording")
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "Failed to stop recording")

            frames = message.get("frames")
            peak = message.get("peak")
            try:
                return int(frames or 0), float(peak or 0.0)
            except (TypeError, ValueError):
                return 0, 0.0

    def cancel_recording(self) -> None:
        with self._lock:
            if not self.is_running():
                return
            req_id = self._next_id
            self._next_id += 1

            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"type": "cancel", "id": req_id}) + "\n")
            self._proc.stdin.flush()

            message = self._wait_for_locked(
                lambda m: m.get("type") in {"canceled", "error"} and m.get("id") == req_id,
                timeout_s=self._CANCEL_TIMEOUT_S,
            )
            if not message:
                raise TimeoutError("Timed out cancelling audio recording")
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "Failed to cancel recording")

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                # Handle waveform messages via callback (don't queue)
                if msg.get("type") == "waveform" and self._waveform_callback:
                    try:
                        self._waveform_callback(msg.get("values", []), msg.get("raw_peak", 0.0))
                    except Exception:
                        pass
                else:
                    messages.put(msg)
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

    def _wait_for_locked(self, predicate, timeout_s: int) -> dict[str, Any] | None:
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
                return {"type": "error", "error": "Audio worker exited unexpectedly"}

            if predicate(message):
                return message

    def _ensure_running_locked(self) -> None:
        if self.is_running():
            return

        self._stop_locked(force=True)

        worker_path = os.path.join(os.path.dirname(__file__), "audio_worker.py")
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing audio worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        last_error: Exception | None = None
        for attempt in range(2):
            proc = subprocess.Popen(
                [sys.executable, "-u", worker_path],
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

            ready = self._wait_for_locked(
                lambda m: m.get("type") in {"ready", "error"},
                timeout_s=self._WORKER_STARTUP_TIMEOUT_S,
            )
            if ready and ready.get("type") == "ready":
                if not self._cleanup_registered:
                    atexit.register(self.stop)
                    self._cleanup_registered = True
                return

            if not ready:
                last_error = TimeoutError("Audio worker did not become ready in time")
            else:
                last_error = RuntimeError(ready.get("error") or "Audio worker failed to start")

            self._stop_locked(force=True)
            if attempt == 0:
                time.sleep(0.1)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Audio worker failed to start")

    def _stop_locked(self, force: bool = False) -> None:
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


class STTApp:
    # Maximum time for any operation before we consider it stuck
    _MAX_OPERATION_TIME_S = 300  # 5 minutes

    def __init__(self, device_name=None, provider=None):
        self.recording = False
        self.device_name = device_name  # Store name, resolve to index at record time
        self.provider = provider or get_provider(PROVIDER)
        self._audio_worker = _AudioWorkerClient()
        self._overlay = get_overlay()

        # Set up waveform callback to update overlay
        self._audio_worker.set_waveform_callback(self._on_waveform)

        # Thread synchronization
        self._lock = threading.Lock()
        self._processing = False  # Guard against concurrent process_recording calls
        self._starting = False  # Guard against concurrent start_recording calls
        self._operation_start_time: Optional[float] = None  # Track when operations start

        # State management for menu bar
        self._state = AppState.IDLE
        self._state_callback: Optional[Callable[[AppState], None]] = None

        # Start watchdog thread to detect stuck states
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _on_waveform(self, values: list[float], raw_peak: float):
        """Handle waveform data from audio worker"""
        above_threshold = raw_peak >= SILENCE_THRESHOLD
        self._overlay.update_waveform(values, above_threshold)

    def _watchdog_loop(self):
        """Monitor for stuck states and recover if needed"""
        while not self._watchdog_stop.wait(timeout=5):  # Check every 5 seconds
            with self._lock:
                if self._operation_start_time is not None:
                    elapsed = time.time() - self._operation_start_time
                    if elapsed > self._MAX_OPERATION_TIME_S:
                        print(f"⚠️  Operation stuck for {elapsed:.0f}s, resetting state...")
                        self._force_reset_state_locked()

    def _force_reset_state_locked(self):
        """Force reset all state flags (must be called with lock held)"""
        self._processing = False
        self._starting = False
        self.recording = False
        self._operation_start_time = None
        # Hide overlay
        try:
            self._overlay.hide()
        except Exception:
            pass
        # Force restart workers
        try:
            self._audio_worker.stop(force=True)
        except Exception:
            pass
        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        # Reset state via callback (schedule on main thread would be ideal but this is ok)
        if self._state_callback:
            try:
                self._state_callback(AppState.IDLE)
            except Exception:
                pass
        self._state = AppState.IDLE
        print("State reset complete. Ready for next recording.")

    def set_state_callback(self, callback: Callable[[AppState], None]):
        """Register callback for state changes (called from any thread)"""
        self._state_callback = callback

    def _set_state(self, new_state: AppState):
        """Update state and notify callback"""
        self._state = new_state
        if self._state_callback:
            self._state_callback(new_state)
        
    def start_recording(self):
        """Start recording audio from microphone"""
        with self._lock:
            if self._processing:
                return
            if self.recording or self._starting:
                return
            self._starting = True
            self.recording = True
            self._operation_start_time = time.time()

        self._set_state(AppState.RECORDING)
        self._overlay.show()
        play_sound(SOUND_START)
        print("Recording...")

        try:
            self._audio_worker.start_recording(device_name=self.device_name, sample_rate=SAMPLE_RATE, channels=CHANNELS)
            with self._lock:
                if not self.recording:
                    # Recording was cancelled while starting
                    try:
                        self._audio_worker.cancel_recording()
                    except Exception:
                        self._audio_worker.stop(force=True)
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self._audio_worker.stop(force=True)
            self._overlay.hide()
            with self._lock:
                self.recording = False
                self._operation_start_time = None
            self._set_state(AppState.IDLE)
        finally:
            with self._lock:
                self._starting = False
    
    def stop_recording(self):
        """Stop recording and return (wav_path, frames, rms)"""
        with self._lock:
            if not self.recording:
                return None, 0, 0.0
            self.recording = False
            starting = self._starting

        self._overlay.set_transcribing(True)
        play_sound(SOUND_STOP)
        print("Stopped")

        if starting:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                with self._lock:
                    if not self._starting:
                        break
                time.sleep(0.01)

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            frames, rms = self._audio_worker.stop_recording(wav_path=wav_path)
            return wav_path, frames, rms
        except TimeoutError:
            print("❌ Audio recording stop timed out. Restarting audio worker...")
            self._audio_worker.stop(force=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return None, 0, 0.0
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            self._audio_worker.stop(force=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return None, 0, 0.0

    def cancel_recording(self):
        """Cancel recording without processing"""
        with self._lock:
            if not self.recording:
                # Even if not recording, ensure state is IDLE (fallback safeguard)
                if self._state == AppState.RECORDING:
                    self._set_state(AppState.IDLE)
                    self._overlay.hide()
                self._operation_start_time = None
                return
            self.recording = False
            self._operation_start_time = None

        self._set_state(AppState.IDLE)
        self._overlay.hide()
        play_sound(SOUND_CANCEL)
        print("❌ Recording cancelled")
        try:
            self._audio_worker.cancel_recording()
        except TimeoutError:
            print("⚠️  Audio cancel timed out. Restarting audio worker...")
            self._audio_worker.stop(force=True)
        except Exception as e:
            print(f"⚠️  Error cancelling audio: {e}")
            self._audio_worker.stop(force=True)

    def cancel_transcription(self):
        """Cancel an in-progress transcription (best-effort)."""
        with self._lock:
            if not self._processing:
                return

        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            print("Cancelling...")
            try:
                cancel()
            except Exception as e:
                print(f"⚠️  Error cancelling transcription: {e}")

    def transcribe_audio(self, audio_file_path, timeout_s: int = 200):
        """Transcribe audio using the configured provider with timeout protection"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.provider.transcribe, audio_file_path, LANGUAGE, PROMPT)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                print("❌ Transcription timed out at STTApp level")
                # Try to cancel the provider if it supports cancellation
                cancel = getattr(self.provider, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        pass
                return None
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                return None

    def print_ready_prompt(self):
        """Print the ready prompt with hotkey name"""
        from rich.console import Console
        console = Console()
        hotkey_name = HOTKEYS[HOTKEY]["name"] if HOTKEY in HOTKEYS else HOTKEY
        console.print(f"\n[bold green]Ready[/bold green] [dim]│[/dim] Hold [cyan]{hotkey_name}[/cyan] to record, +Shift ↵, Esc ✗")

    def transform_text(self, text):
        """Apply text transformations"""
        # Convert "slash command" to "/command"
        text = re.sub(r'^[Ss]lash\s+', '/', text)

        return text

    def type_text(self, text, send_enter=False):
        """Type text into the active text field using clipboard paste (fast)"""
        if not text:
            return

        print(f"Typing: {text}" + (" ↵" if send_enter else ""))

        try:
            # Save current clipboard
            old_clipboard = subprocess.run(
                ["pbpaste"], capture_output=True, timeout=2
            ).stdout

            # Copy text to clipboard
            subprocess.run(
                ["pbcopy"], input=text.encode("utf-8"), check=True, timeout=2
            )

            # Paste via Cmd+V
            paste_script = '''
            tell application "System Events"
                keystroke "v" using command down
            end tell
            '''
            subprocess.run(["osascript", "-e", paste_script], check=True, timeout=5)

            # Small delay to let paste complete before restoring clipboard
            time.sleep(0.05)

            # Restore original clipboard
            subprocess.run(
                ["pbcopy"], input=old_clipboard, check=True, timeout=2
            )

            if send_enter:
                enter_script = '''
                tell application "System Events"
                    key code 36
                end tell
                '''
                subprocess.run(["osascript", "-e", enter_script], check=True, timeout=5)

        except subprocess.TimeoutExpired:
            print("❌ Typing timed out")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to type text: {e}")
    
    def process_recording(self, send_enter=False):
        """Process the recorded audio: transcribe and type"""
        # Guard against concurrent processing calls
        with self._lock:
            if self._processing:
                return
            self._processing = True

        wav_path = None
        try:
            wav_path, frames, peak = self.stop_recording()

            if not wav_path:
                print("⚠️  No audio captured, skipping...")
            elif frames < int(SAMPLE_RATE * 0.5):  # Less than 0.5 seconds
                print("⚠️  Recording too short, skipping...")
            elif peak < SILENCE_THRESHOLD:
                print("⚠️  Audio too quiet (silence), skipping...")
            else:
                self._set_state(AppState.TRANSCRIBING)
                text = self.transcribe_audio(wav_path)

                if text:
                    text = self.transform_text(text)
                    self.type_text(text, send_enter=send_enter)
                    print(f"✓ {text}")
                else:
                    print("No transcription returned")

            self.print_ready_prompt()
        except Exception as e:
            print(f"❌ Error processing recording: {e}")
        finally:
            # Always clean up temp file if created
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            # Hide overlay and reset state
            self._overlay.set_transcribing(False)
            self._overlay.hide()
            self._set_state(AppState.IDLE)
            with self._lock:
                self._processing = False
                self._operation_start_time = None


def main():
    global GROQ_API_KEY, LANGUAGE, HOTKEY, PROMPT, SOUND_ENABLED, PROVIDER, AUDIO_DEVICE

    # Check for --config flag
    if "--config" in sys.argv:
        setup_wizard()
        return

    # Dev flags for testing permission flow
    if "--test-permissions" in sys.argv:
        from onboarding import (
            show_permission_error,
            open_accessibility_settings,
            open_input_monitoring_settings,
            console,
        )
        from rich.prompt import Confirm
        show_permission_error()
        # Step 1: Accessibility
        if Confirm.ask("Open Accessibility settings?", default=True):
            open_accessibility_settings()
            console.print("\n[dim]Enable the permission, then come back here.[/dim]\n")
            Confirm.ask("Done with Accessibility?", default=True)
        # Step 2: Input Monitoring
        if Confirm.ask("Open Input Monitoring settings?", default=True):
            open_input_monitoring_settings()
            console.print("\n[dim]Enable the permission, then come back here.[/dim]\n")
            Confirm.ask("Done with Input Monitoring?", default=True)
        from onboarding import get_terminal_app
        terminal = get_terminal_app()
        console.print(f"\n[green]Permission setup complete.[/green]")
        console.print(f"[yellow]Restart {terminal} and run STT again.[/yellow]\n")
        return

    # Ensure only one instance
    if not acquire_lock():
        from rich.console import Console
        Console().print("[red]Another instance of STT is already running[/red]")
        sys.exit(1)
    atexit.register(release_lock)

    # First-run onboarding
    if is_first_run():
        from onboarding import run_first_time_setup

        def save_and_update(key, value):
            save_config(key, value, force_global=True)
            # Update global vars
            if key == "PROVIDER":
                global PROVIDER
                PROVIDER = value
                os.environ["PROVIDER"] = value
            elif key == "WHISPER_MODEL":
                os.environ["WHISPER_MODEL"] = value
            elif key == "HOTKEY":
                global HOTKEY
                HOTKEY = value
                os.environ["HOTKEY"] = value
            elif key == "AUDIO_DEVICE":
                global AUDIO_DEVICE
                AUDIO_DEVICE = value
                os.environ["AUDIO_DEVICE"] = value

        run_first_time_setup(save_and_update)
        mark_initialized()

        # Reload config after onboarding
        load_dotenv(CONFIG_FILE, override=True)
        PROVIDER = os.environ.get("PROVIDER", "mlx")
        HOTKEY = os.environ.get("HOTKEY", "cmd_r")
        AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "")

    from rich.console import Console
    console = Console()
    console.print()
    console.print("[bold]STT[/bold] [dim]Voice-to-text for macOS[/dim]")
    console.print("[dim]https://github.com/bokan/stt[/dim]")
    console.print()

    # Check for updates in background
    threading.Thread(target=check_for_updates, daemon=True).start()

    # Initialize provider
    try:
        provider = get_provider(PROVIDER)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    if not provider.is_available():
        if PROVIDER == "groq" and not GROQ_API_KEY:
            setup_wizard()
            provider = get_provider(PROVIDER)
        else:
            console.print(f"[red]✗[/red] Provider '{PROVIDER}' is not available")
            if PROVIDER == "mlx":
                console.print("  [dim]Install with: pip install mlx-whisper[/dim]")
            sys.exit(1)

    console.print(f"[green]✓[/green] Provider: [cyan]{provider.name}[/cyan]")

    # Warmup provider (downloads/loads model if needed)
    provider.warmup()

    # Check accessibility permissions
    if not check_accessibility_permissions():
        from onboarding import show_permission_error, open_accessibility_settings, prompt_open_settings, get_terminal_app
        show_permission_error()
        if prompt_open_settings():
            open_accessibility_settings()
        terminal = get_terminal_app()
        console.print(f"\n[yellow]Restart {terminal} and run STT again.[/yellow]")
        sys.exit(1)

    hotkey_name = HOTKEYS[HOTKEY]["name"] if HOTKEY in HOTKEYS else HOTKEY

    # Select audio device (uses saved device or prompts)
    device_name = select_audio_device()
    if device_name:
        console.print(f"[green]✓[/green] Device: [cyan]{device_name}[/cyan]")
    else:
        console.print(f"[green]✓[/green] Device: [cyan]System default[/cyan]")

    console.print()
    console.print(f"[bold green]Ready[/bold green] [dim]│[/dim] Hold [cyan]{hotkey_name}[/cyan] to record, +Shift ↵, Esc ✗")
    console.print()

    app = STTApp(device_name=device_name, provider=provider)
    key_pressed = False
    shift_held = False
    send_enter_flag = False
    trigger_key = HOTKEYS[HOTKEY]["key"] if HOTKEY in HOTKEYS else keyboard.Key.cmd_r

    def on_press(key):
        nonlocal key_pressed, shift_held, send_enter_flag
        try:
            if key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                shift_held = True
                # If already recording, mark for enter and show indicator (stays visible)
                if app.recording and not send_enter_flag:
                    send_enter_flag = True
                    app._overlay.set_shift_held(True)
            elif key == trigger_key:
                if not key_pressed:
                    key_pressed = True
                    # Check if shift is already held when starting
                    send_enter_flag = shift_held
                    if shift_held:
                        app._overlay.set_shift_held(True)
                    # Start recording in background thread to avoid blocking keyboard listener
                    threading.Thread(target=app.start_recording, daemon=True).start()
            elif key == keyboard.Key.esc:
                if app.recording:
                    key_pressed = False
                    send_enter_flag = False
                    # Cancel recording in background thread to avoid blocking keyboard listener
                    threading.Thread(target=app.cancel_recording, daemon=True).start()
                else:
                    threading.Thread(target=app.cancel_transcription, daemon=True).start()
        except Exception as e:
            print(f"⚠️  Error in key press handler: {e}")
            # Reset state on error to prevent stuck keys
            key_pressed = False
            send_enter_flag = False

    def on_release(key):
        nonlocal key_pressed, shift_held, send_enter_flag
        try:
            if key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                shift_held = False
                # Don't hide enter icon - it stays visible once activated
                return
            # Check for trigger key release. Also accept generic 'cmd' when trigger is cmd_r,
            # because macOS reports ambiguous releases (e.g., releasing cmd_r while cmd_l is held)
            # as generic 'cmd' which doesn't match cmd_r specifically.
            is_cmd_trigger = trigger_key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
            is_cmd_release = key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
            if key == trigger_key or (key_pressed and is_cmd_trigger and is_cmd_release):
                if key_pressed:
                    key_pressed = False
                    send_enter = send_enter_flag
                    send_enter_flag = False
                    # Process in a separate thread to not block the listener
                    threading.Thread(target=app.process_recording, args=(send_enter,)).start()
        except Exception as e:
            print(f"⚠️  Error in key release handler: {e}")
            # Reset state on error to prevent stuck keys
            key_pressed = False
            send_enter_flag = False

    def on_click(x, y, button, pressed):
        nonlocal key_pressed, send_enter_flag
        if button == mouse.Button.middle:
            if pressed:
                if not key_pressed:
                    key_pressed = True
                    send_enter_flag = True  # Middle button always sends Enter
                    threading.Thread(target=app.start_recording, daemon=True).start()
            else:
                if key_pressed:
                    key_pressed = False
                    send_enter_flag = False
                    threading.Thread(target=app.process_recording, args=(True,), daemon=True).start()

    # Start the keyboard listener in a background thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Start the mouse listener for middle button trigger
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    # Config change handler
    def on_config_change(changes: dict):
        global GROQ_API_KEY, AUDIO_DEVICE, LANGUAGE, HOTKEY, PROMPT, SOUND_ENABLED, PROVIDER
        nonlocal trigger_key

        # Update global variables
        if "GROQ_API_KEY" in changes:
            GROQ_API_KEY = changes["GROQ_API_KEY"]
        if "AUDIO_DEVICE" in changes:
            AUDIO_DEVICE = changes["AUDIO_DEVICE"]
            # Store name directly; resolved to index at record time
            app.device_name = AUDIO_DEVICE or None
            print(f"   Audio device: {AUDIO_DEVICE or 'default'}")
        if "LANGUAGE" in changes:
            LANGUAGE = changes["LANGUAGE"]
            print(f"   Language: {LANGUAGE}")
        if "HOTKEY" in changes:
            new_hotkey = changes["HOTKEY"]
            if new_hotkey in HOTKEYS:
                HOTKEY = new_hotkey
                trigger_key = HOTKEYS[HOTKEY]["key"]
                print(f"   Hotkey: {HOTKEYS[HOTKEY]['name']}")
            else:
                print(f"   Invalid hotkey: {new_hotkey}, keeping current")
        if "PROMPT" in changes:
            PROMPT = changes["PROMPT"]
            print(f"   Prompt: {PROMPT or '(empty)'}")
        if "SOUND_ENABLED" in changes:
            SOUND_ENABLED = changes["SOUND_ENABLED"]
            menubar.update_sound_enabled(SOUND_ENABLED)
            print(f"   Sound: {'enabled' if SOUND_ENABLED else 'disabled'}")

        # Handle provider changes (requires reinitialization)
        if "PROVIDER" in changes or "WHISPER_MODEL" in changes or "GROQ_API_KEY" in changes:
            new_provider_name = changes.get("PROVIDER", PROVIDER)
            if new_provider_name != PROVIDER:
                PROVIDER = new_provider_name
            try:
                new_provider = get_provider(PROVIDER)
                if new_provider.is_available():
                    new_provider.warmup()
                    app.provider = new_provider
                    menubar.update_provider_name(new_provider.name)
                    print(f"   Provider: {new_provider.name}")
                else:
                    print(f"   Provider '{PROVIDER}' not available")
            except Exception as e:
                print(f"   Failed to reinitialize provider: {e}")

    # Start config watcher
    config_watcher = ConfigWatcher(on_config_change)
    config_watcher.start()

    # Cleanup handler
    def cleanup():
        listener.stop()
        mouse_listener.stop()
        config_watcher.stop()

    atexit.register(cleanup)

    # Callbacks for menu bar
    def on_sound_toggle(enabled):
        global SOUND_ENABLED
        SOUND_ENABLED = enabled
        save_config("SOUND_ENABLED", str(enabled).lower())

    # Create and run menu bar (blocks on main thread)
    menubar = STTMenuBar(
        stt_app=app,
        hotkey_name=hotkey_name,
        provider_name=provider.name,
        sound_enabled=SOUND_ENABLED,
        config_file=CONFIG_FILE,
        on_sound_toggle=on_sound_toggle,
        on_quit=cleanup,
    )
    menubar.run()


if __name__ == "__main__":
    main()
