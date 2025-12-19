#!/usr/bin/env python3
"""
Simple Speech-to-Text app for macOS
Hold Right Command key to record, release to transcribe and type
"""

import os
import re
import sys
import tempfile
import threading
import subprocess
import atexit
import fcntl
from enum import Enum
from typing import Callable, Optional

print("Starting STT...", end="", flush=True)

# Version for update checking
__version__ = "0.1.0"
REPO_URL = "https://api.github.com/repos/bokan/stt/commits/master"

from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from pynput import keyboard
import requests

from providers import get_provider
from menubar import STTMenuBar

print(" ready.", flush=True)


class AppState(Enum):
    """Application state for menu bar icon"""
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


def check_for_updates():
    """Check if a newer version is available on GitHub"""
    version_file = os.path.join(CONFIG_DIR, ".last_commit")

    try:
        response = requests.get(REPO_URL, timeout=5)
        if response.status_code == 200:
            latest_sha = response.json()["sha"][:7]

            # Read stored commit
            stored_sha = None
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    stored_sha = f.read().strip()

            if stored_sha and stored_sha != latest_sha:
                print(f"\n📦 Update available! Run: uv tool upgrade stt", flush=True)

            # Store current commit
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(version_file, "w") as f:
                f.write(latest_sha)
    except Exception:
        pass  # Silently fail if offline


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


def play_sound(sound_path):
    """Play a sound file asynchronously"""
    if not SOUND_ENABLED:
        return
    subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_config(key, value):
    """Save a config value to the config file"""
    # Prefer local .env if it exists, otherwise use global config
    local_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(local_env):
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
    """First-time setup wizard"""
    global GROQ_API_KEY, LANGUAGE, HOTKEY, PROMPT, SOUND_ENABLED, PROVIDER, AUDIO_DEVICE

    print("\n" + "=" * 50)
    print("STT Configuration")
    print("=" * 50)

    # Provider selection
    default_provider = PROVIDER or "mlx"
    print("\nProviders:")
    print("  [1] mlx  - Local (Apple Silicon, no internet required)")
    print("  [2] groq - Cloud (fast, requires API key)")
    provider_choice = input(f"Select provider [{'1' if default_provider == 'mlx' else '2'}]: ").strip()
    if provider_choice == "1":
        PROVIDER = "mlx"
    elif provider_choice == "2":
        PROVIDER = "groq"
    # else keep default

    if PROVIDER != default_provider:
        save_config("PROVIDER", PROVIDER)
        print(f"Provider set to: {PROVIDER}")

    # MLX model selection
    if PROVIDER == "mlx":
        whisper_model = os.environ.get("WHISPER_MODEL", "")
        print("\nMLX Whisper models (larger = more accurate, slower):")
        print("  [1] large-v3 (default, best quality)")
        print("  [2] large")
        print("  [3] medium")
        print("  [4] small")
        print("  [5] base")
        print("  [6] tiny (fastest)")
        model_map = {"1": "large-v3", "2": "large", "3": "medium", "4": "small", "5": "base", "6": "tiny"}
        model_choice = input("Select model [1]: ").strip()
        if model_choice in model_map:
            new_model = model_map[model_choice]
            save_config("WHISPER_MODEL", new_model)
            print(f"Model set to: {new_model}")

    # Groq API Key (only if groq provider selected)
    if PROVIDER == "groq":
        if GROQ_API_KEY:
            masked = mask_api_key(GROQ_API_KEY)
            print(f"\nCurrent API key: {masked}")
            api_key = input("Enter new Groq API key (or press Enter to keep): ").strip()
            if api_key:
                if not api_key.startswith("gsk_"):
                    confirm = input("Key doesn't look like a Groq key (should start with 'gsk_'). Use anyway? [y/N]: ").strip().lower()
                    if confirm != 'y':
                        api_key = ""
                if api_key:
                    save_config("GROQ_API_KEY", api_key)
                    GROQ_API_KEY = api_key
                    print("API key updated")
        else:
            print("\nTo use Groq, you need an API key.")
            print("Get one at: https://console.groq.com")
            while True:
                api_key = input("\nEnter your Groq API key (or 'q' to quit): ").strip()
                if api_key.lower() == 'q':
                    print("\nSetup cancelled.")
                    sys.exit(0)
                if not api_key:
                    print("API key cannot be empty")
                    continue
                if not api_key.startswith("gsk_"):
                    confirm = input("Key doesn't look like a Groq key (should start with 'gsk_'). Use anyway? [y/N]: ").strip().lower()
                    if confirm != 'y':
                        continue
                break
            save_config("GROQ_API_KEY", api_key)
            GROQ_API_KEY = api_key
            print("API key saved")

    # Language
    default_lang = LANGUAGE or "en"
    print(f"\nLanguage codes: en, es, de, fr, it, pt, ja, etc.")
    lang = input(f"Language [{default_lang}]: ").strip().lower()
    if lang and lang != default_lang:
        save_config("LANGUAGE", lang)
        LANGUAGE = lang
        print(f"Language set to: {lang}")
    elif not lang:
        if not LANGUAGE:
            save_config("LANGUAGE", "en")
            LANGUAGE = "en"

    # Hotkey
    default_hotkey = HOTKEY or "cmd_r"
    print(f"\nHotkey options: cmd_r, alt_r, ctrl_r, shift_r")
    hotkey = input(f"Hotkey [{default_hotkey}]: ").strip().lower()
    if hotkey and hotkey != default_hotkey:
        if hotkey in HOTKEYS:
            save_config("HOTKEY", hotkey)
            HOTKEY = hotkey
            print(f"Hotkey set to: {hotkey}")
        else:
            print(f"Invalid hotkey, keeping: {default_hotkey}")

    # Prompt
    default_prompt = PROMPT or ""
    print(f"\nContext prompt helps recognize specific terms (e.g., Claude, TypeScript, React)")
    current = f" [{default_prompt}]" if default_prompt else ""
    prompt = input(f"Prompt{current}: ").strip()
    if prompt and prompt != default_prompt:
        save_config("PROMPT", prompt)
        PROMPT = prompt
        print(f"Prompt set")

    # Sound
    default_sound = "y" if SOUND_ENABLED else "n"
    sound = input(f"\nEnable audio feedback? [{'Y/n' if SOUND_ENABLED else 'y/N'}]: ").strip().lower()
    if sound in ('y', 'n') and sound != default_sound:
        SOUND_ENABLED = sound == 'y'
        save_config("SOUND_ENABLED", str(SOUND_ENABLED).lower())
        print(f"Sound {'enabled' if SOUND_ENABLED else 'disabled'}")

    # Audio device
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev))

    if input_devices:
        print("\nAvailable input devices:")
        default_device_id = sd.default.device[0]
        default_device_name = devices[default_device_id]['name'] if default_device_id is not None else None

        # Find current device by name
        current_device_idx = None
        if AUDIO_DEVICE:
            for i, dev in input_devices:
                if dev['name'] == AUDIO_DEVICE:
                    current_device_idx = i
                    break

        for i, dev in input_devices:
            markers = []
            if i == default_device_id:
                markers.append("system default")
            if current_device_idx is not None and i == current_device_idx:
                markers.append("current")
            marker_str = f" ({', '.join(markers)})" if markers else ""
            print(f"  [{i}] {dev['name']}{marker_str}")

        prompt_default = current_device_idx if current_device_idx is not None else default_device_id
        device_choice = input(f"\nSelect device number [{prompt_default}]: ").strip()
        if device_choice:
            try:
                new_device_idx = int(device_choice)
                matching = [(i, d) for i, d in input_devices if i == new_device_idx]
                if matching:
                    new_device_name = matching[0][1]['name']
                    if new_device_name != AUDIO_DEVICE:
                        save_config("AUDIO_DEVICE", new_device_name)
                        AUDIO_DEVICE = new_device_name
                        print(f"Audio device set to: {new_device_name}")
                else:
                    print("Invalid device number, keeping current setting")
            except ValueError:
                print("Invalid input, keeping current setting")
        elif not AUDIO_DEVICE:
            # Save default device if nothing was configured before
            save_config("AUDIO_DEVICE", default_device_name)
            AUDIO_DEVICE = default_device_name

    print("\nConfiguration complete!\n")


def save_device_to_env(device_name):
    """Save device name to config file"""
    env_path = save_config("AUDIO_DEVICE", device_name)
    print(f"  (saved to {env_path})")

def select_audio_device():
    """List and select an audio input device"""
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev))

    # Check if device is saved in .env (by name)
    if AUDIO_DEVICE:
        for i, dev in input_devices:
            if dev['name'] == AUDIO_DEVICE:
                return i
        print(f"⚠️  Saved device '{AUDIO_DEVICE}' not found, please select again")

    print("\nAvailable input devices:")
    for i, dev in input_devices:
        marker = "*" if i == sd.default.device[0] else " "
        print(f"  {marker} [{i}] {dev['name']}")

    print("\n  (* = default)")

    while True:
        choice = input("\nSelect device number (or press Enter for default): ").strip()
        if choice == "":
            return None  # Use default
        try:
            device_idx = int(choice)
            matching = [(i, d) for i, d in input_devices if i == device_idx]
            if matching:
                device_name = matching[0][1]['name']
                # Ask to save
                save = input("Save this device for future use? [y/N]: ").strip().lower()
                if save == "y":
                    save_device_to_env(device_name)
                return device_idx
            print("Invalid device number")
        except ValueError:
            print("Please enter a number")


class STTApp:
    def __init__(self, device=None, provider=None):
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.device = device
        self.provider = provider or get_provider(PROVIDER)

        # Thread synchronization
        self._lock = threading.Lock()
        self._processing = False  # Guard against concurrent process_recording calls
        self._starting = False  # Guard against concurrent start_recording calls

        # State management for menu bar
        self._state = AppState.IDLE
        self._state_callback: Optional[Callable[[AppState], None]] = None

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
            self.audio_data = []

        self._set_state(AppState.RECORDING)
        play_sound(SOUND_START)
        print("🎤 Recording...")

        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            # Lock-free check is safe here - worst case we miss one chunk
            if self.recording:
                self.audio_data.append(indata.copy())

        try:
            stream = sd.InputStream(
                device=self.device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                callback=callback
            )
            stream.start()
            with self._lock:
                if self.recording:
                    self.stream = stream
                else:
                    # Recording was cancelled while starting
                    stream.abort()
                    stream.close()
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            with self._lock:
                self.recording = False
            self._set_state(AppState.IDLE)
        finally:
            with self._lock:
                self._starting = False
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        with self._lock:
            if not self.recording:
                return None
            self.recording = False
            stream = self.stream
            self.stream = None
            audio_data = self.audio_data
            self.audio_data = []

        play_sound(SOUND_STOP)
        print("⏹️  Stopped recording")

        # Stop stream outside lock to avoid deadlock with audio callback
        # Use abort() instead of stop() - stop() waits for pending buffers which can hang
        if stream:
            try:
                stream.abort()  # Immediate termination, no waiting
                stream.close()
            except Exception as e:
                print(f"⚠️  Error closing stream: {e}")

        if not audio_data:
            return None

        # Concatenate all audio chunks
        return np.concatenate(audio_data, axis=0)

    def cancel_recording(self):
        """Cancel recording without processing"""
        with self._lock:
            if not self.recording:
                # Even if not recording, ensure state is IDLE (fallback safeguard)
                if self._state == AppState.RECORDING:
                    self._set_state(AppState.IDLE)
                return
            self.recording = False
            stream = self.stream
            self.stream = None
            self.audio_data = []

        self._set_state(AppState.IDLE)
        play_sound(SOUND_CANCEL)
        print("❌ Recording cancelled")

        # Use abort() instead of stop() - stop() waits for pending buffers which can hang
        if stream:
            try:
                stream.abort()  # Immediate termination, no waiting
                stream.close()
            except Exception as e:
                print(f"⚠️  Error closing stream: {e}")

    def cancel_transcription(self):
        """Cancel an in-progress transcription (best-effort)."""
        with self._lock:
            if not self._processing:
                return

        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            print("⏹️  Cancelling transcription...")
            try:
                cancel()
            except Exception as e:
                print(f"⚠️  Error cancelling transcription: {e}")

    def save_audio_to_wav(self, audio_data):
        """Save audio data to a temporary WAV file"""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create temp file and close handle before handing to other libraries
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wavfile.write(temp_file.name, SAMPLE_RATE, audio_int16)
            return temp_file.name
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using the configured provider"""
        return self.provider.transcribe(audio_file_path, LANGUAGE, PROMPT)

    def print_ready_prompt(self):
        """Print the ready prompt with hotkey name"""
        hotkey_name = HOTKEYS[HOTKEY]["name"] if HOTKEY in HOTKEYS else HOTKEY
        print(f"Press [{hotkey_name}] to record")

    def transform_text(self, text):
        """Apply text transformations"""
        # Convert "slash command" to "/command"
        text = re.sub(r'^[Ss]lash\s+', '/', text)

        return text

    def type_text(self, text, send_enter=False):
        """Type text into the active text field using AppleScript"""
        if not text:
            return

        print(f"⌨️  Typing: {text}" + (" [+Enter]" if send_enter else ""))

        # Escape special characters for AppleScript
        escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')

        # Use AppleScript to type the text
        script = f'''
        tell application "System Events"
            keystroke "{escaped_text}"
        end tell
        '''

        try:
            subprocess.run(["osascript", "-e", script], check=True)
            if send_enter:
                enter_script = '''
                tell application "System Events"
                    key code 36
                end tell
                '''
                subprocess.run(["osascript", "-e", enter_script], check=True)
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
            audio = self.stop_recording()

            if audio is None or len(audio) < SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
                print("⚠️  Recording too short, skipping...")
            else:
                self._set_state(AppState.TRANSCRIBING)
                wav_path = self.save_audio_to_wav(audio)

                text = self.transcribe_audio(wav_path)

                if text:
                    text = self.transform_text(text)
                    self.type_text(text, send_enter=send_enter)
                    print(f"✅ Done: {text}")
                else:
                    print("⚠️  No transcription returned")

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
            # Always reset state to IDLE
            self._set_state(AppState.IDLE)
            with self._lock:
                self._processing = False


def main():
    # Check for --config flag
    if "--config" in sys.argv:
        setup_wizard()
        return

    # Ensure only one instance
    if not acquire_lock():
        print("❌ Another instance of STT is already running")
        sys.exit(1)
    atexit.register(release_lock)

    print("=" * 50)
    print("STT - Voice-to-Text for macOS")
    print("https://github.com/bokan/stt")
    print("=" * 50)

    # Check for updates in background
    threading.Thread(target=check_for_updates, daemon=True).start()

    # Check accessibility permissions
    if not check_accessibility_permissions():
        print("\n❌ Accessibility permissions required!")
        print("   Grant access to your TERMINAL APP (iTerm2, Terminal, Warp, etc.)")
        print("   — not 'stt' or 'python'.")
        print("\n   System Settings → Privacy & Security → Accessibility")
        print("\n   Then restart this app.")
        sys.exit(1)

    hotkey_name = HOTKEYS[HOTKEY]["name"] if HOTKEY in HOTKEYS else HOTKEY
    print(f"Press [{hotkey_name}] to record, release to transcribe")
    print("Hold LEFT SHIFT while recording to also send Enter")
    print("Press ESC to cancel recording / stuck transcription, Ctrl+C to quit")
    print("=" * 50)

    # Initialize provider
    try:
        provider = get_provider(PROVIDER)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not provider.is_available():
        if PROVIDER == "groq" and not GROQ_API_KEY:
            setup_wizard()
            provider = get_provider(PROVIDER)
        else:
            print(f"❌ Provider '{PROVIDER}' is not available")
            if PROVIDER == "mlx":
                print("   Install with: pip install mlx-whisper")
            sys.exit(1)

    print(f"✓ Provider: {provider.name}")

    # Warmup provider (downloads/loads model if needed)
    provider.warmup()

    # Select audio device
    device = select_audio_device()
    if device is not None:
        print(f"\n✓ Using: {sd.query_devices(device)['name']}")
    else:
        print(f"\n✓ Using default device")

    app = STTApp(device=device, provider=provider)
    app.print_ready_prompt()
    key_pressed = False
    shift_held = False
    send_enter_flag = False
    trigger_key = HOTKEYS[HOTKEY]["key"] if HOTKEY in HOTKEYS else keyboard.Key.cmd_r

    def on_press(key):
        nonlocal key_pressed, shift_held, send_enter_flag
        try:
            if key == keyboard.Key.shift_l:
                shift_held = True
                # If already recording, mark for enter
                if app.recording:
                    send_enter_flag = True
            elif key == trigger_key:
                if not key_pressed:
                    key_pressed = True
                    # Check if shift is already held when starting
                    send_enter_flag = shift_held
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
            if key == keyboard.Key.shift_l:
                shift_held = False
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

    # Start the keyboard listener in a background thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Cleanup handler
    def cleanup():
        listener.stop()

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
