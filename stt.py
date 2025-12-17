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

print(" ready.", flush=True)


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
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1

# Map config string to pynput key
HOTKEY_MAP = {
    "cmd_r": keyboard.Key.cmd_r,
    "cmd_l": keyboard.Key.cmd_l,
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "ctrl_r": keyboard.Key.ctrl_r,
    "ctrl_l": keyboard.Key.ctrl_l,
    "shift_r": keyboard.Key.shift_r,
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
    global GROQ_API_KEY, LANGUAGE, HOTKEY, PROMPT, SOUND_ENABLED

    print("\n" + "=" * 50)
    print("STT Configuration")
    print("=" * 50)

    # API Key
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
        print("\nTo use this app, you need a Groq API key.")
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
    print(f"\nHotkey options: cmd_r, cmd_l, alt_r, alt_l, ctrl_r, ctrl_l, shift_r")
    hotkey = input(f"Hotkey [{default_hotkey}]: ").strip().lower()
    if hotkey and hotkey != default_hotkey:
        if hotkey in HOTKEY_MAP:
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
    if prompt != default_prompt:
        save_config("PROMPT", prompt)
        PROMPT = prompt
        if prompt:
            print(f"Prompt set")

    # Sound
    default_sound = "y" if SOUND_ENABLED else "n"
    sound = input(f"\nEnable audio feedback? [{'Y/n' if SOUND_ENABLED else 'y/N'}]: ").strip().lower()
    if sound in ('y', 'n') and sound != default_sound:
        SOUND_ENABLED = sound == 'y'
        save_config("SOUND_ENABLED", str(SOUND_ENABLED).lower())
        print(f"Sound {'enabled' if SOUND_ENABLED else 'disabled'}")

    print("\nConfiguration complete!\n")


def save_device_to_env(device_id):
    """Save device ID to config file"""
    env_path = save_config("AUDIO_DEVICE", device_id)
    print(f"  (saved to {env_path})")

def select_audio_device():
    """List and select an audio input device"""
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev))

    # Check if device is saved in .env
    if AUDIO_DEVICE:
        try:
            device_id = int(AUDIO_DEVICE)
            if any(d[0] == device_id for d in input_devices):
                return device_id
            print(f"⚠️  Saved device {device_id} not found, please select again")
        except ValueError:
            pass

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
            device_id = int(choice)
            if any(d[0] == device_id for d in input_devices):
                # Ask to save
                save = input("Save this device for future use? [y/N]: ").strip().lower()
                if save == "y":
                    save_device_to_env(device_id)
                return device_id
            print("Invalid device number")
        except ValueError:
            print("Please enter a number")


class STTApp:
    def __init__(self, device=None):
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.device = device
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.recording:
            return

        self.recording = True
        self.audio_data = []
        play_sound(SOUND_START)
        print("🎤 Recording...")
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=callback
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.recording:
            return None

        self.recording = False
        play_sound(SOUND_STOP)
        print("⏹️  Stopped recording")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.audio_data:
            return None
            
        # Concatenate all audio chunks
        audio = np.concatenate(self.audio_data, axis=0)
        return audio

    def cancel_recording(self):
        """Cancel recording without processing"""
        if not self.recording:
            return

        self.recording = False
        play_sound(SOUND_CANCEL)
        print("❌ Recording cancelled")

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.audio_data = []

    def save_audio_to_wav(self, audio_data):
        """Save audio data to a temporary WAV file"""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(temp_file.name, SAMPLE_RATE, audio_int16)
        return temp_file.name
    
    def transcribe_audio(self, audio_file_path):
        """Send audio to Groq Whisper API for transcription"""
        print("🔄 Transcribing...")
        
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3",
                "response_format": "text",
                "language": LANGUAGE,
            }
            if PROMPT:
                data["prompt"] = PROMPT
            
            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                return response.text.strip()
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return None
    
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
        audio = self.stop_recording()

        if audio is None or len(audio) < SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
            print("⚠️  Recording too short, skipping...")
            return

        # Save to temp file
        wav_path = self.save_audio_to_wav(audio)

        try:
            # Transcribe
            text = self.transcribe_audio(wav_path)

            if text:
                # Process text transformations
                text = self.transform_text(text)
                # Type the result
                self.type_text(text, send_enter=send_enter)
                print(f"✅ Done: {text}")
            else:
                print("⚠️  No transcription returned")
        finally:
            # Clean up temp file
            os.unlink(wav_path)


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
        print("   Please grant access in System Settings →")
        print("   Privacy & Security → Accessibility")
        print("\n   Then restart this app.")
        sys.exit(1)

    hotkey_name = HOTKEY.replace("_", " ").upper()
    print(f"Hold {hotkey_name} to record, release to transcribe")
    print("Hold LEFT SHIFT while recording to also send Enter")
    print("Press ESC while recording to cancel, Ctrl+C to quit")
    print("=" * 50)

    if not GROQ_API_KEY:
        setup_wizard()

    # Select audio device
    device = select_audio_device()
    if device is not None:
        print(f"\n✓ Using: {sd.query_devices(device)['name']}")
    else:
        print(f"\n✓ Using default device")

    app = STTApp(device=device)
    key_pressed = False
    shift_held = False
    send_enter_flag = False
    trigger_key = HOTKEY_MAP.get(HOTKEY, keyboard.Key.cmd_r)

    def on_press(key):
        nonlocal key_pressed, shift_held, send_enter_flag

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
                app.start_recording()
        elif key == keyboard.Key.esc:
            if app.recording:
                key_pressed = False
                send_enter_flag = False
                app.cancel_recording()

    def on_release(key):
        nonlocal key_pressed, shift_held, send_enter_flag

        if key == keyboard.Key.shift_l:
            shift_held = False
        elif key == trigger_key:
            if key_pressed:
                key_pressed = False
                send_enter = send_enter_flag
                send_enter_flag = False
                # Process in a separate thread to not block the listener
                threading.Thread(target=app.process_recording, args=(send_enter,)).start()
    
    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
