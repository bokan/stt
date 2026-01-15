"""
Onboarding and setup UI for STT using rich library.
"""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TextColumn,
    TaskProgressColumn,
)
from rich.prompt import IntPrompt, Prompt, Confirm
from rich.table import Table

console = Console()

# Terminal app detection
TERMINAL_APPS = {
    "iTerm.app": "iTerm2",
    "Apple_Terminal": "Terminal",
    "vscode": "Visual Studio Code",
    "WarpTerminal": "Warp",
    "Hyper": "Hyper",
    "Alacritty": "Alacritty",
    "kitty": "Kitty",
    "rio": "Rio",
    "ghostty": "Ghostty",
}


def get_terminal_app() -> str:
    """Detect the current terminal application."""
    term_program = os.environ.get("TERM_PROGRAM", "")

    # If running under tmux/screen, try to find the real terminal
    if term_program in ("tmux", "screen", ""):
        # Try LC_TERMINAL (set by some terminals like iTerm2)
        lc_terminal = os.environ.get("LC_TERMINAL", "")
        if lc_terminal:
            if lc_terminal in TERMINAL_APPS:
                return TERMINAL_APPS[lc_terminal]
            return lc_terminal

        # Try TERM_PROGRAM from before tmux (iTerm2 sets this)
        iterm_profile = os.environ.get("ITERM_PROFILE", "")
        if iterm_profile:
            return "iTerm2"

        # Try to detect from process tree on macOS
        try:
            import subprocess
            # Get parent PIDs up the chain looking for known terminals
            result = subprocess.run(
                ["ps", "-o", "comm=", "-p", str(os.getppid())],
                capture_output=True, text=True
            )
            parent_comm = result.stdout.strip()
            if "iTerm" in parent_comm:
                return "iTerm2"
            if "Terminal" in parent_comm:
                return "Terminal"
            if "Warp" in parent_comm:
                return "Warp"
        except Exception:
            pass

        # If we're in tmux/screen but can't find parent, mention both
        if term_program in ("tmux", "screen"):
            return f"your terminal app (the one running {term_program})"

    # Check known terminal programs
    if term_program in TERMINAL_APPS:
        return TERMINAL_APPS[term_program]

    # Check for common patterns
    if "code" in term_program.lower():
        return "Visual Studio Code"
    if "term" in term_program.lower():
        return term_program

    # Fallback: try to get from parent process
    if term_program:
        return term_program

    return "your terminal app"


# Model information: (id, description, size, download_time_estimate)
WHISPER_MODELS = [
    ("large-v3", "Best quality", "~3.0 GB", "5-10 min"),
    ("large-v3-turbo", "Fast + good quality", "~1.6 GB", "3-5 min"),
    ("medium", "Balanced", "~1.5 GB", "3-5 min"),
    ("small", "Quick start", "~500 MB", "1-2 min"),
    ("base", "Minimal", "~150 MB", "<1 min"),
]

# Provider options
PROVIDERS = [
    ("mlx", "Local MLX Whisper", "Apple Silicon, offline"),
    ("groq", "Groq Cloud", "Fast, requires API key"),
    ("parakeet", "Local Parakeet", "Apple Silicon, English only"),
]


def welcome_banner(reconfigure: bool = False) -> None:
    """Show welcome panel."""
    console.print()
    if reconfigure:
        console.print(
            Panel.fit(
                "[bold blue]STT Configuration[/bold blue]",
                border_style="blue",
                padding=(0, 2),
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold blue]Welcome to STT[/bold blue]\n\n"
                "Voice-to-text for macOS vibe coding.\n"
                "Hold a key, speak, release — words appear.\n\n"
                "[dim]Let's get you set up...[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )
    console.print()


def select_provider(current: str = None) -> str:
    """Interactive provider selection."""
    console.print("[bold]Select transcription provider:[/bold]\n")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Provider", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Notes", style="dim")

    default_choice = 1
    for i, (provider_id, name, notes) in enumerate(PROVIDERS, 1):
        marker = ""
        if current and provider_id == current:
            marker = " [yellow](current)[/yellow]"
            default_choice = i
        elif i == 1 and not current:
            marker = " [bold green](recommended)[/bold green]"
        table.add_row(str(i), provider_id + marker, name, notes)

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask("Select provider", default=default_choice, show_default=True)
        if 1 <= choice <= len(PROVIDERS):
            provider_id = PROVIDERS[choice - 1][0]
            console.print(f"\n[green]Selected:[/green] {provider_id}\n")
            return provider_id
        console.print("[red]Invalid choice.[/red]")


def get_groq_api_key(current: str = None) -> str:
    """Get Groq API key from user."""
    console.print("[bold]Groq API Key[/bold]\n")
    console.print("Get your API key at: [link=https://console.groq.com]https://console.groq.com[/link]\n")

    if current:
        masked = current[:4] + "*" * (len(current) - 8) + current[-4:] if len(current) > 8 else "****"
        console.print(f"Current key: [dim]{masked}[/dim]\n")

    while True:
        key = Prompt.ask("API key", default=current or "", show_default=False)
        if not key:
            if current:
                return current
            console.print("[red]API key required for Groq provider.[/red]")
            continue
        if not key.startswith("gsk_"):
            if not Confirm.ask("Key doesn't look like a Groq key (should start with 'gsk_'). Use anyway?", default=False):
                continue
        console.print()
        return key


def select_model(current: str = None) -> str:
    """Interactive model selection with explanations."""
    console.print("[bold]Select a Whisper model:[/bold]\n")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="cyan")
    table.add_column("Quality", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Download", style="dim")

    default_choice = 1
    for i, (model_id, desc, size, time_est) in enumerate(WHISPER_MODELS, 1):
        marker = ""
        if current and model_id == current:
            marker = " [yellow](current)[/yellow]"
            default_choice = i
        elif i == 1 and not current:
            marker = " [bold green](recommended)[/bold green]"
        table.add_row(str(i), model_id + marker, desc, size, time_est)

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask("Select model", default=default_choice, show_default=True)
        if 1 <= choice <= len(WHISPER_MODELS):
            model_id = WHISPER_MODELS[choice - 1][0]
            console.print(f"\n[green]Selected:[/green] {model_id}\n")
            return model_id
        console.print("[red]Invalid choice. Please select 1-5.[/red]")


def select_audio_device(current: str = None) -> str | None:
    """Interactive audio device selection."""
    try:
        import sounddevice as sd
    except ImportError:
        return None

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            is_default = i == sd.default.device[0]
            is_current = current and dev["name"] == current
            input_devices.append((i, dev["name"], is_default, is_current))

    if not input_devices:
        console.print("[yellow]No input devices found.[/yellow]")
        return None

    console.print("[bold]Select audio input device:[/bold]\n")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Device", style="cyan")
    table.add_column("", style="green")

    default_choice = 1
    for idx, (dev_id, name, is_default, is_current) in enumerate(input_devices, 1):
        markers = []
        if is_default:
            markers.append("default")
        if is_current:
            markers.append("current")
            default_choice = idx
        elif is_default and not current:
            default_choice = idx
        marker = f"[{', '.join(markers)}]" if markers else ""
        table.add_row(str(idx), name, marker)

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask("Select device", default=default_choice, show_default=True)
        if 1 <= choice <= len(input_devices):
            _, name, _, _ = input_devices[choice - 1]
            console.print(f"\n[green]Selected:[/green] {name}\n")
            return name
        console.print("[red]Invalid choice.[/red]")


HOTKEYS = [
    ("cmd_r", "Right Command"),
    ("cmd_l", "Left Command"),
    ("alt_r", "Right Option"),
    ("alt_l", "Left Option"),
    ("ctrl_r", "Right Control"),
    ("shift_r", "Right Shift"),
]


def select_hotkey(current: str = None) -> str:
    """Interactive hotkey selection."""
    console.print("[bold]Select trigger hotkey:[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", style="cyan")
    table.add_column("", style="yellow")

    default_choice = 1
    for i, (key_id, desc) in enumerate(HOTKEYS, 1):
        marker = ""
        if current and key_id == current:
            marker = "(current)"
            default_choice = i
        elif i == 1 and not current:
            marker = "(recommended)"
        table.add_row(str(i), desc, marker)

    console.print(table)
    console.print()

    while True:
        choice = IntPrompt.ask("Select hotkey", default=default_choice, show_default=True)
        if 1 <= choice <= len(HOTKEYS):
            key_id, desc = HOTKEYS[choice - 1]
            console.print(f"\n[green]Selected:[/green] {desc}\n")
            return key_id
        console.print("[red]Invalid choice.[/red]")


def show_permission_error() -> None:
    """Show permission error with specific terminal app name."""
    terminal = get_terminal_app()
    console.print()
    console.print(
        Panel(
            "[bold red]Permissions Required[/bold red]\n\n"
            "STT needs macOS permissions to:\n"
            "  • Capture global hotkey (Input Monitoring)\n"
            "  • Type text into apps (Accessibility)\n\n"
            f"[bold]Grant permissions to: [cyan]{terminal}[/cyan][/bold]\n"
            "[dim](not 'stt' or 'python')[/dim]\n\n"
            "[yellow]System Settings → Privacy & Security → Accessibility[/yellow]\n"
            "[yellow]System Settings → Privacy & Security → Input Monitoring[/yellow]",
            border_style="red",
            padding=(1, 2),
        )
    )
    console.print()


def open_accessibility_settings() -> None:
    """Open macOS Accessibility settings."""
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False,
    )


def open_input_monitoring_settings() -> None:
    """Open macOS Input Monitoring settings."""
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"],
        check=False,
    )


def prompt_open_settings() -> bool:
    """Ask user if they want to open System Settings."""
    return Confirm.ask("Open System Settings now?", default=True)


def verify_permissions() -> bool:
    """Test that accessibility permissions actually work."""
    try:
        from pynput.keyboard import Controller
        # Just instantiate - if permissions are denied, this may raise or later ops will fail
        Controller()
        return True
    except Exception:
        return False


def check_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache, HfFileSystemResolvedPath

        repo_id = f"mlx-community/whisper-{model_id}-mlx"
        # Check for config.json as a proxy for the model being cached
        result = try_to_load_from_cache(repo_id, "config.json")
        return result is not None
    except Exception:
        return False


def get_model_download_size(model_id: str) -> int | None:
    """Get the total download size for a model in bytes."""
    sizes = {
        "large-v3": 3_000_000_000,
        "large-v3-turbo": 1_600_000_000,
        "medium": 1_500_000_000,
        "small": 500_000_000,
        "base": 150_000_000,
    }
    return sizes.get(model_id)


def create_download_progress() -> Progress:
    """Create a rich progress bar for model download."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


class _ProgressCallback:
    """Progress callback that updates a rich progress bar."""

    def __init__(self, progress: Progress, task_id):
        self.progress = progress
        self.task_id = task_id
        self._current_file = None
        self._file_sizes = {}
        self._completed = 0

    def __call__(self, *args, **kwargs):
        # HF Hub progress callback signature varies by version
        # Try to extract useful info
        if args:
            # Newer API: (downloaded_bytes, total_bytes)
            if len(args) >= 2 and isinstance(args[0], (int, float)):
                downloaded, total = args[0], args[1]
                if total and total > 0:
                    self.progress.update(self.task_id, completed=downloaded, total=total)


def download_model_with_progress(model_id: str, progress: Progress, task_id) -> bool:
    """
    Download a model with progress tracking.

    Returns True if download succeeded (or model was already cached).
    """
    import os
    import tqdm

    repo_id = f"mlx-community/whisper-{model_id}-mlx"
    total_size = get_model_download_size(model_id)

    if total_size:
        progress.update(task_id, total=total_size)

    # Custom tqdm class that reports to our rich progress bar
    class RichTqdm(tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            self._rich_progress = progress
            self._rich_task = task_id
            self._rich_total = total_size
            self._rich_completed = 0
            # Suppress default output
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            self._rich_completed += n
            if self._rich_total:
                # Scale to estimated total
                scaled = min(self._rich_completed, self._rich_total)
                self._rich_progress.update(self._rich_task, completed=scaled)

        def close(self):
            super().close()

    # Patch tqdm temporarily for HF Hub downloads
    original_tqdm = tqdm.tqdm
    tqdm.tqdm = RichTqdm

    # Also set environment to not disable progress
    old_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            local_files_only=False,
        )

        # Mark complete
        if total_size:
            progress.update(task_id, completed=total_size)

        return True

    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        return False

    finally:
        # Restore original tqdm
        tqdm.tqdm = original_tqdm
        if old_env is not None:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_env


def run_setup(save_config_fn, current_config: dict = None, reconfigure: bool = False) -> dict:
    """
    Run the setup wizard.

    Args:
        save_config_fn: Function to save config values (key, value)
        current_config: Dict with current config values (for reconfigure mode)
        reconfigure: If True, show "Configuration" instead of "Welcome"

    Returns:
        Dict with setup results
    """
    current = current_config or {}
    welcome_banner(reconfigure=reconfigure)

    # Step 1: Provider selection
    provider = select_provider(current=current.get("provider"))
    save_config_fn("PROVIDER", provider)

    # Step 2: Provider-specific config
    model_id = None
    if provider == "mlx":
        model_id = select_model(current=current.get("model"))
        save_config_fn("WHISPER_MODEL", model_id)

        # Download model if needed
        if not check_model_cached(model_id):
            console.print(f"Downloading [cyan]{model_id}[/cyan]...\n")
            with create_download_progress() as progress:
                task = progress.add_task(f"Downloading {model_id}", total=None)
                success = download_model_with_progress(model_id, progress, task)
                if not success:
                    console.print("[yellow]Model will be downloaded when STT starts.[/yellow]\n")
        else:
            console.print(f"[green]Model '{model_id}' is already downloaded.[/green]\n")

    elif provider == "groq":
        api_key = get_groq_api_key(current=current.get("groq_api_key"))
        save_config_fn("GROQ_API_KEY", api_key)

    elif provider == "parakeet":
        console.print("[dim]Parakeet uses a fixed model (English only).[/dim]\n")

    # Step 3: Audio device & hotkey
    device = select_audio_device(current=current.get("audio_device"))
    if device:
        save_config_fn("AUDIO_DEVICE", device)

    hotkey = select_hotkey(current=current.get("hotkey"))
    save_config_fn("HOTKEY", hotkey)

    # Done
    summary_lines = [
        "[bold green]Setup Complete![/bold green]\n",
        f"Provider: [cyan]{provider}[/cyan]",
    ]
    if model_id:
        summary_lines.append(f"Model: [cyan]{model_id}[/cyan]")
    summary_lines.extend([
        f"Hotkey: [cyan]{hotkey}[/cyan]",
        f"Device: [cyan]{device or 'System default'}[/cyan]",
    ])
    if not reconfigure:
        summary_lines.append("\n[dim]Starting STT...[/dim]")

    console.print(
        Panel(
            "\n".join(summary_lines),
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    return {
        "provider": provider,
        "model": model_id,
        "hotkey": hotkey,
        "device": device,
    }


# Keep old name for compatibility
def run_first_time_setup(save_config_fn) -> dict:
    """Alias for run_setup for first-time setup."""
    return run_setup(save_config_fn, reconfigure=False)


def show_loading_progress(description: str) -> Progress:
    """Create a simple loading progress bar."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
        transient=True,
    )
