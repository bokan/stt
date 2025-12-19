# STT

Voice-to-text for macOS. Hold a key, speak, release — your words appear wherever your cursor is.

Built for vibe coding and conversations with AI agents. Supports cloud (Groq API) or local (MLX Whisper) transcription.

## Features

- **Global hotkey** — works in any application, configurable trigger key
- **Hold-to-record** — no start/stop buttons, just hold and speak
- **Auto-type** — transcribed text is typed directly into the active field
- **Shift+record** — automatically sends Enter after typing (great for chat interfaces)
- **Audio feedback** — subtle system sounds confirm recording state (can be disabled)
- **Slash commands** — say "slash help" to type `/help`
- **Context prompts** — improve accuracy with domain-specific vocabulary
- **Auto-updates** — notifies when a new version is available

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- [UV](https://docs.astral.sh/uv/) package manager
- **For cloud mode (optional):** [Groq API key](https://console.groq.com)

## Installation

```bash
uv tool install git+https://github.com/bokan/stt.git
```

On first run, a setup wizard will guide you through configuration.

To update:

```bash
uv tool upgrade stt
```

## Permissions

STT needs macOS permissions to capture the global hotkey and type text into other apps.

Grant these to **your terminal app** (iTerm2, Terminal, Warp, etc.) — not "stt":

- **Accessibility** — System Settings → Privacy & Security → Accessibility
- **Input Monitoring** — System Settings → Privacy & Security → Input Monitoring

## Usage

```bash
stt
```

| Action | Keys |
|--------|------|
| Record | Hold **Right Command** (default) |
| Record + Enter | Hold **Left Shift** while recording |
| Cancel recording / stuck transcription | **ESC** |
| Quit | **Ctrl+C** |

## Configuration

Settings are stored in `~/.config/stt/.env`. Run `stt --config` to reconfigure, or edit directly:

```bash
# Transcription provider: "mlx" (local, default) or "groq" (cloud)
PROVIDER=mlx

# Required for cloud mode only
GROQ_API_KEY=gsk_...

# Audio device (saved automatically after first selection)
AUDIO_DEVICE=2

# Language code for transcription
LANGUAGE=en

# Trigger key: cmd_r, cmd_l, alt_r, alt_l, ctrl_r, ctrl_l, shift_r
HOTKEY=cmd_r

# Context prompt to improve accuracy for specific terms
PROMPT=Claude, Anthropic, TypeScript, React, Python

# Disable audio feedback sounds
SOUND_ENABLED=true
```

### Local Mode (MLX) — Default

Local transcription uses Apple Silicon GPU acceleration via MLX. On first run, the Whisper large-v3 model (~3GB) will be downloaded and cached. Subsequent runs load from cache.

Runs completely offline — no API key required.

### Cloud Mode (Groq)

To use cloud transcription instead:

```bash
PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Requires a [Groq API key](https://console.groq.com) (free tier available).

### Prompt Examples

The `PROMPT` setting helps Whisper recognize domain-specific terms:

```bash
# Programming
PROMPT=TypeScript, React, useState, async await, API endpoint

# AI tools
PROMPT=Claude, Anthropic, OpenAI, Groq, LLM, GPT
```

## Development

```bash
git clone https://github.com/bokan/stt.git
cd stt
uv sync
uv run stt
```

## License

MIT
