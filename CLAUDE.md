# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xhisper-local is a fork of [xhisper](https://github.com/imaginalnika/xhisper) by imaginalnika. It provides dictation at cursor for Linux using local Whisper transcription and AI-powered formatting.

**Key features:** Local transcription via Parakeet or Whisper (no API keys), AI formatting via Ollama, smart command detection, GPU acceleration, works offline.

## Build & Install

```bash
# Install Python dependencies
pip3 install --break-system-packages 'onnx-asr[cpu]'  # parakeet engine (default); onnxruntime is an extra since onnx-asr 0.12 — a distro onnxruntime package satisfies it too
pip3 install --break-system-packages faster-whisper # whisper engine (optional)

# Build C binary
make

# Install system-wide
sudo make install

# Run from source (without installing)
./xhisper.sh --local
```

To use AI formatting, install Ollama and pull Gemma3:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b
```

## Architecture

### Components

1. **xhispertoold** (C daemon) - Virtual keyboard via Linux uinput, listens on abstract Unix socket `@xhisper_socket`
2. **xhispertool** (C client) - Communicates with daemon via socket; symlinked as `xhispertoold` for daemon mode
3. **xhisper_transcribe.py** (Python) - Transcription via faster-whisper (whisper engine) or onnx-asr (parakeet engine, default); `--engine` dispatch. For parakeet it can also run as a transcription server (`--serve`) that keeps a warmed-up model loaded, on a socket in `$XDG_RUNTIME_DIR/xhisper/`; `--use-server` transcribes through it and falls back to a one-shot run
4. **xhisper_ime.py** (Python) - IBus input method for `insert-method: ibus`. `--serve` runs a service (PyGObject; started under the system Python since the xhisper venv usually lacks `gi`) that registers a pass-through engine at runtime, activates it only during a dictation, shows statuses as preedit and commits the final text in one go; socket in `$XDG_RUNTIME_DIR/xhisper/`. The client commands (`begin`/`status`/`finish`/`end`) need only the standard library
5. **xhisper.sh** (Bash) - Main orchestrator: recording, transcription, AI formatting, typing

### Data Flow

```
User presses hotkey → xhisper.sh toggles recording state
  ├─ First press: start the transcription server if not running (parakeet,
  │  transcription-server on) and preload the Ollama model post-processing
  │  will likely use → pw-record starts → types "(recording...)"
  │  (insert-method ibus: shows it as preedit instead, unless no IBus
  │  text field has focus)
  └─ Second press: kill pw-record → check silence
      ├─ Silent: type "(no sound detected)", exit
      └─ Not silent:
          ├─ transcribe via parakeet (default; through the server when on)
          │  or faster-whisper
          ├─ standard text: cleanup model if cleanup-model set (native
          │  "text:..." prompt; the tuned alias appends the "\nout:" marker),
          │  else instruct LLM per post-process-standard
          ├─ command/email text: instruct LLM (post-process-model)
          └─ type result via daemon (insert-method ibus: commit it via
             xhisper_ime; type only if no IBus text field has focus, or
             if the statuses were typed)
```

### Character Input

- **ASCII (32-126)**: Typed directly via uinput keycodes (US QWERTY layout)
- **Non-ASCII**: Clipboard paste (wl-copy/wl-paste or xclip)

The `ascii2keycode_map` in xhispertool.c maps ASCII to Linux keycodes.

With `insert-method: ibus`, the final text is one IBus commit (any Unicode, no clipboard; newlines become spaces when the field's input purpose is terminal). Focus check: IBus attaches the engine to an internal placeholder context when no app text field has focus and drops commits there. IBus >= 1.5.27 reports it via `focus-in-id` client "fake"; older IBus only via the placeholder's path, `InputContext_1` (an implementation detail). The hotkey's key grab moves IBus focus to the placeholder until the keys are released (GTK counts NotifyGrab as focus-out), so `begin`/`finish` wait for a real field until xhisper.sh's key-release margin (`--focus-wait-until`). One channel per dictation: if the statuses were typed, the text is typed too — a commit overtakes the typed backspaces still erasing the status, which then eat the end of the text. Known limit: nothing observes the actual key release, so a press held past the margin still loses — the start press falls back to typing during the grab (the typed status is lost or garbled, and its 14 backspaces at stop can eat text before the cursor), and a stop press held until the text is ready types it during the grab; README.md documents this under Troubleshooting ("Tap the shortcut"). When a runtime-registered component goes away, ibus-daemon unsets an installed global engine (it only checks runtime-registered engines), so the service restores the user's engine after it exits (detached helper) and on its next start after a crash.

## Configuration

Location: `~/.config/xhisper/xhisperrc` (or `$XDG_CONFIG_HOME/xhisper/xhisperrc`)

Key settings:
- `transcription-engine`: parakeet (default) or whisper
- `parakeet-model`: onnx-asr model (nemo-parakeet-tdt-0.6b-v2 = English, v3 = multilingual)
- `transcription-server`: on (default) or off — keep the parakeet model loaded between dictations; started at record time, falls back to one-shot
- `transcription-server-idle`: minutes without a dictation before the server exits (default 5; 0 = never)
- `model-name`: Whisper size (tiny, base, small, medium, large-v3) — whisper engine only
- `model-device`: auto, cpu, or cuda — whisper always; parakeet stays on CPU for auto/cpu and uses the GPU for cuda via onnxruntime-gpu (pin `==1.24.4`, the last CUDA 12 build; wheels ≥ 1.26 need CUDA 13), plus `nvidia-curand-cu12 nvidia-cufft-cu12`
- `transcription-corrections`: comma-separated from=to phrase map applied to raw transcriptions (ASR mishear fixups, e.g. `clawed code=Claude Code,clawed=Claude`); case-insensitive whole-phrase, longest source first
- `vad-min-silence-ms`: pause length before whisper VAD splits (default 5000)
- `post-process-model`: Ollama instruct model (e.g., gemma3:4b); used for command/email modes, and for standard text per the next two keys
- `cleanup-model`: Ollama model for fast dictation cleanup of standard text — Spokenly cleanup-1 (`xhisper-cleanup-1` alias, auto-created from the `hf.co/spokenly/cleanup-1-gguf:Q8_0` pull with the correct prompt format and greedy decoding). When set, standard text goes through it and `post-process-model` only serves command/email modes
- `post-process-standard`: AI pass for standard text — auto (run cleanup model if set, else skip when parakeet punctuates; default), on (always), off (never; command/email unaffected)
- `post-process-mode`: auto, standard, command, email
- `post-process-timeout`: Max seconds for LLM formatting (on timeout the raw transcription is kept)
- `insert-method`: type (default; key by key via uinput) or ibus (one IBus commit, statuses as preedit, typing fallback where no IBus text field has focus; needs python3-gi + gir1.2-ibus-1.0)
- `smart-insert`: on (default) or off, ibus only — `fit_to_context` in xhisper_ime.py adjusts spacing, first-word case and the final period to the IBus surrounding text. At commit it asks the field (RequireSurroundingText plus a hide/show of the preedit, a GTK trigger point) and fits first if a fresh answer comes within 40ms (IBus >= 1.5.28 clients always answer). Older clients send only changed text, so otherwise it commits as is and fits from the update sent after the commit (delete_surrounding_text + commit). Terminals are skipped; fields that never answer (Chrome) are remembered per context path

## Command Line Options

```
xhisper                  # Toggle recording (auto mode)
xhisper --local          # Run from build dir
xhisper --log            # Show log
xhisper --mode=command   # Force command mode
xhisper --mode=email     # Force email mode
xhisper --mode=standard  # Force standard text mode
xhisper --rightalt       # Use right alt as input switch (non-QWERTY layouts)
```

## AI Formatting Modes

- **auto**: Detects commands vs text automatically (checks for sudo, apt, git, etc.)
- **standard**: Grammar, punctuation, capitalization
- **command**: Linux command syntax correction (e.g., "pseudo" → "sudo")
- **email**: Email body formatting with paragraph breaks (no subject/salutation added)

## Environment

- User must be in `input` group for `/dev/uinput` access
- Requires CUDA toolkit for GPU acceleration
- Models cached in `~/.cache/xhisper/parakeet/` (Parakeet — local dir, not the HF blob cache, to avoid ONNX external-data path errors), `~/.cache/huggingface/hub/` (Whisper) and `~/.ollama/models/` (Ollama)
- Log: `/tmp/xhisper.log`
- Daemon log: `/tmp/xhispertoold.log`

## Testing

Test transcription standalone:
```bash
python3 xhisper_transcribe.py recording.wav --engine parakeet
python3 xhisper_transcribe.py recording.wav --engine whisper --model base --device cuda
```

Test AI formatting:
```bash
echo "fix this text" | ollama run gemma3:4b "Fix grammar. Output only text."
```
