#!/bin/bash

# When the hotkey fired, near enough: see wait_for_key_release
SCRIPT_START_US=$(date +%s%6N)

# xhisper v2.0
# Dictate anywhere in Linux. Transcription at your cursor.
# - Transcription via local Whisper models (faster-whisper)

# Configuration (see default_xhisperrc or ~/.config/xhisper/xhisperrc):
# - transcription-engine : parakeet or whisper (default parakeet)
# - parakeet-model : onnx-asr model for the parakeet engine
# - transcription-server : on or off (default on, parakeet only). Keeps the
#   model loaded and warmed up between dictations (~1s instead of ~5s per
#   transcription on a laptop CPU); started when recording begins so the load
#   overlaps with speaking. Falls back to a one-shot run if unavailable.
# - transcription-server-idle : minutes without a dictation before the server
#   exits and frees the model's ~2.5GB (default 5; 0 = never)
# - model-name : Whisper model size (tiny, base, small, medium, large-v3)
# - model-device : Device to use (auto, cpu, cuda). whisper: auto detects
#   CUDA. parakeet: auto/cpu run on CPU (default, fast); cuda runs on the
#   GPU via onnxruntime-gpu (which replaces onnxruntime)
# - model-language : Language code for faster/more accurate transcription (e.g., en)
# - transcription-prompt : context words for better Whisper accuracy
# - vad-min-silence-ms : silence before the whisper VAD splits segments (whisper only)
# - silence-threshold : max volume in dB to consider silent (e.g., -50)
# - silence-percentage : percentage of recording that must be silent (e.g., 95)
# - non-ascii-initial-delay : sleep after first non-ASCII paste (seconds)
# - non-ascii-default-delay : sleep after subsequent non-ASCII pastes (seconds)
# - transcription-corrections : comma-separated from=to phrase corrections
#   applied to the raw transcription (the ASR cannot be vocabulary-biased, so
#   words it mishears — e.g. "Claude" heard as "clawed" — are fixed here).
#   Case-insensitive whole-phrase matches, longest source first:
#     transcription-corrections : clawed code=Claude Code,clawed=Claude
# - post-process-standard : LLM pass for standard text - auto (skip when
#   parakeet already punctuates), on (always), off (never; command/email
#   modes are unaffected and always use the LLM)
# - cleanup-model : Ollama model for fast dictation cleanup of standard text
#   (filler words, spoken corrections, numbers), e.g. Spokenly cleanup-1.
#   When set it replaces the LLM for standard text (see post-process-standard);
#   post-process-model then only serves command/email modes.
# - insert-method : type (default) or ibus. type: key by key through the
#   uinput daemon (~10ms per character, non-ASCII via the clipboard). ibus:
#   the whole text as one commit through xhisper's IBus input method, with
#   status messages shown as preedit text that never enters the document;
#   falls back to typing where no IBus text field has focus. See
#   xhisper_ime.py.
# - smart-insert : on (default) or off, insert-method ibus only. Fits the
#   dictation to the text around the cursor: a leading space after a word,
#   a lowercase first word mid-sentence (common words only, never names),
#   no period before the words that follow.
# Note: model-language, transcription-prompt, and vad-min-silence-ms apply to
# the whisper engine only; parakeet v2 is English-only, v3 auto-detects.

# Requirements:
# - pipewire with pw-record (pipewire-bin on Debian/Ubuntu, pipewire-utils on Fedora)
# - wl-clipboard (Wayland) or xclip (X11) — required, non-ASCII is pasted via clipboard
# - bc (timing in the log)
# - Python 3 with onnx-asr (parakeet engine) and/or faster-whisper (whisper engine)
# - make to build, sudo make install to install
# Optional: ollama for command/email formatting and/or dictation cleanup
#   (see cleanup-model below); CUDA toolkit for whisper on GPU; IBus with
#   python3-gi and gir1.2-ibus-1.0 for insert-method ibus

# Parse command-line arguments
LOCAL_MODE=0
WRAP_KEY=""
POST_PROCESS_MODE=""  # Empty = use config/default
for arg in "$@"; do
  case "$arg" in
    --local)
      LOCAL_MODE=1
      ;;
    --log)
      if [ -f "/tmp/xhisper.log" ]; then
        cat /tmp/xhisper.log
      else
        echo "No log file found at /tmp/xhisper.log" >&2
      fi
      exit 0
      ;;
    --mode=*)
      POST_PROCESS_MODE="${arg#--mode=}"
      ;;
    --leftalt|--rightalt|--leftctrl|--rightctrl|--leftshift|--rightshift|--super)
      if [ -n "$WRAP_KEY" ]; then
        echo "Error: Multiple wrap keys not yet supported" >&2
        exit 1
      fi
      WRAP_KEY="${arg#--}"
      ;;
    *)
      echo "Error: Unknown option '$arg'" >&2
      echo "Usage: xhisper [--local] [--log] [--mode=auto|standard|command|email] [--leftalt|--rightalt|--leftctrl|--rightctrl|--leftshift|--rightshift|--super]" >&2
      exit 1
      ;;
  esac
done

# Set binary paths based on local mode
if [ "$LOCAL_MODE" -eq 1 ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  XHISPERTOOL="$SCRIPT_DIR/xhispertool"
  XHISPERTOOLD="$SCRIPT_DIR/xhispertoold"
else
  XHISPERTOOL="xhispertool"
  XHISPERTOOLD="xhispertoold"
fi

RECORDING="/tmp/xhisper.wav"
LOGFILE="/tmp/xhisper.log"
PROCESS_PATTERN="pw-record.*$RECORDING"
# How the start press showed "(recording...)": "ime" (preedit) or "type"
# (typed, so the stop press deletes it)
STATUS_MODE_FILE="/tmp/xhisper.status-mode"

# Default configuration
transcription_engine="parakeet"
parakeet_model="nemo-parakeet-tdt-0.6b-v2"
transcription_server="on"
transcription_server_idle=5
model_name="base"
model_device="auto"
model_language=""
transcription_prompt=""
vad_min_silence_ms=5000
silence_threshold=-50
silence_percentage=95
non_ascii_initial_delay=0.1
non_ascii_default_delay=0.025
post_process_model=""
transcription_corrections=""
cleanup_model=""
post_process_timeout=10
post_process_mode="auto"
post_process_standard="auto"
insert_method="type"
smart_insert="on"

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/xhisper/xhisperrc"

if [ -f "$CONFIG_FILE" ]; then
  while IFS=: read -r key value || [ -n "$key" ]; do
    # Skip comments and empty lines
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue

    # Trim whitespace, then one pair of surrounding quotes from the value.
    # Parameter expansion, not echo | sed: two subshells per line added
    # ~0.1s to every keypress.
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value#\"}"
    value="${value%\"}"

    case "$key" in
      transcription-engine) transcription_engine="$value" ;;
      parakeet-model) parakeet_model="$value" ;;
      transcription-server) transcription_server="$value" ;;
      transcription-server-idle) transcription_server_idle="$value" ;;
      model-name) model_name="$value" ;;
      model-device) model_device="$value" ;;
      model-language) model_language="$value" ;;
      transcription-prompt) transcription_prompt="$value" ;;
      transcription-corrections) transcription_corrections="$value" ;;
      vad-min-silence-ms) vad_min_silence_ms="$value" ;;
      silence-threshold) silence_threshold="$value" ;;
      silence-percentage) silence_percentage="$value" ;;
      non-ascii-initial-delay) non_ascii_initial_delay="$value" ;;
      non-ascii-default-delay) non_ascii_default_delay="$value" ;;
      post-process-model) post_process_model="$value" ;;
      cleanup-model) cleanup_model="$value" ;;
      post-process-timeout) post_process_timeout="$value" ;;
      post-process-mode) post_process_mode="$value" ;;
      post-process-standard) post_process_standard="$value" ;;
      insert-method) insert_method="$value" ;;
      smart-insert) smart_insert="$value" ;;
    esac
  done < "$CONFIG_FILE"
fi

# Command-line mode overrides config
[ -n "$POST_PROCESS_MODE" ] && post_process_mode="$POST_PROCESS_MODE"

# Add CUDA library paths for faster-whisper.
# pip-installed nvidia packages (cublas, cudnn, nvrtc) are not on the default
# loader path, and ctranslate2 needs them when it runs the encoder on the GPU.
# Resolve their lib dirs at runtime so this works for any python version.
# Only when a GPU run is possible (whisper's auto device may pick CUDA): the
# lookup starts a python3 process on every keypress.
if [ "$model_device" = "cuda" ] || { [ "$transcription_engine" = "whisper" ] && [ "$model_device" != "cpu" ]; }; then
  NV_LIBS=$(python3 -c "import os, site, sysconfig, glob; dirs = [sysconfig.get_paths()['purelib'], site.getusersitepackages()]; print(':'.join(d for base in dirs for d in glob.glob(os.path.join(base, 'nvidia', '*', 'lib')) if os.path.isdir(d)))" 2>/dev/null)
  # Ollama bundles a CUDA 12 runtime (cublas, cudart) directly in cuda_v12/ —
  # there is no lib/ subdir, and it is a fine libcublas.so.12 source for ctranslate2.
  [ -d /usr/local/lib/ollama/cuda_v12 ] && NV_LIBS="$NV_LIBS:/usr/local/lib/ollama/cuda_v12"
  export LD_LIBRARY_PATH="$NV_LIBS:$LD_LIBRARY_PATH"
fi

# Auto-start daemon if not running
if ! pgrep -x xhispertoold > /dev/null; then
    "$XHISPERTOOLD" 2>> /tmp/xhispertoold.log &
    sleep 1  # Give daemon time to start

    # Verify daemon started successfully
    if ! pgrep -x xhispertoold > /dev/null; then
        echo "Error: Failed to start xhispertoold daemon" >&2
        echo "Check /tmp/xhispertoold.log for details" >&2
        exit 1
    fi
fi

# Check if xhispertool is available
if ! command -v "$XHISPERTOOL" &> /dev/null; then
    echo "Error: xhispertool not found" >&2
    echo "Please either:" >&2
    echo "  - Run 'sudo make install' to install system-wide" >&2
    echo "  - Run 'xhisper --local' from the build directory" >&2
    exit 1
fi

# Resolve transcription script to an absolute path.
# python3 does not search PATH for script files, so a bare name would only
# work when the cwd happens to be the install directory.
if [ "$LOCAL_MODE" -eq 1 ]; then
    TRANSCRIPT_SCRIPT="$SCRIPT_DIR/xhisper_transcribe.py"
else
    TRANSCRIPT_SCRIPT="$(command -v xhisper_transcribe)"
fi

if [ -z "$TRANSCRIPT_SCRIPT" ] || [ ! -f "$TRANSCRIPT_SCRIPT" ]; then
    echo "Error: xhisper_transcribe not found" >&2
    echo "Please either:" >&2
    echo "  - Run 'sudo make install' to install system-wide" >&2
    echo "  - Run 'xhisper --local' from the build directory" >&2
    exit 1
fi

# The input-method client (insert-method ibus); without it, type as usual
IME_SCRIPT=""
if [ "$insert_method" = "ibus" ]; then
  if [ "$LOCAL_MODE" -eq 1 ]; then
    IME_SCRIPT="$SCRIPT_DIR/xhisper_ime.py"
  else
    IME_SCRIPT="$(command -v xhisper_ime)"
  fi
  if [ -z "$IME_SCRIPT" ] || [ ! -f "$IME_SCRIPT" ]; then
    echo "Warning: insert-method is ibus but xhisper_ime is not installed; typing instead" >> /tmp/xhisper.log
    IME_SCRIPT=""
  fi
fi

# Detect clipboard tool
if command -v wl-copy &> /dev/null; then
    CLIP_COPY="wl-copy"
    CLIP_PASTE="wl-paste"
elif command -v xclip &> /dev/null; then
    CLIP_COPY() { xclip -selection clipboard; }
    CLIP_PASTE() { xclip -o -selection clipboard; }
else
    echo "Error: No clipboard tool found. Install wl-clipboard or xclip." >&2
    exit 1
fi

press_wrap_key() {
  if [ -n "$WRAP_KEY" ]; then
    "$XHISPERTOOL" "$WRAP_KEY"
  fi
}

# Type text at the cursor. Named type_text (not paste): a function named
# paste silently shadows the paste(1) binary, which bit this script once —
# apply_corrections' "paste -sd';' -" rule-join typed "-sd;" at the cursor
# on every corrected dictation.
type_text() {
  local text="$1"
  audit "TYPE [${text:0:70}]"
  press_wrap_key
  # Type character by character
  # Use xhispertool type for ASCII (32-126), clipboard+paste for Unicode
  for ((i=0; i<${#text}; i++)); do
    local char="${text:$i:1}"
    local ascii=$(printf '%d' "'$char")

    if [[ $ascii -ge 32 && $ascii -le 126 ]]; then
      # ASCII printable character - use direct key typing (faster)
      "$XHISPERTOOL" type "$char"
    else
      # Unicode or special character - use clipboard
      echo -n "$char" | $CLIP_COPY
      "$XHISPERTOOL" paste
      # On first character (more error-prone), sleep longer
      [ "$i" -eq 0 ] && sleep "$non_ascii_initial_delay" || sleep "$non_ascii_default_delay"
    fi
  done
  press_wrap_key
}

# Audit trail for the typing layer: timestamps every keystroke command the
# script issues so injected-text bugs can be localized (if characters appear
# at the cursor while no audit line covers them, they come from outside
# xhisper — hotkey leakage, compositor, other software).
audit() {
  echo "[$(date +%H:%M:%S.%3N)] $1" >> "$LOGFILE"
}

delete_n_chars() {
  local n="$1"
  audit "BKSP x$n"
  for ((i=0; i<n; i++)); do
    "$XHISPERTOOL" backspace
  done
}

# Apply the transcription-corrections word map to a transcription: the ASR
# engines are pure speech-to-text with no vocabulary biasing, so words they
# mishear (homophones like "Claude"/"clawed") are corrected here, before mode
# detection and cleanup see the text. Matches are case-insensitive whole
# phrases (GNU sed \b + I flag); longest sources are applied first so
# "clawed code=Claude Code" wins over a standalone "clawed" rule.
apply_corrections() {
  local text="$1"
  [ -z "$transcription_corrections" ] && { echo "$text"; return; }

  # Split the comma list into sed rules, sorted by source length descending
  local -a rules=()
  local pair from to
  while IFS= read -r pair; do
    [ -z "$pair" ] && continue
    from="${pair%%=*}"
    to="${pair#*=}"
    # pairs without '=' (or with an empty source) are malformed — skip
    [ "$from" = "$pair" ] || [ -z "$from" ] && continue
    rules+=("s|\b${from}\b|${to}|gI")
  done < <(tr ',' '\n' <<<"$transcription_corrections" | awk -F= '{print length($1)"\t"$0}' | sort -rn | cut -f2-)

  # guard for a corrections list with no valid pairs
  [ "${#rules[@]}" -eq 0 ] && { echo "$text"; return; }

  # Join rules with ';' via IFS — pure bash. Do not use paste(1) here (see
  # the type_text comment: a same-named typing function once shadowed it and
  # typed "-sd;" at the cursor instead of joining the rules).
  local joined
  local IFS=';'
  joined="${rules[*]}"

  echo "$text" | sed -e "$joined"
}

is_silent() {
  local recording="$1"

  # Peak-amplitude check in Python rather than ffmpeg volumedetect:
  # snap-packaged ffmpeg (often symlinked as ffmpeg on Ubuntu) runs in its
  # own mount namespace and cannot read files in /tmp, which made the old
  # check fail open (silence was never detected).
  python3 - "$recording" "$silence_threshold" <<'EOF'
import sys, wave, array

path, threshold = sys.argv[1], float(sys.argv[2])
# dBFS threshold -> 16-bit sample amplitude (e.g., -50 dB -> ~104)
limit = 32768 * 10 ** (threshold / 20)
try:
    w = wave.open(path)
    data = array.array('h', w.readframes(w.getnframes()))
    peak = max(map(abs, data)) if data else 0
except Exception:
    peak = 32768  # unreadable file: assume not silent and let whisper try
sys.exit(0 if peak < limit else 1)
EOF
}

logging_end_and_write_to_logfile() {
  local title="$1"
  local result="$2"
  local logging_start="$3"

  local logging_end=$(date +%s%N)
  local time=$(echo "scale=3; ($logging_end - $logging_start) / 1000000000" | bc)

  echo "=== $title ===" >> "$LOGFILE"
  echo "Result: [$result]" >> "$LOGFILE"
  echo "Time: ${time}s" >> "$LOGFILE"
}

detect_mode() {
  local text="$1"
  local mode="${2:-$post_process_mode}"

  if [ "$mode" != "auto" ]; then
    echo "$mode"
    return
  fi

  # Check for command indicators
  if echo "$text" | grep -qE "^(sudo |apt |apt-get |git |npm |pip |systemctl |docker |podman |cd |ls |mkdir |rm |cp |mv |grep |find |cat |tail |head |ssh |curl |wget |make |cargo |python |node |code |vim |nano |man |chmod |chown |ln |tar |zip |unzip |mount |umount |ps |kill |top |htop |df |du |free |uname |export |alias |source |exit |pseudo )|^(apt|git|npm|pip|sudo|systemctl|docker|cargo) " || \
     echo "$text" | grep -qE " (sudo|apt|apt-get|git|npm|pip|systemctl|docker|podman|cargo|make) (install|update|upgrade|remove|purge|status|start|stop|restart|enable|disable|clone|pull|push|commit|add|log|diff|checkout|branch|merge|rebase|init)( |$)"; then
    echo "command"
  else
    echo "standard"
  fi
}

# Should the AI pass be skipped for standard-mode text?
#   auto (default): run the dedicated cleanup model when configured (it is
#                   cheap, ~0.3s); otherwise skip when the parakeet engine
#                   already punctuates
#   on  : always run the AI pass for standard text (even on parakeet)
#   off : never run any AI pass for standard text (command/email unaffected)
skip_standard_formatting() {
  [ "$1" = "standard" ] || return 1
  [ "$post_process_standard" = "off" ] && return 0
  [ "$post_process_standard" = "on" ] && return 1
  [ -n "$cleanup_model" ] && return 1
  [ "$transcription_engine" = "parakeet" ]
}

# Will any AI pass run for this resolved mode?
#   command/email : only when the instruct model (post-process-model) is set
#   standard      : when either the cleanup model or the instruct model is set
formatting_applies() {
  case "$1" in
    command|email) [ -n "$post_process_model" ] ;;
    *)             [ -n "$cleanup_model" ] || [ -n "$post_process_model" ] ;;
  esac
}

# Spokenly cleanup-1 (https://huggingface.co/spokenly/cleanup-1): a 0.6B
# dictation-cleanup model, ~639MB as Q8_0 GGUF. It is a fine-tune of a BASE
# model (Qwen3): it understands only its native completion format
# "text: ...\nout:" and expects greedy decoding. Two problems with running
# the plain hf.co pull: ollama's default sampling is temperature 0.8, and the
# chat template embedded in the GGUF renders "out: " WITH a trailing space —
# at temperature 0 that trailing-space token makes the model emit EOS
# immediately for many inputs (empty output). So on first use the script
# creates a tuned alias from the pull — instant, no weight copy — that
# overrides the template with the card-exact bare "out:" marker and pins
# greedy decoding plus the card's 4096-token context.
CLEANUP_MODEL_BASE="hf.co/spokenly/cleanup-1-gguf:Q8_0"

ensure_cleanup_model() {
  ollama show "$cleanup_model" &>/dev/null && return 0
  ollama show "$CLEANUP_MODEL_BASE" &>/dev/null || return 1
  local modelfile
  modelfile=$(mktemp)
  cat > "$modelfile" <<EOF
FROM $CLEANUP_MODEL_BASE
TEMPLATE """{{ if .Prompt }}{{ .Prompt }}
out:{{ end }}{{ .Response }}<|endoftext|>"""
PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
PARAMETER stop "<|endoftext|>"
EOF
  ollama create "$cleanup_model" -f "$modelfile" >/dev/null 2>&1
  local status=$?
  rm -f "$modelfile"
  return "$status"
}

# Run one prompt through ollama with a timeout: the cleaned result is written
# to $3 and the timeout/ollama segment's exit status (0 = ok, 124/137 =
# timeout) to $4. Both go through files rather than variables/command
# substitution — pipelines inside $(...) run in a subshell and would leave
# the parent's PIPESTATUS empty (and ollama's spinner escape codes are
# stripped by the sed, which is why it is part of the pipeline).
ollama_run() {
  local model="$1" prompt="$2" result_file="$3" status_file="$4"
  printf '%s' "$prompt" | timeout "$post_process_timeout" ollama run "$model" 2>/dev/null \
    | tr -d '\n\r' \
    | sed -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' > "$result_file"
  echo "${PIPESTATUS[1]}" > "$status_file"
}

post_process() {
  local text="$1"
  local mode="${2:-$post_process_mode}"
  local logging_start=$(date +%s%N)

  # Skip if empty or no model configured
  [ -z "$text" ] && echo "$text" && return
  [ -z "$post_process_model" ] && [ -z "$cleanup_model" ] && echo "$text" && return

  # Check if ollama is available
  if ! command -v ollama &> /dev/null; then
    echo "Error: ollama not found. Install ollama or disable post-processing." >&2
    echo "$text"
    return
  fi

  # Resolve auto-detection
  mode=$(detect_mode "$text" "$mode")

  # command/email modes need the instruct model (post-process-model);
  # if only the cleanup model is configured there is nothing to do for them
  if [ "$mode" != "standard" ] && [ -z "$post_process_model" ]; then
    echo "$text"
    return
  fi

  # Standard text may skip the AI pass — see skip_standard_formatting().
  # command/email always get LLM treatment on both engines.
  if skip_standard_formatting "$mode"; then
    logging_end_and_write_to_logfile "Post-Process [standard] (skipped: post-process-standard=$post_process_standard)" "$text" "$logging_start"
    echo "$text"
    return
  fi

  local prompt model

  if [ "$mode" = "standard" ] && [ -n "$cleanup_model" ]; then
    # Dedicated dictation-cleanup model: the tuned alias (see
    # ensure_cleanup_model) appends the native "\nout:" completion marker,
    # so the prompt is just the text under the "text:" key
    model="$cleanup_model"
    prompt=$(printf 'text: %s' "$text")
    mode="standard/cleanup"
  else
    # Build prompt based on mode
    model="$post_process_model"
    case "$mode" in
      command)
        prompt="You are a Linux command expert. Fix this command transcription. Rules: 1) Correct command names (sudo, apt, git, npm, systemctl, docker, etc.) 2) Keep flags exactly as spoken 3) Keep file paths as spoken 4) Fix pipe syntax 5) Output ONLY the corrected command, no explanations. Input: $text"
        ;;
      email)
        prompt="Fix the grammar, punctuation, and capitalization of this email body text. Rules: 1) Use proper paragraph breaks (double line breaks between paragraphs) 2) Keep the tone natural and conversational 3) Do NOT add subject, salutation, or sign-off - user is typing in the body field 4) Output ONLY the formatted body text. Input: $text"
        ;;
      standard|*)
        prompt="Fix the grammar, punctuation, and capitalization of this text. Important: use apostrophes for contractions like don't, can't, I'm, you're, it's, etc. Keep text natural and conversational. Output ONLY the corrected text. Text: $text"
        ;;
    esac
  fi

  local result_file status_file result
  result_file=$(mktemp)
  status_file=$(mktemp)
  ollama_run "$model" "$prompt" "$result_file" "$status_file"

  # First use of the cleanup alias: a plain run fails with "model not found"
  # — create the tuned alias from the base pull once and retry.
  if [ "$(cat "$status_file")" -ne 0 ] && [ "$mode" = "standard/cleanup" ] && ensure_cleanup_model; then
    ollama_run "$model" "$prompt" "$result_file" "$status_file"
  fi
  result=$(<"$result_file")
  local status
  status=$(<"$status_file")
  rm -f "$result_file" "$status_file"

  # Timeout (124 = SIGTERM, 137 = SIGKILL) or ollama crash: never paste
  # partial output — a killed run leaves whatever it already flushed in the
  # pipe, which would truncate the middle of the text.
  if [ "$status" -ne 0 ]; then
    echo "Warning: post-process failed (exit $status after ${post_process_timeout}s, mode=$mode); keeping original transcription" >> "$LOGFILE"
    [ "$mode" = "standard/cleanup" ] && \
      echo "Hint: run 'ollama pull $CLEANUP_MODEL_BASE' to use the cleanup model" >> "$LOGFILE"
    logging_end_and_write_to_logfile "Post-Process [$mode] FAILED (exit $status)" "$text" "$logging_start"
    echo "$text"
    return
  fi

  logging_end_and_write_to_logfile "Post-Process [$mode]" "$result" "$logging_start"

  # If ollama produced nothing, return original text
  if [ -z "$result" ]; then
    echo "$text"
  else
    echo "$result"
  fi
}

# Load the Ollama model the post-processing pass will most likely use, while
# the user speaks: Ollama unloads idle models after 5 minutes by default,
# and reloading costs ~1s (cleanup-1) when recording stops. Auto mode bets on
# standard text, which most dictations are. Only loads installed models —
# `ollama run` would otherwise start downloading a missing one.
preload_post_process_model() {
  local mode="$post_process_mode" model
  [ "$mode" = "auto" ] && mode="standard"
  formatting_applies "$mode" || return 0
  skip_standard_formatting "$mode" && return 0
  command -v ollama &> /dev/null || return 0

  if [ "$mode" = "standard" ] && [ -n "$cleanup_model" ]; then
    model="$cleanup_model"
    ensure_cleanup_model || return 0  # creates the tuned alias on first use
  else
    model="$post_process_model"
    ollama show "$model" &> /dev/null || return 0
  fi
  # An empty prompt loads the model without generating anything
  ollama run "$model" "" < /dev/null &> /dev/null
}

# The transcription server keeps the parakeet model loaded between
# dictations — see the server section of xhisper_transcribe.py
use_transcription_server() {
  [ "$transcription_engine" = "parakeet" ] && [ "$transcription_server" = "on" ]
}

transcribe() {
  local recording="$1"
  local logging_start=$(date +%s%N)

  # Build command arguments as an array so multi-word values (e.g. a
  # multi-word transcription-prompt) survive as single arguments
  local -a cmd_args=(--engine "$transcription_engine")

  if [ "$transcription_engine" = "parakeet" ]; then
    # device: auto/cpu stay on CPU (pinned providers); cuda opts into the
    # GPU via onnxruntime-gpu — see xhisper_transcribe.py
    cmd_args+=(--parakeet-model "$parakeet_model" --device "$model_device")
    use_transcription_server && cmd_args+=(--use-server)
  else
    cmd_args+=(--model "$model_name" --device "$model_device" --vad-min-silence-ms "$vad_min_silence_ms")
    [ -n "$model_language" ] && cmd_args+=(--language "$model_language")
    [ -n "$transcription_prompt" ] && cmd_args+=(--prompt "$transcription_prompt")
  fi

  # Run transcription — stderr goes to the log so failures are diagnosable
  # via 'xhisper --log' instead of silently producing empty output
  local transcription
  transcription=$(python3 "$TRANSCRIPT_SCRIPT" "$recording" "${cmd_args[@]}" 2>>"$LOGFILE")

  logging_end_and_write_to_logfile "Transcription [$transcription_engine]" "$transcription" "$logging_start"

  echo "$transcription"
}

# The hotkey's modifiers (e.g. the Alt+Shift of Alt+Shift+D) are usually
# still held when this script starts, and keys typed meanwhile arrive
# combined with them — Alt+Backspace deletes a whole word in many apps. The
# old fixed 0.2s pauses happened to put the first keystroke 0.33-0.40s after
# the keypress. Keep that margin, but measure it from the keypress, so work
# done before the first keystroke no longer adds to it.
KEY_RELEASE_MARGIN_US=350000
KEY_RELEASE_DEADLINE_US=$(( SCRIPT_START_US + KEY_RELEASE_MARGIN_US ))

wait_for_key_release() {
  local remaining=$(( KEY_RELEASE_DEADLINE_US - $(date +%s%6N) ))
  if (( remaining > 0 )); then
    sleep "$(printf '%d.%06d' $(( remaining / 1000000 )) $(( remaining % 1000000 )))"
  fi
}

# True if process $1 exits within ~0.3s (15 polls, 20ms apart)
exits_within_status_grace() {
  local i
  for ((i = 0; i < 15; i++)); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 0.02
  done
  return 1
}

# Stop pw-record and wait for it to exit. It finishes the WAV (the header's
# sizes) as it exits, ~5ms after SIGTERM, so this replaces a fixed 0.2s
# "buffer for flush". Gives up after 2s so a stuck recorder can't hang.
stop_recording() {
  local pids pid i
  pids=$(pgrep -f "$PROCESS_PATTERN")
  pkill -f "$PROCESS_PATTERN"
  for pid in $pids; do
    for ((i = 0; i < 200; i++)); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.01
    done
  done
}

# Input method (insert-method ibus) — see xhisper_ime.py
use_ime() {
  [ -n "$IME_SCRIPT" ]
}

# Run the input-method client. It needs only the standard library, so the
# python3 that runs transcription works; it starts the IBus service under a
# Python that has PyGObject. Its errors and the service's log go to the log.
ime_client() {
  python3 "$IME_SCRIPT" "$@" 2>>"$LOGFILE"
}

# Replace the status message with preedit text: shown at the cursor but
# never part of the document, so there is nothing to delete afterwards and
# no hotkey modifiers to wait for
ime_status() {
  audit "PREEDIT [$1]"
  ime_client status -- "$1"
}

# Insert the final text at the cursor. With the input method, commit it as
# one string; type it key by key where no IBus text field has focus (the
# desktop, apps without IBus support such as the VS Code snap) or if the
# input method is unavailable.
#
# If the statuses were typed, type the text too, even if a text field has
# focus by now. The backspaces erasing the typed status reach the app later
# than a commit would (uinput, X and the IBus key handling, vs IBus alone),
# so the commit would land inside the status and the remaining backspaces
# would delete the end of the dictation.
insert_text() {
  local text="$1"
  if use_ime && [ "$STATUS_MODE" = "ime" ]; then
    local -a fit=()
    [ "$smart_insert" = "off" ] && fit=(--no-fit)
    # A quick stop can finish while the hotkey's key grab still holds the
    # focus (see the start press): wait for it, as long as typing would
    ime_client finish --focus-wait-until "$KEY_RELEASE_DEADLINE_US" "${fit[@]}" -- "$text"
    case $? in
      0) audit "COMMIT [${text:0:70}]"; return ;;
      5) audit "COMMIT outcome unknown; not typing it again: [${text:0:70}]"; return ;;
      3) audit "no IBus text field has focus; typing instead" ;;
      *) audit "input method unavailable; typing instead" ;;
    esac
  fi
  wait_for_key_release
  type_text "$text"
}

# Main

# Find recording process, if so then stop it
if pgrep -f "$PROCESS_PATTERN" > /dev/null; then
  stop_recording

  # How the start press showed "(recording...)". No file: the start press is
  # still setting up the input method (a very quick second press), so there
  # is no typed status to delete.
  STATUS_MODE=$(cat "$STATUS_MODE_FILE" 2>/dev/null)
  rm -f "$STATUS_MODE_FILE"
  if [ -z "$STATUS_MODE" ]; then
    use_ime && STATUS_MODE="ime" || STATUS_MODE="type"
  fi

  # Check if recording is silent
  if is_silent "$RECORDING"; then
    if [ "$STATUS_MODE" = "ime" ]; then
      ime_status "(no sound detected)"
      sleep 0.6
      ime_client end
    else
      wait_for_key_release
      delete_n_chars 14 # "(recording...)"
      type_text "(no sound detected)"
      sleep 0.6
      delete_n_chars 19 # "(no sound detected)"
    fi
    rm -f "$RECORDING"
    exit 0
  fi

  # Transcribe in the background while the hotkey is released
  TRANSCRIPTION_FILE=$(mktemp)
  transcribe "$RECORDING" > "$TRANSCRIPTION_FILE" &
  transcribe_pid=$!
  if [ "$STATUS_MODE" = "ime" ]; then
    # A preedit status costs no keystrokes: show it right away
    ime_status "(transcribing...)"
    wait "$transcribe_pid"
  else
    # Show "(transcribing...)" only if it outlasts a grace period as long as
    # typing and deleting the status takes on screen (~0.3s): for a short
    # dictation it would only delay the text.
    wait_for_key_release
    delete_n_chars 14 # "(recording...)"
    if exits_within_status_grace "$transcribe_pid"; then
      wait "$transcribe_pid"
    else
      type_text "(transcribing...)"
      wait "$transcribe_pid"
      delete_n_chars 17 # "(transcribing...)"
    fi
  fi
  TRANSCRIPTION=$(< "$TRANSCRIPTION_FILE")
  rm -f "$TRANSCRIPTION_FILE"

  # ASR mishear fixups (see apply_corrections) — applied before mode
  # detection and cleanup, so everything downstream sees corrected text
  CORRECTED=$(apply_corrections "$TRANSCRIPTION")
  if [ -n "$TRANSCRIPTION" ] && [ "$CORRECTED" != "$TRANSCRIPTION" ]; then
    echo "Corrections applied: [$TRANSCRIPTION] -> [$CORRECTED]" >> "$LOGFILE"
    TRANSCRIPTION="$CORRECTED"
  fi

  # Engine failed (e.g. onnx-asr missing, model error) — errors are in the log
  if [ -z "$TRANSCRIPTION" ]; then
    if [ "$STATUS_MODE" = "ime" ]; then
      ime_status "(transcription failed - see xhisper --log)"
      sleep 1.5
      ime_client end
    else
      type_text "(transcription failed - see xhisper --log)"
      sleep 1.5
      delete_n_chars 42 # "(transcription failed - see xhisper --log)"
    fi
    rm -f "$RECORDING"
    exit 1
  fi

  # Post-process with AI if configured. Skip the whole "(formatting...)"
  # round trip when the pass would no-op anyway: standard text needs the
  # cleanup or instruct model (and the standard skip to allow it), while
  # command/email text needs the instruct model.
  effective_mode=""
  [ -n "$TRANSCRIPTION" ] && effective_mode=$(detect_mode "$TRANSCRIPTION" "$post_process_mode")

  FINAL_TEXT="$TRANSCRIPTION"
  if [ -n "$effective_mode" ] && formatting_applies "$effective_mode" && \
     ! skip_standard_formatting "$effective_mode"; then
    if [ "$STATUS_MODE" = "ime" ]; then
      ime_status "(formatting...)"
      FORMATTED=$(post_process "$TRANSCRIPTION")
    else
      type_text "(formatting...)"
      FORMATTED=$(post_process "$TRANSCRIPTION")
      delete_n_chars 15 # "(formatting...)"
    fi
    # Only use the formatted text if we got a result
    [ -n "$FORMATTED" ] && FINAL_TEXT="$FORMATTED"
  fi
  insert_text "$FINAL_TEXT"

  rm -f "$RECORDING"
else
  # No recording running, so start
  rm -f "$STATUS_MODE_FILE"

  # Load the transcription model while the user speaks, so it is ready (or
  # nearly) when recording stops; a no-op if the server is already up
  if use_transcription_server; then
    python3 "$TRANSCRIPT_SCRIPT" --start-server --parakeet-model "$parakeet_model" \
      --device "$model_device" --idle-minutes "$transcription_server_idle" \
      --log-file "$LOGFILE" 2>>"$LOGFILE" &
  fi
  preload_post_process_model &

  if use_ime; then
    # Record first: the input method's first use starts its service, and
    # speech must not wait for that
    pw-record --channels=1 --rate=16000 "$RECORDING" &
    record_pid=$!
    # Until the hotkey's keys are released, their key grab keeps the focus
    # off the text field, and IBus reports no field: wait for it as long as
    # typing the status would. A warm service answers well before that.
    if ime_client begin --log-file "$LOGFILE" --focus-wait-until "$KEY_RELEASE_DEADLINE_US" \
         -- "(recording...)"; then
      STATUS_MODE="ime"
      audit "PREEDIT [(recording...)]"
    else
      STATUS_MODE="type"  # no IBus text field has focus
    fi
    if ! kill -0 "$record_pid" 2>/dev/null; then
      # The second press already stopped the recording while the input method
      # was starting, and has finished the dictation: show no status (a typed
      # one would stay in the document), and clear the preedit one
      [ "$STATUS_MODE" = "ime" ] && ime_client end
      exit 0
    fi
    echo "$STATUS_MODE" > "$STATUS_MODE_FILE"
    if [ "$STATUS_MODE" = "type" ]; then
      wait_for_key_release
      type_text "(recording...)"
    fi
    wait "$record_pid"
  else
    echo "type" > "$STATUS_MODE_FILE"
    wait_for_key_release
    type_text "(recording...)"
    pw-record --channels=1 --rate=16000 "$RECORDING"
  fi
fi
