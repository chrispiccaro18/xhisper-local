#!/usr/bin/env python3
"""
xhisper transcription module
Transcribes audio files locally with faster-whisper or NVIDIA Parakeet (onnx-asr).
For parakeet it can also run as a server that keeps the model loaded between
recordings (--serve / --start-server / --use-server).
"""

import os
import sys
import json
import stat
import time
import wave
import fcntl
import socket
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging to suppress verbose output
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Compat: PyAV 15+ removed the metadata_errors argument from av.open(), but
# faster-whisper 1.2.x still passes it, which crashes every transcription
# with "open() got an unexpected keyword argument 'metadata_errors'".
# Shim av.open to drop the argument until faster-whisper catches up.
try:
    import av

    if int(av.__version__.split(".")[0]) >= 15:
        _av_open = av.open

        def _av_open_compat(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return _av_open(*args, **kwargs)

        av.open = _av_open_compat
except ImportError:
    pass

def transcribe_file(
    audio_path: str,
    model_size: str = "base",
    device: str = "auto",
    language: str = None,
    prompt: str = None,
    vad_min_silence_ms: int = 5000,
) -> str:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        model_size: Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
        device: Device to use (auto, cpu, cuda)
        language: Language code (e.g., 'en', 'es') or None for auto-detect
        prompt: Optional context text for better accuracy
        vad_min_silence_ms: Silence (ms) before the VAD splits segments; raise
            this if speech after a thinking pause gets dropped

    Returns:
        Transcribed text
    """
    from faster_whisper import WhisperModel

    # Initialize model
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    # Transcribe
    segments, info = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompt,
        beam_size=5,
        vad_filter=True,  # Voice activity detection to remove silence
        vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
        condition_on_previous_text=False,  # avoids repetition drift after pauses
    )

    # Combine all segments
    text = " ".join(segment.text for segment in segments)

    # Clean up extra whitespace
    text = " ".join(text.split())

    return text


# onnx-asr guidance: most models handle only 20-30s of audio per shot, so
# longer recordings must be segmented with a VAD before recognition.
_PARAKEET_VAD_THRESHOLD_S = 25.0

# Download parakeet models as real files into our own cache dir. The HF hub
# cache stores them as content-addressed blobs behind symlinks, which breaks
# ONNX external-data validation ("External data path escapes model directory").
_PARAKEET_MODEL_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "xhisper" / "parakeet"
)


def _wav_duration_s(audio_path: str):
    """Duration of a wav file in seconds, or None if unreadable."""
    try:
        with wave.open(audio_path) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


class TranscriptionError(Exception):
    """A failure whose message is already user-facing; printed as-is."""


class ParakeetEngine:
    """
    A loaded NVIDIA Parakeet model (onnx-asr), reusable across recordings.

    Output is already punctuated and capitalized, which is most of what the
    standard LLM post-processing pass would do.

    device: 'cuda' runs on the GPU via onnxruntime-gpu (which replaces the
    plain onnxruntime package); 'auto'/'cpu' pin the CPU execution provider
    explicitly, so the default stays on CPU even when the GPU wheel is
    installed (ORT would otherwise pick CUDA on its own).
    """

    def __init__(self, model_name: str, device: str = "auto"):
        try:
            import onnx_asr
        except ImportError as exc:
            # onnx_asr imports onnxruntime at module load, so a missing
            # onnxruntime (a separate package / an extra since onnx-asr 0.12)
            # lands here too. Report the real cause, not just "install onnx-asr".
            raise TranscriptionError(
                f"Error: engine 'parakeet' requires onnx-asr with onnxruntime "
                f"({exc}). Install it with: "
                "pip3 install --break-system-packages 'onnx-asr[cpu]' "
                "(or your distro's onnxruntime package)"
            ) from exc

        self.model_name = model_name
        self.device = device
        self._onnx_asr = onnx_asr
        self._vad_model = None  # built on the first long clip

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            import onnxruntime

            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise TranscriptionError(
                    "Error: --device cuda for the parakeet engine requires the "
                    "onnxruntime-gpu package. Install it with: "
                    "pip3 install --break-system-packages onnxruntime-gpu "
                    "(it replaces onnxruntime)"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._model = onnx_asr.load_model(model_name, _PARAKEET_MODEL_DIR, providers=providers)

        if device == "cuda":
            # ORT does NOT error when the CUDA provider fails to initialize (e.g.
            # missing curand/cufft libs) — it silently runs on the CPU instead.
            # Check the encoder session's actual providers and fail loudly.
            try:
                actual = self._model.asr._encoder._sess.get_providers()  # onnx-asr private
            except AttributeError:
                actual = None
            if actual is not None and "CUDAExecutionProvider" not in actual:
                raise TranscriptionError(
                    "Error: --device cuda requested but the model is running on "
                    "the CPU — the CUDA execution provider failed to load. Either "
                    "CUDA runtime libs are missing, or the onnxruntime-gpu wheel "
                    "targets a different CUDA major version (wheels >= 1.26 need "
                    "CUDA 13; the nvidia-*-cu12 packages provide CUDA 12). Known-"
                    "good combo with the cu12 stack: pip3 install "
                    "--break-system-packages 'onnxruntime-gpu==1.24.4' "
                    "nvidia-curand-cu12 nvidia-cufft-cu12"
                )

    def warm_up(self):
        """
        Run one throwaway recognition so the first real one is fast.

        ONNX Runtime's first run is ~4x slower than later ones (3.5s vs 0.8s
        for an 8s clip on a laptop CPU), and a short warm-up input only
        partly helps: a following longer clip is still slow. One run at the
        longest length that skips the VAD makes every later length fast.
        """
        import numpy as np

        seconds = _PARAKEET_VAD_THRESHOLD_S - 1
        noise = np.random.default_rng(0).standard_normal(int(16000 * seconds)) * 0.05
        self._model.recognize(noise.astype(np.float32), sample_rate=16000)

    def transcribe(self, audio_path: str) -> str:
        duration = _wav_duration_s(audio_path)
        if duration is not None and duration > _PARAKEET_VAD_THRESHOLD_S:
            # Long clip: segment with silero VAD so each chunk stays inside the
            # model's window. 5s min-silence keeps thinking pauses intact — a
            # pause becomes a chunk boundary, never dropped audio.
            if self._vad_model is None:
                vad = self._onnx_asr.load_vad("silero")
                self._vad_model = self._model.with_vad(vad, min_silence_duration_ms=5000.0)
            results = self._vad_model.recognize(audio_path)
            if isinstance(results, str):  # defensive: adapter may return a plain str
                return results.strip()
            return " ".join(
                r.text if hasattr(r, "text") else str(r) for r in results
            ).strip()

        return self._model.recognize(audio_path).strip()


def transcribe_parakeet(audio_path: str, model_name: str, device: str = "auto") -> str:
    """Transcribe one audio file with a freshly loaded Parakeet model."""
    return ParakeetEngine(model_name, device).transcribe(audio_path)


# Transcription server (parakeet engine).
#
# A one-shot run spends ~1.7s loading the model and ~3.5s on ONNX Runtime's
# slow first run, for ~0.8s of actual recognition (8s clip, laptop CPU).
# --serve keeps a loaded, warmed-up model in memory instead. xhisper.sh
# starts it with --start-server when recording begins, so the load overlaps
# with speaking, then transcribes with --use-server, which falls back to a
# one-shot run whenever the server is unavailable. The server exits after
# --idle-minutes without a request, freeing the model's ~2.5GB.
#
# Protocol: one JSON line per connection, one JSON line back.
#   {"audio": path, "model": name, "device": dev} -> {"text": ...} | {"error": ...}
#   {"cmd": "stop"}                               -> {"ok": true}

_SERVER_REQUEST_LIMIT = 64 * 1024


def _server_dir():
    """Private per-user directory for the socket and lock file, or None."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    path = Path(runtime) / "xhisper" if runtime else Path(f"/tmp/xhisper-{os.getuid()}")
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        st = path.lstat()
    except OSError:
        return None
    # The /tmp fallback is shared: refuse a directory (or symlink) someone
    # else created, or one others can enter, so only we can reach the socket.
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        return None
    return path


def _try_lock(lock_path: Path):
    """Take an exclusive lock without blocking: the open file, or None if held."""
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def _recv_line(conn: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > _SERVER_REQUEST_LIMIT:
            raise ValueError("message too large")
    return buf.split(b"\n", 1)[0].decode()


def _send_line(conn: socket.socket, obj: dict):
    try:
        conn.sendall((json.dumps(obj) + "\n").encode())
    except OSError:
        pass  # client gave up; nothing to report to


def _server_log(message: str):
    """Timestamped like xhisper.sh's audit lines; stderr is the xhisper log."""
    now = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
    print(f"[{stamp}] server: {message}", file=sys.stderr, flush=True)


def _handle_request(conn: socket.socket, engine):
    """Serve one connection. Returns (engine, keep_running)."""
    conn.settimeout(10)
    try:
        request = json.loads(_recv_line(conn))
        if not isinstance(request, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _server_log(f"bad request: {exc}")
        return engine, True

    if request.get("cmd") == "stop":
        _send_line(conn, {"ok": True})
        _server_log("stop requested, exiting")
        return engine, False

    try:
        model_name, device = request["model"], request["device"]
        if engine is None or (model_name, device) != (engine.model_name, engine.device):
            _server_log(f"loading {model_name} on {device} (config changed)")
            engine = None  # free the old model first: two loaded is ~5GB
            engine = ParakeetEngine(model_name, device)
        start = time.perf_counter()
        text = engine.transcribe(request["audio"])
        duration = _wav_duration_s(request["audio"])
        audio = f"{duration:.1f}s of audio" if duration is not None else "audio"
        _server_log(f"transcribed {audio} in {time.perf_counter() - start:.2f}s")
        _send_line(conn, {"text": text})
    except Exception as exc:  # the client falls back to a one-shot run
        message = f"request missing {exc}" if isinstance(exc, KeyError) else str(exc)
        _server_log(f"error: {message}")
        _send_line(conn, {"error": message})
    return engine, True


def serve(model_name: str, device: str, idle_minutes: float):
    """Run the transcription server in the foreground until idle or stopped."""
    server_dir = _server_dir()
    if server_dir is None:
        raise TranscriptionError(
            "Error: no private runtime directory for the transcription server"
        )
    lock = _try_lock(server_dir / "server.lock")
    if lock is None:
        return  # another server is already running

    sock_path = server_dir / "transcribe.sock"
    if sock_path.exists():  # left by a server that died; we hold the lock now
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            server.bind(str(sock_path))
        except OSError as exc:  # e.g. a runtime dir path over the ~108-byte limit
            raise TranscriptionError(
                f"Error: cannot create the transcription server socket {sock_path}: {exc}"
            ) from exc
        # Listen before loading: a request that arrives during the ~5s load
        # and warm-up waits in the backlog instead of falling back to a
        # second, concurrent model load.
        server.listen(4)

        start = time.perf_counter()
        engine = ParakeetEngine(model_name, device)
        loaded = time.perf_counter()
        engine.warm_up()
        _server_log(
            f"ready: {model_name} on {device} "
            f"(load {loaded - start:.2f}s, warm-up {time.perf_counter() - loaded:.2f}s)"
        )

        server.settimeout(idle_minutes * 60 if idle_minutes > 0 else None)
        keep_running = True
        while keep_running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                _server_log(f"idle for {idle_minutes:g} min, exiting")
                break
            with conn:
                engine, keep_running = _handle_request(conn, engine)
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()
        lock.close()


def start_server(model_name: str, device: str, idle_minutes: float, log_file: str = None):
    """Launch a detached server unless one is running; returns immediately."""
    server_dir = _server_dir()
    if server_dir is None:
        return
    lock = _try_lock(server_dir / "server.lock")
    if lock is None:
        return  # running or starting
    lock.close()  # the server takes the lock itself; a racing twin just exits

    log = open(log_file, "a") if log_file else subprocess.DEVNULL
    subprocess.Popen(
        [
            sys.executable, os.path.abspath(__file__), "--serve",
            "--parakeet-model", model_name,
            "--device", device,
            "--idle-minutes", str(idle_minutes),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=log,
        start_new_session=True,  # outlive the xhisper invocation that started it
    )


def _server_call(request: dict, timeout_s: float) -> dict:
    server_dir = _server_dir()
    if server_dir is None:
        raise OSError("no private runtime directory")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(str(server_dir / "transcribe.sock"))
        conn.sendall((json.dumps(request) + "\n").encode())
        return json.loads(_recv_line(conn))


def transcribe_via_server(audio_path: str, model_name: str, device: str):
    """Transcribe through a running server: the text, or None to fall back."""
    request = {
        "audio": str(Path(audio_path).resolve()),
        "model": model_name,
        "device": device,
    }
    try:
        # Generous timeout: a request can wait out the server's model load
        # and warm-up, then a long VAD-segmented recording.
        reply = _server_call(request, timeout_s=120)
    except (OSError, ValueError) as exc:
        print(f"Transcription server unavailable ({exc}); transcribing in-process", file=sys.stderr)
        return None
    if "error" in reply:
        print(f"Transcription server failed ({reply['error']}); transcribing in-process", file=sys.stderr)
        return None
    return reply.get("text", "")


def stop_server() -> bool:
    try:
        return bool(_server_call({"cmd": "stop"}, timeout_s=5).get("ok"))
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper or Parakeet"
    )
    parser.add_argument(
        "audio_file", nargs="?",
        help="Path to audio file to transcribe (not needed with the server options)",
    )
    parser.add_argument(
        "--engine",
        default="whisper",
        choices=["whisper", "parakeet"],
        help="Transcription engine (default: whisper)",
    )
    parser.add_argument(
        "--parakeet-model",
        default="nemo-parakeet-tdt-0.6b-v2",
        help="onnx-asr model name for the parakeet engine "
        "(nemo-parakeet-tdt-0.6b-v2 = English, nemo-parakeet-tdt-0.6b-v3 = multilingual)",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: base)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto). Whisper: auto detects CUDA. "
        "Parakeet: auto/cpu run on CPU; cuda needs onnxruntime-gpu",
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., en, es) for faster/more accurate transcription",
    )
    parser.add_argument(
        "--prompt",
        help="Context words for better accuracy (whisper engine only)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=5000,
        help="Silence (ms) before the whisper VAD splits segments (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    server = parser.add_argument_group(
        "transcription server (parakeet engine only)",
        "Keep a warmed-up model loaded between recordings instead of "
        "loading it for every one.",
    )
    server.add_argument(
        "--use-server",
        action="store_true",
        help="Transcribe through a running server; fall back to a one-shot "
        "run if none answers",
    )
    actions = server.add_mutually_exclusive_group()
    actions.add_argument(
        "--serve", action="store_true",
        help="Run the server in the foreground",
    )
    actions.add_argument(
        "--start-server", action="store_true",
        help="Start a server in the background unless one is running",
    )
    actions.add_argument(
        "--stop-server", action="store_true",
        help="Stop the running server",
    )
    server.add_argument(
        "--idle-minutes",
        type=float,
        default=5,
        help="Server exits after this long without a request; 0 = never (default: 5)",
    )
    server.add_argument(
        "--log-file",
        help="Append the started server's log to this file (--start-server)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    try:
        if args.serve:
            serve(args.parakeet_model, args.device, args.idle_minutes)
            return
        if args.start_server:
            start_server(args.parakeet_model, args.device, args.idle_minutes, args.log_file)
            return
    except TranscriptionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    if args.stop_server:
        if not stop_server():
            print("No transcription server running", file=sys.stderr)
            sys.exit(1)
        return

    if args.audio_file is None:
        parser.error("audio_file is required")

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.engine == "parakeet":
            result = None
            if args.use_server:
                result = transcribe_via_server(args.audio_file, args.parakeet_model, args.device)
            if result is None:
                result = transcribe_parakeet(args.audio_file, args.parakeet_model, args.device)
        else:
            result = transcribe_file(
                args.audio_file,
                model_size=args.model,
                device=args.device,
                language=args.language,
                prompt=args.prompt,
                vad_min_silence_ms=args.vad_min_silence_ms,
            )
        print(result)
    except TranscriptionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
