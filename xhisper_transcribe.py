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
import struct
import wave
import fcntl
import socket
import threading
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
        self._vad = None  # silero, loaded on first use
        self._vad_model = None  # built on the first long clip
        self._ts_model = None  # built on the first streamed recording
        # The server transcribes on two threads: streaming passes during a
        # recording, and requests. One model run at a time.
        self.lock = threading.Lock()

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

    def _silero(self):
        if self._vad is None:
            self._vad = self._onnx_asr.load_vad("silero")
        return self._vad

    def _segmented(self, audio, **kwargs) -> str:
        """
        Long clip: segment with silero VAD so each chunk stays inside the
        model's window. 5s min-silence keeps thinking pauses intact — a pause
        becomes a chunk boundary, never dropped audio.
        """
        if self._vad_model is None:
            self._vad_model = self._model.with_vad(self._silero(), min_silence_duration_ms=5000.0)
        results = self._vad_model.recognize(audio, **kwargs)
        if isinstance(results, str):  # defensive: adapter may return a plain str
            return results.strip()
        return " ".join(
            r.text if hasattr(r, "text") else str(r) for r in results
        ).strip()

    def transcribe(self, audio_path: str) -> str:
        duration = _wav_duration_s(audio_path)
        with self.lock:
            if duration is not None and duration > _PARAKEET_VAD_THRESHOLD_S:
                return self._segmented(audio_path)
            return self._model.recognize(audio_path).strip()

    def transcribe_samples(self, samples) -> str:
        """Like transcribe(), for 16kHz mono float32 samples."""
        with self.lock:
            if len(samples) / 16000 > _PARAKEET_VAD_THRESHOLD_S:
                return self._segmented(samples, sample_rate=16000)
            return self._model.recognize(samples, sample_rate=16000).strip()

    def transcribe_tokens(self, samples):
        """Tokens and their start times (s) for 16kHz mono float32 samples."""
        with self.lock:
            if self._ts_model is None:
                self._ts_model = self._model.with_timestamps()
            result = self._ts_model.recognize(samples, sample_rate=16000)
        return list(result.tokens), list(result.timestamps)

    def vad_session(self):
        """
        Silero's ONNX session, for running the VAD a frame at a time as audio
        arrives. onnx-asr only runs it over whole clips, so this reaches into
        a private attribute; streaming stays off if it is missing.
        """
        with self.lock:  # the request thread may be loading it for a long clip
            vad = self._silero()
        session = getattr(vad, "_model", None)  # onnx-asr private
        if not hasattr(session, "run"):
            raise TranscriptionError("this onnx-asr version's VAD cannot be run incrementally")
        return session


def transcribe_parakeet(audio_path: str, model_name: str, device: str = "auto") -> str:
    """Transcribe one audio file with a freshly loaded Parakeet model."""
    return ParakeetEngine(model_name, device).transcribe(audio_path)


# Streaming (parakeet engine, through the server).
#
# While recording, the server follows the WAV that pw-record is writing and
# transcribes at pauses, so little audio is left to transcribe at stop. Text
# is still typed only at stop; streaming just moves the work earlier.
#
# Pieces transcribed separately get punctuated at the cut and can lose words
# there, so it only cuts where the model itself ended a sentence. At each
# pause of 1s or more it re-transcribes the audio since the last cut, then
# cuts after the latest sentence end that the previous pass produced
# identically and that has words after it (LocalAgreement-2, as in
# Whisper-Streaming). It cuts nothing in the first 20s, so a shorter
# dictation is transcribed whole, never in pieces. Without an agreed
# sentence end it cuts at a pause once 30s are uncut.
#
# At stop, if nothing was said since the latest pass, that pass's text for
# the audio since the last cut is reused instead of transcribing it again:
# usually so when the user pauses a second before stopping. For a dictation
# with no cuts that is the whole dictation, so it is done almost instantly.
# The reused pass only lacks the silence after the pause that triggered it;
# Parakeet's punctuation can shift with the length of trailing silence, so
# this also stops the text depending on how long the user waited.
#
# Tested on real dictations of 37-67s: 0-0.6s left to transcribe at stop
# instead of 2-3s, with fewer punctuation errors than the >25s VAD path, which
# cuts every 20s even mid-word. Cutting at 600ms pauses split sentences. On
# 6-14s dictations with a 1.5s+ pause before stop: 0.05-0.3s instead of
# 0.6-1.3s.

_STREAM_RATE = 16000
_STREAM_PAUSE_S = 1.0  # a pause this long triggers a pass
_STREAM_HOLD_S = 20.0  # no cuts before this much recording
_STREAM_MIN_PASS_S = 2.0  # don't bother with less uncut audio than this
_STREAM_MAX_UNCUT_S = 30.0  # then cut at a pause, sentence end or not
_STREAM_MAX_PASS_S = 40.0  # a longer pause-free stretch is VAD-segmented instead
_STREAM_STALL_S = 30.0  # recording unchanged this long without a stop: abandoned
_STREAM_POLL_S = 0.02  # pw-record writes about every 21ms
_VAD_HOP, _VAD_CONTEXT = 512, 64  # silero at 16kHz: 512 new samples + 64 of context
_VAD_SPEECH, _VAD_SILENCE = 0.5, 0.35  # onnx-asr's silero thresholds
_PAUSE_HOPS = int(_STREAM_PAUSE_S * _STREAM_RATE / _VAD_HOP)
_SENTENCE_ENDS = (".", "?", "!")


def _wav_data_chunk(f):
    """
    (offset, size) of the data chunk of a 16kHz mono 16-bit PCM WAV, or None
    while the header is still incomplete. pw-record leaves size 0 until it
    exits.
    """
    f.seek(0)
    head = f.read(1024)
    if len(head) < 12:
        return None
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise TranscriptionError("not a WAV file")
    pos, fmt = 12, None
    while pos + 8 <= len(head):
        chunk_id = head[pos:pos + 4]
        size = int.from_bytes(head[pos + 4:pos + 8], "little")
        if chunk_id == b"data":
            if fmt != (1, 1, _STREAM_RATE, 16):  # PCM, mono, 16kHz, 16-bit
                raise TranscriptionError(f"cannot stream WAV format {fmt}")
            return pos + 8, size
        if chunk_id == b"fmt ":
            if pos + 24 > len(head):
                return None
            tag, channels, rate = struct.unpack_from("<HHI", head, pos + 8)
            fmt = (tag, channels, rate, struct.unpack_from("<H", head, pos + 22)[0])
        pos += 8 + size + (size & 1)
    return None


def _join_tokens(tokens) -> str:
    return "".join(tokens).strip()


class StreamSession:
    """Transcribes one recording at its pauses while pw-record writes it."""

    def __init__(self, engine, path: str):
        import numpy as np

        self.engine, self.path = engine, path
        self.note = ""  # how it went, for the log
        self._vad = None  # loaded on the session's thread: the request returns at once
        self._stopping = threading.Event()
        self._done = threading.Event()
        self._cancelled = False
        self._text = None

        self._raw = bytearray()  # PCM read so far
        self._data_offset = 0
        self._vad_pos = 0  # samples run through the VAD
        self._vad_state = np.zeros((2, 1, 128), np.float32)
        self._vad_context = np.zeros(_VAD_CONTEXT, np.float32)
        self._silent_hops, self._spoke, self._last_speech = 0, False, 0

        self._parts = []  # text of the audio before _cut
        self._cut = 0  # sample where the uncut audio starts
        self._prev = None  # tokens of the previous pass from _cut
        self._last_pass = None  # (end sample, text) of the latest pass from _cut
        self._passes = 0

        self._thread = threading.Thread(target=self._run, name="xhisper-stream", daemon=True)
        self._thread.start()

    def finish(self, timeout_s: float):
        """
        The recording's text, once pw-record has exited, or None to
        transcribe the file as usual.
        """
        self._stopping.set()
        if not self._done.wait(timeout_s):
            self._cancelled = True
            self.note = "streaming timed out"
            return None
        return self._text

    def cancel(self, timeout_s: float = 5):
        self._cancelled = True
        self._stopping.set()
        self._thread.join(timeout_s)

    def done(self) -> bool:
        return self._done.is_set()

    def _run(self):
        try:
            self._vad = self.engine.vad_session()
            self._text = self._follow()
        except Exception as exc:  # a failed stream must never cost the dictation
            self.note = f"streaming failed ({exc})"
            self._text = None
        finally:
            self._done.set()
        if not self._stopping.is_set() and self.note:  # ended on its own, unasked
            _server_log(f"streaming ended: {self.note}")

    def _follow(self):
        f = self._open()
        if f is None:
            return None
        with f:
            unchanged_since = time.monotonic()
            while not self._cancelled:
                # The stop request comes after pw-record exits, so once it
                # has, one more read gets the rest of the audio.
                stopping = self._stopping.is_set()
                data = f.read()
                if data:
                    self._raw += data
                    unchanged_since = time.monotonic()
                if stopping:
                    return self._final(f)
                if data:
                    self._feed(transcribe=True)
                elif time.monotonic() - unchanged_since > _STREAM_STALL_S or self._replaced(f):
                    self.note = "recording abandoned"
                    return None
                else:
                    time.sleep(_STREAM_POLL_S)
        return None

    def _open(self):
        """The recording, positioned at its first sample, once pw-record has written its header."""
        deadline = time.monotonic() + _STREAM_STALL_S
        while not self._stopping.is_set():
            try:
                f = open(self.path, "rb")
            except FileNotFoundError:
                f = None
            if f is not None:
                try:
                    chunk = _wav_data_chunk(f)
                except Exception:
                    f.close()
                    raise
                if chunk is not None:
                    self._data_offset, size = chunk
                    if size:  # finished before we got here: nothing to gain
                        f.close()
                        self.note = "recording already finished"
                        return None
                    f.seek(self._data_offset)
                    return f
                f.close()
            if time.monotonic() > deadline:
                self.note = "recording never started"
                return None
            time.sleep(_STREAM_POLL_S)
        self.note = "stopped before the recording started"
        return None

    def _replaced(self, f) -> bool:
        """Was the recording deleted (silent, discarded), replaced or truncated?"""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return True
        return st.st_ino != os.fstat(f.fileno()).st_ino or st.st_size < f.tell()

    def _samples(self, start: int, end: int):
        import numpy as np

        pcm = np.frombuffer(self._raw[2 * start:2 * end], "<i2")
        return pcm.astype(np.float32) / 32768  # as onnx-asr reads WAV files

    def _feed(self, transcribe: bool):
        """Run the VAD over the new audio; transcribe at each pause it finds."""
        import numpy as np

        available = len(self._raw) // 2
        while self._vad_pos + _VAD_HOP <= available and not self._cancelled:
            hop = self._samples(self._vad_pos, self._vad_pos + _VAD_HOP)
            out, self._vad_state = self._vad.run(["output", "stateN"], {
                "input": np.concatenate([self._vad_context, hop])[None],
                "state": self._vad_state,
                "sr": np.array([_STREAM_RATE]),
            })
            self._vad_context = hop[-_VAD_CONTEXT:]
            self._vad_pos += _VAD_HOP
            p = float(out[0, 0])
            if p >= _VAD_SPEECH:
                self._spoke, self._silent_hops = True, 0
                self._last_speech = self._vad_pos
            elif p < _VAD_SILENCE:
                self._silent_hops += 1
                if self._spoke and self._silent_hops >= _PAUSE_HOPS:
                    self._spoke = False
                    if transcribe and not self._stopping.is_set():
                        mid = self._vad_pos - _PAUSE_HOPS // 2 * _VAD_HOP
                        self._on_pause(self._vad_pos, mid)

    def _on_pause(self, end: int, mid: int):
        """A pause was detected at sample `end`; `mid` is mid-pause."""
        uncut = end - self._cut
        if uncut < _STREAM_MIN_PASS_S * _STREAM_RATE:
            return
        if uncut > _STREAM_MAX_PASS_S * _STREAM_RATE:
            self._cut_at(mid, self.engine.transcribe_samples(self._samples(self._cut, mid)))
            return
        tokens, starts = self.engine.transcribe_tokens(self._samples(self._cut, end))
        self._passes += 1
        self._last_pass = (end, _join_tokens(tokens))

        agreed = None
        if end >= _STREAM_HOLD_S * _STREAM_RATE and self._prev is not None:
            for j in range(len(tokens) - 1):
                if (tokens[j] in _SENTENCE_ENDS
                        and tokens[:j + 1] == self._prev[:j + 1]
                        and any(t.startswith(" ") for t in tokens[j + 1:])):
                    agreed = j
        self._prev = tokens

        if agreed is not None:
            # cut between the sentence end and the next word
            following = next(j for j in range(agreed + 1, len(tokens)) if tokens[j].startswith(" "))
            offset = int((starts[agreed] + starts[following]) / 2 * _STREAM_RATE)
            self._cut_at(self._cut + offset, _join_tokens(tokens[:agreed + 1]))
            # This pass already has the words after the cut: if nothing is
            # said before the stop, reuse them rather than redo the tail
            self._last_pass = (end, _join_tokens(tokens[following:]))
        elif uncut > _STREAM_MAX_UNCUT_S * _STREAM_RATE:
            before = (mid - self._cut) / _STREAM_RATE
            self._cut_at(mid, _join_tokens([t for t, s in zip(tokens, starts) if s < before]))
            self._last_pass = (end, _join_tokens([t for t, s in zip(tokens, starts) if s >= before]))

    def _cut_at(self, sample: int, text: str):
        self._parts.append(text)
        self._cut, self._prev, self._last_pass = sample, None, None

    def _final(self, f):
        if self._cancelled:
            return None
        # pw-record wrote the real data size as it exited: anything past it
        # is not audio
        f.seek(self._data_offset - 4)
        size = int.from_bytes(f.read(4), "little")
        if 0 < size <= len(self._raw):
            del self._raw[size:]
        self._feed(transcribe=False)  # VAD only, for the checks below

        end = len(self._raw) // 2
        # How long the user paused before stopping: a pass covers the whole
        # tail only if a pause of _STREAM_PAUSE_S came after the last speech
        pause = f"{(end - self._last_speech) / _STREAM_RATE:.1f}s pause before stop"
        # nothing said since the latest pass, which ran at a pause after it
        covered = self._last_pass is not None and self._last_speech <= self._last_pass[0]
        if not self._parts:
            # Nothing cut (always so in the first 20s). If the latest pass
            # already covers every word, reuse it instead of transcribing the
            # whole file again: it misses only the non-speech after its pause.
            if covered:
                self.note = (f"streamed: {self._passes} passes, nothing cut, "
                             f"all {end / _STREAM_RATE:.1f}s already done, {pause}")
                return self._last_pass[1]
            self.note = f"{self._passes} passes, nothing cut, {pause}"
            return None
        uncut = (end - self._cut) / _STREAM_RATE
        if covered:
            tail, how = self._last_pass[1], "already done"
        else:
            tail = self.engine.transcribe_samples(self._samples(self._cut, end))
            how = "done at stop"
        self.note = (f"streamed: {self._passes} passes, {len(self._parts)} cuts, "
                     f"last {uncut:.1f}s {how}, {pause}")
        return " ".join(p for p in self._parts + [tail] if p)


# Transcription server (parakeet engine).
#
# A one-shot run spends ~1.7s loading the model and ~3.5s on ONNX Runtime's
# slow first run, for ~0.8s of actual recognition (8s clip, laptop CPU).
# --serve keeps a loaded, warmed-up model in memory instead. xhisper.sh
# starts it with --start-server when recording begins, so the load overlaps
# with speaking, then transcribes with --use-server, which falls back to a
# one-shot run whenever the server is unavailable. The server exits after
# --idle-minutes without a request, freeing the model's ~2.5GB. With
# --watch, the start also asks the server to stream the new recording (see
# Streaming above); the stop's request then collects the result.
#
# Protocol: one JSON line per connection, one JSON line back.
#   {"audio": path, "model": name, "device": dev} -> {"text": ...} | {"error": ...}
#   {"cmd": "watch", "audio": path, "model": name, "device": dev}
#                                                 -> {"ok": true} | {"error": ...}
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


class _ServerState:
    """The loaded model and the recording being streamed, if any."""

    def __init__(self, engine):
        self.engine = engine
        self.session = None

    def engine_for(self, model_name: str, device: str):
        if self.engine is None or (model_name, device) != (self.engine.model_name, self.engine.device):
            self.end_session()
            _server_log(f"loading {model_name} on {device} (config changed)")
            self.engine = None  # free the old model first: two loaded is ~5GB
            self.engine = ParakeetEngine(model_name, device)
        return self.engine

    def end_session(self):
        if self.session is not None:
            self.session.cancel()
            self.session = None


def _handle_request(conn: socket.socket, state: _ServerState) -> bool:
    """Serve one connection. Returns whether to keep running."""
    conn.settimeout(10)
    try:
        request = json.loads(_recv_line(conn))
        if not isinstance(request, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _server_log(f"bad request: {exc}")
        return True

    cmd = request.get("cmd")
    if cmd == "stop":
        _send_line(conn, {"ok": True})
        _server_log("stop requested, exiting")
        return False

    try:
        if cmd not in (None, "watch"):
            raise ValueError(f"unknown command {cmd!r}")
        engine = state.engine_for(request["model"], request["device"])
        audio_path = request["audio"]
        if cmd == "watch":
            state.end_session()
            state.session = StreamSession(engine, audio_path)
            _send_line(conn, {"ok": True})
            return True

        start = time.perf_counter()
        text, note = None, ""
        session, state.session = state.session, None
        if session is not None and session.path == audio_path:
            text = session.finish(timeout_s=60)
            note = f" ({session.note})" if session.note else ""
        elif session is not None:
            session.cancel()
        if text is None:
            text = engine.transcribe(audio_path)
        duration = _wav_duration_s(audio_path)
        audio = f"{duration:.1f}s of audio" if duration is not None else "audio"
        _server_log(f"transcribed {audio} in {time.perf_counter() - start:.2f}s{note}")
        _send_line(conn, {"text": text})
    except Exception as exc:  # the client falls back to a one-shot run
        message = f"request missing {exc}" if isinstance(exc, KeyError) else str(exc)
        _server_log(f"error: {message}")
        _send_line(conn, {"error": message})
    return True


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

        state = _ServerState(engine)
        server.settimeout(idle_minutes * 60 if idle_minutes > 0 else None)
        keep_running = True
        while keep_running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                if state.session is not None and not state.session.done():
                    continue  # still recording
                _server_log(f"idle for {idle_minutes:g} min, exiting")
                break
            with conn:
                keep_running = _handle_request(conn, state)
        state.end_session()
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


def watch_recording(audio_path: str, model_name: str, device: str) -> bool:
    """
    Have the server stream a recording that is starting (see Streaming).
    Run in the background at record start; on failure the recording is just
    transcribed whole at stop.
    """
    request = {
        "cmd": "watch",
        "audio": str(Path(audio_path).resolve()),
        "model": model_name,
        "device": device,
    }
    deadline = time.monotonic() + 10  # a just-started server binds in well under 1s
    while True:
        try:
            # The reply waits out a model load and warm-up (~5s)
            reply = _server_call(request, timeout_s=60)
            break
        except (FileNotFoundError, ConnectionRefusedError) as exc:  # not listening yet
            if time.monotonic() > deadline:
                print(f"Streaming unavailable ({exc})", file=sys.stderr)
                return False
            time.sleep(0.05)
        except (OSError, ValueError) as exc:
            print(f"Streaming unavailable ({exc})", file=sys.stderr)
            return False
    if "error" in reply:
        print(f"Streaming unavailable ({reply['error']})", file=sys.stderr)
        return False
    return True


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
        "--watch", metavar="AUDIO",
        help="Have the server transcribe AUDIO at its pauses while it is "
        "being recorded, so less is left at stop (with --start-server)",
    )
    server.add_argument(
        "--log-file",
        help="Append the started server's log to this file (--start-server)",
    )

    args = parser.parse_args()
    if args.watch and (args.serve or args.stop_server):
        parser.error("--watch cannot be combined with --serve or --stop-server")

    if args.debug:
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    try:
        if args.serve:
            serve(args.parakeet_model, args.device, args.idle_minutes)
            return
        if args.start_server or args.watch:
            if args.start_server:
                start_server(args.parakeet_model, args.device, args.idle_minutes, args.log_file)
            if args.watch:
                watch_recording(args.watch, args.parakeet_model, args.device)
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
