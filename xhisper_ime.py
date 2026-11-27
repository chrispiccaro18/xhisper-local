#!/usr/bin/env python3
"""xhisper's IBus input method: insert a dictation as one commit.

Typing a dictation key by key through the uinput daemon costs ~10ms per
character, loses keys when IBus (via XIM) receives fast bursts, and needs
the clipboard for non-ASCII characters; pasting needs the clipboard and a
different shortcut in terminals. An input method avoids all of that: IBus
hands committed text to the focused text field as one string, the way
Chinese or Japanese typists enter text, and "preedit" text shows status
messages at the cursor without ever entering the document.

The service registers an engine with ibus-daemon at runtime (no system-wide
component file, no root) and makes it the active engine only for the length
of a dictation, then restores the previous engine. The engine passes every
key through untouched, so typing during a dictation behaves as usual. If the
service dies, IBus is left without an engine and keys still pass through.

Where no application text field has focus (the desktop, an app without
input-method support such as the VS Code snap, a web page with no text field
selected), IBus attaches the engine to an internal placeholder context and
drops anything committed there. The service detects that case and xhisper
types the text instead; see is_real_focus().

Client commands (used by xhisper.sh; the client needs only the standard
library, the service needs PyGObject with the IBus typelib):

  begin STATUS     start the service if needed, activate the engine and
                   show STATUS as preedit. Exit 0: the input method will
                   show statuses; 3: no IBus text field has focus (type
                   instead); 4: service unavailable
  status STATUS    replace the preedit status (best effort)
  finish TEXT      commit TEXT (fitted to the text around the cursor
                   unless --no-fit), then restore the previous engine. Exit 0:
                   committed; 3: no IBus text field has focus, type it;
                   4: service unavailable, type it; 5: outcome unknown (the
                   reply was lost), do not type it again
  end              clear the status and restore the previous engine
  probe            print the focus state as JSON (diagnostics)
  --serve          run the service in the foreground
  --stop-server    stop the running service

Put -- before a text argument so a dictation that looks like an option
stays text.

Protocol: one JSON line per connection, one JSON line back.
"""

import fcntl
import importlib.util
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

ENGINE_NAME = "xhisper"
COMPONENT_NAME = "org.freedesktop.IBus.Xhisper"
SOCKET_NAME = "ime.sock"
LOCK_NAME = "ime.lock"
PREV_ENGINE_FILE = "ime-prev-engine"
DEFAULT_IDLE_MINUTES = 30

# IBus keeps an internal placeholder ("fake") input context and attaches the
# active engine to it whenever no application text field has focus; text
# committed there is dropped. IBus >= 1.5.27 says so directly: the client
# argument of IBusEngine::focus-in-id is "fake" ("focus is on desktop
# background or other programs where no input is possible"). Older IBus only
# exposes the focused context's object path. The placeholder is the first
# context ibus-daemon creates at startup (bus_ibus_impl_init in
# bus/ibusimpl.c), and context paths come from a counter starting at 1
# (bus_input_context_new in bus/inputcontext.c), so it is InputContext_1.
# That numbering is an implementation detail, not documented API — it holds
# for IBus 1.5.26 and current IBus, and newer IBus uses focus-in-id anyway.
FAKE_CONTEXT_PATH = "/org/freedesktop/IBus/InputContext_1"
FAKE_CLIENT = "fake"

FOCUS_WAIT_S = 0.5      # activation until the engine is attached to the focused field
SURROUNDING_FRESH_WAIT_S = 0.04  # an answer to asking for the text around the cursor
SURROUNDING_WAIT_S = 0.15  # the update a field sends after a commit
START_WAIT_S = 5.0      # service start until its socket answers
REQUEST_LIMIT = 1024 * 1024

EXIT_OK, EXIT_ERROR, EXIT_NO_FIELD, EXIT_UNAVAILABLE, EXIT_UNKNOWN = 0, 1, 3, 4, 5


# ---------------------------------------------------------------------------
# Shared helpers (same conventions as the transcription server)

def runtime_dir():
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


def try_lock(lock_path: Path):
    """Take an exclusive lock without blocking: the open file, or None if held."""
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def recv_line(conn: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > REQUEST_LIMIT:
            raise ValueError("message too large")
    return buf.split(b"\n", 1)[0].decode()


def send_line(conn: socket.socket, obj: dict):
    try:
        conn.sendall((json.dumps(obj) + "\n").encode())
    except OSError:
        pass  # client gave up; nothing to report to


def log(message: str):
    """Timestamped like xhisper.sh's audit lines; stderr is the xhisper log."""
    now = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
    print(f"[{stamp}] ime: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Fitting a dictation to the text around the cursor (smart insertion)
#
# The transcription is punctuated as if it were a whole text: it starts with
# a capital and ends with a period. Inserted after a word or mid-sentence it
# needs a leading space, often a lowercase first word, and no period before
# the words that follow. The focused field reports the text around the
# cursor (IBus surrounding text), which says which case applies.

_OPENERS = "([{“‘«"
_CLOSERS = "\"'”’)]}»"
_SENTENCE_ENDS = ".!?…"
_NO_SPACE_BEFORE = ",.;:!?)]}…%"

# First words lowercased when a dictation continues a sentence. Only common
# words that are not names; anything else (a name, "I", an acronym) keeps
# its capital, since Parakeet can't be asked which it is. Ambiguous words
# such as "May", "Will" or "March" are left out on purpose.
_LOWERCASE_MID_SENTENCE = frozenset("""
a an the and but or nor so yet for then also plus because although though
while if unless until since whereas whether as than like which that who
whom whose what when where why how there here this these those it its
it's that's there's here's what's who's let's we you he she they them us
him her me my your our their his we're you're they're we've you've
they've we'll you'll they'll it'll he's she's is are was were be been
being am do does did don't doesn't didn't isn't aren't wasn't weren't
can't won't wouldn't couldn't shouldn't have has had would could should
might must can to of in on at by with from about into onto over under
after before without within through between around against during
toward towards just maybe probably really actually now not still even
only too very some any all each every both either neither more most much
many such no yes okay ok well anyway though instead otherwise however
""".split())


def _first_word(text: str):
    """(start, end) of the first word of text, past opening quotes."""
    m = re.search(r"[^\W\d_][\w'’-]*", text)
    return (m.start(), m.end()) if m else None


def fit_to_context(text: str, before: str, after: str) -> str:
    """
    Adjust a dictation to the text before and after the cursor: spacing,
    the case of its first word, and its final period. Text it can't judge
    (an empty dictation, no letters) is returned unchanged.
    """
    if not text.strip():
        return text
    text = text.strip()

    # Where the cursor is: the start of a line or sentence, or mid-sentence
    trimmed = before.rstrip(" \t")
    core = trimmed.rstrip(_CLOSERS)
    sentence_start = (not trimmed or trimmed.endswith("\n")
                      or (core[-1:] in _SENTENCE_ENDS if core else True))

    # Leading space: after a word or closing punctuation, not after a space,
    # a line start or an opening bracket/quote
    last = before[-1:]
    if last and not last.isspace() and text[0] not in _NO_SPACE_BEFORE:
        opening_quote = last in "\"'" and (len(before) == 1 or before[-2].isspace()
                                            or before[-2] in _OPENERS)
        if last not in _OPENERS and not opening_quote:
            text = " " + text

    # Case of the first word
    word = _first_word(text)
    if word:
        s, e = word
        first = text[s:e]
        if sentence_start:
            if first[0].islower():
                text = text[:s] + first[0].upper() + text[s + 1:]
        elif (first[0].isupper() and first[1:] == first[1:].lower()
              and first.lower().replace("’", "'") in _LOWERCASE_MID_SENTENCE):
            text = text[:s] + first[0].lower() + text[s + 1:]

    # The end: the words after the cursor decide whether the dictation ends
    # a sentence there
    nxt = after[:1]
    following = after.lstrip(" \t")
    continues = bool(following) and following[0].islower()
    if nxt and nxt in ",.;:!?":  # ("" is "in" every string)
        text = re.sub(r"(?<![.!?…])[.,;:!?]$", "", text)  # no "word.," / "word.."
    elif continues:
        text = re.sub(r"(?<![.!?…])\.$", "", text)  # mid-sentence: no period
    if nxt and (nxt.isalnum() or nxt in _OPENERS):
        text = text + " "
    return text


# ---------------------------------------------------------------------------
# Service (runs under a Python that has PyGObject; see service_python())

def _make_engine_class(IBus, service):
    class XhisperEngine(IBus.Engine):
        """Pass-through engine: never consumes a key, reports focus changes."""

        __gtype_name__ = "XhisperEngine"

        def do_process_key_event(self, keyval, keycode, state):
            return False  # the application handles every key as usual

        # IBus < 1.5.27, or an engine created without has-focus-id
        def do_focus_in(self):
            service.focus_in(self, None, None)

        def do_focus_out(self):
            service.focus_out(self)

        # IBus >= 1.5.27 with has-focus-id: replaces focus-in/focus-out
        def do_focus_in_id(self, object_path, client):
            service.focus_in(self, object_path, client)

        def do_focus_out_id(self, object_path):
            service.focus_out(self)

        def do_set_content_type(self, purpose, hints):
            service.content_type(self, purpose, hints)

        def do_set_capabilities(self, caps):
            service.caps = caps

    return XhisperEngine


class ImeService:
    """IBus engine plus the socket server xhisper.sh talks to.

    All IBus calls run on the GLib main thread; the socket thread hands work
    to it with on_main(). Engine switches use the async API: ibus-daemon
    answers SetGlobalEngine only after it has created the engine through
    this process's factory, so a synchronous call from the main thread would
    deadlock.
    """

    def __init__(self, GLib, IBus, server_dir: Path, idle_minutes: float):
        self.GLib, self.IBus = GLib, IBus
        self.server_dir = server_dir
        self.idle_minutes = idle_minutes

        IBus.init()
        self.bus = IBus.Bus()
        if not self.bus.is_connected():
            raise RuntimeError("cannot connect to ibus-daemon (is IBus running?)")
        self.bus.connect("disconnected", self._on_disconnected)
        self.daemon_gone = False
        self.prev_engine = self._load_prev_engine()
        current = self.bus.get_global_engine()
        if current is None and self.prev_engine:
            # A previous service died without the restore below running
            # (see restore_engine_after_exit): put the user's engine back
            log(f"no input method set; restoring {self.prev_engine}")
            self.bus.set_global_engine(self.prev_engine)
        elif current is not None and current.get_name() != ENGINE_NAME:
            # Known before registering: if this service exits, even before
            # its first dictation, the engine it must restore is this one
            self.prev_engine = current.get_name()
            self._save_prev_engine(self.prev_engine)

        self.has_focus_id = "has-focus-id" in {p.name for p in IBus.Engine.list_properties()}
        self.Engine = _make_engine_class(IBus, self)
        self.engines = set()
        self.engine_serial = 0
        self.factory = IBus.Factory.new(self.bus.get_connection())
        self.factory.connect("create-engine", self._create_engine)

        component = IBus.Component(
            name=COMPONENT_NAME, description="xhisper dictation input",
            version="1.0", license="MIT", author="xhisper",
            homepage="https://github.com/imaginalnika/xhisper",
            command_line="", textdomain="")
        component.add_engine(IBus.EngineDesc(
            name=ENGINE_NAME, longname="xhisper dictation",
            description="Inserts xhisper dictations; passes every key through",
            language="other", license="MIT", author="xhisper", icon="",
            layout="default"))
        if not self.bus.register_component(component):
            raise RuntimeError("ibus-daemon refused to register the xhisper engine")

        self.engine = None    # engine instance attached to the focused context
        self.client = None    # focus-in-id client name (IBus >= 1.5.27)
        self.path = None      # focus-in-id context path (IBus >= 1.5.27)
        self.purpose = None   # IBus.InputPurpose of the focused field
        self.status = None    # preedit status, re-shown when focus moves
        self.caps = 0         # IBus.Capabilite flags of the focused field
        self.surrounding = None  # (text, cursor, anchor) around the cursor
        self.surrounding_updates = 0
        self.updates_at_commit = 0
        # Fields that sent no surrounding text even after a commit (Chrome,
        # for one), by context path: don't make every dictation wait for them
        self.silent_fields = set()
        self.active = False
        self.last_request = time.monotonic()
        self.loop = GLib.MainLoop()

    # -- engine callbacks (main thread) ------------------------------------

    def _create_engine(self, factory, engine_name):
        self.engine_serial += 1
        props = {
            "engine_name": engine_name,
            "object_path": f"/org/freedesktop/IBus/Engine/Xhisper/{self.engine_serial}",
            "connection": self.bus.get_connection(),
        }
        if self.has_focus_id:
            props["has_focus_id"] = True
        engine = self.Engine(**props)
        engine.connect("destroy", self._engine_destroyed)
        engine.connect("set-surrounding-text", self._surrounding_text)
        self.engines.add(engine)  # keep the Python wrapper alive with the engine
        return engine

    def _engine_destroyed(self, engine):
        self.engines.discard(engine)
        if engine is self.engine:
            self.engine = None

    def _switch_engine(self, name: str):
        def done(bus, result):
            try:
                if not bus.set_global_engine_async_finish(result):
                    log(f"could not switch the input method to {name}")
            except Exception as exc:
                log(f"could not switch the input method to {name}: {exc}")
        self.bus.set_global_engine_async(name, -1, None, done)

    def focus_in(self, engine, path, client):
        self.engine, self.path, self.client = engine, path, client
        self.surrounding = None  # belongs to the previous field
        if self.status is not None:
            self._show_status()  # follow the user to the newly focused field

    def focus_out(self, engine):
        if engine is self.engine:
            self.engine = self.path = self.client = self.purpose = None
            self.surrounding = None

    def _surrounding_text(self, engine, text, cursor_pos, anchor_pos):
        self.surrounding = (text.get_text(), cursor_pos, anchor_pos)
        self.surrounding_updates += 1

    def request_surrounding(self):
        """
        Ask the focused field for the text around the cursor. Returns the
        update count to wait past, or None if no answer can be expected.

        Clients of IBus >= 1.5.28 answer at once. Older ones send it only at
        their own trigger points, and only if it changed since they last
        computed it, even for another engine (ibus_input_context_set_
        surrounding_text compares with its cache): right after the user
        typed, they send nothing. A commit always changes it, which is what
        finish falls back on (fit_after_commit).
        """
        if not self.is_real_focus() or not self.caps & self.IBus.Capabilite.SURROUNDING_TEXT:
            return None
        if self.purpose == self.IBus.InputPurpose.TERMINAL:
            return None  # terminals advertise it but never answer
        if self._field() in self.silent_fields:
            return None
        self.engine.get_surrounding_text()  # emits RequireSurroundingText
        # That only flags the field (IBus 1.5.26 GTK module): it sends the text
        # at its own trigger points, such as key presses, commits, and preedit
        # going from hidden to shown (_ibus_context_show_preedit_text_cb). The
        # last is one the engine can cause. The preedit is cleared at commit.
        self.engine.hide_preedit_text()
        self.engine.show_preedit_text()
        return self.surrounding_updates

    def _field(self):
        """The focused input context's path: identifies the field."""
        try:
            return self.path or self.bus.current_input_context()
        except Exception:
            return None

    def content_type(self, engine, purpose, hints):
        # Stored whatever its order relative to focus-in: only the engine
        # attached to the focused field receives it
        self.purpose = purpose

    # -- state (main thread) -------------------------------------------------

    def is_real_focus(self) -> bool:
        """True if an application text field (not IBus's placeholder) has focus."""
        if self.engine is None:
            return False
        if self.client is not None:  # IBus >= 1.5.27: documented client type
            return self.client != FAKE_CLIENT
        path = self.bus.current_input_context()  # None if nothing is focused
        return bool(path) and path != FAKE_CONTEXT_PATH

    def focus_info(self) -> dict:
        try:
            path = self.path or self.bus.current_input_context()
        except Exception:
            path = None
        purpose = None
        if self.purpose is not None:  # the vfunc passes a plain integer
            try:
                purpose = self.IBus.InputPurpose(self.purpose).value_nick
            except (ValueError, AttributeError):
                purpose = str(self.purpose)
        return {"real": self.is_real_focus(), "path": path, "client": self.client,
                "placeholder": path == FAKE_CONTEXT_PATH if path else None,
                "purpose": purpose, "focus_id": self.has_focus_id}

    def is_attached(self) -> bool:
        return self.engine is not None

    def activate(self):
        if self.active:
            return
        current = self.bus.get_global_engine()
        name = current.get_name() if current else None
        if name and name != ENGINE_NAME:
            self.prev_engine = name
            self._save_prev_engine(name)
        self.active = True
        if name != ENGINE_NAME:
            # wait for the focus-in (and content type) of the new engine; a
            # purpose left from an earlier field must not reach this one
            self.engine = self.purpose = None
            self._switch_engine(ENGINE_NAME)

    def deactivate(self):
        if not self.active:
            return
        self._clear_status()
        self.status = None
        self.active = False
        current = self.bus.get_global_engine()
        if not current or current.get_name() != ENGINE_NAME:
            return  # the user switched input sources meanwhile; leave theirs
        if self.prev_engine:
            self._switch_engine(self.prev_engine)
            self.engine = self.purpose = None
        else:
            log("no previous input method known; leaving the pass-through engine active")

    def set_status(self, text: str) -> bool:
        self.status = text
        return self._show_status()

    def _show_status(self) -> bool:
        if not self.is_real_focus():
            return False
        IBus = self.IBus
        # CLEAR: if focus moves away, the preedit is dropped, never committed
        self.engine.update_preedit_text_with_mode(
            IBus.Text.new_from_string(self.status), len(self.status), True,
            IBus.PreeditFocusMode.CLEAR)
        return True

    def _clear_status(self):
        if self.engine is not None:
            self.engine.update_preedit_text_with_mode(
                self.IBus.Text.new_from_string(""), 0, False, self.IBus.PreeditFocusMode.CLEAR)

    def commit(self, text: str, fit: bool = False, context: str = None):
        """Commit text, fitted to fresh surrounding text if fit; the text committed, or None."""
        if not self.is_real_focus():
            return None
        self.status = None
        self._clear_status()
        if self.purpose == self.IBus.InputPurpose.TERMINAL:
            # A committed newline is Enter in a terminal: it would run the
            # command line. Join lines with spaces instead.
            text = re.sub(r"[ \t]*\r?\n\s*", " ", text).strip()
        if fit and self.surrounding is not None:
            around, cursor, anchor = self.surrounding
            start, end = sorted((cursor, anchor))  # a selection gets replaced
            fitted = fit_to_context(text, around[:start], around[end:])
            if fitted != text:
                log(f"fitted to the text around the cursor ({context}): {text!r} -> {fitted!r}")
                text = fitted
        self.updates_at_commit = self.surrounding_updates
        self.engine.commit_text(self.IBus.Text.new_from_string(text))
        return text

    def fit_after_commit(self, committed: str):
        """
        Fit a dictation already committed as is, using the surrounding text
        the field sent after the commit: delete it and commit the fitted
        version. Only if that text shows the dictation right before the
        cursor; otherwise (a multi-line dictation, an app that edits what it
        receives) it stays as is.
        """
        around, cursor, anchor = self.surrounding
        n = len(committed)
        if cursor != anchor or cursor < n or around[cursor - n:cursor] != committed:
            log("the text around the cursor doesn't end with the dictation; left as is")
            return
        fitted = fit_to_context(committed, around[:cursor - n], around[cursor:])
        if fitted == committed:
            return
        self.engine.delete_surrounding_text(-n, n)
        self.engine.commit_text(self.IBus.Text.new_from_string(fitted))
        log(f"fitted to the text around the cursor (after inserting): {committed!r} -> {fitted!r}")

    def _load_prev_engine(self):
        try:
            return (self.server_dir / PREV_ENGINE_FILE).read_text().strip() or None
        except OSError:
            return None

    def _save_prev_engine(self, name: str):
        # Survives a service restart: if the service dies while active, IBus
        # is left with no engine and the next run still knows what to restore
        try:
            (self.server_dir / PREV_ENGINE_FILE).write_text(name + "\n")
        except OSError:
            pass

    # -- requests (socket thread) --------------------------------------------

    def on_main(self, fn, timeout_s: float = 5.0):
        """Run fn on the GLib main thread and return its result."""
        done = threading.Event()
        box = {}

        def run():
            try:
                box["value"] = fn()
            except Exception as exc:
                box["error"] = exc
            done.set()
            return False

        self.GLib.idle_add(run)
        if not done.wait(timeout_s):
            raise TimeoutError("IBus main loop not responding")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _wait_surrounding(self, seen: int, timeout_s: float):
        """Wait for a surrounding-text update past count seen: ms it took, or None."""
        start = time.monotonic()
        while self.on_main(lambda: self.surrounding_updates) == seen:
            if time.monotonic() - start > timeout_s:
                return None
            time.sleep(0.005)
        return (time.monotonic() - start) * 1000

    def _activate_and_wait(self):
        self.on_main(self.activate)
        deadline = time.monotonic() + FOCUS_WAIT_S
        while not self.on_main(self.is_attached) and time.monotonic() < deadline:
            time.sleep(0.005)

    def handle(self, request: dict) -> dict:
        cmd = request.get("cmd")
        if cmd == "ping":
            return {"ok": True}
        if cmd == "probe":
            return {"ok": True, "active": self.on_main(lambda: self.active),
                    **self.on_main(self.focus_info)}
        if cmd == "begin":
            self._activate_and_wait()
            info = self.on_main(self.focus_info)
            if not info["real"]:
                self.on_main(self.deactivate)
                return {"ok": True, "mode": "type", **info}
            self.on_main(lambda: self.set_status(str(request.get("status", ""))))
            return {"ok": True, "mode": "ime", **info}
        if cmd == "status":
            if not self.on_main(lambda: self.active):
                return {"ok": True, "shown": False}
            return {"ok": True, "shown": self.on_main(lambda: self.set_status(str(request["text"])))}
        if cmd == "finish":
            self._activate_and_wait()
            text = str(request["text"])
            fit = bool(request.get("fit", True))
            # Ask the field for the text around the cursor. IBus >= 1.5.28
            # clients answer at once; older ones only when it changed since
            # they last computed it, and a cached copy may be stale, so only
            # a fresh answer counts.
            seen = self.on_main(self.request_surrounding) if fit else None
            fresh = self._wait_surrounding(seen, SURROUNDING_FRESH_WAIT_S) if seen is not None else None
            context = None if fresh is None else f"answered in {fresh:.0f}ms"
            committed = self.on_main(lambda: self.commit(text, fresh is not None, context))
            if committed is not None and seen is not None and fresh is None:
                # No fresh answer: after a commit the field's text has
                # changed, so even older clients send it. Fit from that.
                after = self.on_main(lambda: self.updates_at_commit)
                if self._wait_surrounding(after, SURROUNDING_WAIT_S) is not None:
                    self.on_main(lambda: self.fit_after_commit(committed))
                else:
                    field = self.on_main(self._field)
                    self.on_main(lambda: self.silent_fields.add(field))
                    log(f"no surrounding text from the focused field ({field}) within "
                        f"{SURROUNDING_WAIT_S * 1000:.0f}ms; inserted as is, and won't ask it again")
            info = self.on_main(self.focus_info)
            self.on_main(self.deactivate)
            return {"ok": True, "committed": committed is not None, "context": context, **info}
        if cmd == "end":
            self.on_main(self.deactivate)
            return {"ok": True}
        if cmd == "stop":
            return {"ok": True}  # serve_socket quits once the reply is out
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    def serve_socket(self, server: socket.socket):
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return  # socket closed: shutting down
            with conn:
                conn.settimeout(10)
                try:
                    request = json.loads(recv_line(conn))
                    if not isinstance(request, dict):
                        raise ValueError("not a JSON object")
                except (OSError, ValueError) as exc:
                    log(f"bad request: {exc}")
                    continue
                self.last_request = time.monotonic()
                try:
                    reply = self.handle(request)
                except Exception as exc:
                    log(f"{request.get('cmd')} failed: {exc}")
                    reply = {"ok": False, "error": str(exc)}
                send_line(conn, reply)
            if request.get("cmd") == "stop":
                log("stop requested, exiting")
                self.on_main(self.quit)
                return

    # -- lifecycle (main thread) ---------------------------------------------

    def _idle_check(self):
        idle_s = time.monotonic() - self.last_request
        # Never exit mid-dictation: a long recording sends no requests
        if self.idle_minutes > 0 and not self.active and idle_s > self.idle_minutes * 60:
            log(f"idle for {self.idle_minutes:g} min, exiting")
            self.quit()
            return False
        return True

    def _on_disconnected(self, bus):
        log("ibus-daemon disconnected, exiting")
        self.active = False  # nothing to restore: the daemon is gone
        self.daemon_gone = True
        self.loop.quit()

    def quit(self):
        self.deactivate()
        self.loop.quit()
        return False

    def _on_signal(self, signum):
        log(f"{signal.Signals(signum).name} received, exiting")
        return self.quit()

    def run(self):
        GLib = self.GLib
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            GLib.unix_signal_add(GLib.PRIORITY_HIGH, signum, self._on_signal, signum)
        GLib.timeout_add_seconds(30, self._idle_check)
        self.loop.run()


def restore_engine_after_exit(engine_name: str):
    """Have a detached helper restore engine_name once this process is gone.

    When a component registered at runtime goes away, ibus-daemon re-checks
    the global engine (bus_ibus_impl_check_global_engine in bus/ibusimpl.c),
    but only against other runtime-registered engines: installed engines
    such as xkb:us::eng count as missing, so it unsets the global engine.
    Keys still pass through with no engine, but the user's input method
    (e.g. a layout or a CJK engine) is off. The component only goes away
    with this process's connection, so the restore must happen afterwards.
    """
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--restore-engine",
             engine_name, str(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        log(f"could not start the input-method restore helper: {exc}")


def restore_engine(engine_name: str, after_pid: int) -> int:
    """Wait for after_pid to exit; if IBus then has no engine, set engine_name."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and Path(f"/proc/{after_pid}").exists():
        time.sleep(0.02)
    # ibus-daemon unsets the engine shortly after the connection closes
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        current = subprocess.run(["ibus", "engine"], capture_output=True, text=True)
        if current.returncode != 0:  # "No engine is set."
            subprocess.run(["ibus", "engine", engine_name], capture_output=True)
            return EXIT_OK
        time.sleep(0.05)
    return EXIT_OK  # an engine is set: nothing to restore


def serve(idle_minutes: float) -> int:
    """Run the input-method service in the foreground until idle or stopped."""
    server_dir = runtime_dir()
    if server_dir is None:
        log("no private runtime directory for the input-method service")
        return EXIT_ERROR
    lock = try_lock(server_dir / LOCK_NAME)
    if lock is None:
        return EXIT_OK  # another service is already running

    try:
        import gi
        gi.require_version("IBus", "1.0")
        from gi.repository import GLib, IBus
    except (ImportError, ValueError) as exc:
        log(f"cannot load the IBus Python bindings ({exc}); install python3-gi "
            "and gir1.2-ibus-1.0 (Debian/Ubuntu) or python3-gobject and ibus-libs (Fedora)")
        return EXIT_ERROR

    sock_path = server_dir / SOCKET_NAME
    if sock_path.exists():  # left by a service that died; we hold the lock now
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    service = None
    try:
        # Bind before registering with IBus: once registered, exiting has a
        # side effect on the user's input method (see restore_engine_after_exit)
        try:
            server.bind(str(sock_path))
        except OSError as exc:  # e.g. a runtime dir path over the ~108-byte limit
            log(f"cannot create the input-method socket {sock_path}: {exc}")
            return EXIT_ERROR
        server.listen(8)
        start = time.perf_counter()
        try:
            service = ImeService(GLib, IBus, server_dir, idle_minutes)
        except RuntimeError as exc:
            log(str(exc))
            return EXIT_ERROR
        threading.Thread(target=service.serve_socket, args=(server,), daemon=True).start()
        log(f"ready: engine registered in {time.perf_counter() - start:.2f}s "
            f"(IBus {IBus.MAJOR_VERSION}.{IBus.MINOR_VERSION}.{IBus.MICRO_VERSION}, "
            f"focus detection: {'focus-in-id' if service.has_focus_id else 'context path'})")
        service.run()
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()
        if service is not None and service.prev_engine and not service.daemon_gone:
            restore_engine_after_exit(service.prev_engine)
        lock.close()
    return EXIT_OK


# ---------------------------------------------------------------------------
# Client (standard library only: runs under whatever python3 xhisper uses)

class Unavailable(Exception):
    """The service could not be reached; nothing was sent."""


def call(request: dict, timeout_s: float) -> dict:
    """One request. Raises Unavailable if nothing was sent, OSError/ValueError
    if the request was sent but the reply was lost."""
    server_dir = runtime_dir()
    if server_dir is None:
        raise Unavailable("no private runtime directory")
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(timeout_s)
        try:
            conn.connect(str(server_dir / SOCKET_NAME))
        except OSError as exc:
            raise Unavailable(str(exc)) from exc
        conn.sendall((json.dumps(request) + "\n").encode())
        return json.loads(recv_line(conn))
    finally:
        conn.close()


def service_python() -> str:
    """An interpreter that can import PyGObject. xhisper often runs from a
    virtualenv (for onnx-asr) that cannot see the distribution's gi package."""
    if importlib.util.find_spec("gi") is not None:
        return sys.executable
    for candidate in ("/usr/bin/python3", "/bin/python3"):
        if os.access(candidate, os.X_OK):
            return candidate
    return sys.executable  # the service logs the import failure


def start_service(idle_minutes: float, log_file: str = None) -> bool:
    """Start the service unless it is running; True once it answers."""
    try:
        if call({"cmd": "ping"}, timeout_s=2).get("ok"):
            return True
    except Unavailable:
        pass
    except (OSError, ValueError):
        return False  # running but not answering

    server_dir = runtime_dir()
    if server_dir is None:
        return False
    lock = try_lock(server_dir / LOCK_NAME)
    proc = None
    if lock is not None:  # not running or starting: launch it
        lock.close()  # the service takes the lock itself; a racing twin exits
        log_target = open(log_file, "a") if log_file else subprocess.DEVNULL
        proc = subprocess.Popen(
            [service_python(), os.path.abspath(__file__), "--serve",
             "--idle-minutes", str(idle_minutes)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log_target,
            start_new_session=True,  # outlive the xhisper invocation that started it
        )

    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        try:
            if call({"cmd": "ping"}, timeout_s=1).get("ok"):
                return True
        except Unavailable:
            pass
        except (OSError, ValueError):
            return False
        if proc is not None and proc.poll() not in (None, EXIT_OK):
            return False  # it failed to start (its log says why); don't wait it out
        time.sleep(0.02)
    return False


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return EXIT_ERROR
    cmd, args = argv[1], argv[2:]

    idle_minutes, log_file, fit = DEFAULT_IDLE_MINUTES, None, True
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--":  # the rest is text, even if it looks like an option
            rest += args[i + 1:]
            break
        if args[i] == "--idle-minutes" and i + 1 < len(args):
            idle_minutes = float(args[i + 1])
            i += 2
        elif args[i] == "--log-file" and i + 1 < len(args):
            log_file = args[i + 1]
            i += 2
        elif args[i] == "--no-fit":
            fit = False
            i += 1
        else:
            rest.append(args[i])
            i += 1
    text = rest[0] if rest else ""

    if cmd == "--serve":
        return serve(idle_minutes)
    if cmd == "--restore-engine" and len(rest) == 2:
        return restore_engine(rest[0], int(rest[1]))
    if cmd == "--stop-server":
        try:
            return EXIT_OK if call({"cmd": "stop"}, timeout_s=5).get("ok") else EXIT_ERROR
        except (OSError, ValueError, Unavailable):
            print("No input-method service running", file=sys.stderr)
            return EXIT_ERROR

    if cmd in ("begin", "probe"):
        if not start_service(idle_minutes, log_file):
            log("input-method service unavailable; typing instead")
            return EXIT_UNAVAILABLE

    try:
        if cmd == "begin":
            reply = call({"cmd": "begin", "status": text}, timeout_s=5)
            return EXIT_OK if reply.get("mode") == "ime" else EXIT_NO_FIELD
        if cmd == "status":
            call({"cmd": "status", "text": text}, timeout_s=2)
            return EXIT_OK
        if cmd == "finish":
            reply = call({"cmd": "finish", "text": text, "fit": fit}, timeout_s=5)
            if not reply.get("ok"):
                log(f"finish failed: {reply.get('error')}; typing instead")
                return EXIT_UNAVAILABLE
            return EXIT_OK if reply.get("committed") else EXIT_NO_FIELD
        if cmd == "end":
            call({"cmd": "end"}, timeout_s=2)
            return EXIT_OK
        if cmd == "probe":
            print(json.dumps(call({"cmd": "probe"}, timeout_s=5)))
            return EXIT_OK
    except Unavailable:
        return EXIT_UNAVAILABLE
    except (OSError, ValueError) as exc:
        if cmd == "finish":
            log(f"no reply to finish ({exc}); the text may already be in place, not typing it again")
            return EXIT_UNKNOWN
        return EXIT_ERROR

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main(sys.argv))
