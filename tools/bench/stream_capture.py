"""Opt-in binary stream archival; healthy progressing local files are required.

Only the reader owns the source and archive. finish grants an EOF drain interval,
then cuts off even a surviving producer. The archival byte frontier is authoritative;
console mirroring is nonblocking best effort and never participates in completeness.
Lifecycle: reading -> drain requested -> EOF/cutoff/error -> resources retired.
The supervisor alone calls finish; after successful construction source ownership
belongs to the reader. This Linux host capture does not claim bounded I/O on a
stalled archival filesystem.
"""
from __future__ import annotations

import math
import os
import selectors
import threading
import time
from pathlib import Path
from typing import BinaryIO


class StreamCapture:
    def __init__(self, source: BinaryIO, path: Path, console_fd: int | None = None):
        self.source = source
        self.path = path
        self.capture_complete = False
        self.capture_failed = False
        self.capture_stop_reason: str | None = None
        self.archived_bytes = 0
        self.error: BaseException | None = None
        self.console_disabled = False
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._finished = False
        self._retired = threading.Event()
        self._console: int | None = None
        self._archive = path.open("wb", buffering=0)
        try:
            self._wake_read, self._wake_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
            # FileIO owns the writer: close is idempotent even if cancellation
            # arrives after the descriptor closes but before finish advances.
            self._wake_writer = os.fdopen(self._wake_write, "wb", buffering=0)
            if console_fd is not None:
                try:
                    # Reopen rather than dup: O_NONBLOCK must not change the caller's
                    # open-file description (including inherited console flags).
                    self._console = os.open(f"/proc/self/fd/{console_fd}",
                                            os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                except OSError:
                    self.console_disabled = True
            os.set_blocking(source.fileno(), False)
            self.worker = threading.Thread(target=self._run, name="forensic-stream-reader")
            self.worker.start()
        except BaseException:
            self._archive.close()
            writer = getattr(self, "_wake_writer", None)
            if writer is not None:
                writer.close()
            for fd in (getattr(self, "_wake_read", None),
                       None if writer is not None else getattr(self, "_wake_write", None),
                       self._console):
                if fd is not None:
                    os.close(fd)
            raise

    def _fail(self, exc: BaseException) -> None:
        self.capture_failed = True
        self.capture_complete = False
        self.capture_stop_reason = "capture_error"
        if self.error is None:
            self.error = exc

    def _mirror(self, chunk: bytes) -> None:
        if self._console is None:
            return
        try:
            # One nonblocking attempt: partial or blocked output is deliberately
            # not retried. Archival, not console delivery, is the contract.
            if os.write(self._console, chunk) != len(chunk):
                self.console_disabled = True
        except OSError:
            self.console_disabled = True
        if self.console_disabled:
            os.close(self._console)
            self._console = None

    def _run(self) -> None:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.source.fileno(), selectors.EVENT_READ, "source")
                selector.register(self._wake_read, selectors.EVENT_READ, "control")
                while True:
                    with self._lock:
                        deadline = self._deadline
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        self.capture_stop_reason = "drain_timeout"
                        break
                    ready = selector.select(remaining)
                    for key, _ in ready:
                        if key.data == "control":
                            os.read(self._wake_read, 4096)
                            continue
                        # Check the cutoff even under continuous readiness.
                        with self._lock:
                            deadline = self._deadline
                        if deadline is not None and time.monotonic() >= deadline:
                            self.capture_stop_reason = "drain_timeout"
                            return
                        try:
                            chunk = os.read(self.source.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            self.capture_complete = True
                            self.capture_stop_reason = "eof"
                            return
                        view = memoryview(chunk)
                        while view:
                            count = self._archive.write(view)
                            if not count:
                                raise OSError("archive made no write progress")
                            self.archived_bytes += count
                            view = view[count:]
                        self._mirror(chunk)
        except BaseException as exc:
            self._fail(exc)
        finally:
            for resource in (self._archive, self.source):
                try:
                    resource.close()
                except BaseException as exc:
                    self._fail(exc)
            for fd in (self._wake_read, self._console):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as exc:
                        self._fail(exc)
            self._console = None
            self._retired.set()

    def finish(self, drain_timeout: float = 1.0) -> dict[str, object]:
        if not math.isfinite(drain_timeout) or drain_timeout < 0:
            raise ValueError("drain_timeout must be finite and nonnegative")
        interruption: BaseException | None = None
        # Retry the complete sequence, not just the retirement wait. The reader
        # alone closes its resources; join must succeed before the writer closes
        # and finish becomes idempotent. A retry never extends the drain deadline.
        while not self._finished:
            try:
                with self._lock:
                    if self._deadline is None:
                        self._deadline = time.monotonic() + drain_timeout
                if not self._retired.is_set():
                    # Reader may already have closed its end after EOF/failure.
                    try:
                        os.write(self._wake_write, b"x")
                    except BrokenPipeError:
                        pass
                    self._retired.wait()
                self.worker.join()
                self._wake_writer.close()
                self._finished = True
            except BaseException as exc:
                if interruption is None:
                    interruption = exc
                elif exc is not interruption:
                    interruption.add_note(f"secondary capture interruption: {exc!r}")
        if interruption is not None:
            raise interruption
        return self.status()

    def status(self) -> dict[str, object]:
        return {"capture_complete": self.capture_complete,
                "capture_failed": self.capture_failed,
                "capture_stop_reason": self.capture_stop_reason,
                "archived_bytes": self.archived_bytes,
                "console_disabled": self.console_disabled,
                "capture_error": None if self.error is None else repr(self.error)}
