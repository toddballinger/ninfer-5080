#!/usr/bin/env python3
"""Linux host-only durable external campaign owner; never modifies the workload.

Use run or daemonise with --run, --output, optional --deadline-min and --restore.
Detached launch acknowledges readiness from the re-executed supervisor itself,
not from a stale pidfile. The campaign owns concurrent pipe capture, group
retirement, durable return-code evidence and restoration independently of its
launcher. Output directories are exclusive while a campaign is active.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]


def _persist_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _subreaper() -> None:
    # Adopt orphaned runner grandchildren so retirement includes reaping, not
    # merely signalling. Linux is the deployment and host-proof platform.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    # A child can create another session. Subreaper adoption preserves our
    # ownership even when group membership does not. This standalone owner
    # has no unrelated children; signal its entire descendant tree as well.
    def signal_children(parent):
        try:
            raw = Path(f'/proc/{parent}/task/{parent}/children').read_text()
        except FileNotFoundError:
            return
        for token in raw.split():
            child = int(token)
            signal_children(child)
            try:
                os.kill(child, sig)
            except ProcessLookupError:
                pass
    signal_children(os.getpid())


def _open_archive(path):
    # A FIFO or device must not defeat the campaign watchdog before spawn.
    # Reject symlinks/devices and open nonblocking before checking file type.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC |
                 os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError('archive must be a regular file')
        return os.fdopen(fd, 'wb', buffering=0)
    except BaseException:
        os.close(fd)
        raise


def _close(resource, label, errors):
    # A close that raises before releasing its descriptor must not skip later
    # resources; one bounded retry also covers transient injected failures.
    for attempt in range(2):
        try:
            resource.close()
            return
        except Exception as exc:
            errors.append(f'{label}: {exc!r}')


def _execute(command, directory, deadline, grace, env, on_spawn=None, campaign_signals=None):
    """Drain both pipes during execution, then retire the whole owned group.

    A dead group leader is not evidence that its descendants have retired.
    Continue capture through TERM/KILL escalation even after leader exit.
    Failed sinks are recorded; their pipe is still drained/discarded to avoid
    turning a forensic error into a blocked runner. No capture threads exist.
    """
    directory = Path(directory)
    result = dict(command=list(command), rc=None, signal=None,
                  stdout_archived_bytes=0, stderr_archived_bytes=0)
    errors = []
    proc = None
    streams = []
    logs = {}
    start = time.monotonic()
    deadline_fired = False
    stopping = None
    killed_at = None
    pending_signals = campaign_signals if campaign_signals is not None else []
    term_wait = float(os.environ.get('DSS_TERM_WAIT', '2'))
    if not math.isfinite(term_wait) or term_wait < 0:
        raise ValueError('DSS_TERM_WAIT must be finite and nonnegative')
    selector = selectors.DefaultSelector()
    try:
        for name in ('stdout', 'stderr'):
            try:
                logs[name] = _open_archive(directory / ('runner_' + name + '.log'))
            except OSError as exc:
                logs[name] = None
                errors.append(f'{name}: {exc!r}')
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        result['pid'] = proc.pid
        streams = [proc.stdout, proc.stderr]
        for name, stream in (('stdout', proc.stdout), ('stderr', proc.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        if on_spawn:
            on_spawn()
        while True:
            now = time.monotonic()
            rc = proc.poll()
            if rc is not None:
                result['rc'] = rc
                result['signal'] = -rc if rc < 0 else None
            if pending_signals:
                stopping = now
            if deadline is not None and now - start >= deadline and rc is None:
                deadline_fired = True
            if stopping is None and (rc is not None or deadline_fired):
                # Grace is only for a deadline, never an unbounded drain.
                stopping = now + (grace if deadline_fired else 0)
            if stopping is not None and now >= stopping and killed_at is None:
                _signal_group(proc.pid, signal.SIGTERM)
                killed_at = now + term_wait
            if killed_at is not None and now >= killed_at:
                _signal_group(proc.pid, signal.SIGKILL)
                if rc is not None and not selector.get_map():
                    break
                if now >= killed_at + 2:
                    errors.append('pipe EOF absent after group retirement')
                    break
            # Bound work per event: an active producer cannot starve the
            # deadline or the other stream by keeping one pipe readable.
            for key, _ in selector.select(0.02):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    _close(key.fileobj, 'stream', errors)
                    continue
                name = key.data
                handle = logs[name]
                if handle is not None:
                    try:
                        view = memoryview(chunk)
                        while view:
                            count = handle.write(view)
                            if not count:
                                raise OSError('short archive write')
                            view = view[count:]
                            result[name + '_archived_bytes'] += count
                        os.fsync(handle.fileno())
                    except OSError as exc:
                        errors.append(f'{name}: {exc!r}')
                        _close(handle, f'close {name}', errors)
                        logs[name] = None
    except Exception as exc:
        errors.append(f'execution: {exc!r}')
    finally:
        if proc is not None:
            # Independent stages preserve primary facts despite cleanup errors.
            for attempt in range(2):
                try:
                    _signal_group(proc.pid, signal.SIGKILL)
                    break
                except Exception as exc:
                    errors.append(f'retire signal: {exc!r}')
            for attempt in range(2):
                try:
                    proc.wait(timeout=5)
                    break
                except Exception as exc:
                    errors.append(f'retire wait: {exc!r}')
            if proc.returncode is not None:
                result['rc'] = proc.returncode
                result['signal'] = -proc.returncode if proc.returncode < 0 else None
            until = time.monotonic() + 5
            reap_failures = 0
            while proc.returncode is not None:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                except Exception as exc:
                    errors.append(f'reap: {exc!r}')
                    reap_failures += 1
                    if reap_failures >= 2:
                        break
                    continue
                if not pid:
                    if time.monotonic() >= until:
                        errors.append('reap: owned descendants did not retire')
                        break
                    time.sleep(0.01)
        for stream in streams:
            _close(stream, 'stream', errors)
        for name, handle in list(logs.items()):
            if handle is not None:
                _close(handle, f'close {name}', errors)
                logs[name] = None
        _close(selector, 'selector', errors)
    result['duration_seconds'] = time.monotonic() - start
    return result, deadline_fired, errors


def _run(command, root, deadline_seconds, deadline_drain_seconds, restore,
         extra_env=None, restore_marker=None, ready_fd=None, campaign_signals=None, finalize_dispositions=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.supervise.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = root / 'supervision'
        directory.mkdir(exist_ok=True)
        (directory / 'report.json.done').unlink(missing_ok=True)
        if restore_marker:
            restore_marker.unlink(missing_ok=True)
        _subreaper()
        _persist_file(root / '.supervise_child.pid', f'{os.getpid()}\n'.encode())
        if ready_fd is not None:
            try:
                os.write(ready_fd, f'{os.getpid()}\n'.encode())
            except BrokenPipeError:
                pass  # Launcher loss must not cancel the campaign.
            finally:
                os.close(ready_fd)
        env = {**os.environ, **(extra_env or {})}
        report = dict(schema_version=1, started_at=time.time(), runner_cmd=command,
                      deadline_seconds=deadline_seconds, cleanup_errors=[])
        try:
            runner, fired, errors = _execute(command, directory, deadline_seconds,
                                             deadline_drain_seconds, env, campaign_signals=campaign_signals)
        except Exception as exc:
            runner = dict(command=list(command), rc=None, signal=None)
            fired, errors = False, [f'supervision: {exc!r}']
        report.update(runner=runner, deadline_fired=fired,
                      stop_reason='campaign_deadline' if fired else
                      ('runner_exited' if runner['rc'] is not None else 'supervision_failure'))
        report['cleanup_errors'].extend(errors)
        rc = 120 if fired else (runner['rc'] if runner['rc'] is not None else 1)
        if rc < 0:
            rc = 128 - rc
        # Commit runner evidence BEFORE restoration, even if restore times out.
        try:
            _persist_file(directory / 'runner_rc.json', (json.dumps(runner, indent=2) + '\n').encode())
        except OSError as exc:
            report['cleanup_errors'].append(f'runner RC persistence: {exc!r}')
        report['restore'] = dict(command=restore, dispatched=False, rc=0)
        if restore.strip():
            restore_dir = directory / 'restore'
            restore_dir.mkdir(exist_ok=True)
            def dispatched():
                report['restore']['dispatched'] = True
                if restore_marker:
                    try:
                        _persist_file(restore_marker, b'1')
                    except OSError as exc:
                        report['cleanup_errors'].append(f'restore marker: {exc!r}')
            try:
                restored, timed_out, errs = _execute(shlex.split(restore), restore_dir, 120, 0, env, dispatched)
                report['restore'].update(rc=restored['rc'], deadline_fired=timed_out)
                report['cleanup_errors'].extend('restore: ' + e for e in errs)
                if timed_out or restored['rc'] != 0:
                    report['cleanup_errors'].append(f"restore: rc={restored['rc']}, timeout={timed_out}")
            except Exception as exc:
                report['cleanup_errors'].append(f'restore: {exc!r}')
        # Disposition cleanup belongs inside the lock and before final evidence.
        collect_signals = finalize_dispositions(report) if finalize_dispositions else lambda: None
        report.update(runner_exit_code=rc, finished_at=time.time())
        failures = 0
        committed = False
        for attempt in range(5):
            collect_signals()
            if campaign_signals:
                evidence = 'campaign signals: ' + ','.join(map(str, campaign_signals))
                report['cleanup_errors'] = [e for e in report['cleanup_errors']
                                            if not e.startswith('campaign signals: ')]
                report['cleanup_errors'].append(evidence)
            if rc == 0 and report['cleanup_errors']:
                rc = 1
            report['runner_exit_code'] = rc
            # Retry serialization must include the preceding failed commit.
            payload = (json.dumps(report, indent=2) + '\n').encode()
            try:
                _persist_file(directory / 'report.json', payload)
            except OSError as exc:
                report['cleanup_errors'].append(f'final report commit: {exc!r}')
                failures += 1
                if failures >= 2:
                    break
                continue
            collect_signals()
            if campaign_signals:
                evidence = 'campaign signals: ' + ','.join(map(str, campaign_signals))
                if evidence not in report['cleanup_errors']:
                    continue
            committed = True
            break
        if not committed:
            # Withhold completion over stale or uncommitted evidence.
            return rc, report
        # Final ownership cutoff: later signals remain contained until release,
        # but cannot reopen an already completed campaign's publication.
        _persist_file(directory / 'report.json.done', b'1')
        return rc, report


def run(command, root, deadline_seconds, deadline_drain_seconds, restore,
        extra_env=None, restore_marker=None, ready_fd=None):
    received = []
    original = {}
    contained = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    old_mask = None
    def record_signal(sig, frame):
        if sig not in received:
            received.append(sig)
    def collect_signals():
        while True:
            pending = signal.sigtimedwait(contained, 0)
            if pending is None:
                break
            record_signal(pending.si_signo, None)
    def finalize_dispositions(report):
        nonlocal old_mask
        # Original/default dispositions are never live during final publication.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, contained)
        for sig, previous in original.items():
            for attempt in range(2):
                try:
                    signal.signal(sig, previous)
                    break
                except Exception as exc:
                    report['cleanup_errors'].append(f'restore disposition {int(sig)}: {exc!r}')
        return collect_signals
    try:
        for sig in contained:
            original[sig] = signal.getsignal(sig)
            signal.signal(sig, record_signal)
        return _run(command, root, deadline_seconds, deadline_drain_seconds,
                    restore, extra_env, restore_marker, ready_fd, received,
                    finalize_dispositions)
    finally:
        # Never rewrite after lock release/done. Consume post-cutoff signals
        # before unmasking, including on an abnormal persistence path.
        if old_mask is None:
            # Only abnormal pre-finalization exits need this best-effort cleanup.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, contained)
            for sig, previous in original.items():
                for attempt in range(2):
                    try:
                        signal.signal(sig, previous)
                        break
                    except Exception:
                        pass
        collect_signals()
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def daemonise(args):
    # Explicit interpreter + resolved script + ordinary run arguments. No
    # argv/path heuristics, and readiness comes only AFTER successful re-exec.
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    argv = [sys.executable, str(Path(__file__).resolve()), 'run',
            '--run', args.run, '--output', str(args.root),
            '--deadline-drain-seconds', str(args.deadline_drain_seconds),
            '--restore', args.restore, '--ready-fd', str(write_fd)]
    if args.deadline_min is not None:
        argv += ['--deadline-min', str(args.deadline_min)]
    if args.restore_marker:
        argv += ['--restore-marker', str(args.restore_marker)]
    for entry in args.env:
        argv += ['--env', entry]
    first = os.fork()
    if first == 0:
        try:
            os.close(read_fd)
            os.setsid()
            if os.fork():
                os._exit(0)
            args.root.mkdir(parents=True, exist_ok=True)
            null = os.open(os.devnull, os.O_RDWR)
            err = os.open(args.root / 'supervisor_stderr.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            for fd in (0, 1):
                os.dup2(null, fd)
            os.dup2(err, 2)
            os.close(null)
            os.close(err)
            os.execv(sys.executable, argv)
        except BaseException:
            os._exit(127)
    os.close(write_fd)
    try:
        os.waitpid(first, 0)
        with selectors.DefaultSelector() as selector:
            selector.register(read_fd, selectors.EVENT_READ)
            if not selector.select(10):
                raise RuntimeError('detached readiness timed out')
            raw = os.read(read_fd, 64)
        if not raw:
            raise RuntimeError('detached supervisor failed before readiness')
        return int(raw)
    finally:
        os.close(read_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'daemonise'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-min', type=float)
    parser.add_argument('--deadline-drain-seconds', type=float, default=1)
    parser.add_argument('--restore', default='')
    parser.add_argument('--restore-marker', type=Path)
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('--ready-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for value in (args.deadline_min, args.deadline_drain_seconds):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error('deadline values must be finite and nonnegative')
    try:
        command = shlex.split(args.run)
        env = dict(entry.split('=', 1) for entry in args.env)
        if not command or any(not key for key in env):
            raise ValueError('nonempty command and environment keys required')
    except ValueError as exc:
        parser.error(str(exc))
    args.root = Path(args.output).expanduser().resolve()
    if args.restore_marker:
        args.restore_marker = args.restore_marker.resolve()
    try:
        if args.mode == 'daemonise':
            print(f'DAEMONISED_CHILD_PID={daemonise(args)}', flush=True)
            return 0
        return run(command, args.root,
                   None if args.deadline_min is None else args.deadline_min * 60,
                   args.deadline_drain_seconds, args.restore, env,
                   args.restore_marker, args.ready_fd)[0]
    except Exception as exc:
        print(f'SUPERVISION_FAILED={exc!r}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
