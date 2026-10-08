"""Real CLI lifecycle regressions; every case has an external process watchdog.

The case process is a Linux subreaper: detached supervisors are adopted and
reaped. Watchdog cleanup enumerates descendants across session boundaries;
killpg alone would miss the very double-fork being tested. No GPU/service I/O.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / 'tools/bench/durable_supervise.py'
PAYLOAD = bytes(range(256)) * 2048


def command(code, *args):
    return shlex.join([sys.executable, '-c', code, *map(str, args)])


def children(pid):
    try:
        return [int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except FileNotFoundError:
        return []


def kill_tree(pid):
    descendants = children(pid)
    for child in descendants:
        kill_tree(child)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_for(predicate, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('bounded condition wait expired')


def cli(root, code, mode='run', extra=(), restore=None):
    argv = [sys.executable, str(SUP), mode, '--run', code, '--output', str(root),
            '--deadline-drain-seconds', '0', *extra]
    if restore is not None:
        argv += ['--restore', restore]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**os.environ, 'DSS_TERM_WAIT': '.05'})
    try:
        out, err = p.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        kill_tree(p.pid)
        p.communicate()
        raise AssertionError(f'CLI watchdog expired: {argv}')
    print('CLI', shlex.join(argv), 'RC', p.returncode, 'STDOUT', out, 'STDERR', err, flush=True)
    return p.returncode, out


def report(root):
    return json.loads((root / 'supervision/report.json').read_text())


def verify(root, rc, stdout=b'', stderr=b''):
    data = report(root)
    assert data['runner']['rc'] == rc, data
    assert json.loads((root / 'supervision/runner_rc.json').read_text()) == data['runner']
    assert (root / 'supervision/runner_stdout.log').read_bytes() == stdout
    assert (root / 'supervision/runner_stderr.log').read_bytes() == stderr
    assert (root / 'supervision/report.json.done').exists()
    print('REPORT', json.dumps(data, sort_keys=True), flush=True)
    return data


def run_case(case, root):
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    os.environ['DSS_TERM_WAIT'] = '.05'
    if case == 'identity':
        # A stale pidfile and done marker must NOT be used as a new launch's
        # identity/readiness. Each new supervisor runs actual campaign code.
        root.mkdir(exist_ok=True)
        (root / '.supervise_child.pid').write_text('123456789')
        (root / 'supervision').mkdir()
        (root / 'supervision/report.json.done').write_text('stale')
        identities = []
        for value in (b'first', b'second'):
            rc, out = cli(root, command(f'import os; os.write(1,{value!r})'), 'daemonise')
            assert rc == 0
            pid = int(out.strip().split(b'=')[1])
            identities.append(pid)
            assert int((root / '.supervise_child.pid').read_text()) == pid
            wait_for(lambda: (root / 'supervision/report.json.done').exists())
            verify(root, 0, value)
            wait_for(lambda: Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z' if Path(f'/proc/{pid}/stat').exists() else True)
            assert os.waitpid(pid, 0)[0] == pid
        assert len(set(identities)) == 2
    elif case in ('parent_success', 'parent_deadline'):
        # Living launcher acknowledges the CLI PID, then waits. Runner retains
        # an adverse gate until that launcher is killed and reaped.
        gate = root / 'gate'
        ready = root / 'runner_ready'
        marker = root / 'restored'
        code = "import pathlib,time,os,signal,sys; "
        if case == 'parent_deadline':
            code += "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        code += f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        gated = f"while not pathlib.Path({str(gate)!r}).exists(): time.sleep(.01)"
        code += f"exec({gated!r}); "
        code += "os.write(1,b'after-parent'); os.write(2,b'err'); sys.exit(23)"
        argv = [sys.executable, str(SUP), 'daemonise', '--run', command(code),
                '--output', str(root), '--restore', command('pass'),
                '--restore-marker', str(marker), '--deadline-drain-seconds', '0']
        if case == 'parent_deadline':
            argv += ['--deadline-min', '.03']
        launcher = subprocess.Popen([sys.executable, '-c',
            'import subprocess,sys,time; p=subprocess.run(sys.argv[1:],capture_output=True); '
            'sys.stdout.buffer.write(p.stdout); sys.stdout.flush(); time.sleep(300)', *argv],
            stdout=subprocess.PIPE, start_new_session=True)
        line = launcher.stdout.readline()
        assert line.startswith(b'DAEMONISED_CHILD_PID='), line
        supervisor = int(line.split(b'=')[1])
        wait_for(ready.exists)
        runner = int(ready.read_text())
        assert launcher.poll() is None
        assert os.getsid(supervisor) != launcher.pid
        assert not marker.exists()
        os.killpg(launcher.pid, signal.SIGKILL)
        launcher.wait(timeout=2)
        launcher.stdout.close()
        if case == 'parent_success':
            gate.touch()  # completion is causally AFTER the parent's death
        wait_for(lambda: (root / 'supervision/report.json.done').exists(), 6)
        data = verify(root, 23 if case == 'parent_success' else -9,
                      b'after-parent' if case == 'parent_success' else b'',
                      b'err' if case == 'parent_success' else b'')
        assert marker.read_bytes() == b'1'
        assert data['restore']['dispatched'] and data['restore']['rc'] == 0
        assert data['runner_exit_code'] == (23 if case == 'parent_success' else 120)
        wait_for(lambda: not Path(f'/proc/{runner}').exists())
        os.waitpid(supervisor, 0)
    elif case in ('deadline_cooperative', 'deadline_noncooperative'):
        ready = root / 'ready'
        code = 'import signal,time,pathlib,os; '
        if case.endswith('noncooperative'):
            code += 'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        code += f'pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)'
        rc, _ = cli(root, command(code), extra=['--deadline-min', '.01'], restore=command('pass'))
        assert ready.exists()  # signal behaviour installed before deadline
        assert rc == 120
        data = verify(root, -9 if case.endswith('noncooperative') else -15)
        assert data['deadline_fired'] and data['stop_reason'] == 'campaign_deadline'
        assert not Path(f'/proc/{int(ready.read_text())}').exists()
    elif case == 'large_capture':
        rc, _ = cli(root, command('import os; data=bytes(range(256))*2048; '
            'os.write(1,data); os.write(2,data[::-1]); raise SystemExit(42)'), restore=command('raise SystemExit(7)'))
        assert rc == 42
        data = verify(root, 42, PAYLOAD, PAYLOAD[::-1])
        assert data['runner']['stdout_archived_bytes'] == len(PAYLOAD)
        assert data['restore']['rc'] == 7 and data['cleanup_errors']
    elif case in ('signal_term', 'signal_hup', 'signal_int'):
        # A real signal delivered to the LIVE supervisor (its own process) must
        # not bypass durable completion: the runner still retires through the
        # campaign, the known RC/deadline evidence is preserved, the restore is
        # still dispatched, and descriptors are retired.
        sig = {'signal_term': signal.SIGTERM, 'signal_hup': signal.SIGHUP,
               'signal_int': signal.SIGINT}[case]
        ready = root / 'ready'
        descendant = root / 'signal_descendant'
        body = (f'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                f'pathlib.Path({str(descendant)!r}).write_text(str(os.getpid())); time.sleep(300)')
        gate = f'while not pathlib.Path({str(descendant)!r}).exists(): time.sleep(.01)'
        code = ('import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                f'pid=os.fork(); exec({body!r}) if pid==0 else exec({gate!r}); '
                f'pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(300)')
        marker = root / 'restored'
        argv = [sys.executable, str(SUP), 'run', '--run', command(code),
                '--output', str(root), '--restore', command(
                    f'import pathlib,time; pathlib.Path({str(root / "restore_ready")!r}).touch(); '
                    f'exec("while not pathlib.Path({str(root / "restore_gate")!r}).exists(): time.sleep(.01)")'),
                '--restore-marker', str(marker), '--deadline-drain-seconds', '0',
                '--env', 'DSS_TERM_WAIT=.05']
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=dict(os.environ), start_new_session=True)
        try:
            wait_for(ready.exists, 10)
            runner_pid = int(ready.read_text())  # the runner process
            # The live supervisor is the runner's parent (field 4 of /proc stat);
            # a signal to it exercises the campaign handler that must contain it
            # through durable completion.
            supervisor_pid = int(Path(f'/proc/{runner_pid}/stat').read_text().split()[3])
            assert supervisor_pid == p.pid  # PPID, not runner pgrp
            os.kill(supervisor_pid, sig)
            wait_for((root / 'restore_ready').exists)
            assert p.poll() is None and not (root / 'supervision/report.json.done').exists()
            for late_sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                os.kill(p.pid, late_sig)
            (root / 'restore_gate').touch()
            out, err = p.communicate(timeout=8)
            print('SIGNAL_CHILD_RC', p.returncode, 'SIGNAL', sig, flush=True)
            assert p.returncode == 137, (p.returncode, out, err)
            data = verify(root, -9)
            assert data['stop_reason'] == 'runner_exited' and not data['deadline_fired'], data
            assert not Path(f'/proc/{runner_pid}').exists()
            assert not Path(f'/proc/{int(descendant.read_text())}').exists()
            assert data['restore']['dispatched'] and data['restore']['rc'] == 0, data
            assert marker.read_bytes() == b'1'
            assert any('campaign signals' in e for e in data['cleanup_errors']), data['cleanup_errors']
            assert data['runner_exit_code'] == 137
        finally:
            if p.poll() is None:
                kill_tree(p.pid)
                p.communicate(timeout=3)
    elif case in ('descendant_idle', 'descendant_active', 'descendant_escaped'):
        childfile = root / 'descendant'
        code = f"import os,signal,time,pathlib; pid=os.fork(); "
        body = ("os.setsid(); " if case == "descendant_escaped" else "")
        body += f"signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(childfile)!r}).write_text(str(os.getpid())); "
        body += "exec(" + repr("while True: os.write(1,bytes([65])*4096)") + ")" if case == 'descendant_active' else 'time.sleep(300)'
        parent_body = f"while not pathlib.Path({str(childfile)!r}).exists(): time.sleep(.01)"
        code += f"exec({body!r}) if pid==0 else exec({parent_body!r}); "
        code += "raise SystemExit(31)"
        rc, _ = cli(root, command(code))
        assert rc == 31
        data = report(root)
        assert data['runner']['rc'] == 31 and not data['cleanup_errors'], data
        assert not Path(f'/proc/{int(childfile.read_text())}').exists()
        if case == 'descendant_active':
            archived = (root / 'supervision/runner_stdout.log').read_bytes()
            assert archived and set(archived) == {65}
    elif case in ('restore_failure', 'spawn_failure', 'sink_failure'):
        marker = root / 'marker'
        if case == 'sink_failure':
            (root / 'supervision').mkdir(parents=True)
            (root / 'supervision/runner_stdout.log').symlink_to('/dev/full')
            rc, _ = cli(root, command('import os; os.write(1,b"x"*200000); raise SystemExit(23)'),
                        extra=['--restore-marker', str(marker)], restore=command('pass'))
            assert rc == 23
            data = report(root)
            assert data['runner']['rc'] == 23 and data['cleanup_errors']
            assert marker.exists()
        elif case == 'spawn_failure':
            rc, _ = cli(root, '/definitely/not/a/runner',
                        extra=['--restore-marker', str(marker)], restore=command('pass'))
            assert rc == 1 and marker.exists()
            assert report(root)['stop_reason'] == 'supervision_failure'
        else:
            for restore in (command('raise SystemExit(7)'), '/definitely/not/a/restore'):
                rc, _ = cli(root, command('pass'), restore=restore)
                assert rc == 1
                data = verify(root, 0)
                assert data['cleanup_errors']
    elif case == 'exclusive_output':
        ready = root / 'ready'
        code = command(f'import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(300)')
        rc, out = cli(root, code, 'daemonise', extra=['--deadline-min', '.025'])
        assert rc == 0
        pid = int(out.strip().split(b'=')[1])
        wait_for(ready.exists)
        first_identity = (root / '.supervise_child.pid').read_text()
        rc, _ = cli(root, command('pass'), 'daemonise')
        assert rc == 1
        assert (root / '.supervise_child.pid').read_text() == first_identity
        wait_for(lambda: (root / 'supervision/report.json.done').exists())
        assert report(root)['runner_exit_code'] == 120
        os.waitpid(pid, 0)
    elif case == 'env_and_arguments':
        marker = root / 'marker with spaces'
        rc, out = cli(root, command('import os; os.write(1,os.environ["PROOF_VALUE"].encode())'), 'daemonise',
                      extra=['--env', 'PROOF_VALUE=a b', '--restore-marker', str(marker)], restore=command('pass'))
        assert rc == 0
        pid = int(out.strip().split(b'=')[1])
        wait_for(lambda: (root / 'supervision/report.json.done').exists())
        verify(root, 0, b'a b')
        assert marker.exists()
        os.waitpid(pid, 0)
    elif case == 'malformed':
        for extra in (['--deadline-min', '-1'], ['--deadline-min', 'nan'],
                      ['--deadline-drain-seconds', 'inf'], ['--env', 'broken']):
            rc, _ = cli(root, command('pass'), extra=extra)
            assert rc == 2
        rc, _ = cli(root, ' ')
        assert rc == 2
        assert not (root / 'supervision').exists()
    elif case == 'blocked_sink':
        (root / 'supervision').mkdir(parents=True)
        os.mkfifo(root / 'supervision/runner_stdout.log')  # no reader, retained
        rc, _ = cli(root, command('import os; os.write(1,b"x"*200000)'), restore=command('pass'))
        assert rc == 1
        data = report(root)
        assert data['runner']['rc'] == 0 and data['cleanup_errors']
        assert data['restore']['rc'] == 0
    elif case == 'restore_timeout':
        sys.path.insert(0, str(ROOT))
        from tools.bench import durable_supervise as module
        real_execute = module._execute
        def short_restore(command, directory, deadline, *args, **kwargs):
            return real_execute(command, directory, .3 if directory.name == 'restore' else deadline, *args, **kwargs)
        module._execute = short_restore
        marker = root / 'marker'
        rc, data = module.run([sys.executable, '-c', 'raise SystemExit(23)'], root, None, 0,
            command('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)'),
            restore_marker=marker)
        assert rc == 23 and marker.exists()
        assert data['restore']['deadline_fired'] and data['restore']['rc'] == -9
        assert data['cleanup_errors'] and not children(os.getpid())
        print('RESTORE_TIMEOUT_PRIMARY_PRESERVED', json.dumps(data), flush=True)
    elif case.startswith('failure_'):
        sys.path.insert(0, str(ROOT))
        from tools.bench import durable_supervise as mod
        primary, fault = case.removeprefix('failure_').rsplit('_', 1)
        before = set(os.listdir('/proc/self/fd'))
        original_signals = {sig: signal.getsignal(sig) for sig in
                            (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        injected = []
        def poison():
            if not injected:
                injected.append(fault)
                raise OSError('injected ' + fault)
        real_signal, real_wait, real_reap = mod._signal_group, subprocess.Popen.wait, os.waitpid
        real_disposition = signal.signal
        real_open = mod._open_archive
        handles = []
        class Archive:
            def __init__(self, handle):
                self.handle = handle
                handles.append(handle)
            def __getattr__(self, name):
                return getattr(self.handle, name)
            def close(self):
                poison()  # fail before closure; require independent retry
                self.handle.close()
        if fault == 'retire':
            import inspect
            lines, first = inspect.getsourcelines(mod._execute)
            final_kill_line = max(first + index for index, line in enumerate(lines)
                                  if '_signal_group(proc.pid, signal.SIGKILL)' in line)
            def bad_signal(pid, sig):
                # Inject in final retirement only, never a preceding loop KILL.
                if sig == signal.SIGKILL and sys._getframe(1).f_lineno == final_kill_line:
                    poison()
                return real_signal(pid, sig)
            mod._signal_group = bad_signal
        elif fault == 'wait':
            def bad_wait(self, *args, **kwargs):
                poison()
                return real_wait(self, *args, **kwargs)
            subprocess.Popen.wait = bad_wait
        elif fault == 'reap':
            def bad_reap(pid, options):
                if pid == -1:
                    poison()
                return real_reap(pid, options)
            os.waitpid = bad_reap
        elif fault == 'close':
            mod._open_archive = lambda path: Archive(real_open(path))
        elif fault == 'disposition':
            def bad_disposition(sig, handler):
                if handler == original_signals[sig]:
                    poison()
                return real_disposition(sig, handler)
            signal.signal = bad_disposition
        else:
            raise AssertionError(fault)
        deadline = primary == 'deadline'
        ready = root / 'runner_ready'
        code = ('import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                f'pathlib.Path({str(ready)!r}).touch(); time.sleep(300)') if deadline else 'raise SystemExit(23)'
        marker = root / 'marker'
        try:
            rc, data = mod.run([sys.executable, '-c', code], root, .3 if deadline else None,
                               0, command('pass'), restore_marker=marker)
        finally:
            mod._signal_group, subprocess.Popen.wait, os.waitpid = real_signal, real_wait, real_reap
            mod._open_archive = real_open
            signal.signal = real_disposition
        assert injected == [fault]
        assert rc == (120 if deadline else 23), data
        assert data['runner']['rc'] == (-9 if deadline else 23), data
        assert data['deadline_fired'] == deadline
        assert data['stop_reason'] == ('campaign_deadline' if deadline else 'runner_exited')
        assert any('injected ' + fault in e for e in data['cleanup_errors']), data
        assert data['restore']['dispatched'] and data['restore']['rc'] == 0
        assert marker.read_bytes() == b'1'
        assert json.loads((root / 'supervision/runner_rc.json').read_text()) == data['runner']
        assert (root / 'supervision/report.json.done').exists()
        assert all(handle.closed for handle in handles)
        assert set(os.listdir('/proc/self/fd')) == before
        assert not children(os.getpid())
        assert all(signal.getsignal(sig) == prev for sig, prev in original_signals.items())
        if deadline:
            assert ready.exists()
        print('PRIMARY_CLOSURES_RESTORE_PROVEN', case, json.dumps(data), flush=True)
    elif case.startswith('publication_'):
        import importlib.util
        import fcntl
        source = Path(os.environ.get('DSS_PROOF_SUPERVISOR', str(SUP)))
        spec = importlib.util.spec_from_file_location('publication_owner', source)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mode = case.removeprefix('publication_')
        real_persist, real_signal = mod._persist_file, signal.signal
        original = {sig: signal.getsignal(sig) for sig in
                    (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        before = set(os.listdir('/proc/self/fd'))
        before_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        injected, attempts, done = [], [], []
        def disposition(sig, handler):
            if mode == 'combined' and handler == original[sig] and not injected:
                injected.append('disposition')
                raise OSError('injected disposition')
            return real_signal(sig, handler)
        def persist(path, payload):
            if path.name in ('report.json', 'report.json.done'):
                # Probe the actual flock with an independent open description.
                with (root / '.supervise.lock').open('a') as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        raise AssertionError('publication outside ownership')
            if path.name == 'report.json':
                assert not (path.parent / 'report.json.done').exists(), 'early done'
                attempts.append(json.loads(payload))
                if mode == 'combined':
                    assert injected == ['disposition'], 'disposition evidence too late'
                if len(attempts) == 1 and mode in ('combined', 'late_signal'):
                    os.kill(os.getpid(), signal.SIGTERM)
                if mode == 'exhausted' or (len(attempts) == 1 and mode != 'late_signal'):
                    raise OSError('injected report persistence')
            if path.name == 'report.json.done':
                persisted = report(root)
                assert persisted == attempts[-1]
                if mode in ('combined', 'stale'):
                    assert any('injected report persistence' in e for e in persisted['cleanup_errors'])
                if mode in ('combined', 'late_signal'):
                    assert any('campaign signals: 15' == e for e in persisted['cleanup_errors'])
                if mode == 'combined':
                    assert any('injected disposition' in e for e in persisted['cleanup_errors'])
                done.append(True)
            return real_persist(path, payload)
        mod._persist_file = persist
        signal.signal = disposition
        try:
            rc, data = mod.run([sys.executable, '-c', 'raise SystemExit(23)'], root,
                               None, 0, command('raise SystemExit(7)'))
        finally:
            signal.signal = real_signal
        assert rc == 23 and data['runner']['rc'] == 23
        assert data['restore']['rc'] == 7
        assert len(attempts) == 2, attempts
        if mode == 'exhausted':
            assert not done and not (root / 'supervision/report.json.done').exists()
        else:
            assert done == [True] and report(root) == data
        assert json.loads((root / 'supervision/runner_rc.json').read_text()) == data['runner']
        assert all(signal.getsignal(sig) == prev for sig, prev in original.items())
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before_mask
        assert set(os.listdir('/proc/self/fd')) == before and not children(os.getpid())
        print('FINAL_PUBLICATION_PROOF', mode, 'ATTEMPTS', json.dumps(attempts),
              'DONE', done, 'REPORT', json.dumps(data), flush=True)
    elif case == 'signal_evidence':
        sys.path.insert(0, str(ROOT))
        from tools.bench import durable_supervise as mod
        real_persist = mod._persist_file
        delivered = []
        before = set(os.listdir('/proc/self/fd'))
        original = {sig: signal.getsignal(sig) for sig in
                    (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        def interrupted_persist(path, payload):
            # Execute real signals inside the actual supervisor at each durable
            # commit stage; no timing sleep or surrogate process group.
            if path.name in ('runner_rc.json', 'report.json', 'report.json.done'):
                for sig in original:
                    delivered.append((path.name, int(sig)))
                    os.kill(os.getpid(), sig)
            real_persist(path, payload)
        mod._persist_file = interrupted_persist
        marker = root / 'marker'
        rc, data = mod.run([sys.executable, '-c', 'raise SystemExit(23)'], root,
                           None, 0, command('pass'), restore_marker=marker)
        assert rc == 23 and data['runner']['rc'] == 23
        assert len(delivered) == 9
        assert data['restore']['rc'] == 0 and marker.read_bytes() == b'1'
        assert (root / 'supervision/report.json.done').exists()
        assert json.loads((root / 'supervision/runner_rc.json').read_text()) == data['runner']
        assert all(signal.getsignal(sig) == prev for sig, prev in original.items())
        assert set(os.listdir('/proc/self/fd')) == before and not children(os.getpid())
        print('SIGNALS_CONTAINED_DURING_EVIDENCE', delivered, json.dumps(data), flush=True)
    elif case == 'marker_failure':
        marker, restored = root / 'marker', root / 'actual_restore'
        rc, _ = cli(root, command(f'import pathlib; pathlib.Path({str(marker)!r}).mkdir(); raise SystemExit(23)'),
                    extra=['--restore-marker', str(marker)],
                    restore=command(f'import pathlib; pathlib.Path({str(restored)!r}).touch()'))
        assert rc == 23 and restored.exists()
        assert any('restore marker' in e for e in report(root)['cleanup_errors'])
        assert report(root)['restore']['rc'] == 0
    elif case == 'descriptor_retirement':
        sys.path.insert(0, str(ROOT))
        from tools.bench.durable_supervise import run
        before = set(os.listdir('/proc/self/fd'))
        for index in range(3):
            rc, data = run([sys.executable, '-c', 'print("finite")'], root / str(index), None, 0, '')
            assert rc == 0 and data['runner']['rc'] == 0
            assert set(os.listdir('/proc/self/fd')) == before
            assert not children(os.getpid())
        print('DESCRIPTORS_EXACTLY_RETIRED', sorted(before), flush=True)
    elif case == 'corpus_invariance':
        # Full tracked tree equality is stronger than checking insertion counts:
        # deletion, replacement or inference/profile edits all fail this proof.
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=ROOT, check=True)
        paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).split(b'\0')
        digest = hashlib.sha256()
        for raw in paths:
            if raw:
                path = os.fsdecode(raw)
                expected = subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=ROOT)
                actual = (ROOT / path).read_bytes()
                assert actual == expected, path
                digest.update(raw + b'\0' + hashlib.sha256(actual).digest())
        print('TRACKED_CORPUS_INFERENCE_PROFILE_BYTE_IDENTITY', digest.hexdigest(), flush=True)
    else:
        raise AssertionError(case)
    # Includes adopted supervisor and orphaned intermediate children; every
    # lifecycle case must actually retire/reap resources before PASS.
    until = time.monotonic() + 3
    while children(os.getpid()):
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if not pid:
            assert time.monotonic() < until, children(os.getpid())
            time.sleep(.01)
    assert not children(os.getpid())
    print('NO_OWNED_CHILDREN', case, flush=True)


CASES = ('identity', 'parent_success', 'parent_deadline', 'deadline_cooperative',
         'deadline_noncooperative', 'large_capture', 'descendant_idle',
         'descendant_active', 'descendant_escaped', 'restore_failure', 'spawn_failure', 'sink_failure',
         *(f'failure_{primary}_{fault}' for primary in ('completed_nonzero', 'deadline')
           for fault in ('retire', 'wait', 'reap', 'close', 'disposition')),
         'publication_combined', 'publication_stale', 'publication_exhausted', 'publication_late_signal',
         'signal_term', 'signal_hup', 'signal_int', 'signal_evidence', 'exclusive_output', 'env_and_arguments', 'malformed', 'marker_failure',
         'descriptor_retirement', 'blocked_sink', 'restore_timeout', 'corpus_invariance')


@pytest.mark.parametrize('case', CASES)
def test_durable_supervision_lifecycle(case, tmp_path):
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), case, str(tmp_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = child.communicate(timeout=25)
    except subprocess.TimeoutExpired:
        kill_tree(child.pid)
        out, err = child.communicate(timeout=3)
        pytest.fail(f'external watchdog expired {case}: {out!r} {err!r}')
    print(out.decode(errors='backslashreplace'), end='')
    assert child.returncode == 0, (out, err)
    assert b'PASS ' + case.encode() in out


if __name__ == '__main__':
    try:
        run_case(sys.argv[1], Path(sys.argv[2]))
        print('PASS', sys.argv[1], flush=True)
    finally:
        for pid in children(os.getpid()):
            kill_tree(pid)
        while True:
            try:
                os.waitpid(-1, 0)
            except ChildProcessError:
                break
