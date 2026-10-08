"""Host-only capture proofs, each isolated behind an external process watchdog."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.bench.stream_capture import StreamCapture
from tools.bench.run_serve_corpus import CampaignError, RunningServer


def retired(capture):
    assert not capture.worker.is_alive()
    assert capture.source.closed and capture._archive.closed
    for fd in (capture._wake_read, capture._wake_write):
        with pytest.raises(OSError):
            os.fstat(fd)


def run_case(case, root):
    baseline_fds = len(os.listdir('/proc/self/fd'))
    path = root / 'archive'
    if case in ('idle', 'active'):
        read, write = os.pipe()
        ack_read, ack_write = os.pipe()
        gate_read, gate_write = os.pipe()
        payload = (bytes(range(256)) * 16)[8:]
        prefix = (0).to_bytes(8, "little") + payload
        pid = os.fork()
        if pid == 0:
            os.close(read)
            os.close(ack_read)
            os.close(gate_write)
            os.write(write, prefix)
            os.write(ack_write, b'A')  # Acknowledged before the cutoff request.
            if case == 'active':
                try:
                    sequence = 1
                    while True:
                        os.write(write, sequence.to_bytes(8, 'little') + payload)
                        sequence += 1
                except BrokenPipeError:
                    pass
            os.read(gate_read, 1)  # Retain producer even after reader retirement.
            os._exit(0)
        os.close(write)
        os.close(ack_write)
        os.close(gate_read)
        capture = StreamCapture(os.fdopen(read, 'rb', buffering=0), path)
        try:
            assert os.read(ack_read, 1) == b'A'
            status = capture.finish(0.15)
            assert os.waitpid(pid, os.WNOHANG) == (0, 0)
            assert status['capture_stop_reason'] == 'drain_timeout'
            assert not status['capture_complete'] and not status['capture_failed']
            data = path.read_bytes()
            assert len(data) == status['archived_bytes']
            assert data[:len(prefix)] == prefix
            expected = b''.join(i.to_bytes(8, 'little') + payload
                                for i in range((len(data) + len(prefix) - 1) // len(prefix)))
            assert data == expected[:len(data)]
            if case == 'idle':
                assert data == prefix
            else:
                assert len(data) > len(prefix)  # Writer actually progressed during drain.
            retired(capture)
            assert capture.finish(0) == status
        finally:
            os.close(ack_read)
            os.write(gate_write, b'X')
            os.close(gate_write)
            assert os.waitpid(pid, 0)[1] == 0
    elif case in ('finite', 'closed_console', 'blocked_console', 'archive_failure'):
        console = None
        console_read = console_write = None
        if case in ('closed_console', 'blocked_console'):
            console_read, console_write = os.pipe()
            if case == 'closed_console':
                os.close(console_read)
                console_read = None
            else:
                os.set_blocking(console_write, False)
                try:
                    while True:
                        os.write(console_write, b'c' * 4096)
                except BlockingIOError:
                    pass
                os.set_blocking(console_write, True)
            console = console_write
        producer = subprocess.Popen([sys.executable, '-c',
            "import os; data=bytes(range(256))*8192+bytes([0])+b'tail'+bytes([255]); "
            "v=memoryview(data)\n"
            "while v:\n n=os.write(1,v)\n v=v[n:]"], stdout=subprocess.PIPE)
        assert producer.stdout is not None
        capture = StreamCapture(producer.stdout, Path('/dev/full') if case == 'archive_failure' else path, console)
        if case == 'archive_failure':
            capture.worker.join()
            status = capture.finish()
            assert status['capture_failed'] and not status['capture_complete']
            assert status['capture_stop_reason'] == 'capture_error'
            assert status['capture_error'] and status['archived_bytes'] == 0
        else:
            assert producer.wait(timeout=4) == 0
            status = capture.finish(1)
            data = bytes(range(256)) * 8192 + b'\x00tail\xff'
            assert path.read_bytes() == data
            assert status['archived_bytes'] == len(data)
            assert status['capture_complete'] and not status['capture_failed']
            assert status['capture_stop_reason'] == 'eof'
            if console is not None:
                assert status['console_disabled']
                assert os.get_blocking(console_write)  # Caller flags unchanged.
        retired(capture)
        producer.wait(timeout=4)
        if console_write is not None:
            os.close(console_write)
        if console_read is not None:
            os.close(console_read)
    elif case in ('rc', 'primary', 'failure_without_primary', 'setup_failure', 'disabled'):
        kwargs = {} if case == 'disabled' else dict(
            capture_paths=(root / 'stdout', root / 'stderr'), capture_drain_timeout=0.2)
        if case in ('primary', 'failure_without_primary'):
            kwargs['capture_paths'] = (Path('/dev/full'), root / 'stderr')
        if case == 'setup_failure':
            kwargs['capture_paths'] = (root / 'stdout', root / 'missing' / 'stderr')
        server = RunningServer([sys.executable, '-c',
            "import os; os.write(1,b'OUT'); os.write(2,b'ERR'); raise SystemExit(23)"],
            '127.0.0.1', 0, root / 'log', **kwargs)
        primary = KeyboardInterrupt('original cancellation')
        if case == 'setup_failure':
            with pytest.raises(FileNotFoundError):
                server.__enter__()
            assert server.process.poll() is not None
            assert server.capture_status[-1]['capture_failed']
            assert server.capture_status[-1]['capture_stop_reason'] == 'setup_error'
        elif case == 'primary':
            with pytest.raises(KeyboardInterrupt) as caught:
                with server:
                    server.process.wait(timeout=4)
                    raise primary
            assert caught.value is primary
            assert any('archival failed' in note for note in primary.__notes__)
            assert server.process.returncode == 23
        elif case == 'failure_without_primary':
            with pytest.raises(CampaignError, match='archival failed'):
                with server:
                    server.process.wait(timeout=4)
            assert server.process.returncode == 23
        else:
            with server:
                server.process.wait(timeout=4)
            assert server.process.returncode == 23
            if case == 'disabled':
                assert server.process.stdout is None and server.process.stderr is None
                assert not server.captures
            else:
                assert (root / 'stdout').read_bytes() == b'OUT'
                assert (root / 'stderr').read_bytes() == b'ERR'
                assert all(s['capture_complete'] for s in server.capture_status)
        for capture in server.captures:
            retired(capture)
    elif case == 'partial_archive_failure':
        from unittest.mock import patch
        read, write = os.pipe()
        handle = path.open('wb', buffering=0)
        class Sink:
            closed = False
            first = True
            def write(self, data):
                if self.first:
                    self.first = False
                    return handle.write(data[:97])
                raise OSError('injected disk failure after partial write')
            def close(self):
                handle.close()
                self.closed = True
        sink = Sink()
        with patch.object(Path, 'open', return_value=sink):
            capture = StreamCapture(os.fdopen(read, 'rb', buffering=0), path)
        os.write(write, bytes(range(256)))
        capture._retired.wait()
        status = capture.finish()
        assert status['capture_failed'] and not status['capture_complete']
        assert status['archived_bytes'] == 97
        assert path.read_bytes() == bytes(range(97))
        retired(capture)
        os.close(write)
    elif case == 'boundary':
        read, write = os.pipe()
        capture = StreamCapture(os.fdopen(read, 'rb', buffering=0), path)
        for bad in (-1, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                capture.finish(bad)
            with pytest.raises(ValueError):
                RunningServer([], 'localhost', 0, root / 'log', capture_drain_timeout=bad)
        capture.finish(0)
        assert capture.status()['capture_stop_reason'] == 'drain_timeout'
        retired(capture)
        os.close(write)
    elif case == 'interrupted_finish':
        from unittest.mock import patch
        read, write = os.pipe()
        capture = StreamCapture(os.fdopen(read, 'rb', buffering=0), path)
        original_wait = capture._retired.wait
        interruption = KeyboardInterrupt('cleanup interruption')
        calls = []
        def wait(timeout=None):
            if not calls:
                calls.append(1)
                raise interruption
            return original_wait(timeout)
        with patch.object(capture._retired, 'wait', wait):
            with pytest.raises(KeyboardInterrupt) as caught:
                capture.finish(0.15)
        assert caught.value is interruption
        assert capture.status()['capture_stop_reason'] == 'drain_timeout'
        retired(capture)
        os.close(write)
    elif case in ('interrupted_join', 'interrupted_signal_join', 'interrupted_wait_join',
                  'interrupted_join_exit', 'interrupted_close_before',
                  'interrupted_close_after'):
        import threading
        from unittest.mock import patch
        read, write = os.pipe()  # Keep idle producer open through all assertions.
        gate = threading.Event()
        worker_tail_done = threading.Event()
        original_run = StreamCapture._run
        def run(capture):
            original_run(capture)
            # _retired is true, but the actual worker is deliberately still alive.
            gate.wait()
            worker_tail_done.set()
        with patch.object(StreamCapture, '_run', run):
            capture = StreamCapture(os.fdopen(read, 'rb', buffering=0), path)
        original_join = capture.worker.join
        original_wait = capture._retired.wait
        writer = getattr(capture, '_wake_writer', None)
        first = KeyboardInterrupt('first retirement interruption')
        join_error = (SystemExit('join cancellation') if case == 'interrupted_join_exit'
                      else KeyboardInterrupt('final join interruption'))
        calls = []
        wait_calls = []
        deadline = []
        signal_senders = []
        old_handler = signal.getsignal(signal.SIGINT)
        def interrupt(signum, frame):
            raise join_error
        if case == 'interrupted_signal_join':
            signal.signal(signal.SIGINT, interrupt)
        def wait(timeout=None):
            if case == 'interrupted_wait_join' and not wait_calls:
                wait_calls.append(1)
                deadline.append(capture._deadline)
                raise first
            return original_wait(timeout)
        def join(timeout=None):
            calls.append(1)
            if len(calls) == 1:
                assert capture._retired.is_set()
                assert capture.worker.ident in sys._current_frames()
                assert not worker_tail_done.is_set()
                assert capture.source.closed and capture._archive.closed
                with pytest.raises(OSError):
                    os.fstat(capture._wake_read)
                os.fstat(capture._wake_write)  # Supervisor still owns the writer.
                if case.startswith('interrupted_close'):
                    gate.set()
                    return original_join(timeout)
                if case == 'interrupted_signal_join':
                    main_ident = threading.get_ident()
                    def send_signal():
                        # No timing sleep: retain worker gate until the main
                        # frame is inside the genuine blocking Thread.join.
                        while True:
                            frame = sys._current_frames()[main_ident]
                            while frame is not None:
                                if frame.f_code is original_join.__func__.__code__:
                                    os.kill(os.getpid(), signal.SIGINT)
                                    return
                                frame = frame.f_back
                    sender = threading.Thread(target=send_signal, name='join-interrupt-injector')
                    signal_senders.append(sender)
                    sender.start()
                    return original_join(timeout)
                raise join_error
            gate.set()  # Only a retried join permits the real worker to exit.
            return original_join(timeout)
        close_calls = []
        replacement_fd = []
        closure_owners = {}
        source_close = capture.source.close
        archive_close = capture._archive.close
        raw_close = os.close
        def close_source():
            closure_owners['source'] = threading.get_ident()
            return source_close()
        def close_archive():
            closure_owners['archive'] = threading.get_ident()
            return archive_close()
        def close_fd(fd):
            if fd == capture._wake_read:
                closure_owners['reader_control'] = threading.get_ident()
            return raw_close(fd)
        class Writer:
            def close(self):
                assert threading.current_thread() is threading.main_thread()
                assert worker_tail_done.is_set()
                assert capture.worker.ident not in sys._current_frames()
                close_calls.append(1)
                if len(close_calls) == 1:
                    if case == 'interrupted_close_before':
                        raise first
                    writer.close()
                    if case == 'interrupted_close_after':
                        # Retry must not close the descriptor number's new owner.
                        fd = os.open('/dev/null', os.O_RDONLY)
                        if fd != capture._wake_write:
                            os.dup2(fd, capture._wake_write)
                            os.close(fd)
                        replacement_fd.append(capture._wake_write)
                        raise first
                writer.close()
        try:
            with (patch.object(capture.worker, 'join', join),
                  patch.object(capture._retired, 'wait', wait),
                  patch.object(capture.source, 'close', close_source),
                  patch.object(capture._archive, 'close', close_archive),
                  patch.object(os, 'close', close_fd)):
                if writer is not None:
                    capture._wake_writer = Writer()
                elif case.startswith('interrupted_close'):
                    raise AssertionError('control writer must have an idempotent owner')
                expected = (first if case == 'interrupted_wait_join' or
                            case.startswith('interrupted_close') else join_error)
                with pytest.raises(type(expected)) as caught:
                    capture.finish(0.05)
            assert caught.value is expected
            assert worker_tail_done.is_set()
            assert capture.worker.ident not in sys._current_frames()
            assert closure_owners == dict.fromkeys(
                ('source', 'archive', 'reader_control'), capture.worker.ident)
            assert capture._finished
            if deadline:
                assert capture._deadline == deadline[0]
                assert any(repr(join_error) in n for n in first.__notes__)
            if replacement_fd:
                os.fstat(replacement_fd[0])
                os.close(replacement_fd.pop())
            retired(capture)
            assert capture.finish(0) == capture.status()
            assert capture.status()['capture_stop_reason'] == 'drain_timeout'
            if writer is not None:
                assert writer.closed
            print('RETIREMENT', len(calls), 'joins; worker frame absent; finished; writer closed; identical exception', flush=True)
        finally:
            gate.set()
            original_join()
            for sender in signal_senders:
                sender.join()
                assert not sender.is_alive()
            signal.signal(signal.SIGINT, old_handler)
            for fd in replacement_fd:
                os.close(fd)
            os.close(write)
    elif case == 'primary_join':
        server = RunningServer([sys.executable, '-c',
            "import os; os.write(1,b'OUT'); os.write(2,b'ERR'); raise SystemExit(23)"],
            '127.0.0.1', 0, root / 'log',
            capture_paths=(root / 'stdout', root / 'stderr'))
        primary = KeyboardInterrupt('original context interruption')
        cleanup = KeyboardInterrupt('injected final join interruption')
        calls = []
        with pytest.raises(KeyboardInterrupt) as caught:
            with server:
                server.process.wait(timeout=4)
                capture = server.captures[0]
                original_join = capture.worker.join
                def join(timeout=None):
                    calls.append(1)
                    if len(calls) == 1:
                        raise cleanup
                    return original_join(timeout)
                # Keep injection installed through __exit__.
                capture.worker.join = join
                raise primary
        assert caught.value is primary
        assert any(repr(cleanup) in n for n in primary.__notes__)
        assert len(calls) == 2
        assert server.process.returncode == 23
        assert (root / 'stdout').read_bytes() == b'OUT'
        assert (root / 'stderr').read_bytes() == b'ERR'
        for capture in server.captures:
            retired(capture)
            assert capture._finished and capture._wake_writer.closed
            assert capture.worker.ident not in sys._current_frames()
    elif case in ('ordered', 'concurrent'):
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools.bench import run_serve_concurrency as bench
        sent = []
        closed = []
        both = threading.Barrier(2)
        job_one_seen = threading.Event()
        class Connection:
            def __init__(self, *args, **kwargs): pass
            def connect(self): pass
            def close(self): closed.append(self)
        def payload(point, job):
            if job.index == 0:
                job_one_seen.wait()  # Deliberately invert payload preparation.
            else:
                job_one_seen.set()
            return job.index
        def send(connection, data):
            sent.append(data)
        def receive(connection):
            both.wait()  # First response cannot finish until both sends occurred.
            return {}
        def post(connection, data):
            send(connection, data)
            return receive(connection)
        point = SimpleNamespace(suite='corpus-makespan' if case == 'ordered' else 'decode-saturation', concurrency=2)
        jobs = [SimpleNamespace(index=i) for i in range(2)]
        def parse(job, response, started, finished):
            return SimpleNamespace(job=job, finished_at=finished)
        with patch.object(bench.http.client, 'HTTPConnection', Connection), \
             patch.object(bench, 'request_payload', payload), \
             patch.object(bench.corpus, 'send_json', send), \
             patch.object(bench.corpus, 'receive_json', receive), \
             patch.object(bench.corpus, 'post_json', post), \
             patch.object(bench, 'parse_client_response', parse):
            results, start, end = bench.run_clients(point, jobs, 0)
        assert [r.job.index for r in results] == [0, 1]
        assert len(closed) == 2 and end >= start
        if case == 'ordered':
            assert sent == [0, 1]
        else:
            assert sorted(sent) == [0, 1]
    elif case == 'termination_policy':
        calls = []
        class Process:
            def poll(self): return None
            def terminate(self): calls.append('terminate')
            def wait(self, timeout=None):
                calls.append(('wait', timeout))
                if timeout is not None:
                    raise subprocess.TimeoutExpired('fake', timeout)
            def kill(self): calls.append('kill')
        server = RunningServer([], 'localhost', 0, root / 'log')
        server.process = Process()
        server.stop()
        assert calls == ['terminate', ('wait', 15.0), 'kill', ('wait', None)]
    else:
        raise AssertionError(case)
    assert len(os.listdir('/proc/self/fd')) == baseline_fds
    if 'status' in locals():
        print('STATUS', status, flush=True)
    if 'server' in locals():
        print('SERVER_STATUS', server.capture_status, flush=True)
    print('PASS', case, flush=True)


CASES = ('finite', 'active', 'idle', 'closed_console', 'blocked_console',
         'archive_failure', 'rc', 'primary', 'failure_without_primary',
         'setup_failure', 'disabled', 'termination_policy', 'interrupted_finish',
         'ordered', 'concurrent', 'partial_archive_failure', 'boundary',
         'interrupted_join', 'interrupted_signal_join', 'interrupted_wait_join', 'interrupted_join_exit',
         'interrupted_close_before', 'interrupted_close_after', 'primary_join')


@pytest.mark.parametrize('case', CASES)
def test_capture_watchdog(case):
    with tempfile.TemporaryDirectory() as directory:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), case, directory],
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True)
        try:
            stdout, stderr = child.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate()
            pytest.fail(f'external watchdog expired for {case}: {stdout!r} {stderr!r}')
        print(stdout.decode(errors='backslashreplace'), end='')
        if stderr:
            print(stderr.decode(errors='backslashreplace'), end='', file=sys.stderr)
        assert child.returncode == 0, (stdout, stderr)
        assert b'PASS ' + case.encode() in stdout


if __name__ == '__main__':
    run_case(sys.argv[1], Path(sys.argv[2]))
