"""Host-only CLI proofs with external process watchdogs.

Historical behavior executes full byte-identical immutable commit source, not HEAD,
a handmade substitute, or candidate-delegated analysis. Works after candidate commit.
Host fixtures replace external I/O and identity only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.bench import run_serve_concurrency as bench
from tools.bench import run_serve_corpus as corpus

BASELINE_COMMIT = 'cab5bbbc50575d70241ed0a419a32655fedf4a21'
BASELINE_PATH = 'tools/bench/run_serve_concurrency.py'


def cli(root):
    return ['--artifact', 'qwen3_8_27b=/models/placeholder.ninfer', '--mode', 'mtp3',
            '--suite', 'decode-saturation', '--concurrency', '1', '--output', str(root / 'out')]


def old_module(root):
    # Fetch full immutable object bytes; never HEAD or candidate working source.
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE_COMMIT}:{BASELINE_PATH}'], cwd=ROOT)
    directory = tempfile.TemporaryDirectory(prefix='issue32-pinned-', dir=root)
    path = Path(directory.name) / 'tools' / 'bench' / 'run_serve_concurrency.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    assert path.read_bytes() == source
    name = '_issue32_pinned_' + hashlib.sha256(source + str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its defining module here.
    import_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = import_path
    module._source_directory = directory
    assert Path(module.__file__).read_bytes() == source
    assert module.analyze_point.__code__.co_filename == str(path)
    assert module.run_point.__code__.co_filename == str(path)
    assert module.analyze_point is not bench.analyze_point
    assert len(inspect.signature(module.analyze_point).parameters) == 9
    # Shared lifecycle helpers are independently byte-verified against the pin.
    # Historical analysis resolves its own globals, never candidate analysis.
    for relative in ('tools/bench/run_serve_corpus.py', 'tools/bench/stream_capture.py'):
        pinned = subprocess.check_output(
            ['git', 'show', f'{BASELINE_COMMIT}:{relative}'], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == pinned
    print('PINNED_SOURCE', BASELINE_COMMIT, hashlib.sha256(source).hexdigest(), flush=True)
    return module


def point(concurrency=1):
    return bench.Point('qwen3_8_27b', corpus.TARGET_MODEL_IDS['qwen3_8_27b'],
                       Path('/models/placeholder.ninfer'), 'mtp3', 'mtp', 3,
                       'stochastic', 'decode-saturation', concurrency)


def identity(event):
    return dict(artifact_type=corpus.SERVER_LOG_ARTIFACT_TYPE,
                schema_version=corpus.SERVER_LOG_SCHEMA_VERSION,
                server_instance_id='host-stub', event=event)


def startup(p):
    return dict(identity('server_start'), engine=dict(
        device=0, max_context=131072, kv_capacity=131072, prefill_chunk=1792,
        speculative_draft_window=3, max_concurrency=p.concurrency,
        max_pending_requests=16, pending_timeout_ms=180000, log_stats_interval_ms=1000),
        memory={name: 0 for name in corpus.MEMORY_METRICS})


def events(p):
    done = dict(identity('request_done'), result=dict(prompt_tokens=1, completion_tokens=2,
                computed_prefill_tokens=0, finish_reason='output_limit'),
                timings_seconds={name: 0.1 for name in
                                 ('prepare', 'ttft', 'vision', 'prefill', 'decode', 'total')},
                speculative=dict(backend='mtp', draft_window=3, rounds=1,
                                 drafted_tokens=3, accepted_tokens=1, fallback_steps=0))
    throughput = dict(identity('throughput'), interval_seconds=1.0,
                      tokens=dict(computed_prefill=0, committed_decode=p.concurrency),
                      decode_batch=dict(rounds=1, row_rounds=p.concurrency),
                      scheduler=dict(running=p.concurrency, prefilling=0,
                                     decode_ready=p.concurrency))
    return [done.copy() for _ in range(p.concurrency)] + [throughput]


def payload(p, stream):
    return (p.key + stream).encode() + bytes(range(256)) * 2048 + b'\x00tail\xff'


def retired(server):
    assert server.process.poll() is not None
    for capture in server.captures:
        assert not capture.worker.is_alive()
        assert capture.source.closed and capture._archive.closed
        for fd in (capture._wake_read, capture._wake_write):
            with pytest.raises(OSError):
                os.fstat(fd)


def run_point(module, root, p, enabled, failure=False):
    output = root / 'out'
    for name in ('server', 'points'):
        (output / name).mkdir(parents=True, exist_ok=True)
    args = module.parse_args(cli(root))
    if enabled:
        # Input attribute only: actual historical run_point must ignore it.
        # No historical source or capture wiring is modified.
        args.forensics_dir = root / 'nested' / 'forensics'
    fixture = SimpleNamespace(name='host-fixture', max_new=2, min_new=1)
    servers = []
    real_server = corpus.RunningServer

    def make_server(*a, **kw):
        server = real_server(*a, **kw)
        servers.append(server)
        return server

    def command(*unused):
        # Large finite binary output: exact archival, not an allowance for byte loss.
        script = "import os\nprefix=" + repr(p.key) + "\n"
        script += "for fd, stream in ((1, 'stdout'), (2, 'stderr')):\n"
        script += " data=(prefix+stream).encode()+bytes(range(256))*2048+bytes([0])+b'tail'+bytes([255])\n"
        script += " v=memoryview(data)\n while v:\n  n=os.write(fd,v); v=v[n:]\n"
        return [sys.executable, '-c', script]

    def clients(pt, jobs, port):
        assert servers[-1].process.wait(timeout=5) == 0
        if failure:
            raise RuntimeError('original client failure')
        return ([module.ClientResult(job, 1.0, 2.0, 1, 2, 'output_limit')
                 for job in jobs], 1.0, 2.0)

    with (patch.object(module, 'server_command', command),
          patch.object(corpus, 'RunningServer', make_server),
          patch.object(real_server, 'wait_until_ready', lambda self: startup(p)),
          patch.object(module, 'validate_server_start', return_value=('host-stub', 'host-weights')),
          patch.object(module, 'run_clients', clients),
          patch.object(module, 'load_server_events', return_value=events(p))):
        try:
            report = module.run_point(Path('/not-a-gpu-server'), p,
                                     {module.SATURATION_FIXTURE: fixture}, output, args)
        finally:
            for server in servers:
                retired(server)
    server = servers[-1]
    if not enabled or module is not bench:
        assert server.capture_paths is None and not server.captures
        assert server.process.stdout is None and server.process.stderr is None
        assert not (root / 'nested').exists()
        assert not any(output.rglob('*.log'))
    return report, server


def assert_capture(report, server, p):
    assert report['forensics_dir'] == str(server.capture_paths[0].parent)
    assert report['capture'] == server.capture_status
    assert report['capture_paths'] == dict(zip(('stdout', 'stderr'),
                                              map(str, server.capture_paths)))
    for stream, path, status in zip(('stdout', 'stderr'), server.capture_paths, report['capture']):
        assert path.read_bytes() == payload(p, stream)
        assert status['archived_bytes'] == len(payload(p, stream))
        assert status['capture_complete'] and not status['capture_failed']
        assert status['capture_stop_reason'] == 'eof' and status['capture_error'] is None


def case(name, root):
    if name == 'multipoint':
        paths = set()
        for concurrency in (1, 2):
            p = point(concurrency)
            report, server = run_point(bench, root, p, True)
            assert_capture(report, server, p)
            paths.update(server.capture_paths)
            assert json.loads((root / 'out' / 'points' / (p.key + '.json')).read_text()) == report
        assert len(paths) == 4
        for concurrency in (1, 2):
            for stream in ('stdout', 'stderr'):
                p = point(concurrency)
                assert (root / 'nested' / 'forensics' / (p.key + '.' + stream + '.log')).read_bytes() == payload(p, stream)
    elif name == 'no_flag':
        old = old_module(root)
        new_report, _ = run_point(bench, root, point(), False)
        old_report, _ = run_point(old, root, point(), False)
        assert json.dumps(new_report, sort_keys=True) == json.dumps(old_report, sort_keys=True)
        assert not {'capture', 'capture_paths', 'forensics_dir'} & new_report.keys()
        assert not {'capture', 'capture_paths', 'forensics_dir'} & old_report.keys()
        print('INDEPENDENT_NO_FLAG_JSON_EQUAL=YES', flush=True)
    elif name in ('prior_failure', 'post_commit'):
        if name == 'post_commit':
            # Model the sole relevant post-commit change: HEAD now resolves to
            # candidate bytes. No real index/commit/worktree mutation is needed.
            real_show = subprocess.check_output
            candidate = (ROOT / BASELINE_PATH).read_bytes()
            def committed_show(command, **kwargs):
                if command == ['git', 'show', f'HEAD:{BASELINE_PATH}']:
                    return candidate
                return real_show(command, **kwargs)
            with patch.object(subprocess, 'check_output', committed_show):
                assert subprocess.check_output(
                    ['git', 'show', f'HEAD:{BASELINE_PATH}'], cwd=ROOT) == candidate
                old = old_module(root)
            assert Path(old.__file__).read_bytes() != candidate
            print('POST_COMMIT HEAD=candidate; immutable baseline distinct', flush=True)
        else:
            old = old_module(root)
        report, server = run_point(old, root, point(), True)
        assert not {'capture', 'capture_paths', 'forensics_dir'} & report.keys()
        assert server.capture_paths is None and not server.captures
        with pytest.raises(KeyError, match='forensics_dir'):
            assert_capture(report, server, point())
        print('HISTORICAL_CAPTURE_ASSERTION=FAIL (expected); no archives/wiring', flush=True)
        report, server = run_point(bench, root, point(), True)
        assert_capture(report, server, point())
    elif name == 'client_failure':
        with pytest.raises(RuntimeError, match='original client failure'):
            run_point(bench, root, point(), True, failure=True)
        assert not list((root / 'out' / 'points').iterdir())
    print('PASS', name, flush=True)


@pytest.mark.parametrize('name', ['multipoint', 'no_flag', 'prior_failure', 'post_commit', 'client_failure'])
def test_run_point_watchdog(name, tmp_path):
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), name, str(tmp_path)],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
        pytest.fail(f'external watchdog expired: {stdout[-2000:]!r} {stderr[-2000:]!r}')
    assert child.returncode == 0, (stdout[-2000:], stderr[-2000:])
    assert b'PASS ' + name.encode() in stdout
    print(stdout.decode(errors='replace'))
    print('WATCHDOG PASS', name)


def test_cli_and_old_parser(tmp_path):
    args = bench.parse_args(cli(tmp_path) + ['--forensics-dir', str(tmp_path / 'new')])
    bench.validate_args(args)
    assert args.forensics_dir == (tmp_path / 'new').resolve()
    assert bench.parse_args(cli(tmp_path)).forensics_dir is None
    old = old_module(tmp_path)
    with pytest.raises(SystemExit) as caught:
        old.parse_args(cli(tmp_path) + ['--forensics-dir', str(tmp_path / 'new')])
    assert caught.value.code == 2
    assert not hasattr(old.parse_args(cli(tmp_path)), 'forensics_dir')
    file = tmp_path / 'file'
    file.write_text('not a directory')
    args.forensics_dir = file
    with pytest.raises(corpus.CampaignError, match='must be a directory'):
        bench.validate_args(args)


if __name__ == '__main__':
    case(sys.argv[1], Path(sys.argv[2]))