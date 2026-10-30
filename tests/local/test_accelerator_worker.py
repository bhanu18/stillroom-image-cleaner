from types import SimpleNamespace

import pytest
from conftest import upload
from sqlalchemy import text

from cleaner.domain import Problem
from cleaner.worker import fallback_warnings


def claimed(setup):
    client, worker, app = setup
    asset = upload(client, worker)
    client.post('/v1/jobs', json={'input_asset_id': asset}, headers={'Idempotency-Key': 'device-test'})
    job = worker.db.claim(worker.owner)
    worker.runner = None
    worker.settings.device = 'cuda'
    return worker, job


@pytest.mark.parametrize('requested', ['cuda', 'mps', 'auto'])
@pytest.mark.parametrize('code', ['ACCELERATOR_OOM', 'ACCELERATOR_UNAVAILABLE'])
def test_retry_once_same_inputs_deadline_and_record_warning(setup, monkeypatch, code, requested):
    worker, job = claimed(setup)
    worker.settings.device = requested
    calls = []

    def attempt(job, operation, args, device, deadline):
        calls.append((args.copy(), device, deadline))
        if device != 'cpu':
            raise Problem(code, 'GPU failure', 503)
        return {'vector': [1.], 'diagnostics': {'device': 'cpu'}}

    monkeypatch.setattr(worker, '_model_attempt', attempt)
    result = worker.run_model(job, 'embed', query='red bag')
    assert [call[1] for call in calls] == [requested, 'cpu']
    assert calls[0][0] == calls[1][0] == {'query': 'red bag'}
    assert calls[0][2] == calls[1][2] == job['deadline']
    assert result['diagnostics']['requested_device'] == requested
    assert result['diagnostics']['fallback_reason'] == code
    assert fallback_warnings(result) == ['accelerator_cpu_fallback']


@pytest.mark.parametrize('code', ['MODEL_INTEGRITY', 'DEVICE_UNQUALIFIED', 'MODEL_UNAVAILABLE',
                                  'INVALID_IMAGE', 'DEADLINE_EXCEEDED', 'LEASE_LOST', 'RESOURCE_PRESSURE'])
def test_no_retry_for_unrelated_errors(setup, monkeypatch, code):
    worker, job = claimed(setup)
    calls = []

    def attempt(*args):
        calls.append(args)
        raise Problem(code, 'failure')

    monkeypatch.setattr(worker, '_model_attempt', attempt)
    with pytest.raises(Problem) as error:
        worker.run_model(job, 'embed', query='red bag')
    assert error.value.code == code
    assert len(calls) == 1


def test_cpu_failure_not_retried(setup, monkeypatch):
    worker, job = claimed(setup)
    calls = []

    def attempt(job, operation, args, device, deadline):
        calls.append(device)
        raise Problem('ACCELERATOR_OOM' if device == 'cuda' else 'MODEL_UNAVAILABLE', 'failure')

    monkeypatch.setattr(worker, '_model_attempt', attempt)
    with pytest.raises(Problem) as error:
        worker.run_model(job, 'embed', query='red bag')
    assert error.value.code == 'MODEL_UNAVAILABLE'
    assert calls == ['cuda', 'cpu']


def fake_context(events, cancel=None):
    class Pipe:
        def poll(self, timeout):
            return True

        def recv(self):
            return 'error', {'code': 'ACCELERATOR_OOM', 'detail': 'OOM', 'status': 503}

        def close(self):
            events.append('close')

    class Process:
        pid = 12345

        def __init__(self, **kwargs):
            self.alive = True

        def start(self):
            events.append('start')
            if cancel:
                cancel()

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append('terminate')

        def join(self, timeout=None):
            events.append('join')

        def kill(self):
            events.append('kill')
            self.alive = False

    return SimpleNamespace(Pipe=lambda duplex: (Pipe(), Pipe()), Process=Process)


def test_cleanup_completes_before_cpu_retry(setup, monkeypatch):
    worker, job = claimed(setup)
    events = []
    monkeypatch.setattr('cleaner.worker.mp.get_context', lambda _: fake_context(events))
    real_attempt = worker._model_attempt

    def attempt(job, operation, args, device, deadline):
        if device == 'cpu':
            assert events[-5:] == ['terminate', 'join', 'kill', 'join', 'close']
            return {'diagnostics': {'device': 'cpu'}}
        return real_attempt(job, operation, args, device, deadline)

    monkeypatch.setattr(worker, '_model_attempt', attempt)
    worker.run_model(job, 'embed', query='red bag')


def test_cancellation_prevents_retry_and_cleans_up(setup, monkeypatch):
    worker, job = claimed(setup)
    events = []

    def cancel():
        with worker.db.tx() as c:
            c.execute(text('UPDATE jobs SET canceled=1 WHERE id=:id'), {'id': job['id']})

    monkeypatch.setattr('cleaner.worker.mp.get_context', lambda _: fake_context(events, cancel))
    with pytest.raises(Problem) as error:
        worker.run_model(job, 'embed', query='red bag')
    assert error.value.code == 'LEASE_LOST'
    assert events.count('start') == 1
    assert 'kill' in events


def test_deadline_prevents_new_child(setup, monkeypatch):
    worker, job = claimed(setup)
    monkeypatch.setattr('cleaner.worker.now', lambda: job['deadline'] + 1)
    events = []
    monkeypatch.setattr('cleaner.worker.mp.get_context', lambda _: fake_context(events))
    with pytest.raises(Problem) as error:
        worker.run_model(job, 'embed', query='red bag')
    assert error.value.code == 'DEADLINE_EXCEEDED'
    assert not events


def simulated_accelerator_child(conn, root, operation, args, device):
    """Exercise real spawn/IPC/teardown without loading model weights or CUDA."""
    import os
    import time
    from pathlib import Path

    if device != 'cpu':
        (Path(root) / 'gpu-child.pid').write_text(str(os.getpid()))
        conn.send(('error', {'code': 'ACCELERATOR_OOM', 'detail': 'simulated GPU OOM', 'status': 503}))
        time.sleep(30)  # Parent must reap this child before entering the CPU attempt.
    else:
        import psutil

        gpu_pid = int((Path(root) / 'gpu-child.pid').read_text())
        assert not psutil.pid_exists(gpu_pid)
        conn.send(('ok', {'vector': [1.], 'diagnostics': {'device': 'cpu'}}))
    conn.close()


def test_real_spawn_fallback_reaps_child(setup, monkeypatch):
    worker, job = claimed(setup)
    monkeypatch.setattr('cleaner.worker.child_call', simulated_accelerator_child)
    result = worker.run_model(job, 'embed', query='red bag')
    assert result['diagnostics']['device'] == 'cpu'
    assert result['diagnostics']['fallback_reason'] == 'ACCELERATOR_OOM'
