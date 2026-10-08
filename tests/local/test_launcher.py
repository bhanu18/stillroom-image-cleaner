import os
import subprocess
import sys
import time

import httpx
import psutil
import pytest


@pytest.mark.skipif(not os.getenv("RUN_PROCESS_TESTS"), reason="Requires permission to bind a temporary loopback port")
def test_worker_restart_and_clean_server_shutdown(tmp_path):
    env = {**os.environ, "IMAGE_CLEANING_DATA": str(tmp_path), "IMAGE_CLEANING_ORIGIN": "http://127.0.0.1:8766"}
    with (tmp_path / "launcher.log").open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "cleaner.cli", "serve"], env=env, stdout=log, stderr=log)
        children = []
        try:
            for _ in range(100):
                try:
                    if httpx.get("http://127.0.0.1:8766/health/ready", timeout=0.5, trust_env=False).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("Server never became ready")
            parent = psutil.Process(server.pid)
            children = parent.children(recursive=True)
            worker = next(p for p in children if p.cmdline()[-1] == "worker")
            worker.terminate()
            for _ in range(100):
                alive = [
                    p for p in parent.children(recursive=True) if p.pid != worker.pid and p.cmdline()[-1] == "worker"
                ]
                if alive:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Supervisor did not restart worker")
            children += alive
            server.terminate()
            server.wait(timeout=15)
            for child in children:
                assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        finally:
            if server.poll() is None:
                server.kill()
                server.wait()
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
