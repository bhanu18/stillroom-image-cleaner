"""Supervise only processes created by this launcher, including their model children."""

import os
import signal
import subprocess
import sys
import threading


class Supervisor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.process = None
        self.thread = threading.Thread(target=self.run, name="worker-supervisor", daemon=True)

    def start(self):
        self.thread.start()

    def terminate_group(self):
        if self.process:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()

    def run(self):
        while not self.stop_event.is_set():
            self.process = subprocess.Popen(
                [sys.executable, "-m", "cleaner.cli", "worker"], env=os.environ.copy(), start_new_session=True
            )
            while self.process.poll() is None and not self.stop_event.wait(1):
                pass
            self.terminate_group()
            if self.stop_event.wait(2):
                break

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
