import fcntl
from contextlib import contextmanager


@contextmanager
def lifecycle_lock(storage, exclusive=False):
    with storage.path("lifecycle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
