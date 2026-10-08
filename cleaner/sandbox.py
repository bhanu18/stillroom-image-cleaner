"""Native macOS child egress denial and CPU bounds; no external inference calls."""

import ctypes
import platform
import resource

from .domain import Problem


def restrict_child(operation):
    cpu = 30 if operation == "decode" else 1800
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    if platform.system() == "Darwin":
        lib = ctypes.CDLL("/usr/lib/libsandbox.dylib")
        lib.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
        error = ctypes.c_char_p()
        if lib.sandbox_init(b"(version 1)(allow default)(deny network*)", 0, ctypes.byref(error)) != 0:
            raise Problem("SANDBOX_UNAVAILABLE", "Native inference sandbox could not be initialized", 503)
