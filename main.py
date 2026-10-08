"""Convenience launcher for the native local application."""

import sys

from cleaner.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("serve")
    main()
