from __future__ import annotations

import sys

from .application import VoxaApplication


def main() -> int:
    return VoxaApplication().run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
