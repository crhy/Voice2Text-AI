from __future__ import annotations

import sys

from .application import Voice2TextApplication


def main() -> int:
    return Voice2TextApplication().run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
