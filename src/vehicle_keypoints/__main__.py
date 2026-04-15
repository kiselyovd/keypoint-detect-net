"""CLI entrypoint: python -m vehicle_keypoints"""
from __future__ import annotations

import sys


def main() -> int:
    print("vehicle-keypoints — use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
