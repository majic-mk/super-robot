"""Run the v8 offline H1 repair-grid sentinel after Profile-bound qualification."""

from __future__ import annotations

from run_v7_h1_pilot import main


if __name__ == "__main__":
    raise SystemExit(main(protocol_version=8))
