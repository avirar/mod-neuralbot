#!/usr/bin/env python3
"""
Analyze behavior-cloning demonstration records written by the worldserver.

Record layout (NeuralBotFrame.h `NeuralBotBcRecord`, little-endian, packed):
    botGuid(8) targetGuid(8) seq(8) name[64] | NeuralBotFrame (FRAME_BYTES)
    total = 88 + FRAME_BYTES = 5997 bytes

The recorder appends one record per executed playerbot action. Frames carry the full
structured observation at action time; the reward tail is zeroed (progress is
reconstructed here from consecutive per-bot self xp/money/level deltas).

Usage:
    python3 bc_analyze.py hist     <file> [--limit N]   # action-name histogram
    python3 bc_analyze.py progress <file> [--limit N]   # per-action progress stats
"""
import argparse
import collections
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from neuralbot_client import FRAME_DTYPE, FRAME_BYTES  # noqa: E402

HEADER_BYTES = 88  # 8 + 8 + 8 + 64
RECORD_BYTES = HEADER_BYTES + FRAME_BYTES

REC_DTYPE = np.dtype([
    ("botGuid", "<u8"),
    ("targetGuid", "<u8"),
    ("seq", "<u8"),
    ("name", "S64"),
    ("frame", FRAME_DTYPE),
])


def load_records(path: str, limit: int = 0) -> np.ndarray:
    data = open(path, "rb").read()
    n = len(data) // RECORD_BYTES
    if n == 0:
        return np.empty(0, dtype=REC_DTYPE)
    data = data[: n * RECORD_BYTES]  # drop a partial tail from a crash
    if limit and limit < n:
        data = data[: limit * RECORD_BYTES]
    return np.frombuffer(data, dtype=REC_DTYPE)


def _decode(arr: np.ndarray) -> np.ndarray:
    return np.array([b.decode("utf-8", "replace").rstrip("\x00") for b in arr], dtype=object)


def cmd_hist(records: np.ndarray) -> None:
    names = _decode(records["name"])
    counts = collections.Counter(names)
    total = sum(counts.values())
    print(f"{len(records)} records, {len(counts)} distinct action names\n")
    print(f"{'count':>8}  {'pct':>6}  action")
    for name, c in counts.most_common():
        print(f"{c:>8}  {100.0 * c / total:>5.1f}%  {name}")


def cmd_progress(records: np.ndarray) -> None:
    """Attribute per-bot self-deltas between consecutive records to the action taken."""
    if len(records) < 2:
        print("too few records")
        return
    # Stable sort by (bot, seq) so consecutive frames are per-bot.
    order = np.lexsort((records["seq"], records["botGuid"]))
    rec = records[order]

    same_bot = rec["botGuid"][1:] == rec["botGuid"][:-1]
    xp_delta = (rec["frame"]["self"]["xp"][1:].astype(np.int64)
                - rec["frame"]["self"]["xp"][:-1].astype(np.int64))
    lvl_delta = (rec["frame"]["self"]["level"][1:].astype(np.int64)
                 - rec["frame"]["self"]["level"][:-1].astype(np.int64))
    money_delta = (rec["frame"]["self"]["money"][1:].astype(np.int64)
                   - rec["frame"]["self"]["money"][:-1].astype(np.int64))
    # Action taken at time t -> progress observed at t+1.
    names = _decode(rec["name"][:-1])
    xp_delta = xp_delta[same_bot]
    lvl_delta = lvl_delta[same_bot]
    money_delta = money_delta[same_bot]
    names = names[same_bot]

    stats = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])  # n, xp, lvl, money
    for name, xp, lvl, money in zip(names, xp_delta, lvl_delta, money_delta):
        s = stats[name]
        s[0] += 1
        s[1] += xp
        s[2] += lvl
        s[3] += money
        s[4] += 1.0 if lvl > 0 else 0.0

    print(f"{'action':<44} {'n':>7} {'avg_xp':>9} {'avg_lvl':>8} {'avg_money':>10} {'lvl_ups':>8}")
    for name, (n, xp, lvl, money, lvl_ups) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        if n < 5:  # skip actions with too few samples to be meaningful
            continue
        print(f"{name:<44} {n:>7} {xp / n:>9.2f} {lvl / n:>8.4f} {money / n:>10.1f} {lvl_ups:>8.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hist", "progress"])
    ap.add_argument("file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_records(args.file, args.limit)
    if len(records) == 0:
        print(f"no records in {args.file}")
        return
    print(f"parsed {len(records)} records from {args.file}\n")
    if args.cmd == "hist":
        cmd_hist(records)
    else:
        cmd_progress(records)


if __name__ == "__main__":
    main()
