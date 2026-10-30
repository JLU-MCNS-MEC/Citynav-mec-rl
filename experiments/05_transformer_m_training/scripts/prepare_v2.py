#!/usr/bin/env python3
"""Build the stage-05 episode-level frozen-visual cache (candidates24).

Each finished episode is one ``.npz`` written atomically and journalled with its
size and SHA-256.  A restart re-hashes what is on disk and only rebuilds the
episodes that are missing or damaged - a file that merely exists is not trusted.

The split is prepared in shards (one process per contiguous episode range); each
shard writes its own journal file, and the journals are merged in teacher order
afterwards, so the result is identical to a single-process build.

    python prepare_v2.py --split train_seen --limit 8      # speed test
    python prepare_v2.py                                   # all three splits
    python prepare_v2.py --verify-only                     # re-hash, no build

Output: features_v2/<split>/episodes/<key>.npz, journal.jsonl, index.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_v2 import (  # noqa: E402
    CACHE, CANDS, CANDIDATE_VERSION, CACHE_VERSION, ENCODER_ID, SPLITS,
    CandidateStore, FrozenEncoder, RasterStore, build_index, episode_meta,
    episode_path, load_split, prepare_arrays, read_journal, sha256_file,
    verify_journal_entry, write_episode,
)

BYTES_PER_STATE = 2048 * 2 + 12 * 4 + 2 + 3 * 4 + 1      # pooled, belief, valid, states, action
BYTES_PER_EPISODE = 2048 * 2 + 1024                      # target_emb + npz overhead


def estimate(episodes) -> dict:
    states = sum(len(e.actions) for e in episodes)
    return {"episodes": len(episodes), "states": states,
            "bytes": states * BYTES_PER_STATE + len(episodes) * BYTES_PER_EPISODE}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _worker(args: tuple) -> dict:
    split, indices, journal_path, limit = args
    episodes = load_split(split)
    rasters, cands = RasterStore(), CandidateStore()
    encoder = FrozenEncoder("cuda")
    started = time.time()
    n = 0
    with Path(journal_path).open("w", encoding="utf-8") as jf:
        for i in indices:
            ep = episodes[i]
            arrays = prepare_arrays(ep, rasters.get(ep.map), cands.get(ep.map), encoder)
            entry = write_episode(ep, arrays, split)
            jf.write(json.dumps(entry) + "\n")
            jf.flush()
            n += 1
            if n % 200 == 0:
                rate = n / max(time.time() - started, 1e-6)
                print(f"  [{split} shard {os.getpid()}] {n}/{len(indices)} "
                      f"episodes, {rate:.1f} ep/s", flush=True)
    return {"split": split, "pid": os.getpid(), "episodes": n,
            "seconds": round(time.time() - started, 1)}


def prepare_serial(split: str, limit: int) -> None:
    """Small --limit build in this process (used to time the encoder)."""
    episodes = load_split(split)[:limit]
    rasters, cands = RasterStore(), CandidateStore()
    encoder = FrozenEncoder("cuda")
    journal = CACHE / split / "journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    states = 0
    with journal.open("a", encoding="utf-8") as jf:
        for ep in episodes:
            arrays = prepare_arrays(ep, rasters.get(ep.map), cands.get(ep.map), encoder)
            entry = write_episode(ep, arrays, split)
            jf.write(json.dumps(entry) + "\n")
            states += len(ep.actions)
    seconds = time.time() - started
    print(f"{split}: {len(episodes)} episodes / {states} states in {seconds:.1f}s "
          f"({states / max(seconds, 1e-6):.0f} states/s)", flush=True)


def verify_split(split: str) -> tuple[int, int, dict[str, int]]:
    entries = read_journal(split)
    ok, bad = 0, 0
    reasons: dict[str, int] = {}
    for entry in entries.values():
        good, why = verify_journal_entry(entry)
        if good:
            ok += 1
        else:
            bad += 1
            reasons[why] = reasons.get(why, 0) + 1
    return ok, bad, reasons


def write_journal(split: str, entries: dict[int, dict], episodes) -> None:
    """Rewrite the canonical journal in teacher order (atomic)."""
    path = CACHE / split / "journal.jsonl"
    tmp = path.with_name(f".journal.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(entries[ep.key]) + "\n")
    os.replace(tmp, path)


def prepare_split(split: str, shards: int) -> None:
    episodes = load_split(split)
    est = estimate(episodes)
    print(f"{split}: {est['episodes']} episodes / {est['states']} states, "
          f"estimated {human(est['bytes'])}", flush=True)

    entries = read_journal(split)
    ok, bad, reasons = verify_split(split)
    print(f"{split}: journal {len(entries)} episodes -> {ok} valid, {bad} to rebuild {reasons}",
          flush=True)
    done = {k for k, e in entries.items() if verify_journal_entry(e)[0]}
    todo = [i for i, ep in enumerate(episodes) if ep.key not in done]
    if not todo:
        index = build_index(split)
        print(f"{split}: cache complete ({index['states']} states)", flush=True)
        return

    parts_dir = CACHE / split / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    shards = max(1, min(shards, len(todo)))
    bounds = [round(i * len(todo) / shards) for i in range(shards + 1)]
    jobs = [(split, todo[bounds[i]:bounds[i + 1]], str(parts_dir / f"journal.part{i:02d}.jsonl"), 0)
            for i in range(shards)]
    journal_paths = [Path(j[2]) for j in jobs]
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(shards) as pool:
        for msg in pool.imap_unordered(_worker, jobs):
            print(f"  {msg}", flush=True)
    seconds = time.time() - started
    states = sum(len(episodes[i].actions) for i in todo)
    print(f"{split}: built {len(todo)} episodes / {states} states in {seconds / 60:.1f} min "
          f"({states / max(seconds, 1e-6):.0f} states/s)", flush=True)

    for path in journal_paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    entries[int(e["key"])] = e
        path.unlink(missing_ok=True)
    write_journal(split, entries, episodes)
    index = build_index(split)
    ok, bad, reasons = verify_split(split)
    print(f"{split}: index built, {index['states']} states, journal {ok} ok / {bad} bad {reasons}",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", default=list(SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(f"candidate library: {CANDS} ({CANDIDATE_VERSION})", flush=True)
    print(f"encoder: {ENCODER_ID} | cache version: {CACHE_VERSION}", flush=True)
    for split in args.split:
        if args.verify_only:
            ok, bad, reasons = verify_split(split)
            print(f"{split}: verify {ok} ok / {bad} bad {reasons}", flush=True)
        elif args.limit:
            prepare_serial(split, args.limit)
        else:
            prepare_split(split, args.shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
