"""
Spread subset of the organic split for a minutes-scale smoke test of the
training pipeline (the files are grouped by molecule, so a head-of-file cut
would give one molecule).

Usage:
  python data/make_smoke_subset.py data/organic8 --every 20
"""

import argparse
import os

import ase.io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--every", type=int, default=20)
    args = ap.parse_args()
    for name in ("train", "val"):
        src = os.path.join(args.dir, f"{name}.xyz")
        dst = os.path.join(args.dir, f"smoke_{name}.xyz")
        frames = [a for i, a in enumerate(ase.io.iread(src, index=":", format="extxyz")) if i % args.every == 0]
        ase.io.write(dst, frames, format="extxyz")
        print(name, len(frames), "->", dst)


if __name__ == "__main__":
    main()
