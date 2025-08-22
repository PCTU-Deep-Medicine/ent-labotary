#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task dataset splitter.

Reads a CSV containing at least: filename,label (fine class index)
Optional column: label_type (coarse class index). If absent it's derived from label by fixed ranges:
    0-4  -> 0 (ear)
    5-8  -> 1 (nose)
    9-11 -> 2 (throat)

Outputs a folder tree with BOTH fine and coarse ImageFolder layouts so that you can
instantiate two ImageFolders or derive coarse labels on the fly.

OUT_ROOT/
    fine/ train/<fine_name>/*, val/<fine_name>/*, test/<fine_name>/*
    type/ train/<type_name>/*, val/<type_name>/*, test/<type_name>/*
    splits.csv (filename,label,label_type,split,rel_path_fine,rel_path_type)

Modes for placing files: copy | symlink | move (default symlink to save space).

Ratios are enforced per FINE class using largest remainder; integrity checks printed.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ================== CONFIG ==================
CSV_PATH = "data/data-filter.csv"  # columns: filename,label[,label_type]
FINE_LABEL_MAP_TXT = "data/label_map_v.txt"  # lines: "0: class_name"
SRC_IMG_DIR = "data/data-filter"  # source images directory
OUT_ROOT = "data/12endo_multitask"  # output root (will contain fine/ & type/)
RATIOS = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42
COPY_MODE = "symlink"  # symlink | copy | move
TOL = 0.02  # tolerance for ratio reporting
COARSE_MAP = {0: "ear", 1: "nose", 2: "throat"}
# If True: raise AssertionError when any mismatch in counts detected
STRICT_CHECK = True
# ============================================

rng = np.random.default_rng(SEED)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_label_map(txt_path: str) -> dict:
    if not os.path.isfile(txt_path):
        return {}
    mapping = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            v_norm = re.sub(r"\s+", "_", v)
            mapping[int(k.strip())] = v_norm
    return mapping


def place(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError("COPY_MODE must be copy|symlink|move")


def alloc_counts(n: int, ratios: Tuple[float, float, float]):
    r = np.array(ratios, dtype=float)
    r = r / r.sum()
    raw = r * n
    base = np.floor(raw).astype(int)
    remainder = n - base.sum()
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remainder):
        base[order[i % len(base)]] += 1
    if n >= len(base):
        zeros = np.where(base == 0)[0]
        for z in zeros:
            m = np.argmax(base)
            if base[m] > 1:
                base[m] -= 1
                base[z] += 1
    assert base.sum() == n
    return tuple(base.tolist())


def derive_label_type(y_label_series: pd.Series) -> pd.Series:
    # vectorized mapping using bins
    y = y_label_series.astype(int)
    conds = [y <= 4, (y >= 5) & (y <= 8), y >= 9]
    choices = [0, 1, 2]
    return np.select(conds, choices).astype(int)


def split_per_fine(df: pd.DataFrame, ratios):
    splits = {"train": [], "val": [], "test": []}
    for _fine_label, sub in df.groupby(
        "label", sort=True
    ):  # _fine_label not used explicitly
        sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(sub)
        n_tr, n_va, n_te = alloc_counts(n, ratios)
        splits["train"].append(sub.iloc[:n_tr])
        splits["val"].append(sub.iloc[n_tr : n_tr + n_va])
        splits["test"].append(sub.iloc[n_tr + n_va : n_tr + n_va + n_te])
    for k, parts in splits.items():
        splits[k] = pd.concat(parts, ignore_index=True)
    return splits


def build_dirs(out_root: Path, fine_map: dict, mode_map: dict):
    for branch in ["fine", "type"]:
        for split in ["train", "val", "test"]:
            names = fine_map.values() if branch == "fine" else mode_map.values()
            for name in sorted(names):
                ensure_dir(out_root / branch / split / name)


def perform_placement(
    splits: dict,
    out_root: Path,
    fine_map: dict,
    mode_map: dict,
    src_dir: str,
    copy_mode: str,
):
    records = []
    missing = 0
    for split, sdf in splits.items():
        for _, row in sdf.iterrows():
            fname = row["filename"]
            fine_id = int(row["label"])
            type_id = int(row["label_type"])
            fine_name = fine_map.get(fine_id, str(fine_id))
            type_name = mode_map.get(type_id, str(type_id))
            src = Path(src_dir) / fname
            if not src.is_file():
                missing += 1
                continue
            dst_fine = out_root / "fine" / split / fine_name / src.name
            dst_type = out_root / "type" / split / type_name / src.name
            place(src, dst_fine, copy_mode)
            place(src, dst_type, copy_mode)
            records.append(
                {
                    "filename": fname,
                    "label": fine_id,
                    "label_type": type_id,
                    "split": split,
                    "rel_path_fine": str(dst_fine.relative_to(out_root)),
                    "rel_path_type": str(dst_type.relative_to(out_root)),
                }
            )
    return records, missing


def ratio_report(df_all: pd.DataFrame, ratios, tol: float):
    print("\n=== RATIO CHECK (per fine class) ===")
    train_r, val_r, test_r = ratios
    for fine, sub in df_all.groupby("label"):
        total = len(sub)

        def frac(
            split_name: str, subset=sub, total_count=total
        ):  # bind loop vars (flake8 B023)
            n_local = (subset["split"] == split_name).sum()
            return n_local / total_count if total_count else 0.0

        r_tr = frac("train")
        r_va = frac("val")
        r_te = frac("test")

        def flag(r_val, target, tolerance=tol):  # bind tol
            return "" if abs(r_val - target) <= tolerance else " (!)"

        print(
            f"class {fine:2d} | train={r_tr:.3f}{flag(r_tr, train_r)} val={r_va:.3f}{flag(r_va, val_r)} test={r_te:.3f}{flag(r_te, test_r)} n={total}"
        )


def count_branch_files(out_root: Path, branch: str):
    """Count files (incl. symlinks) under a branch (fine/ or type/).
    Returns dict: {split: count} and total.
    """
    split_counts = {}
    for split in ["train", "val", "test"]:
        split_dir = out_root / branch / split
        c = 0
        if split_dir.is_dir():
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                for item in cls_dir.iterdir():
                    # Count file or symlink (even if broken skip broken symlink check)
                    if item.is_file() or item.is_symlink():
                        c += 1
        split_counts[split] = c
    return split_counts, sum(split_counts.values())


def integrity_checks(df_original: pd.DataFrame, df_out: pd.DataFrame, out_root: Path):
    orig_total = len(df_original)
    placed_rows = len(df_out)
    print("\n=== INTEGRITY SUMMARY ===")
    print(f"Original rows (CSV): {orig_total}")
    print(f"Metadata rows (splits.csv): {placed_rows}")
    if orig_total != placed_rows:
        print("⚠️ MISMATCH: some rows not placed (see missing warnings above).")
    # Count filesystem for fine & type
    fine_counts, fine_total = count_branch_files(out_root, "fine")
    type_counts, type_total = count_branch_files(out_root, "type")
    print(f"Fine tree totals per split: {fine_counts} -> total={fine_total}")
    print(f"Type tree totals per split: {type_counts} -> total={type_total}")
    # Each sample should appear exactly once per branch
    expected = placed_rows
    ok_fine = fine_total == expected
    ok_type = type_total == expected
    print(f"Fine branch count {'OK' if ok_fine else 'MISMATCH'} (expected {expected})")
    print(f"Type branch count {'OK' if ok_type else 'MISMATCH'} (expected {expected})")
    if STRICT_CHECK and (not ok_fine or not ok_type or orig_total != placed_rows):
        raise AssertionError(
            "Sample loss detected: enable COPY_MODE='copy' to re-run or inspect warnings."
        )


def main():
    assert abs(sum(RATIOS) - 1.0) < 1e-6, "RATIOS must sum to 1.0"
    fine_map = load_label_map(FINE_LABEL_MAP_TXT)
    # read csv
    df = pd.read_csv(CSV_PATH)
    needed = {"filename", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}")
    # Always derive numeric coarse label_type from fine label regardless of existing column.
    # This avoids issues when CSV has textual label_type like 'ear'.
    if "label_type" in df.columns:
        # If existing column has non-numeric entries, just warn.
        if not np.issubdtype(df["label_type"].dtype, np.number):
            print(
                "ℹ️ Existing 'label_type' column is non-numeric; overriding with derived indices (ear=0,nose=1,throat=2)."
            )
        else:
            # Even if numeric, we enforce mapping for consistency.
            print(
                "ℹ️ Overriding existing numeric 'label_type' with derived mapping (ensuring consistency)."
            )
    df["label_type"] = derive_label_type(df["label"]).astype(int)

    splits = split_per_fine(df, RATIOS)

    out_root = Path(OUT_ROOT)
    build_dirs(out_root, fine_map, COARSE_MAP)
    records, missing = perform_placement(
        splits, out_root, fine_map, COARSE_MAP, SRC_IMG_DIR, COPY_MODE
    )
    if missing:
        print(f"⚠️ Missing {missing} source files (skipped).")

    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(out_root / "splits.csv", index=False)

    print(f"\nSaved split metadata: {out_root / 'splits.csv'}")
    ratio_report(df_out, RATIOS, TOL)
    integrity_checks(df, df_out, out_root)
    print(f"\nOutput root: {out_root.resolve()}")


if __name__ == "__main__":
    main()
