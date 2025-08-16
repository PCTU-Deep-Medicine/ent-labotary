#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ==== CONFIG =====
CSV_PATH = "data/data-filter.csv"  # cần cột: filename, label
LABEL_MAP_TXT = "data/label_map_v.txt"  # "0: class_name" (để đặt tên folder & báo cáo)
SRC_IMG_DIR = "data/data-filter"  # thư mục ảnh gốc
OUT_ROOT = "data/12endo"  # sẽ tạo train/ val/ test/<class_name>/*
RATIOS = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42
COPY_MODE = "copy"  # "copy" | "symlink" | "move"
TOL = 0.02  # dung sai khi so tỉ lệ (±2%)
# =================

rng = np.random.default_rng(SEED)


def load_label_map(txt_path):
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
            # chuẩn hoá tên folder (giữ unicode VN, bỏ ký tự lạ)
            v_norm = re.sub(
                r"[^\w\s\-\(\)áàạãảăắằặẵẳâấầậẫẩéèẹẽẻêếềệễểíìịĩỉóòọõỏôốồộỗổơớờợỡởúùụũủưứừựữửýỳỵỹỷđĐ]",
                "",
                v,
            )
            v_norm = re.sub(r"\s+", "_", v_norm)
            mapping[int(k.strip())] = v_norm
    return mapping


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def alloc_counts(n, ratios):
    """
    Phân bổ n mẫu theo tỉ lệ (train,val,test) sao cho tổng đúng n.
    Dùng largest remainder (làm tròn thông minh) + cố gắng mỗi split >=1 nếu n>=3.
    """
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
    return tuple(base.tolist())  # (n_train, n_val, n_test)


def split_relative_no_loss(df, ratios, label_map, out_root, src_img_dir, copy_mode):
    # chia theo từng lớp, giữ nguyên 100% dữ liệu
    splits = {"train": [], "val": [], "test": []}
    classes = sorted(df["label"].unique())

    for y in classes:
        sub = (
            df[df["label"] == y]
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        n = len(sub)
        n_tr, n_va, n_te = alloc_counts(n, ratios)

        train_part = sub.iloc[:n_tr]
        val_part = sub.iloc[n_tr : n_tr + n_va]
        test_part = sub.iloc[n_tr + n_va : n_tr + n_va + n_te]

        splits["train"].append(train_part)
        splits["val"].append(val_part)
        splits["test"].append(test_part)

    # gộp & tạo cây thư mục
    for k in splits:
        splits[k] = pd.concat(splits[k], ignore_index=True)

    out_root = Path(out_root)
    ensure_dir(out_root)
    for split, sdf in splits.items():
        for y in sorted(sdf["label"].unique()):
            cls_name = label_map.get(int(y), str(y))
            ensure_dir(out_root / split / cls_name)

    # copy/symlink/move file
    missing = 0
    for split, sdf in splits.items():
        for _, row in sdf.iterrows():
            y = int(row["label"])
            cls_name = label_map.get(y, str(y))
            src = Path(src_img_dir) / row["filename"]
            if not src.is_file():
                missing += 1
                print(f"[WARN] Missing: {src}")
                continue
            dst = out_root / split / cls_name / src.name
            place(src, dst, copy_mode)

    # lưu CSV tham chiếu
    for split, sdf in splits.items():
        sdf.to_csv(out_root / f"{split}.csv", index=False)

    return splits, missing


def count_original(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    assert {"filename", "label"}.issubset(df.columns)
    # kiểm tra ảnh tồn tại
    missing_files = [
        fn for fn in df["filename"] if not os.path.isfile(os.path.join(img_dir, fn))
    ]
    counts = df["label"].value_counts().sort_index()
    total = int(counts.sum())
    return df, counts, total, missing_files


def count_after_split(out_root):
    res = {}
    for split in ["train", "val", "test"]:
        d = Path(out_root) / split
        if not d.is_dir():
            res[split] = pd.Series(dtype=int)
            continue
        rows = []
        for cls in sorted([p for p in d.iterdir() if p.is_dir()]):
            n = sum(1 for f in cls.iterdir() if f.is_file())
            rows.append((cls.name, n))
        res[split] = pd.Series({k: v for k, v in rows}).sort_index()
    return res


def check_no_data_loss(orig_total, split_counts):
    split_total = sum(int(ser.sum()) for ser in split_counts.values())
    ok = split_total == orig_total
    msg = (
        "✅ No data loss."
        if ok
        else f"⚠️ Data loss: original={orig_total}, after_split={split_total}"
    )
    print("\n=== INTEGRITY CHECK ===")
    print(msg)
    return ok


def check_balance_per_class(orig_df, split_counts, label_map, ratios, tol):
    print("\n=== BALANCE CHECK (relative by class) ===")
    train_r, val_r, test_r = ratios
    # build reverse map: folder_name -> label_id
    rev_map = {v: k for k, v in label_map.items()} if label_map else {}  # noqa: F841

    for y, group in orig_df.groupby("label"):
        n_orig = len(group)
        name = label_map.get(int(y), str(y)) if label_map else str(y)
        folder = name  # class folder name

        n_tr = split_counts["train"].get(folder, 0)
        n_va = split_counts["val"].get(folder, 0)
        n_te = split_counts["test"].get(folder, 0)

        r_tr = n_tr / n_orig if n_orig else 0.0
        r_va = n_va / n_orig if n_orig else 0.0
        r_te = n_te / n_orig if n_orig else 0.0

        def flag(r, target):
            return "" if abs(r - target) <= tol else " (!)"

        print(
            f"- class {int(y):2d} | {name}: orig={n_orig} | "
            f"train={n_tr} ({r_tr:.3f}{flag(r_tr, train_r)})  "
            f"val={n_va} ({r_va:.3f}{flag(r_va, val_r)})  "
            f"test={n_te} ({r_te:.3f}{flag(r_te, test_r)})"
        )


def pretty_print(orig_counts, orig_total, split_counts, label_map):
    print("\n=== ORIGINAL DATASET (from CSV) ===")
    print(f"Total: {orig_total}")
    for y, c in orig_counts.items():
        name = label_map.get(int(y), str(y)) if label_map else str(y)
        print(f"  class {int(y):2d} | {name:<30} : {int(c)}")

    print("\n=== AFTER SPLIT (folder counts) ===")
    for split, ser in split_counts.items():
        total = int(ser.sum()) if len(ser) > 0 else 0
        print(f"[{split}] total={total}")
        if len(ser) == 0:
            continue
        for cls_name, c in ser.items():
            print(f"  {cls_name:<30} : {int(c)}")


def main():
    assert abs(sum(RATIOS) - 1.0) < 1e-6, "RATIOS must sum to 1.0"

    label_map = load_label_map(LABEL_MAP_TXT)
    orig_df, orig_counts, orig_total, missing_src = count_original(
        CSV_PATH, SRC_IMG_DIR
    )
    if missing_src:
        print(
            f"⚠️ Missing in source folder: {len(missing_src)} files (first 5): {missing_src[:5]}"
        )

    # split + copy/symlink/move
    splits_df, missing_after = split_relative_no_loss(
        df=orig_df,
        ratios=RATIOS,
        label_map=label_map,
        out_root=OUT_ROOT,
        src_img_dir=SRC_IMG_DIR,
        copy_mode=COPY_MODE,
    )
    if missing_after:
        print(f"⚠️ Missing files during split: {missing_after}")

    # count after split
    split_counts = count_after_split(OUT_ROOT)

    # reports
    pretty_print(orig_counts, orig_total, split_counts, label_map)
    check_no_data_loss(orig_total, split_counts)
    check_balance_per_class(orig_df, split_counts, label_map, ratios=RATIOS, tol=TOL)

    print(f"\n✅ Done. Output root: {Path(OUT_ROOT).resolve()}")


if __name__ == "__main__":
    main()
