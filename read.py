#!/usr/bin/env python3
"""从 parquet 切片生成数据集统计信息 (论文标准格式)."""

import argparse
import os
import numpy as np
import pandas as pd

DATA_SPECS = {
    "criteo": {
        "name": "criteo",
        "data_dir": "/NAS/shith/datasets2024/uplift/criteo",
        "feature_cols": [f"f{i}" for i in range(12)],
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "mediator_col": "visit",
    },
    "hillstrom": {
        "name": "hillstrom",
        "data_dir": "/NAS/shith/datasets2024/uplift/hillstrom",
        "feature_cols": [
            "recency",
            "history_segment_id",
            "mens",
            "womens",
            "zip_code_id",
            "newbie",
            "channel_id",
        ],
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "mediator_col": "visit",
    },
}

SPLITS = ("train", "valid", "test")


def load_split(data_dir, data_spec, split):
    path = os.path.join(data_dir, f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file at: {path}")
    
    df = pd.read_parquet(path)
    feat_cols = list(data_spec["feature_cols"])
    t_col = data_spec["treatment_col"]
    y_col = data_spec["outcome_col"]
    m_col = data_spec["mediator_col"]

    for c in feat_cols + [t_col, y_col, m_col]:
        if c not in df.columns:
            raise KeyError(f"Split={split} missing column '{c}'")

    return {
        "split": split,
        "n": int(len(df)),
        "t": df[t_col].to_numpy(dtype=np.float64),
        "y": df[y_col].to_numpy(dtype=np.float64),
        "m": df[m_col].to_numpy(dtype=np.float64),
        "n_feat": len(feat_cols),
    }


def summarize(name, t, y, m, n_feat=None):
    t, y, m = np.asarray(t), np.asarray(y), np.asarray(m)
    n = int(t.shape[0])
    
    mask0, mask1 = (t == 0), (t == 1)
    e_y_t0 = float(y[mask0].mean()) if mask0.any() else float("nan")
    e_y_t1 = float(y[mask1].mean()) if mask1.any() else float("nan")

    return {
        "name": name,
        "n": n,
        "n_m": n / 1e6,
        "p_t": float(t.mean()),
        "p_m": float((m > 0).mean()),
        "p_y": float((y > 0).mean()),
        "e_y_t0": e_y_t0,
        "e_y_t1": e_y_t1,
        "ate": e_y_t1 - e_y_t0,
        "n_feat": n_feat,
    }


def fmt_m(n_m):
    return f"{n_m:.1f} M" if n_m >= 10 else f"{n_m:.2f} M"


def print_markdown(rows, overall, n_feat):
    print("\n| Dataset | Total Samples | Train | Valid | Test | P(T=1) | P(M=1) | P(Y>0) | E[Y|T=0] | ATE | # Features |")
    print("|---------|---------------|-------|-------|------|--------|--------|--------|-----------|-----|------------|")
    by = {r["name"]: r for r in rows}
    print(
        f"| Criteo | {fmt_m(overall['n_m'])} | {fmt_m(by['train']['n_m'])} | {fmt_m(by['valid']['n_m'])} | "
        f"{fmt_m(by['test']['n_m'])} | {overall['p_t']:.3f} | {overall['p_m']:.3f} | {overall['p_y']:.4f} | "
        f"{overall['e_y_t0']:.4f} | {overall['ate']:+.4f} | {n_feat} |"
    )

    print("\nPer-split detail:")
    print("| Split | n | P(T=1) | P(M=1) | P(Y>0) | E[Y|T=0] | ATE |")
    print("|-------|---|--------|--------|--------|-----------|-----|")
    for r in rows + [overall]:
        print(
            f"| {r['name']} | {r['n']:,} | {r['p_t']:.4f} | {r['p_m']:.4f} | "
            f"{r['p_y']:.4f} | {r['e_y_t0']:.6f} | {r['ate']:+.6f} |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-name", default="criteo", choices=["criteo", "hillstrom"])
    args = ap.parse_args()

    spec = DATA_SPECS[args.data_name]
    data_dir = args.data_dir or spec["data_dir"]

    loaded = [load_split(data_dir, spec, s) for s in SPLITS]
    rows = [summarize(s["split"], s["t"], s["y"], s["m"]) for s in loaded]

    pooled_t = np.concatenate([s["t"] for s in loaded])
    pooled_y = np.concatenate([s["y"] for s in loaded])
    pooled_m = np.concatenate([s["m"] for s in loaded])
    overall = summarize("pooled", pooled_t, pooled_y, pooled_m)

    print_markdown(rows, overall, loaded[0]["n_feat"])


if __name__ == "__main__":
    main()