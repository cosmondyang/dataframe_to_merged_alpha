"""
Refactored version of `examine_featureonebyonetocsv.py` – robust to
missing keys, avoids crashing workers, and now also exports
`orth_lw_ic_v2` & `selfcorr` (plus keeps original QoL patches).

新增
====
* analyze()  :  metric_dict 里加入 "lw_ic_v2" 与 "orth_lw_ic_v2"
* process_column():
      - 取 `orth_lw_ic_v2` 均值
      - 把 `selfcorr` 写进 metrics 行
* main() : 输出 CSV 时强制列顺序
"""

import os
import time
import argparse
import datetime
import multiprocessing as mp
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from touchstone_client import TouchstoneClient
from config import UtilConfig as cfg

# ------------------------------------------------------------------
# helpers -----------------------------------------------------------
# ------------------------------------------------------------------


def _as_float(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, dict):
        x = list(x.values())
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype="float64")
        return float(np.nanmean(arr)) if arr.size else np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_get_dates(res_dict: Dict[str, Any]):
    dates_raw = res_dict.get("date")
    if dates_raw is None:
        return []
    try:
        return [datetime.datetime.strptime(d, "%Y%m%d") for d in dates_raw]
    except Exception:
        return []


# ------------------------------------------------------------------
# plotting (unchanged) ---------------------------------------------
# ------------------------------------------------------------------

# ...  —— draw(), _plot_series(), _plot_longside(), _plot_layered()
# （与原版本完全一致，省略，见上一版 189 / 190 行） ...
def draw(res_dict: Dict[str, Any], T: int, save_dir: str):
    """Generate figures; skip gracefully if required data missing."""
    date_list = safe_get_dates(res_dict)
    if not date_list:
        print("[draw] Warning – 'date' field missing or unparsable; skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    draw_list = [
        "ic", "lw_ic_v2", "longside_return", "coef",
        "layered_return", "layer_diffret",
    ]
    draw_list += [f"orth_{n}" for n in draw_list]
    draw_list += [f"neutral_{n}" for n in draw_list]

    for k in ["max_corr_with_factor_pool", "vif", "selfcorr", "mean_skew", "mean_kurt"]:
        if k not in res_dict:
            continue
        val = res_dict[k]
        print(f"---- {k} ----")
        if np.isscalar(val):
            print(f"{k} = {val}")
            continue
        data = np.asarray(val, dtype="float64")
        print(f"valid {(~np.isnan(data)).sum()} / total {data.size}")
        print(f"mean {k}: {np.nanmean(data)}  |  std {k}: {np.nanstd(data)}")

    # --- individual plot types -----------------------------------
    for k in draw_list:
        if k not in res_dict:
            continue
        try:
            if "ic" in k or "coef" in k:
                _plot_series(res_dict[k], date_list, save_dir, k)
            elif "longside_return" in k:
                _plot_longside(res_dict[k], date_list, save_dir, k, T)
            elif "layered" in k:
                _plot_layered(res_dict[k], date_list, save_dir, k, T)
        except Exception as e:
            print(f"[draw] Failed to draw {k}: {e}")


def _plot_series(arr_like, dates, save_dir, label):
    arr = np.asarray(arr_like, dtype="float64")
    if arr.size == 0:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax0.plot(dates, arr)
    ax0.set_title(label)
    ax1.plot(dates, np.nancumsum(arr))
    ax1.set_title("CumSum " + label)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"figure_{label}.png"))
    plt.close(fig)


def _plot_longside(ret_dict, dates, save_dir, label_root, T):
    for cfg_num, arr_like in ret_dict.items():
        arr = np.asarray(arr_like, dtype="float64")
        if arr.size == 0:
            continue
        label = f"{label_root}_{cfg_num}"
        ann_mean = np.nanmean(arr) * (244 / T)
        ann_std  = np.nanstd(arr) * np.sqrt(244 / T)
        print(f"{label}: ann_mu={ann_mean:.4%}, ann_std={ann_std:.4%}")
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax0.plot(dates, arr)
        ax1.plot(dates, np.nancumsum(arr))
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"figure_{label}.png"))
        plt.close(fig)


def _plot_layered(mat_like, dates, save_dir, key, T):
    mat = np.asarray(mat_like, dtype="float64")
    if mat.ndim != 2:
        return
    centered = mat - np.nanmean(mat, axis=1, keepdims=True)
    # plot layers cumsum
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, np.nancumsum(centered, axis=0))
    ax.set_title(key)
    ax.legend([f"L{i+1}" for i in range(centered.shape[1])])
    fig.savefig(os.path.join(save_dir, f"figure_{key}_layers.png"))
    plt.close(fig)

    # diff plot
    diff_series = centered[:, -1] - centered[:, 0] + 0.8 * (centered[:, -2] - centered[:, 1])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0.plot(dates, diff_series)
    ax0.set_title("Layer diff")
    ax1.plot(dates, np.nancumsum(diff_series))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"figure_{key}_diff.png"))
    plt.close(fig)

# ------------------------------------------------------------------
# touchstone wrapper ------------------------------------------------
# ------------------------------------------------------------------


def analyze(factor_df: pd.DataFrame, state: str = "core", T: int = 20) -> Dict[str, Any]:
    """Wrapper around TouchstoneClient.analyze with extra self-stats."""
    unstacked = factor_df.unstack()
    selfcorr = float(unstacked.corrwith(unstacked.shift(1)).mean())
    mean_skew = float(unstacked.skew(axis=1).mean())
    mean_kurt = float(unstacked.kurt(axis=1).mean())

    client = TouchstoneClient(cfg.TOUCHSTONE_PARAMS_IP, cfg.TOUCHSTONE_PARAMS_PORT)
    analyze_cfg = dict(
        buy_time_desc="1000.average",
        sell_time_desc="1000.average",
        T=T,
        freq=30,
        state=state,
        standarize=False,
        compress_method=1,
    )
    metric_dict = {
        "max_corr_with_factor_pool": {},
        "ic": {},
        "lw_ic_v2": {},                # ❶ 新增
        "layered_return": {"layers": 10},
        "layer_diffret": {},
        "orth_ic": {},
        "orth_lw_ic_v2": {},           # ❶ 新增
        "orth_layered_return": {"layers": 10},
        "orth_layer_diffret": {},
        "neutral_ic": {},
        "neutral_layered_return": {"layers": 10},
    }

    res = client.analyze(analyze_cfg, metric_dict, factor_df)
    if res.get("status", 0) != 0:
        print("[analyze] Warning – non-zero status:", res.get("status"))
    res["selfcorr"] = selfcorr
    res["mean_skew"] = mean_skew
    res["mean_kurt"] = mean_kurt
    return res


# ------------------------------------------------------------------
# multiprocessing worker -------------------------------------------
# ------------------------------------------------------------------


def process_column(task: Tuple):
    idx, col, state, T, feature_path, save_root, cover_rate = task
    try:
        df = pd.read_parquet(feature_path, columns=[col]).dropna()
        std_name = "standardized"
        df[std_name] = df.groupby(level=df.index.names[0])[col] \
            .transform(lambda x: (x - x.mean()) / x.std())
        res = analyze(df[[std_name]], state, T)

        metrics = dict(
            alpha=col,
            ic=np.nanmean(res.get("ic", [])),
            orthic=np.nanmean(res.get("orth_ic", [])),
            neutic=np.nanmean(res.get("neutral_ic", [])),
            coverrate=cover_rate,
            orth_lw_ic_v2=np.nanmean(res.get("orth_lw_ic_v2", [])),  # ❷ 新增
            selfcorr=_as_float(res.get("selfcorr")),                  # ❷ 新增
            stdic=np.nanstd(res.get("ic", [])),
            stdorthic=np.nanstd(res.get("orth_ic", [])),
            stdneutic=np.nanstd(res.get("neutral_ic", [])),
            maxcorr=_as_float(res.get("max_corr_with_factor_pool")),
            meanskew=_as_float(res.get("mean_skew")),
            meankurt=_as_float(res.get("mean_kurt")),
        )

        # optional drawings
        try:
            col_dir = os.path.join(save_root, f"col_{idx}_{col}")
            draw(res, T, col_dir)
        except Exception as e:
            print(f"[worker] plot error for {col}: {e}")

        return metrics

    except Exception as e:
        print(f"[worker] fatal error in column {col}: {e}")
        return dict(alpha=col, ic=np.nan)


# ------------------------------------------------------------------
# main --------------------------------------------------------------
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="parquet name")
    parser.add_argument("--state", type=str, default="core")
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=min(os.cpu_count(), 12))
    args = parser.parse_args()

    feature_path = os.path.join(cfg.feature_dir, f"{args.name}.parquet")
    full_df = pd.read_parquet(feature_path)
    # print(full_df)
    cover_rates = full_df.groupby("pred_date").apply(lambda df: df.count() / len(df)).mean()

    save_root = os.path.join(os.path.dirname(__file__), "figures", args.name)
    os.makedirs(save_root, exist_ok=True)

    tasks: List[Tuple] = []
    for idx, col in enumerate(full_df.columns):
        tasks.append((idx, col, args.state, args.T, feature_path, save_root,
                      cover_rates.get(col, np.nan)))

    start = time.time()
    with mp.Pool(processes=args.n_jobs) as pool:
        results = pool.map(process_column, tasks)

    metrics_df = pd.DataFrame(results).set_index("alpha")

    # ❸ 按要求重排列顺序
    ordered_cols = [
        "ic", "orthic", "neutic", "coverrate",
        "orth_lw_ic_v2", "selfcorr",
        "stdic", "stdorthic", "stdneutic",
        "maxcorr", "meanskew", "meankurt",
    ]
    metrics_df = metrics_df[ordered_cols]

    out_csv = os.path.join(os.path.dirname(__file__),
                           f"metricsclassify_summary_compareall19_{args.name}.csv")
    metrics_df.to_csv(out_csv)
    print("Saved summary to", out_csv)
    print(metrics_df)
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
