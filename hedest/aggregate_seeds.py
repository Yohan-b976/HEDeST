from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from hedest.analysis import PredAnalyzer  # adjust import path if needed

# Import PredAnalyzer the same way run_model.py does

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_seed_info(run_dir: Path) -> list[dict]:
    """Load info.pickle for all seed_* subdirectories found in run_dir."""
    seed_dirs = sorted(run_dir.glob("seed_*"))

    if not seed_dirs:
        logger.error(f"No seed_* directories found in {run_dir}")
        return []

    # Ensure the folder contains only seed_* subdirectories (and no unexpected ones)
    all_subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    unexpected = [p for p in all_subdirs if not p.name.startswith("seed_")]
    if unexpected:
        logger.warning(
            f"Unexpected subdirectories found: {[p.name for p in unexpected]}. Aggregating only seed_* folders."
        )

    infos = []
    for seed_dir in seed_dirs:
        seed_path = seed_dir / "info.pickle"
        if not seed_path.exists():
            logger.warning(f"[SKIP] No info.pickle in {seed_dir}")
            continue
        with open(seed_path, "rb") as f:
            info = pickle.load(f)
        logger.info(f"[LOAD] {seed_dir.name} loaded.")
        infos.append(info)
    return infos


def aggregate_seeds(run_dir: Path):
    infos = load_seed_info(run_dir)

    if not infos:
        logger.error("No seed results found. Aborting aggregation.")
        return

    logger.info(f"Aggregating {len(infos)} seeds...")

    # --- Extract predictions per seed ---
    preds_best = []
    preds_best_adjusted = []

    for info in infos:
        # Use PredAnalyzer to access predictions consistently
        preds_best.append(PredAnalyzer(model_info=info, adjusted=False).predictions)
        preds_best_adjusted.append(PredAnalyzer(model_info=info, adjusted=True).predictions)

    # --- Average across seeds ---
    avg_pred_best = np.mean(preds_best, axis=0)
    avg_pred_best_adjusted = np.mean(preds_best_adjusted, axis=0)

    # Wrap back into DataFrames if predictions are DataFrames
    ref = infos[0]
    if isinstance(ref["preds"]["pred_best"], pd.DataFrame):
        avg_pred_best = pd.DataFrame(
            avg_pred_best,
            index=ref["preds"]["pred_best"].index,
            columns=ref["preds"]["pred_best"].columns,
        )
        avg_pred_best_adjusted = pd.DataFrame(
            avg_pred_best_adjusted,
            index=ref["preds"]["pred_best_adjusted"].index,
            columns=ref["preds"]["pred_best_adjusted"].columns,
        )

    # --- Build aggregated model_info ---
    agg_info = {
        "model_name": ref["model_name"],
        "hidden_dims": ref["hidden_dims"],
        "norm": ref["norm"],
        "dropout": ref["dropout"],
        "spot_dict": ref["spot_dict"],
        "train_spot_dict": ref["train_spot_dict"],
        "proportions": ref["proportions"],
        "num_seeds": len(infos),
        # No history — aggregation across seeds makes history meaningless
        "preds": {
            "pred_best": avg_pred_best,
            "pred_best_adjusted": avg_pred_best_adjusted,
        },
    }

    # --- Save aggregated info.pickle ---
    agg_pickle_path = run_dir / "info_aggregated.pickle"
    with open(agg_pickle_path, "wb") as f:
        pickle.dump(agg_info, f)
    logger.info(f"Saved aggregated info to {agg_pickle_path}")

    # --- Extract and save aggregated stats.xlsx ---
    stats_best_predicted = PredAnalyzer(model_info=agg_info, adjusted=False).extract_stats(metric="predicted")
    stats_best_all = PredAnalyzer(model_info=agg_info, adjusted=False).extract_stats(metric="all")
    stats_best_adj_predicted = PredAnalyzer(model_info=agg_info, adjusted=True).extract_stats(metric="predicted")
    stats_best_adj_all = PredAnalyzer(model_info=agg_info, adjusted=True).extract_stats(metric="all")

    agg_stats_path = run_dir / "stats_aggregated.xlsx"
    with pd.ExcelWriter(agg_stats_path) as writer:
        stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
        stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
        stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
        stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)
    logger.info(f"Saved aggregated stats to {agg_stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HEDeST Seed Aggregator")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the RUN_OUT directory")
    args = parser.parse_args()

    aggregate_seeds(args.run_dir)
