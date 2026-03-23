from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hedest.analysis.pred_analyzer import PredAnalyzer

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


def aggregate_seeds(run_dir: Path, json_path: Optional[str] = None, color_dict_file: Optional[str] = None):

    infos = load_seed_info(run_dir)

    if not infos:
        logger.error("No seed results found. Aborting aggregation.")
        return

    logger.info(f"Aggregating {len(infos)} seeds...")

    preds_best = []
    preds_best_adjusted = []
    for info in infos:
        preds_best.append(PredAnalyzer(model_info=info, adjusted=False).predictions)
        preds_best_adjusted.append(PredAnalyzer(model_info=info, adjusted=True).predictions)

    avg_pred_best = np.mean(preds_best, axis=0)
    avg_pred_best_adjusted = np.mean(preds_best_adjusted, axis=0)

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

    agg_info = {
        "model_name": ref["model_name"],
        "hidden_dims": ref["hidden_dims"],
        "norm": ref["norm"],
        "dropout": ref["dropout"],
        "spot_dict": ref["spot_dict"],
        "train_spot_dict": ref["train_spot_dict"],
        "proportions": ref["proportions"],
        "preds": {
            "pred_best": avg_pred_best,
            "pred_best_adjusted": avg_pred_best_adjusted,
        },
    }

    agg_pickle_path = run_dir / "info_aggregated.pickle"
    with open(agg_pickle_path, "wb") as f:
        pickle.dump(agg_info, f)
    logger.info(f"Saved aggregated info to {agg_pickle_path}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    seg_dict_raw = None
    if json_path is not None:
        with open(json_path, "r") as f:
            seg_dict_raw = json.load(f)

    analyzer_best = PredAnalyzer(model_info=agg_info, adjusted=False, seg_dict=seg_dict_raw)
    analyzer_best_adj = PredAnalyzer(model_info=agg_info, adjusted=True, seg_dict=seg_dict_raw)

    stats_best_predicted = analyzer_best.extract_stats(metric="predicted")
    stats_best_all = analyzer_best.extract_stats(metric="all")
    stats_best_adj_predicted = analyzer_best_adj.extract_stats(metric="predicted")
    stats_best_adj_all = analyzer_best_adj.extract_stats(metric="all")

    agg_stats_path = run_dir / "stats_aggregated.xlsx"
    with pd.ExcelWriter(agg_stats_path) as writer:
        stats_best_predicted.to_excel(writer, sheet_name="best_predicted", index=False)
        stats_best_all.to_excel(writer, sheet_name="best_all", index=False)
        stats_best_adj_predicted.to_excel(writer, sheet_name="best_adj_predicted", index=False)
        stats_best_adj_all.to_excel(writer, sheet_name="best_adj_all", index=False)
    logger.info(f"Saved aggregated stats to {agg_stats_path}")

    # ── GeoJSON export ────────────────────────────────────────────────────────
    if seg_dict_raw is not None:
        import yaml
        from hedest.utils import seg_dict_to_geojson, generate_color_dict

        ct_list = list(ref["proportions"].columns)

        if color_dict_file is not None:
            with open(color_dict_file, "r") as f:
                color_dict = yaml.load(f)
        else:
            color_dict = generate_color_dict(ct_list, format="special")
            auto_color_path = run_dir / "auto_color_dict.yaml"
            with open(auto_color_path, "w") as f:
                yaml.dump(color_dict, f)
            logger.info(f"Auto-generated color dict saved to {auto_color_path}")

        seg_dict_to_geojson(
            analyzer_best.seg_dict_w_class,
            str(run_dir / "hedest_predictions_aggregated.geojson"),
            color_dict=color_dict,
        )
        logger.info(f"GeoJSON (unadjusted) exported to {run_dir / 'hedest_predictions_aggregated.geojson'}")

        seg_dict_to_geojson(
            analyzer_best_adj.seg_dict_w_class,
            str(run_dir / "hedest_predictions_adj_aggregated.geojson"),
            color_dict=color_dict,
        )
        logger.info(f"GeoJSON (adjusted) exported to {run_dir / 'hedest_predictions_adj_aggregated.geojson'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HEDeST Seed Aggregator")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument("--color-dict-file", type=str, default=None)
    args = parser.parse_args()

    aggregate_seeds(
        run_dir=args.run_dir,
        json_path=args.json_path,
        color_dict_file=args.color_dict_file,
    )
