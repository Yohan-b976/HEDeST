from __future__ import annotations

import json
import os
from PIL import Image
from typing import Dict
from typing import Optional
import numpy as np
import anndata as ad
import openslide
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from hedest.config import TqdmToLogger
from scipy.spatial import KDTree

tqdm_out = TqdmToLogger(logger, level="INFO")


def filter_by_st_proximity(nuc_dict, adata_path, mpp, dist_thresh_um=300):
    """Removes cells further than threshold from the nearest ST spot."""
    logger.info(f"Loading AnnData from {adata_path} for spatial filtering...")
    adata = ad.read_h5ad(adata_path)
    
    # Extract spot coordinates from obsm['spatial']
    spot_coords = np.array(adata.obsm["spatial"].astype("int64"))
    tree = KDTree(spot_coords)
    
    # Convert microns to pixels: threshold_px = microns / microns_per_pixel
    dist_thresh_px = dist_thresh_um / mpp
    
    filtered_nuc = {}
    original_count = len(nuc_dict)
    
    for nuc_id, nuc_info in nuc_dict.items():
        # HoVer-Net centroids are usually [y, x] or [x, y]
        # Coordinates must match the orientation in adata.obsm['spatial']
        centroid = np.array(nuc_info['centroid'])
        
        dist, _ = tree.query(centroid)
        
        if dist <= dist_thresh_px:
            filtered_nuc[nuc_id] = nuc_info
            
    logger.info(f"ST Filtering: Kept {len(filtered_nuc)}/{original_count} cells "
             f"(Threshold: {dist_thresh_um}um / {dist_thresh_px:.2f}px)")
    return filtered_nuc


def extract_images_hn(
    image_path: str,
    json_path: str,
    level: int = 0,
    size_px: int = 64,
    size_um: Optional[float] = None,
    mpp: Optional[float] = None,
    dict_types: Optional[Dict[int, str]] = None,
    save_images: Optional[str] = None,
    save_dict: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts tiles from a whole slide image (WSI) given a JSON file with cell centroids.

    Args:
        image_path: Path to the WSI file.
        json_path: Path to the JSON file with cell centroids.
        level: Level of the WSI to extract the tiles.
        size_px: Size of the tile in pixels (used if size_um is None).
        size_um: Size of the tile in micrometers.
        mpp: Resolution of the original WSI (Microns per pixel).
        dict_types: Optional dictionary mapping cell types to names.
        save_images: Path to save extracted image tiles.
        save_dict: Path to save the dictionary of extracted tiles.
        
    Returns:
        Dictionary containing extracted tiles as tensors.
    """

    slide = openslide.open_slide(image_path)

    if size_um is not None:
        if mpp is None:
            raise ValueError("If size_um is provided, `mpp` must also be provided.")
        crop_px = int(round(size_um / mpp))
    else:
        crop_px = size_px

    centroid_list_wsi = []
    type_list_wsi = []
    image_dict = {}

    # Extract nuclear info
    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data["nuc"]
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = inst_info["centroid"]
            centroid_list_wsi.append(inst_centroid)
            if dict_types is not None:
                inst_type = inst_info["type"]
                type_list_wsi.append(inst_type)

    cell_table = pd.DataFrame(centroid_list_wsi, columns=["x", "y"])
    if dict_types is not None:
        cell_table["class"] = type_list_wsi

    for i in tqdm(range(len(cell_table)), file=tqdm_out, desc="Extracting tiles"):
        cell_line = cell_table[cell_table.index == i]

        x = int(cell_line["x"].values[0])
        y = int(cell_line["y"].values[0])

        img_cell = slide.read_region(
            (x - crop_px // 2, y - crop_px // 2),
            level,
            (crop_px, crop_px),
        )
        img_cell = img_cell.convert("RGB")

        if size_um is not None and crop_px != size_px:
            img_cell = img_cell.resize((size_px, size_px))

        if save_images is not None:
            if dict_types is not None:
                cell_class = dict_types[cell_line["class"].values[0]]
                save_dir = os.path.join(save_images, cell_class)
            else:
                save_dir = save_images

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img_cell.save(os.path.join(save_dir, f"cell{i}.jpg"))

        img_tensor = torch.tensor(np.array(img_cell)).permute(2, 0, 1)
        image_dict[str(i)] = img_tensor

    if save_images is not None:
        logger.info("-> Tile images saved.")

    if save_dict is not None:
        torch.save(image_dict, save_dict)
        logger.info("-> image_dict saved.")

    return image_dict