from __future__ import annotations

import argparse
import os

from loguru import logger

from external.mocov3.run_infer import infer_embed
from external.mocov3.utils import image_dict_to_h5
from external.mocov3.utils import run_ssl


def main(
    image_path: str,
    save_path: str,
    tag: str,
    batch_size_infer: int = 2048,
    num_workers: int = 4,
) -> None:
    """
    Main function to run SSL on a dataset.

    Args:
        image_path: Path to the image dict or WSI.
        save_path: Folder to save the results.
        tag: Tag for the run.
        batch_size_infer: Batch size for inference.
        num_workers: Number of workers for data loading.
    """

    # Check your image path
    if image_path.endswith(".pt"):
        logger.info(f"Your image path is a dictionary: {image_path}")
    else:
        raise ValueError("Please ensure the image path is a .pt file.")

    # Save h5 file
    h5_folder = os.path.join(save_path, "cell_images")
    if not os.path.exists(h5_folder):
        os.makedirs(h5_folder)

    sample_id = f"slide_{tag}"
    h5_path = os.path.join(h5_folder, f"{sample_id}.h5")
    image_dict_to_h5(image_path, h5_path)

    # Run SSL with Moco-v3
    run_ssl(save_path, None, [sample_id], tag, 1, 4)

    # Run inference to get embeddings
    logger.info("-> Running inference to get embeddings")
    infer_embed(image_path, save_path, tag, model_name="resnet50", batch_size=batch_size_infer, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SSL on a dataset")
    parser.add_argument("--image_path", type=str, help="Path to the image dict or WSI")
    parser.add_argument("--save_path", type=str, help="Folder to save the results")
    parser.add_argument("--tag", type=str, help="Tag for the run")
    parser.add_argument("--batch_size_infer", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args.image_path, args.save_path, args.tag, args.batch_size_infer, args.num_workers)
