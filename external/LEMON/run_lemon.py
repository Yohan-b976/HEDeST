from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from lemon_utils import get_vit_feature_extractor
from lemon_utils import prepare_transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm


# -------------------------
# Dataset
# -------------------------
class ImageDictDataset(Dataset):
    """Dataset for loading images from a pre-saved image_dict.pt"""

    def __init__(self, image_dict: Dict, transform):
        self.image_dict = image_dict
        self.transform = transform
        self.cell_ids = list(image_dict.keys())
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        image = self.image_dict[cell_id]

        image = image.float() / 255.0
        image = self.to_pil(image)
        image = self.transform(image)

        return image, cell_id


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViT embeddings using a pretrained LEMON model")

    parser.add_argument("--image-dict", type=Path, required=True, help="Path to image_dict.pt")
    parser.add_argument("--output-path", type=Path, required=True, help="Output filename")
    parser.add_argument("--model-name", type=str, default="vits8", help="ViT model variant")
    parser.add_argument("--cell-size", type=int, default=40, help="Target image size")
    parser.add_argument(
        "--weights", type=Path, default=Path("pretrained/lemon.pth.tar"), help="Path to pretrained weights"
    )
    parser.add_argument("--stats", type=Path, default=Path("mean_std.json"), help="Path to mean/std statistics JSON")

    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")

    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load model & transforms
    # -------------------------
    transform = prepare_transform(args.stats, size=args.cell_size)

    model = get_vit_feature_extractor(args.weights, args.model_name, img_size=args.cell_size)
    model.eval()
    model.to(device)

    # -------------------------
    # Load data
    # -------------------------
    image_dict = torch.load(args.image_dict)

    dataset = ImageDictDataset(image_dict, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -------------------------
    # Inference
    # -------------------------
    embeddings_dict = {}

    for images, cell_ids in tqdm(dataloader, desc="Extracting embeddings"):
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float32):
            images = images.to(device)
            embeddings = model(images)
            embeddings = embeddings.view(embeddings.shape[0], -1)

            for i, cell_id in enumerate(cell_ids):
                embeddings_dict[cell_id] = embeddings[i].cpu()

    # -------------------------
    # Save
    # -------------------------
    torch.save(embeddings_dict, args.output_path)

    print("Feature extraction completed")
    print(f"Saved embeddings to: {args.output_path}")
    print(f"Total cells embedded: {len(embeddings_dict)}")


if __name__ == "__main__":
    main()
