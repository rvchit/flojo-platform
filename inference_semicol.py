import os
import yaml
import torch
import cupy as cp
import cucim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from multi_task_learner import MultiTaskLearner
from utils.config_templates.train import TrainConfig


def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return TrainConfig(**config_dict)


def tile_wsi(wsi_path, patch_size):
    slide = cucim.CuImage(wsi_path)
    width, height = slide.shape[:2]
    tiles = []
    coords = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            if x + patch_size <= width and y + patch_size <= height:
                tile = slide.read_region((x, y), (patch_size, patch_size), level=0)
                tile = cp.array(tile, dtype=cp.float32)
                tile = ToTensor()(tile.get())  # Convert to tensor on CPU
                tiles.append(tile)
                coords.append((x, y))

    return tiles, coords, (width, height)


def run_inference(model, tiles, batch_size):
    tile_dataset = TensorDataset(torch.stack(tiles))
    tile_loader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    for batch in tile_loader:
        with torch.no_grad():
            tiles = batch[0].to(model.device)  # Move to the appropriate device
            logits_sgm, _ = model(tiles)  # Only get the segmentation logits
            all_predictions.append(logits_sgm.cpu())

    all_logits_sgm = torch.cat(all_predictions)
    
    return all_logits_sgm


def reconstruct_wsi(predictions, coords, wsi_size, patch_size, num_classes):
    reconstructed_sgm = np.zeros((wsi_size[1], wsi_size[0], num_classes))  # Assuming num_classes for segmentation

    for i, (x, y) in enumerate(coords):
        logits_sgm = predictions[i]
        sgm_pred = logits_sgm.argmax(dim=0).numpy()
        
        reconstructed_sgm[y:y+patch_size, x:x+patch_size] = sgm_pred

    return reconstructed_sgm


def main(config_path, wsi_path, checkpoint_path):
    # Load configuration
    config = load_config(config_path)

    # Tile the WSI
    patch_size = config.experiment.patch_size
    tiles, coords, wsi_size = tile_wsi(wsi_path, patch_size)

    # Load model from checkpoint
    model = MultiTaskLearner.load_from_checkpoint(checkpoint_path, config=config)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Run inference
    batch_size = config.experiment.batch_size
    logits_sgm = run_inference(model, tiles, batch_size)

    # Reconstruct WSI
    num_classes = config.model.arch.num_classes
    reconstructed_sgm = reconstruct_wsi(
        logits_sgm,
        coords, wsi_size, patch_size, num_classes
    )

    # Optionally, save or visualize the reconstructed WSI
    # For example, save the reconstructed WSI as numpy arrays
    output_path = config.paths.output_path
    np.save(os.path.join(output_path, 'reconstructed_sgm.npy'), reconstructed_sgm)


if __name__ == "__main__":
    config_path = 'configs/your_config_file.yaml'  # Replace with the path to your YAML file
    wsi_path = 'path_to_your_wsi'  # Replace with the path to your WSI file
    checkpoint_path = 'ckpts/arm1.ckpt'  # Replace with the path to your checkpoint file
    main(config_path, wsi_path, checkpoint_path)
