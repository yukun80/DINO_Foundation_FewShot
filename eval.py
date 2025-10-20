"""
This script is dedicated to evaluating a trained Few-Shot Semantic Segmentation model.
It has been refactored to use Sacred for experiment tracking.

--- Sacred Integration Details ---

Purpose:
    - To evaluate a pre-trained model checkpoint in a reproducible manner.
    - Each evaluation run is saved to `experiments/FSS_Evaluation`.
    - Logs detailed performance metrics (OA, Precision, Recall, F1, mIoU) to Sacred.

--- Example Usage ---

python3 eval.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' method=multilayer nb_shots=20

"""

# --- Example Command ---
# python3 eval.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
# -----------------------

import os
import yaml
import torch
import warnings
from typing import Dict, Any
from sacred import Experiment
from sacred.observers import FileStorageObserver

# --- Project-specific Imports ---
from utils.train_utils import get_dataset_loaders
from models.backbones.dino import DINO_linear

warnings.filterwarnings("ignore")

# --- Sacred Experiment Setup ---
ex = Experiment("FSS_Evaluation")
ex.observers.append(FileStorageObserver('experiments/FSS_Evaluation'))


@ex.config
def cfg():
    """
    Defines the default configuration for the evaluation experiment.
    """
    # Load base configuration from the YAML file
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    # --- Command-line accessible parameters ---
    checkpoint_path = None  # REQUIRED: Path to the trained model .pth file
    model_name = "DINO"
    method = "linear"
    dataset = "disaster"
    nb_shots = 10
    input_size = 512

    # Merge CLI-accessible parameters into the main config dictionary
    config.update({
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "method": method,
        "dataset": dataset,
        "number_of_shots": nb_shots,
        "input_size": input_size,
    })


class Metrics:
    """
    A comprehensive metrics calculator for semantic segmentation.
    This class is designed to be clear, correct, and provide a thorough evaluation.
    """
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer.")
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Updates the confusion matrix with a new batch of predictions and targets."""
        pred = pred.cpu().flatten()
        target = target.cpu().flatten()
        mask = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[mask] + pred[mask]
        self.mat += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        """Computes all relevant metrics from the confusion matrix."""
        h = self.mat.float()
        epsilon = 1e-6

        tp = torch.diag(h)
        fp = h.sum(0) - tp
        fn = h.sum(1) - tp

        iou = (tp + epsilon) / (tp + fp + fn + epsilon)
        precision = (tp + epsilon) / (tp + fp + epsilon)
        recall = (tp + epsilon) / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        overall_accuracy = torch.diag(h).sum() / (h.sum() + epsilon)

        # Per-class metrics for the "landslide" class (class 1)
        landslide_iou = iou[1] if self.num_classes > 1 else iou[0]
        landslide_precision = precision[1] if self.num_classes > 1 else precision[0]
        landslide_recall = recall[1] if self.num_classes > 1 else recall[0]
        landslide_f1 = f1[1] if self.num_classes > 1 else f1[0]

        metrics = {
            "Overall_Accuracy": round(overall_accuracy.item() * 100, 2),
            "Mean_IoU": round(iou.mean().item() * 100, 2),
            "Mean_Precision": round(precision.mean().item() * 100, 2),
            "Mean_Recall": round(recall.mean().item() * 100, 2),
            "Mean_F1-Score": round(f1.mean().item() * 100, 2),
            "Landslide_IoU": round(landslide_iou.item() * 100, 2),
            "Landslide_Precision": round(landslide_precision.item() * 100, 2),
            "Landslide_Recall": round(landslide_recall.item() * 100, 2),
            "Landslide_F1-Score": round(landslide_f1.item() * 100, 2),
        }
        return metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    """
    The main evaluation function.

    Args:
        model: The model to evaluate.
        data_loader: The DataLoader for the validation set.
        device: The device to run inference on.
        num_classes: The number of classes in the dataset.

    Returns:
        A dictionary containing the computed performance metrics.
    """
    model.eval()
    metrics_calculator = Metrics(num_classes)
    
    print("Starting evaluation...")
    for image, target, _ in data_loader:
        image, target = image.to(device), target.to(device)
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
        metrics_calculator.update(output.argmax(1), target)

    computed_metrics = metrics_calculator.compute()
    return computed_metrics


@ex.automain
def main(_run, config: Dict[str, Any]):
    """
    The main entry point for the evaluation script, managed by Sacred.
    """
    if config["checkpoint_path"] is None:
        raise ValueError("A `checkpoint_path` must be provided via the command line. Ex: `with checkpoint_path='path/to/model.pth'`")

    # --- Environment Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluating model from: {config['checkpoint_path']}")

    # --- Model Initialization and Loading ---
    print("Initializing model...")
    if config["model_name"] == "DINO":
        model = DINO_linear(
            version=config.get("dino_version", 2),
            method=config["method"],
            num_classes=config["num_classes"],
            input_size=config["input_size"],
            model_repo_path=config["model_repo_path"],
            pretrain_dir=config["pretrain_dir"],
            dinov2_size=config.get("dinov2_size", "base"),
            enable_frequency_adapter=config.get("enable_frequency_adapter", True),
            freq_mask_mode=config.get("freq_mask_mode", "per_layer"),
        )
    else:
        raise NotImplementedError(f"Model '{config['model_name']}' is not supported.")

    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # --- Data Loading ---
    print("Loading validation (query) dataset...")
    _, val_loader, _ = get_dataset_loaders(config)

    # --- Run Evaluation ---
    computed_metrics = evaluate(model, val_loader, device, config["num_classes"])

    # --- Log Metrics and Display Results ---
    print("\n" + "="*50)
    print(" " * 18 + "Evaluation Results")
    print("="*50)
    for name, value in computed_metrics.items():
        # Log each metric to Sacred
        _run.log_scalar(f"metrics.{name}", value)
        print(f"{name:<20}: {value}%")
    print("="*50 + "\n")

    # Add the model path as an artifact for traceability
    _run.add_artifact(config["checkpoint_path"])

    print(f"Evaluation complete. Results logged to Sacred run {_run._id}.")
    return computed_metrics["Mean_IoU"]
