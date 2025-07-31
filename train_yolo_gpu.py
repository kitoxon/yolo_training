#!/usr/bin/env python3
"""
YOLO Logo Detection Training Script
Optimized for AMD Ryzen 7 7735HS with integrated graphics
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_environment():
    """Setup environment and check system capabilities"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        device = 'cuda'
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Training device: {device}")
    return device

def create_dataset_yaml(train_path, val_path, class_names, save_path="dataset.yaml"):
    """Create dataset configuration file"""
    dataset_config = {
        'path': os.path.abspath('dataset_fc_tokyo_home'),  # Root directory for dataset
        'train': 'images/train',  # Relative to dataset root
        'val': 'images/val',  # Relative to dataset root
        'nc': len(class_names),  # Number of classes
        'names': class_names
    }

    with open(save_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Dataset configuration saved to {save_path}")
    return save_path

def train_yolo_model(dataset_yaml, model_size='n', epochs=100, batch_size=24, img_size=640, device='cuda'):
    """
    Train YOLO model for logo detection

    Args:
        dataset_yaml: Path to dataset configuration file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size (optimized for GPU)
        img_size: Input image size
        device: Training device ('cuda' or 'cpu')
    """

    # Load pre-trained model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Determine optimal batch size based on GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory:.1f} GB")

        # Suggested batch sizes based on GPU memory and model size
        batch_suggestions = {
            'n': min(batch_size, int(gpu_memory * 16)),  # Nano model
            's': min(batch_size, int(gpu_memory * 12)),  # Small model
            'm': min(batch_size, int(gpu_memory * 8)),   # Medium model
            'l': min(batch_size, int(gpu_memory * 6)),   # Large model
            'x': min(batch_size, int(gpu_memory * 4))    # Extra Large model
        }

        suggested_batch = batch_suggestions.get(model_size, batch_size)
        if suggested_batch != batch_size:
            print(f"Adjusting batch size from {batch_size} to {suggested_batch} based on GPU memory")
            batch_size = suggested_batch

    # Training parameters optimized for GPU
    training_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 4 if device == 'cuda' else 8,  # Fewer workers for GPU to avoid bottleneck
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': True,
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'label_smoothing': 0.1,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True,
        'amp': True,  # Automatic Mixed Precision for faster GPU training
        'fraction': 1.0,  # Use full dataset
        'profile': False,  # Disable profiling for faster training
        'freeze': None,   # Don't freeze any layers
        'multi_scale': True,  # Multi-scale training for better generalization
        'optimizer': 'auto',  # Let YOLO choose optimal optimizer
        'close_mosaic': 10,   # Disable mosaic augmentation for last 10 epochs
    }

    print("=== Starting GPU Training ===")
    print(f"Model: YOLOv8{model_size}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Workers: {training_args['workers']}")
    print(f"Mixed Precision: {training_args['amp']}")

    # Start training
    results = model.train(**training_args)

    print("=== Training Completed ===")
    return model, results

def validate_model(model, dataset_yaml):
    """Validate the trained model"""
    print("=== Validating Model ===")
    results = model.val(data=dataset_yaml, plots=True)
    return results

def export_model(model, format='onnx'):
    """Export model to different formats"""
    print(f"=== Exporting Model to {format.upper()} ===")
    model.export(format=format)
def continue_training_from_checkpoint(checkpoint_path, dataset_yaml, epochs=50, batch_size=24, img_size=640, device='cuda'):
    """
    Continue training from existing checkpoint with optimized parameters

    Args:
        checkpoint_path: Path to existing model (.pt file)
        dataset_yaml: Path to dataset configuration file
        epochs: Additional epochs to train
        batch_size: Batch size for training
        img_size: Input image size
        device: Training device ('cuda' or 'cpu')
    """

    print(f"Loading existing model from: {checkpoint_path}")
    model = YOLO(checkpoint_path)

    # Adjust batch size for GPU if needed
    if device == 'cuda' and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU memory: {gpu_memory:.2f} GB")

        if gpu_memory < 8:  # Adjust for smaller GPUs
            batch_size = min(batch_size, 16)
        elif gpu_memory < 12:
            batch_size = min(batch_size, 24)

    # Continue training parameters (optimized for fine-tuning on GPU)
    continue_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 4 if device == 'cuda' else 8,
        'patience': 15,
        'save': True,
        'save_period': 5,
        'cache': True,
        'augment': True,
        'mosaic': 0.8,
        'mixup': 0.05,
        'copy_paste': 0.05,
        'degrees': 5,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 1.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'label_smoothing': 0.05,
        'lr0': 0.005,        # Lower learning rate for fine-tuning
        'lrf': 0.05,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.05,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True,
        'resume': False,
        'amp': True,         # Mixed precision for GPU
        'multi_scale': False, # Disable for stability during fine-tuning
        'optimizer': 'auto',
        'close_mosaic': 5,   # Disable mosaic earlier for fine-tuning
    }

    print("=== Continue Training on GPU ===")
    print(f"Base model: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Additional epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {continue_args['lr0']} (reduced for fine-tuning)")
    print(f"Mixed Precision: {continue_args['amp']}")

    # Start continue training
    results = model.train(**continue_args)

    print("=== Continue Training Completed ===")
    return model, results

def main():
    """Main training pipeline with continue training option"""

    # Setup
    device = setup_environment()

    # Dataset configuration - Updated to match your structure
    train_path = "dataset_fc_tokyo_home/images/train"  # Your training folder
    val_path = "dataset_fc_tokyo_home/images/val"  # Your validation folder

    # Define your logo classes here
    # Example classes - modify according to your dataset
    class_names = [
        'aichi_toyota', 'RB_back_stand_sign_rear', 'RB_uniform_chest', 'RB_uniform_upper_back', 'RB_led_sign(top)',
        'ntt_uniform_collarbone', 'ntt_led_sign_top',
        'ntt_back_stand_sign_front', 'ntt_back_stand_sign_rear', 'fuji_uniform_sleeve', 'fuji_led_sign_top',
        'shimamura_uniform_lower_back', 'shimamura_led_sign_top', 'musashi_uniform_pants_hem(front)',
        'musashi_led_sign_top', 'saikan_uniform_pants_hem(back)', 'saikan_led_sign_top',
        'koudensha_led_sign_patternC', 'shimura_train_uniform', 'tokyo_gas_uniform_chest', 'mixi_uniform_right_collarbone', 'mixi_uniform_pants_hem_front',
        'mitsui_co_uniform_left_collarbone', 'keio_electric_railway_uniform_sleeves', 'mitsubushi_uniform_upper_back', 'new_balance_uniform_pants_hem_back',
        'kiraboshi_FG_led_signboard', 'konami_front_back_stand_signboard', 'ichigo_front_back_stand_signboard', 'ajek_behind_back_stand_signboard'
        # Add all your logo class names here
    ]

    # Verify dataset structure
    if not os.path.exists(train_path):
        print(f"Error: Training directory '{train_path}' not found!")
        return

    if not os.path.exists(val_path):
        print(f"Error: Validation directory '{val_path}' not found!")
        return

    # Create dataset configuration
    dataset_yaml = create_dataset_yaml(train_path, val_path, class_names)

    # Training configuration - GPU optimized
    MODEL_SIZE = 's' if device == 'cuda' else 'n'  # Use larger model for GPU
    EPOCHS = 200 if device == 'cuda' else 100      # More epochs for GPU

    # GPU-optimized batch sizes
    if device == 'cuda':
        BATCH_SIZE = 24  # Larger batch size for GPU
    else:
        BATCH_SIZE = 12  # Smaller batch size for CPU

    IMG_SIZE = 640

    print(f"\n=== Dataset Structure ===")
    print(f"Train folder: {train_path}")
    print(f"Validation folder: {val_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Check for existing trained model
    best_model_path = "runs/detect/train8/weights/best.pt"
    last_model_path = "runs/detect/train8/weights/last.pt"

    # Ask user if they want to continue training or start fresh
    if os.path.exists(best_model_path) or os.path.exists(last_model_path):
        print(f"\n=== Existing Model Found ===")
        if os.path.exists(best_model_path):
            print(f"Best model found: {best_model_path}")
        if os.path.exists(last_model_path):
            print(f"Last model found: {last_model_path}")

        print("\nTraining Options:")
        print("1. Continue training from existing model (recommended)")
        print("2. Start fresh training (will overwrite existing)")

        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Please enter 1 or 2")

        if choice == '1':
            # Continue training
            continue_epochs = int(input(f"Enter additional epochs to train (default: {EPOCHS//2}): ") or EPOCHS//2)
            try:
                model, results = continue_training_from_checkpoint(
                    checkpoint_path=best_model_path if os.path.exists(best_model_path) else last_model_path,
                    dataset_yaml=dataset_yaml,
                    epochs=continue_epochs,
                    batch_size=BATCH_SIZE,
                    img_size=IMG_SIZE,
                    device=device
                )
                print(f"\n=== Continue Training Completed ===")
                print(f"Updated model saved in new training folder")
                return
            except Exception as e:
                print(f"Error during continue training: {e}")
                print("Falling back to fresh training...")

    try:
        # Train the model (fresh start)
        model, results = train_yolo_model(
            dataset_yaml=dataset_yaml,
            model_size=MODEL_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=device
        )

        # Validate the model
        val_results = validate_model(model, dataset_yaml)

        # Export model
        export_model(model, 'onnx')

        print("\n=== Training Summary ===")
        print(f"Best model saved to: runs/detect/train/weights/best.pt")
        print(f"Last model saved to: runs/detect/train/weights/last.pt")
        print("Training plots saved to: runs/detect/train/")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your dataset structure and configuration.")

if __name__ == "__main__":
    main()


def predict_single_image(model_path, image_path, conf_threshold=0.25):
    """Predict on a single image"""
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)
    results[0].show()  # Display results
    return results

def batch_predict(model_path, image_folder, output_folder, conf_threshold=0.25):
    """Batch prediction on multiple images"""
    model = YOLO(model_path)

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, image_file)
            results = model(image_path, conf=conf_threshold)

            # Save results
            output_path = os.path.join(output_folder, f"predicted_{image_file}")
            results[0].save(output_path)

# Usage examples:
"""
# Basic training
python train_yolo_logo.py

# Resume training
resume_training('runs/detect/train/weights/last.pt', epochs=50)

# Predict single image
predict_single_image('runs/detect/train/weights/best.pt', 'test_image.jpg')

# Batch prediction
batch_predict('runs/detect/train/weights/best.pt', 'test_images/', 'predictions/')
"""
