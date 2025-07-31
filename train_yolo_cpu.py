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
    print(f"Number of CPUs: {os.cpu_count()}")

    # For AMD integrated graphics, we'll use CPU training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training device: {device}")
    return device

def create_dataset_yaml(train_path, val_path, class_names, save_path="dataset.yaml"):
    """Create dataset configuration file"""
    dataset_config = {
        'path': os.path.abspath('dataset2'),  # Root directory for dataset
        'train': 'images/Train',  # Relative to dataset root
        'val': 'images/Validation',  # Relative to dataset root
        'nc': len(class_names),  # Number of classes
        'names': class_names
    }

    with open(save_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Dataset configuration saved to {save_path}")
    return save_path

def train_yolo_model(dataset_yaml, model_size='n', epochs=100, batch_size=16, img_size=640):
    """
    Train YOLO model for logo detection

    Args:
        dataset_yaml: Path to dataset configuration file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size (adjusted for 32GB RAM)
        img_size: Input image size
    """

    # Load pre-trained model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Training parameters optimized for your system
    training_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': 'cpu',  # Use CPU for AMD integrated graphics
        'workers': 8,     # Utilize your 8 cores
        'patience': 20,   # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,    # Cache images for faster training
        'augment': True,  # Data augmentation
        'mosaic': 1.0,    # Mosaic augmentation
        'mixup': 0.1,     # Mixup augmentation
        'copy_paste': 0.1, # Copy-paste augmentation
        'degrees': 10,    # Rotation augmentation
        'translate': 0.1, # Translation augmentation
        'scale': 0.5,     # Scale augmentation
        'shear': 2.0,     # Shear augmentation
        'perspective': 0.0, # Perspective augmentation
        'flipud': 0.0,    # Vertical flip
        'fliplr': 0.5,    # Horizontal flip
        'hsv_h': 0.015,   # HSV-Hue augmentation
        'hsv_s': 0.7,     # HSV-Saturation augmentation
        'hsv_v': 0.4,     # HSV-Value augmentation
        'label_smoothing': 0.1,  # Label smoothing
        'lr0': 0.01,      # Initial learning rate
        'lrf': 0.1,       # Final learning rate factor
        'momentum': 0.937, # SGD momentum
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,    # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,   # Warmup initial bias learning rate
        'box': 7.5,       # Box loss gain
        'cls': 0.5,       # Classification loss gain
        'dfl': 1.5,       # Distribution focal loss gain
        'pose': 12.0,     # Pose loss gain
        'kobj': 2.0,      # Keypoint object loss gain
        'nbs': 64,        # Nominal batch size
        'overlap_mask': True,    # Overlap masks
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,   # Use dropout regularization
        'val': True,      # Validate during training
        'plots': True,    # Save training plots
        'verbose': True   # Verbose output
    }

    print("=== Starting Training ===")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Workers: {training_args['workers']}")

    # Start training
    results = model.train(**training_args)

    print("=== Training Completed ===")
    return model, results
def continue_training_from_checkpoint(checkpoint_path, dataset_yaml, epochs=50, batch_size=16, img_size=640):
    """
    Continue training from existing checkpoint with optimized parameters

    Args:
        checkpoint_path: Path to existing model (.pt file)
        dataset_yaml: Path to dataset configuration file
        epochs: Additional epochs to train
        batch_size: Batch size for training
        img_size: Input image size
    """

    print(f"Loading existing model from: {checkpoint_path}")
    model = YOLO(checkpoint_path)

    # Continue training parameters (optimized for fine-tuning)
    continue_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': 'cpu',
        'workers': 8,
        'patience': 15,      # Reduced patience for fine-tuning
        'save': True,
        'save_period': 5,    # Save more frequently
        'cache': True,
        'augment': True,
        'mosaic': 0.8,       # Reduced mosaic for fine-tuning
        'mixup': 0.05,       # Reduced mixup
        'copy_paste': 0.05,  # Reduced copy-paste
        'degrees': 5,        # Reduced rotation
        'translate': 0.05,   # Reduced translation
        'scale': 0.3,        # Reduced scale
        'shear': 1.0,        # Reduced shear
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.01,       # Reduced HSV augmentation
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'label_smoothing': 0.05,  # Reduced label smoothing
        'lr0': 0.005,        # Lower initial learning rate for fine-tuning
        'lrf': 0.05,         # Lower final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 1.0,     # Shorter warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.05,   # Lower warmup bias lr
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
        'resume': False      # Don't resume from exact point, use as pretrained
    }

    print("=== Continue Training ===")
    print(f"Base model: {checkpoint_path}")
    print(f"Additional epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {continue_args['lr0']} (reduced for fine-tuning)")
    print(f"Augmentation: Reduced intensity for stability")

    # Start continue training
    results = model.train(**continue_args)

    print("=== Continue Training Completed ===")
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

def main():
    """Main training pipeline with continue training option"""

    # Setup
    device = setup_environment()

    # Dataset configuration - Updated to match your structure
    train_path = "dataset2/images/Train"  # Your training folder
    val_path = "dataset2/images/Validation"  # Your validation folder

    # Define your logo classes here
    # Example classes - modify according to your dataset
    class_names = [
        'RB_back_stand_sign_rear', 'RB_uniform_chest', 'RB_uniform_upper_back', 'RB_led_sign(top)',
        'ntt_uniform_collarbone', 'ntt_uniform_chest', 'ntt_uniform_upper_back', 'ntt_led_sign_top',
        'ntt_back_stand_sign_front', 'ntt_back_stand_sign_rear', 'fuji_uniform_sleeve', 'fuji_led_sign_top',
        'shimamura_uniform_lower_back', 'shimamura_led_sign_top', 'musashi_uniform_pants_hem(front)',
        'musashi_led_sign_top', 'saikan_uniform_pants_hem(back)', 'saikan_led_sign_top',
        'koudensha_led_sign_patternC'
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

    # Training configuration
    MODEL_SIZE = 'n'  # Start with nano model for faster training
    EPOCHS = 100
    BATCH_SIZE = 16   # Optimized for your 32GB RAM
    IMG_SIZE = 640

    print(f"\n=== Dataset Structure ===")
    print(f"Train folder: {train_path}")
    print(f"Validation folder: {val_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Check for existing trained model
    best_model_path = "runs/detect/train2/weights/best.pt"
    last_model_path = "runs/detect/train2/weights/last.pt"

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
                    img_size=IMG_SIZE
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
            img_size=IMG_SIZE
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
