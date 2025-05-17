# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:22:44 2025

@author: ARNES
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple

def split_dataset(
    source_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
):
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
    
    random.seed(seed)  # Set the random seed for reproducibility
    source_dir = Path(source_dir)  # Convert source directory to Path object
    output_dir = Path(output_dir)  # Convert output directory to Path object

    # Create output folders for train, validate, and test splits
    for split in ['train', 'validate', 'test']:
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir():  # Check if it is a directory
                target_dir = output_dir / split / class_dir.name  # Define target directory
                target_dir.mkdir(parents=True, exist_ok=True)  # Create target directory if it doesn't exist

    # Process each class directory
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():  # Skip if not a directory
            continue
        images = list(class_dir.glob("*"))  # Get all image files in the class directory
        random.shuffle(images)  # Shuffle the images randomly
        
        total = len(images)  # Total number of images
        train_end = int(split_ratio[0] * total)  # End index for training set
        val_end = train_end + int(split_ratio[1] * total)  # End index for validation set

        # Split data into train, validate, and test sets
        split_data = {
            'train': images[:train_end],
            'validate': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Copy files to their respective directories
        for split, files in split_data.items():
            for file_path in files:
                dest = output_dir / split / class_dir.name / file_path.name  # Destination path
                shutil.copy(file_path, dest)  # Copy file to destination

    print("Dataset successfully split and copied!")

# Example usage:
# For two folders (train/test only), change ratio to (0.8, 0.0, 0.2)
split_dataset(
    source_dir=r'C:\deep_learning\dataset\flower-dataset',  # Path to the source dataset folder
    output_dir=r'C:\deep_learning\dataset\flower-dataset',  # Path to the output folder
    split_ratio=(0.7, 0.15, 0.15)  # Adjust as needed
)