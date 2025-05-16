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
    
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Membuat folder output
    for split in ['train', 'validate', 'test']:
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir():
                target_dir = output_dir / split / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

    # Memproses setiap kelas
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*"))
        random.shuffle(images)
        
        total = len(images)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)

        split_data = {
            'train': images[:train_end],
            'validate': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, files in split_data.items():
            for file_path in files:
                dest = output_dir / split / class_dir.name / file_path.name
                shutil.copy(file_path, dest)

    print("Dataset successfully split and copied!")

# Contoh penggunaan:
# Untuk dua folder (train/test saja), ubah ratio menjadi (0.8, 0.0, 0.2)
split_dataset(
    source_dir=r'C:\deep_learning\dataset\flower-dataset',           # path ke folder dataset asal
    output_dir=r'C:\deep_learning\dataset\flower-dataset',   # path folder tujuan
    split_ratio=(0.7, 0.15, 0.15)   # ubah sesuai kebutuhan
)
