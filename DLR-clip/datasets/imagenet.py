import random
from collections import defaultdict
import os
from pathlib import Path

import torchvision
from typing import Any, Tuple
from PIL import Image

from .utils import Datum, DatasetBase


# ImageNet 类别列表和模板
imagenet_classes = []  # 由 _load_official_classnames() 从 LOC_synset_mapping.txt 动态加载

imagenet_templates = ["itap of a {}.",
                        "a bad photo of the {}.",
                        "a origami {}.",
                        "a photo of the large {}.",
                        "a {} in a video game.",
                        "art of the {}.",
                        "a photo of the small {}."]


gradacam_templates = ["itap of {} in black background.",
                        "a bad photo of the {} in black background.",
                        "a origami {} in black background.",
                        "a photo of the large {} in black background.",
                        "a {} in a video game in black background.",
                        "art of the {} in black background.",
                        "a photo of the small {} in black background."]

    
class MyImageNet(DatasetBase):
    """ImageNet dataset using DatasetBase structure like SUN397, Caltech101, etc."""

    def __init__(self, root: str, num_shots: int, split: str = 'train', transform: Any = None, **kwargs: Any) -> None:
        if split.lower() in ['test', 'val']:
            split = 'val'
        
        self.dataset_dir = root
        self.template = imagenet_templates
        self.transform = transform
        
        # Load official ImageNet class ID to index mapping from LOC_synset_mapping.txt
        self.synset_to_idx = self._load_synset_mapping(root)
        self.official_classnames = self._load_official_classnames(root)
        
        # Load data based on split
        if split == 'train':
            train_items = self._load_train_split(root)
            # Apply few-shot sampling like DatasetBase.generate_fewshot_dataset
            train_items = self.generate_fewshot_dataset(train_items, num_shots=num_shots)
            super().__init__(train_x=train_items, val=None, test=None)
        else:  # val/test
            val_items = self._load_val_split(root)
            # 提供一个非空的占位符以便 DatasetBase 初始化
            train_placeholder = self._load_train_split(root)[:1]
            super().__init__(train_x=train_placeholder, val=val_items, test=None)
            # 实际数据会被 DataLoader 使用
            self._train_x = val_items
        
        # 为 ImageNet 设置完整的 classnames 列表
        self._classnames = self.official_classnames if self.official_classnames else imagenet_classes
        self._lab2cname = {i: self._classnames[i] for i in range(len(self._classnames))}
        self._num_classes = len(self._classnames)
    
    def _load_synset_mapping(self, root: str):
        """Load ImageNet official synset to class index mapping from LOC_synset_mapping.txt"""
        synset_to_idx = {}
        mapping_file = os.path.join(root, 'LOC_synset_mapping.txt')
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    for idx, line in enumerate(f):
                        synset = line.split()[0]  # Extract synset ID like n01440764
                        synset_to_idx[synset] = idx
                print(f"[ImageNet] 成功加载{len(synset_to_idx)}个synset映射")
            except Exception as e:
                print(f"[Warning] 读取synset映射文件失败: {e}")
        else:
            print(f"[Warning] 未找到synset映射文件: {mapping_file}")
        
        return synset_to_idx
    
    def _load_official_classnames(self, root: str):
        """Load official ImageNet class names from LOC_synset_mapping.txt"""
        classnames = []
        mapping_file = os.path.join(root, 'LOC_synset_mapping.txt')
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) >= 2:
                            classname = parts[1].split(',')[0].strip()
                            classnames.append(classname)
                print(f"[ImageNet] 成功加载{len(classnames)}个官方类别名称")
            except Exception as e:
                print(f"[Warning] 读取官方类别名称失败: {e}")
        
        return classnames if classnames else None
    
    def _load_train_split(self, root: str):
        """Load training split from train/ folder"""
        items = []
        train_dir = os.path.join(root, 'train')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        
        # 按排序顺序遍历类别目录
        for class_dir in sorted(os.listdir(train_dir)):
            class_path = os.path.join(train_dir, class_dir)
            if os.path.isdir(class_path):
                if self.synset_to_idx and class_dir in self.synset_to_idx:
                    class_idx = self.synset_to_idx[class_dir]
                else:
                    class_idx = 0
                
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        item = Datum(
                            impath=img_path,
                            label=class_idx,
                            classname=self.official_classnames[class_idx] if self.official_classnames else f"class_{class_idx}"
                        )
                        items.append(item)
        
        return items
    
    def _load_val_split(self, root: str):
        """Load validation split from val/ folder using official label file"""
        items = []
        val_dir = os.path.join(root, 'val')
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Val directory not found: {val_dir}")
        
        # Load official validation labels from ILSVRC2012_validation_ground_truth.txt
        label_map = self._load_validation_labels(root)
        
        # 检查val目录的结构：是否已经按synset分类
        val_contents = os.listdir(val_dir)
        has_synset_dirs = any(item.startswith('n') and os.path.isdir(os.path.join(val_dir, item)) for item in val_contents)
        
        if has_synset_dirs:
            # 新结构：val目录下有1000个synset子文件夹
            print("[ImageNet] 检测到val已按synset分类")
            for synset_dir in sorted(os.listdir(val_dir)):
                synset_path = os.path.join(val_dir, synset_dir)
                if os.path.isdir(synset_path) and synset_dir.startswith('n'):
                    # 获取该synset对应的类别索引
                    if synset_dir in self.synset_to_idx:
                        class_idx = self.synset_to_idx[synset_dir]
                        # 遍历该synset文件夹中的所有图像
                        for img_name in os.listdir(synset_path):
                            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                                img_path = os.path.join(synset_path, img_name)
                                item = Datum(
                                    impath=img_path,
                                    label=class_idx,
                                    classname=self.official_classnames[class_idx] if self.official_classnames else f"class_{class_idx}"
                                )
                                items.append(item)
        else:
            # 如果val目录未按synset分类，则尝试使用标签文件读取
            # 不过正常情况下应该已经按synset分类了
            print("[ImageNet] 警告：val目录未按synset分类")
        
        print(f"[ImageNet] 成功加载{len(items)}个验证集样本")
        return items
    
    def _load_validation_labels(self, root: str):
        """Load validation ground truth labels from official ILSVRC2012 file"""
        label_map = {}
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(root, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'),
            os.path.join(root, 'ILSVRC2012_validation_ground_truth.txt'),
            os.path.join(os.path.dirname(root), 'ILSVRC2012_validation_ground_truth.txt'),
        ]
        
        label_file = None
        for path in possible_paths:
            if os.path.exists(path):
                label_file = path
                break
        
        if label_file:
            try:
                with open(label_file, 'r') as f:
                    for idx, line in enumerate(f):
                        # Convert from 1-1000 to 0-999
                        label = int(line.strip()) - 1
                        label_map[idx] = label
                print(f"[ImageNet] 成功加载{len(label_map)}个验证集标签 from {label_file}")
            except Exception as e:
                print(f"[Warning] 读取验证集标签文件失败: {e}")
        else:
            print(f"[Warning] 未找到验证集标签文件")
        
        return label_map
    
    def __len__(self):
        """Return the number of training samples"""
        return len(self.train_x) if self.train_x else 0
    
    def __getitem__(self, index):
        """Return image and label at given index"""
        item = self.train_x[index]
        img = Image.open(item.impath).convert('RGB')
        if hasattr(self, 'transform') and self.transform is not None:
            img = self.transform(img)
        return img, item.label
