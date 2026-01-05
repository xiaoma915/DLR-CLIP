import os
from torch.utils.data import Dataset

from typing import Any, Tuple
from PIL import Image
from .imagenet import imagenet_classes, imagenet_templates


class ImageNetSketch(Dataset):
    """ImageNetSketch.

    This dataset is used for testing only.
    """
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.template = imagenet_templates
        self.classnames = imagenet_classes
        self.transform = transform
        self.samples = []
        print(f"{root=}")
        
        # 创建WordNet ID到索引的映射
        synset_to_idx = {}
        mapping_file = "/media/yang/Elements SE/ImageNet/images/LOC_synset_mapping.txt"
        if os.path.exists(mapping_file):
            for idx, line in enumerate(open(mapping_file)):
                synset = line.split()[0]
                synset_to_idx[synset] = idx
        else:
            # 备用方案：使用目录名的排序顺序作为索引
            folders = sorted([f.name for f in os.scandir(root) if f.is_dir() and f.name.startswith('n')])
            for idx, folder in enumerate(folders):
                synset_to_idx[folder] = idx
        
        folders = sorted(f.name for f in os.scandir(root) if f.is_dir())
        for folder in folders:
            if folder.startswith('n'):  # 确保是WordNet ID格式
                # 使用WordNet ID获取类别索引
                class_id = synset_to_idx.get(folder, -1)
                if class_id != -1:  # 只处理有效的类别
                    for cur_root, dirs, fnames in os.walk(os.path.join(root, folder)):
                        if len(fnames) > 0:
                            for fname in fnames:
                                img_path = os.path.join(cur_root, fname)
                                self.samples.append((img_path, class_id))
        
    def loader(self, path )-> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        new_sample = self.transform(sample)
        return new_sample, target
    
    def __len__(self) -> int:
        return len(self.samples)