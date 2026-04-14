"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None, img_size: int = 224):
        """
        Args:
            root_dir: Path to dataset root (containing 'images' and 'annotations').
            split: 'train' or 'test'.
            transform: Albumentations transforms.
            img_size: Target image size.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.images_dir = os.path.join(root_dir, 'images')
        self.trimaps_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xmls_dir = os.path.join(root_dir, 'annotations', 'xmls')
        
        # Always read from trainval.txt (test.txt has no XML bounding boxes)
        # then do a deterministic random 80/20 train/val split
        list_path = os.path.join(root_dir, 'annotations', 'trainval.txt')
        
        all_samples = []
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        basename = parts[0]
                        class_id = int(parts[1]) - 1  # 0-indexed
                        # Only keep samples that have bounding boxes for multi-task
                        xml_path = os.path.join(self.xmls_dir, f"{basename}.xml")
                        trimap_path = os.path.join(self.trimaps_dir, f"{basename}.png")
                        if os.path.exists(xml_path) and os.path.exists(trimap_path):
                            all_samples.append({
                                'basename': basename,
                                'class_id': class_id
                            })
        else:
            # Fallback if list file missing, just read all xmls
            if os.path.exists(self.xmls_dir):
                for xml_file in os.listdir(self.xmls_dir):
                    if xml_file.endswith('.xml'):
                        basename = xml_file[:-4]
                        all_samples.append({'basename': basename, 'class_id': 0})

        # Deterministic 80/20 split using fixed seed (ensures train/val never overlap)
        rng = random.Random(42)
        all_samples_shuffled = all_samples.copy()
        rng.shuffle(all_samples_shuffled)
        split_idx = int(0.8 * len(all_samples_shuffled))

        if split == 'train':
            self.samples = all_samples_shuffled[:split_idx]
        else:
            self.samples = all_samples_shuffled[split_idx:]
        
        # Default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            else:
                self.transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = transform

    def _parse_xml(self, xml_path, img_w, img_h):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to YOLO format (x_center, y_center, width, height) normalized to [0, 1]
        w = xmax - xmin
        h = ymax - ymin
        x_c = xmin + w / 2
        y_c = ymin + h / 2
        
        return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        basename = sample['basename']
        
        img_path = os.path.join(self.images_dir, f"{basename}.jpg")
        trimap_path = os.path.join(self.trimaps_dir, f"{basename}.png")
        xml_path = os.path.join(self.xmls_dir, f"{basename}.xml")
        
        # Load Image
        image_pil = Image.open(img_path).convert('RGB')
        img_w, img_h = image_pil.size
        image = np.array(image_pil)
        
        # Load Trimap (1: foreground, 2: background, 3: non-classified)
        # Convert to 0, 1, 2 for PyTorch CrossEntropyLoss
        mask = Image.open(trimap_path)
        mask = np.array(mask) - 1
        
        # Load BBox
        bbox = self._parse_xml(xml_path, img_w, img_h)
        
        class_id = sample['class_id']
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image, 
                mask=mask, 
                bboxes=[bbox], 
                class_labels=[class_id]
            )
            image = transformed['image']
            mask = transformed['mask']
            transformed_bbox = transformed['bboxes'][0]
            # Absolute pixel scale matching image dimensions
            bbox_tensor = torch.tensor([
                transformed_bbox[0] * self.img_size, 
                transformed_bbox[1] * self.img_size, 
                transformed_bbox[2] * self.img_size, 
                transformed_bbox[3] * self.img_size
            ], dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
            
        return {
            'image': image,
            'class_id': torch.tensor(class_id, dtype=torch.long),
            'bbox': bbox_tensor,
            'mask': torch.tensor(mask, dtype=torch.long),
            'img_path': img_path
        }