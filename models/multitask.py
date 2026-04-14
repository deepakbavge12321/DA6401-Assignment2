"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import gdown
        import os

        os.makedirs(os.path.dirname(classifier_path) if os.path.dirname(classifier_path) else '.', exist_ok=True)

        if not os.path.exists(classifier_path):
            gdown.download(
                url=f"https://drive.google.com/uc?id=1rD0WYTwXeBdbWvz4TuOYa8eanMGloiid",
                output=classifier_path,
                quiet=False
            )

        if not os.path.exists(localizer_path):
            gdown.download(
                url=f"https://drive.google.com/uc?id=1MBS2oWNmprQfzFvR5si5TuA9H1g72KMR",
                output=localizer_path,
                quiet=False
            )

        if not os.path.exists(unet_path):
            gdown.download(
                url=f"https://drive.google.com/uc?id=1HI6AfVyUUs1DN67DCzcG1t9fUdp0ERmA",
                output=unet_path,
                quiet=False
            )

        self.encoder = VGG11(in_channels=in_channels)
        
        # Classification head components
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classification_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )
        
        # Localization head components
        self.localization_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4),
            nn.Sigmoid()
        )
        
        # Segmentation decoder components
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, seg_classes, kernel_size=1)
        )

        def _load(path, label):
            if not os.path.exists(path):
                print(f"[MultiTask] WARNING: {label} checkpoint not found at {path}")
                return
            ckpt = torch.load(path, map_location='cpu')
            state = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"[MultiTask] Loaded {label} — missing: {len(missing)}, unexpected: {len(unexpected)}")

        # Load encoder + classification head from classifier checkpoint
        _load(classifier_path, 'classifier')

        # Load encoder + localization head from localizer checkpoint
        _load(localizer_path, 'localizer')

        # Load encoder + segmentation decoder from unet checkpoint
        _load(unet_path, 'unet')

    def set_transfer_learning_strategy(self, strategy: str):
        """
        Set the parameter requires_grad based on strategy.
        Args:
            strategy: 'strict_extractor', 'partial_fine_tune', 'full_fine_tune'
        """
        if strategy == 'full_fine_tune':
            for param in self.parameters():
                param.requires_grad = True
        elif strategy == 'strict_extractor':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif strategy == 'partial_fine_tune':
            for param in self.parameters():
                param.requires_grad = True
            for param in self.encoder.conv1.parameters():
                param.requires_grad = False
            for param in self.encoder.conv2.parameters():
                param.requires_grad = False
            for param in self.encoder.conv3.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        _, _, h, w = x.shape
        
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Base representation for flat heads
        base_feat = self.avgpool(bottleneck)
        base_feat = torch.flatten(base_feat, 1)
        
        # Classification
        class_logits = self.classification_head(base_feat)
        
        # Localization
        loc_out = self.localization_head(base_feat)
        scale = torch.tensor([w, h, w, h], dtype=loc_out.dtype, device=loc_out.device)
        loc_boxes = loc_out * scale
        
        # Segmentation
        d5 = self.upconv5(bottleneck)
        d5 = torch.cat((d5, features['stage5']), dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat((d4, features['stage4']), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, features['stage3']), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, features['stage2']), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, features['stage1']), dim=1)
        seg_logits = self.dec1(d1)
        
        return {
            'classification': class_logits,
            'localization': loc_boxes,
            'segmentation': seg_logits
        }