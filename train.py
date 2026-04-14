"""Training entrypoint
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import f1_score
from models.multitask import MultiTaskPerceptionModel
from models.vgg11 import VGG11
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset


def dice_score(pred_logits, target, num_classes=3):
    """Per-batch macro Dice, averaged across classes."""
    pred = torch.argmax(pred_logits, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        intersection = (p & t).float().sum((1, 2))
        union = p.float().sum((1, 2)) + t.float().sum((1, 2))
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dices.append(dice.mean().item())
    return float(np.mean(dices))


def pixel_accuracy(pred_logits, target):
    """Pixel-wise accuracy."""
    pred = torch.argmax(pred_logits, dim=1)
    correct = (pred == target).float().sum()
    total = torch.numel(target).float() if hasattr(torch.numel(target), 'float') else float(target.numel())
    return (correct / target.numel()).item()


def log_activation_distributions(model_with_bn, model_without_bn, sample_batch, device, epoch):
    """For Section 2.1 — plot activation distributions with/without BatchNorm."""
    imgs = sample_batch['image'].to(device)
    hooks = {}

    def make_hook(name):
        def hook(module, input, output):
            hooks[name] = output.detach().cpu().flatten().numpy()
        return hook

    h1 = model_with_bn.encoder.conv3[0].register_forward_hook(make_hook('with_bn'))
    with torch.no_grad():
        model_with_bn(imgs)
    h1.remove()

    h2 = model_without_bn.encoder.conv3[0].register_forward_hook(make_hook('without_bn'))
    with torch.no_grad():
        model_without_bn(imgs)
    h2.remove()

    wandb.log({
        "2.1 Activation Dist with BN": wandb.Histogram(hooks.get('with_bn', [0])),
        "2.1 Activation Dist without BN": wandb.Histogram(hooks.get('without_bn', [0])),
        "Epoch": epoch
    })


def main(args):
    run_name = args.run_name or f"dropout{args.dropout_p}_{args.strategy}"
    wandb.init(
        project="da6401-assignment-2",
        entity="da6401-fdl",
        name=run_name,
        config=vars(args)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = OxfordIIITPetDataset(args.data_dir, split='train')
    val_dataset = OxfordIIITPetDataset(args.data_dir, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    model_no_bn = None
    if args.log_batchnorm_comparison:
        class VGG11NoBN(nn.Module):
            """VGG11 backbone without BatchNorm — used only for 2.1 activation plots."""
            def __init__(self):
                super().__init__()
                self.encoder = _make_vgg11_no_bn()
            def forward(self, x):
                return self.encoder(x)

        def _make_vgg11_no_bn():
            enc = VGG11()
       
            for name, module in enc.named_children():
                if isinstance(module, nn.Sequential):
                    new_layers = []
                    for layer in module:
                        if isinstance(layer, nn.BatchNorm2d):
                            new_layers.append(nn.Identity())
                        else:
                            new_layers.append(layer)
                    setattr(enc, name, nn.Sequential(*new_layers))
            return enc

        model_no_bn = MultiTaskPerceptionModel(
            num_breeds=37, seg_classes=3, dropout_p=args.dropout_p
        ).to(device)
        
        for seq_name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            seq = getattr(model_no_bn.encoder, seq_name)
            new_layers = []
            for layer in seq:
                new_layers.append(nn.Identity() if isinstance(layer, nn.BatchNorm2d) else layer)
            setattr(model_no_bn.encoder, seq_name, nn.Sequential(*new_layers))


    model = MultiTaskPerceptionModel(
        num_breeds=37, seg_classes=3, dropout_p=args.dropout_p
    )
    model.set_transfer_learning_strategy(args.strategy)
    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    criterion_class = nn.CrossEntropyLoss()
    criterion_loc_iou = IoULoss(reduction='mean')
    criterion_loc_mse = nn.MSELoss()
    criterion_seg = nn.CrossEntropyLoss()
    


    fixed_batch = next(iter(val_loader))

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # ---- Train --------------------------------------------------------
        model.train()
        train_loss_total = 0.0
        train_loss_class = 0.0
        train_loss_loc = 0.0
        train_loss_seg = 0.0

        for batch in train_loader:
            imgs = batch['image'].to(device)
            target_class = batch['class_id'].to(device)
            target_bbox = batch['bbox'].to(device)
            target_mask = batch['mask'].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            l_class = criterion_class(out['classification'], target_class)
            l_loc = criterion_loc_mse(out['localization'], target_bbox) + criterion_loc_iou(out['localization'], target_bbox)
            l_seg = criterion_seg(out['segmentation'], target_mask)
            loss = l_class + l_loc + l_seg

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_loss_class += l_class.item()
            train_loss_loc += l_loc.item()
            train_loss_seg += l_seg.item()

        lr_scheduler.step()
        n_train = len(train_loader)

        model.eval()
        val_loss_total = 0.0
        val_loss_class = 0.0
        val_loss_loc = 0.0
        val_loss_seg = 0.0
        val_dice_acc = 0.0
        val_pixel_acc = 0.0
        preds_class, targets_class = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                target_class = batch['class_id'].to(device)
                target_bbox = batch['bbox'].to(device)
                target_mask = batch['mask'].to(device)

                out = model(imgs)

                l_class = criterion_class(out['classification'], target_class)
                l_loc = criterion_loc_mse(out['localization'], target_bbox) + criterion_loc_iou(out['localization'], target_bbox)
                l_seg = criterion_seg(out['segmentation'], target_mask)

                val_loss_total += (l_class + l_loc + l_seg).item()
                val_loss_class += l_class.item()
                val_loss_loc += l_loc.item()
                val_loss_seg += l_seg.item()

        
                val_dice_acc += dice_score(out['segmentation'], target_mask)
                val_pixel_acc += pixel_accuracy(out['segmentation'], target_mask)

                preds_class.extend(torch.argmax(out['classification'], dim=1).cpu().numpy())
                targets_class.extend(target_class.cpu().numpy())

        n_val = len(val_loader)
        val_f1 = f1_score(targets_class, preds_class, average='macro', zero_division=0)

        
        log_dict = {

            "Train/Loss Total": train_loss_total / n_train,
            "Val/Loss Total": val_loss_total / n_val,

            "Train/Loss Classification": train_loss_class / n_train,
            "Train/Loss Localization": train_loss_loc / n_train,
            "Train/Loss Segmentation": train_loss_seg / n_train,
            "Val/Loss Classification": val_loss_class / n_val,
            "Val/Loss Localization": val_loss_loc / n_val,
            "Val/Loss Segmentation": val_loss_seg / n_val,
           
            "Val/Macro F1": val_f1,
           
            "Val/Dice Score": val_dice_acc / n_val,
            
            "Val/Pixel Accuracy": val_pixel_acc / n_val,
            "Epoch": epoch
        }
        wandb.log(log_dict)


        if args.log_batchnorm_comparison and model_no_bn is not None:
            log_activation_distributions(model, model_no_bn, fixed_batch, device, epoch)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss_total/n_train:.4f} | "
            f"Val Loss: {val_loss_total/n_val:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Dice: {val_dice_acc/n_val:.4f} | "
            f"Val PixAcc: {val_pixel_acc/n_val:.4f}"
        )


        if epoch == 0 or val_loss_total / n_val < best_val_loss:
            best_val_loss = val_loss_total / n_val
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pth")
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_metric': best_val_loss,
            }, best_ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--strategy', type=str, default='full_fine_tune',
        choices=['strict_extractor', 'partial_fine_tune', 'full_fine_tune'],
        help="Transfer learning strategy (Section 2.3)"
    )
    parser.add_argument(
        '--dropout_p', type=float, default=0.5,
        help="Dropout probability (0.0 = no dropout). Section 2.2"
    )
    parser.add_argument(
        '--log_batchnorm_comparison', action='store_true',
        help="Also log activation distributions with/without BN for Section 2.1"
    )
    parser.add_argument(
        '--run_name', type=str, default='',
        help="W&B run display name. Auto-generated if empty."
    )
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
