"""Inference and evaluation
"""

import argparse
import io
import torch
import numpy as np
import wandb
from PIL import Image
from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_feature_maps(model, image_tensor, device='cpu'):
    """Log first and last conv layer feature maps to W&B (Section 2.4)."""
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        _, features = model.encoder(img, return_features=True)

    first_fm = features['stage1'][0].cpu()   # [64, H, W]
    last_fm  = features['stage5'][0].cpu()   # [512, H, W]

    def fm_to_image(fm):
        avg = fm.mean(dim=0).numpy()
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-6)
        return (avg * 255).astype(np.uint8)

    wandb.log({
        "2.4_Feature_Map_First_Conv_Layer":
            wandb.Image(fm_to_image(first_fm), caption="Conv Stage-1 (edges & colours)"),
        "2.4_Feature_Map_Last_Conv_Layer":
            wandb.Image(fm_to_image(last_fm),  caption="Conv Stage-5 (semantic shapes)"),
    })



def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf).copy()


def evaluate_predictions(model, dataset, device='cpu',
                         det_samples=10, seg_samples=5):
    """Build and log W&B Tables for Sections 2.5 and 2.6."""
    model.eval()

    table_det = wandb.Table(
        columns=["Image (overlay)", "GT Box", "Pred Box", "IoU", "Confidence"])
    table_seg = wandb.Table(
        columns=["Original", "Ground Truth Trimap", "Predicted Mask"])

    with torch.no_grad():
        for i in range(max(det_samples, seg_samples)):
            sample  = dataset[i]
            img_t   = sample['image'].unsqueeze(0).to(device)
            gt_box  = sample['bbox'].cpu()
            gt_mask = sample['mask'].cpu()

            out      = model(img_t)
            pred_box = out['localization'][0].cpu()
            pred_mask = torch.argmax(out['segmentation'][0], dim=0).cpu()
            conf      = torch.softmax(out['classification'][0].cpu(), dim=0).max().item()

            def _corners(b):
                return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2

            px1,py1,px2,py2 = _corners(pred_box)
            tx1,ty1,tx2,ty2 = _corners(gt_box)
            ix = max(0, min(px2,tx2) - max(px1,tx1))
            iy = max(0, min(py2,ty2) - max(py1,ty1))
            inter = float(ix * iy)
            union = float(pred_box[2]*pred_box[3] + gt_box[2]*gt_box[3] - inter)
            iou   = inter / (union + 1e-6)

            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np = (img_t[0].cpu().permute(1,2,0).numpy() * std + mean).clip(0, 1)

            if i < det_samples:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_np)
                ax.add_patch(patches.Rectangle(
                    (tx1, ty1), float(gt_box[2]), float(gt_box[3]),
                    linewidth=2, edgecolor='lime', facecolor='none', label='GT'))
                ax.add_patch(patches.Rectangle(
                    (px1, py1), float(pred_box[2]), float(pred_box[3]),
                    linewidth=2, edgecolor='red', facecolor='none', label='Pred'))
                ax.axis('off')
                ax.legend(loc='upper right', fontsize=8)
                pil_overlay = fig_to_pil(fig)
                plt.close(fig)

                table_det.add_data(
                    wandb.Image(pil_overlay),
                    gt_box.tolist(),
                    pred_box.tolist(),
                    round(iou, 4),
                    round(conf, 4)
                )

            if i < seg_samples:
                table_seg.add_data(
                    wandb.Image(img_np),
                    wandb.Image((gt_mask.numpy() / 2.0).astype(np.float32),
                                caption="GT Trimap"),
                    wandb.Image((pred_mask.numpy() / 2.0).astype(np.float32),
                                caption="Pred Mask"),
                )

    wandb.log({
        "2.5_Object_Detection_Table": table_det,
        "2.6_Segmentation_Table":     table_seg,
    })

def main(args):
    wandb.init(project="da6401-assignment-2", entity="da6401-fdl", job_type="eval")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    # if args.checkpoint:
    #     state = torch.load(args.checkpoint, map_location='cpu')
    #     model.load_state_dict(state)
    model.to(device)

    dataset = OxfordIIITPetDataset(args.data_dir, split='test')

    # Section 2.4
    visualize_feature_maps(model, dataset[0]['image'], device=device)

    # Sections 2.5 & 2.6
    evaluate_predictions(model, dataset, device=device,
                         det_samples=10, seg_samples=5)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DA6401 Assignment-2 Inference")
    parser.add_argument('--data_dir',   type=str, default='./data/')
    parser.add_argument('--checkpoint', type=str, default='',
                        help="Path to trained .pth checkpoint")
    args = parser.parse_args()
    main(args)


