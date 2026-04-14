import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.multitask import MultiTaskPerceptionModel

wandb.init(project="da6401-assignment-2", entity="da6401-fdl", job_type="wild_inference")

device = torch.device('cpu')
model = MultiTaskPerceptionModel()
model.eval()
model.to(device)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

CLASS_NAMES = [
    'Abyssinian','Bengal','Birman','Bombay','British Shorthair',
    'Egyptian Mau','Maine Coon','Persian','Ragdoll','Russian Blue',
    'Siamese','Sphynx','american bulldog','american pit bull terrier',
    'basset hound','beagle','boxer','chihuahua','english cocker spaniel',
    'english setter','german shorthaired','great pyrenees','havanese',
    'japanese chin','keeshond','leonberger','miniature pinscher',
    'newfoundland','pomeranian','pug','saint bernard','samoyed',
    'scottish terrier','shiba inu','staffordshire bull terrier',
    'wheaten terrier','yorkshire terrier'
]

import os
image_files = [f for f in os.listdir('./wild_images') if f.endswith(('.jpg','.jpeg','.png'))]

table = wandb.Table(columns=["Image", "Predicted Breed", "Confidence", "Bounding Box", "Segmentation Mask"])

for img_file in image_files:
    img_path = os.path.join('./wild_images', img_file)
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)

    transformed = transform(image=img_np)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)

    # Classification
    probs = torch.softmax(out['classification'][0], dim=0)
    conf, pred_class = probs.max(0)
    breed = CLASS_NAMES[pred_class.item()]

    # Bounding box
    box = out['localization'][0].cpu().numpy()
    x1 = box[0] - box[2]/2
    y1 = box[1] - box[3]/2

    # Segmentation mask
    mask = torch.argmax(out['segmentation'][0], dim=0).cpu().numpy()

    # Visualize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_show = (img_tensor[0].cpu().permute(1,2,0).numpy() * std + mean).clip(0,1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Original + bbox
    axes[0].imshow(img_show)
    axes[0].add_patch(patches.Rectangle(
        (x1, y1), box[2], box[3],
        linewidth=2, edgecolor='red', facecolor='none'))
    axes[0].set_title(f'{breed} ({conf.item():.2f})')
    axes[0].axis('off')
    # Segmentation
    axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    # Original
    axes[2].imshow(img_show)
    axes[2].set_title('Original')
    axes[2].axis('off')

    plt.tight_layout()
    safe_name = img_file.replace(".", "_").replace(" ", "_")

    wandb.log({f"wild_images/{safe_name}": wandb.Image(fig)})   
    plt.close(fig)

    table.add_data(
        wandb.Image(img_show),
        breed,
        round(conf.item(), 4),
        box.tolist(),
        wandb.Image((mask / 2.0).astype(np.float32))
    )

wandb.log({"2.7_Wild_Images_Table": table})
wandb.finish()
print("Done! Check your W&B report.")