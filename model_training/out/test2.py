import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# --- Configuration ---
TRAIN_IMG_FOLDER = './kaggleSAR/SARscope/train/'
TRAIN_ANN_FILE = './kaggleSAR/SARscope/train/_annotations.coco.json'
VAL_IMG_FOLDER = './kaggleSAR/SARscope/valid/'
VAL_ANN_FILE = './kaggleSAR/SARscope/valid/_annotations.coco.json'
TEST_IMG_FOLDER = './kaggleSAR/SARscope/test/'
TEST_ANN_FILE = './kaggleSAR/SARscope/test/_annotations.coco.json'

MODEL_CHECKPOINT = "facebook/detr-resnet-50"
BATCH_SIZE = 4
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ID_TO_LABEL = {0: "ship"}
LABEL_TO_ID = {"ship": 0}

# --- 1. Load Processor and Model ---
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
model_test = AutoModelForObjectDetection.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    ignore_mismatched_sizes=True
)
state_dict = torch.load("final_model.pt", map_location=device)
model_test.load_state_dict(state_dict, strict=False)
model_test.to(device).eval()

print("✅ Model and processor loaded")

# --- 2. Dataset and Dataloader ---
class DetrCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, image_processor):
        super().__init__(img_folder, ann_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        for ann in target:
            ann['category_id'] = LABEL_TO_ID['ship']

        formatted_target = {'image_id': image_id, 'annotations': target}
        encoding = self.image_processor(images=img, annotations=formatted_target, return_tensors="pt")

        if 'labels' not in encoding or not encoding['labels']:
            encoding['labels'] = [{'class_labels': torch.tensor([], dtype=torch.long),
                                   'boxes': torch.tensor([], dtype=torch.float32)}]

        return img, encoding["pixel_values"].squeeze(), encoding["labels"][0], image_id

def collate_fn(batch):
    original_images = [b[0] for b in batch]
    pixel_values = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    image_ids = [b[3] for b in batch]

    padded = image_processor.pad(pixel_values, return_tensors="pt", return_pixel_mask=True)
    return {
        'original_images': original_images,
        'pixel_values': padded['pixel_values'],
        'pixel_mask': padded['pixel_mask'],
        'labels': labels,
        'image_ids': image_ids
    }

test_dataset = DetrCocoDetection(TEST_IMG_FOLDER, TEST_ANN_FILE, image_processor)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)

# --- 3. Test Loop ---
print("\nStarting test inference...")
total_test_loss = 0.0
sample_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        outputs = model_test(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        total_test_loss += outputs.loss.item()

        target_sizes = torch.tensor([img.size[::-1] for img in batch['original_images']]).to(device)
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.6
        )

        if len(sample_predictions) < 5:
            for i in range(len(batch['original_images'])):
                if len(sample_predictions) < 5:
                    sample_predictions.append({
                        'original_image': batch['original_images'][i],
                        'image_id': batch['image_ids'][i],
                        'predictions': {
                            'boxes': results[i]['boxes'].cpu().numpy(),
                            'scores': results[i]['scores'].cpu().numpy(),
                            'labels': results[i]['labels'].cpu().numpy(),
                        },
                        'ground_truths': {
                            'boxes': batch['labels'][i]['boxes'].cpu().numpy(),
                            'labels': batch['labels'][i]['class_labels'].cpu().numpy()
                        }
                    })

avg_test_loss = total_test_loss / len(test_dataloader)
print(f"\n✅ Average Test Loss: {avg_test_loss:.4f}")

# --- 4. Visualization (Save Images) ---
print(f"\nSaving visualizations to: {RESULTS_DIR}")

def plot_and_save_comparison(pil_img, predictions, ground_truths, id2label, save_path, confidence_threshold=0.8):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    img_w, img_h = pil_img.size

    # Left: Ground Truth
    ax_gt = axes[0]
    ax_gt.imshow(pil_img)
    ax_gt.set_title("Ground Truth")
    ax_gt.axis("off")

    for gt_box, gt_label in zip(ground_truths['boxes'], ground_truths['labels']):
        cx, cy, w, h = gt_box
        xmin, ymin = (cx - w/2)*img_w, (cy - h/2)*img_h
        xmax, ymax = (cx + w/2)*img_w, (cy + h/2)*img_h
        label = id2label[int(gt_label)]
        ax_gt.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                          fill=False, edgecolor='blue', linestyle='--', linewidth=2))
        ax_gt.text(xmin, ymin - 10, f"GT: {label}", color='white',
                   bbox=dict(facecolor='blue', alpha=0.7), fontsize=8)

    # Right: Predictions
    ax_pr = axes[1]
    ax_pr.imshow(pil_img)
    ax_pr.set_title(f"Predictions (conf>{confidence_threshold})")
    ax_pr.axis("off")

    for score, label_id, (xmin, ymin, xmax, ymax) in zip(predictions['scores'],
                                                         predictions['labels'],
                                                         predictions['boxes']):
        if score > confidence_threshold:
            label = id2label[int(label_id)]
            ax_pr.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              fill=False, edgecolor='red', linewidth=2))
            ax_pr.text(xmin, ymin - 10, f"{label}: {score:.2f}",
                       bbox=dict(facecolor='red', alpha=0.7), fontsize=8, color='white')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # prevent GUI blocking

# Save top samples
for i, s in enumerate(sample_predictions):
    filename = os.path.join(RESULTS_DIR, f"sample_{i+1}_id_{s['image_id']}.png")
    plot_and_save_comparison(
        s['original_image'], s['predictions'], s['ground_truths'],
        ID_TO_LABEL, filename, confidence_threshold=0.8
    )
    print(f"✅ Saved: {filename}")

print("\n✅ Test inference and visualization complete! All images saved to ./results/")
