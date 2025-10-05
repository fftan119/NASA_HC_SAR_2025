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

# --- Configuration (can be imported from above, but duplicated for standalone execution) ---
TEST_IMG_FOLDER = './kaggleSAR/SARscope/test/'
TEST_ANN_FILE = './kaggleSAR/SARscope/test/_annotations.coco.json'
MODEL_CHECKPOINT = "facebook/detr-resnet-50" # This is the base checkpoint, you'd ideally load your fine-tuned model weights here if saved.
BATCH_SIZE = 4 # Same as validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_TO_LABEL = {0: "ship"}
LABEL_TO_ID = {"ship": 0}

# --- Data Loading Setup (replicated for test block independence) ---
# image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
# image_processor.load_state_dict(torch.load("final_model.pt"))

image_processor = AutoModelForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    id2label={0:"ship"},
    label2id={"ship":0},
    ignore_mismatched_sizes=True
)
state_dict = torch.load("final_model.pt", map_location="cpu")
image_processor.load_state_dict(state_dict)

class DetrCocoDetectionTest(CocoDetection): # Renamed class to avoid conflict
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
        
        if 'labels' not in encoding:
            encoding['labels'] = [{'class_labels': torch.tensor([], dtype=torch.long), 
                                   'boxes': torch.tensor([], dtype=torch.float32)}]
        elif not encoding['labels']:
             encoding['labels'] = [{'class_labels': torch.tensor([], dtype=torch.long), 
                                   'boxes': torch.tensor([], dtype=torch.float32)}]

        return encoding["pixel_values"].squeeze(), encoding["labels"][0], img, image_id # Also return original image and ID

def collate_fn_test(batch): # Separate collate_fn for test to handle additional returns
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    original_images = [item[2] for item in batch]
    image_ids = [item[3] for item in batch]

    padded_batch = image_processor.pad(
        pixel_values,
        return_tensors="pt",
        return_pixel_mask=True
    )
    return {
        'pixel_values': padded_batch['pixel_values'],
        'pixel_mask': padded_batch['pixel_mask'],
        'labels': labels,
        'original_images': original_images, # Include original images for plotting
        'image_ids': image_ids
    }

try:
    test_dataset = DetrCocoDetectionTest(
        img_folder=TEST_IMG_FOLDER,
        ann_file=TEST_ANN_FILE,
        image_processor=image_processor
    )
except Exception as e:
    print(f"Error loading test dataset: {e}")
    exit()

test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn_test, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of test samples: {len(test_dataset)}")

# --- Model Loading (Load the fine-tuned model) ---
# IMPORTANT: If you saved your fine-tuned model from the training script, load it here!
# For example:
# model = AutoModelForObjectDetection.from_pretrained(
#     MODEL_CHECKPOINT,
#     id2label=ID_TO_LABEL,
#     label2id=LABEL_TO_ID,
#     ignore_mismatched_sizes=True
# )
# model.load_state_dict(torch.load("path/to/your_finetuned_model.pth")) # Load saved weights
# model.to(device)

# For this example, we'll re-initialize the model with the same setup as training,
# assuming you'd run the training script first in the same session,
# or if you are just testing the architecture after configuring it for 1 class.
# If running this block separately, you MUST load trained weights.
model_test = AutoModelForObjectDetection.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    ignore_mismatched_sizes=True
)

# Freeze all parameters (as done during fine-tuning)
for param in model_test.parameters():
    param.requires_grad = False

# Unfreeze the heads and query embeddings (as done during fine-tuning)
if hasattr(model_test, 'class_predictor'):
    for param in model_test.class_predictor.parameters():
        param.requires_grad = True

if hasattr(model_test, 'bbox_predictor'):
    for param in model_test.bbox_predictor.parameters():
        param.requires_grad = True

if hasattr(model_test, 'detr') and hasattr(model_test.detr, 'query_embed'):
    for param in model_test.detr.query_embed.parameters():
        param.requires_grad = True

model_test.to(device)
model_test.eval() # Set to evaluation mode