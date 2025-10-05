import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

# --- Configuration ---
TRAIN_IMG_FOLDER = './kaggleSAR/SARscope/train/'
TRAIN_ANN_FILE = './kaggleSAR/SARscope/train/_annotations.coco.json'
VAL_IMG_FOLDER = './kaggleSAR/SARscope/valid/'
VAL_ANN_FILE = './kaggleSAR/SARscope/valid/_annotations.coco.json'
TEST_IMG_FOLDER = './kaggleSAR/SARscope/test/'
TEST_ANN_FILE = './kaggleSAR/SARscope/test/_annotations.coco.json'

MODEL_CHECKPOINT = "facebook/detr-resnet-50"
BATCH_SIZE = 60 # Reduced batch size for fine-tuning
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4 # Standard learning rate for fine-tuning DETR
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ID_TO_LABEL = {0: "ship"}
LABEL_TO_ID = {"ship": 0}

# --- 1. Data Loading and Preprocessing Setup ---

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

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
        
        # Ensure 'labels' key is always present, even if no objects
        if 'labels' not in encoding:
            encoding['labels'] = [{'class_labels': torch.tensor([], dtype=torch.long), 
                                   'boxes': torch.tensor([], dtype=torch.float32)}] # Empty tensors for no objects
        elif not encoding['labels']: # If labels list is empty
             encoding['labels'] = [{'class_labels': torch.tensor([], dtype=torch.long), 
                                   'boxes': torch.tensor([], dtype=torch.float32)}]


        return encoding["pixel_values"].squeeze(), encoding["labels"][0]

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded_batch = image_processor.pad(
        pixel_values,
        return_tensors="pt",
        return_pixel_mask=True
    )
    return {
        'pixel_values': padded_batch['pixel_values'],
        'pixel_mask': padded_batch['pixel_mask'],
        'labels': labels
    }

try:
    train_dataset = DetrCocoDetection(
        img_folder=TRAIN_IMG_FOLDER,
        ann_file=TRAIN_ANN_FILE,
        image_processor=image_processor
    )
    val_dataset = DetrCocoDetection(
        img_folder=VAL_IMG_FOLDER,
        ann_file=VAL_ANN_FILE,
        image_processor=image_processor
    )
    test_dataset = DetrCocoDetection( # Added test dataset
        img_folder=TEST_IMG_FOLDER,
        ann_file=TEST_ANN_FILE,
        image_processor=image_processor
    )
except Exception as e:
    print(f"Error loading datasets. Please ensure paths are correct and COCO JSON is valid: {e}")
    exit()

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE) # Test Dataloader

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# --- 2. Model, Optimizer, and Scheduler Setup ---

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    ignore_mismatched_sizes=True
)
model.to(device)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the heads and query embeddings for fine-tuning
# These are the direct prediction layers and the learnable object queries.
# This approach ensures minimal changes to the original DETR architecture while fine-tuning.
if hasattr(model, 'class_predictor'):
    for param in model.class_predictor.parameters():
        param.requires_grad = True
    print("Unfrozen model.class_predictor parameters.")

if hasattr(model, 'bbox_predictor'):
    for param in model.bbox_predictor.parameters():
        param.requires_grad = True
    print("Unfrozen model.bbox_predictor parameters.")

if hasattr(model, 'detr') and hasattr(model.detr, 'query_embed'):
    for param in model.detr.query_embed.parameters():
        param.requires_grad = True
    print("Unfrozen model.detr.query_embed parameters.")

# Verify trainable parameters
print("\nTrainable parameters after freezing:")
trainable_params_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  - {name}: {param.numel()} parameters")
        trainable_params_count += param.numel()
print(f"Total trainable parameters: {trainable_params_count}")


optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_losses = []
val_losses = []

# --- 3. Training Loop ---

print("\nStarting training loop...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # --- 4. Validation Loop ---
    # model.eval()
    # total_val_loss = 0.0
    
    # with torch.no_grad():
    #     for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)")):
    #         pixel_values = batch['pixel_values'].to(device)
    #         pixel_mask = batch['pixel_mask'].to(device)
    #         labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

    #         outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    #         loss = outputs.loss
    #         total_val_loss += loss.item()

    # avg_val_loss = total_val_loss / len(val_dataloader)
    # val_losses.append(avg_val_loss)
    # print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

print("\nTraining complete!")
torch.save(model.state_dict(), "./final_model.pt")
# --- 5. Plotting Loss Curves ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()