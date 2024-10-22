import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Define paths
DATA_DIR = r'C:\Users\Osama\Downloads\archive\cisol_TD-TSR\TD-TSR'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')

class CISOLDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_processor):
        self.image_dir = image_dir
        self.image_processor = image_processor
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = [img['file_name'] for img in self.annotations['images']]
        
        if not self.image_files:
            raise ValueError(f"No images found in the annotation file")

        # Create a mapping from image_id to annotations
        self.image_id_to_annotations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Create category id to label mapping
        self.cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(self.annotations['categories'])}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        img_id = img_info['id']
        anns = self.image_id_to_annotations.get(img_id, [])

        # Process annotations to match the model's expected format
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(self.cat_id_to_label[ann['category_id']])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([img_id])
        }

        # Process the image using the image processor
        encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()  # Remove batch dimension
        target = encoding['labels'][0]  # Remove batch dimension

        return pixel_values, target

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pad images to the same size
    max_size = max(img.shape[1:] for img in pixel_values)
    padded_images = []
    for img in pixel_values:
        pad_width = [(0, max_size[i] - img.shape[i+1]) for i in range(2)]
        padded_img = torch.nn.functional.pad(img, [pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1]])
        padded_images.append(padded_img)
    
    pixel_values = torch.stack(padded_images)
    return {'pixel_values': pixel_values, 'labels': targets}

# Load pre-trained Table Transformer model and image processor
model = TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-detection')
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")

# Create datasets and dataloaders
try:
    train_dataset = CISOLDataset(os.path.join(IMAGE_DIR, 'train'), os.path.join(ANNOTATION_DIR, 'train.json'), image_processor)
    val_dataset = CISOLDataset(os.path.join(IMAGE_DIR, 'val'), os.path.join(ANNOTATION_DIR, 'val.json'), image_processor)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
except Exception as e:
    print(f"Error creating dataset or dataloader: {e}")
    raise

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# Train model
trainer.train()

# Save the model
trainer.save_model("./table_transformer_model")

# Test the model on a sample image
def test_model(model, image_processor, image_path):
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Process outputs
    pred_boxes = outputs.pred_boxes[0].detach().cpu().numpy()
    scores = outputs.scores[0].detach().cpu().numpy()
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for box, score in zip(pred_boxes, scores):
        if score > 0.7:  # You can adjust this threshold
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(box[0], box[1], f'{score:.2f}', color='white', 
                     backgroundcolor='red', fontsize=8)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test the model on all images in the test folder
def test_on_folder(model, image_processor, test_folder):
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, filename)
            print(f"Testing on image: {filename}")
            test_model(model, image_processor, image_path)

# Test the model on all images in the test folder
test_folder = os.path.join(IMAGE_DIR, 'test')
test_on_folder(model, image_processor, test_folder)