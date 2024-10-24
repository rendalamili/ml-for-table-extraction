import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    TableTransformerForObjectDetection, 
    Trainer, 
    TrainingArguments, 
    AutoImageProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define category mapping
CATEGORY_ID_MAP = {
    1: 0,  # table -> 0
    2: 1,  # row -> 1
    3: 1,  # cell -> 1 (mapping cells to same class as rows for simplicity)
    4: 1,  # header -> 1 (mapping headers to same class as rows for simplicity)
    5: 2,  # column -> 2
}

def convert_coco_bbox_to_normal(bbox, image_width, image_height):
    """Convert COCO format bbox [x, y, width, height] to normalized [x1, y1, x2, y2]"""
    x, y, width, height = bbox
    x1 = x / image_width
    y1 = y / image_height
    x2 = (x + width) / image_width
    y2 = (y + height) / image_height
    
    # Ensure coordinates are within [0, 1]
    x1 = max(0, min(1, x1))
    y1 = max(0, min(1, y1))
    x2 = max(0, min(1, x2))
    y2 = max(0, min(1, y2))
    
    return [x1, y1, x2, y2]

def load_dataset_with_labels(images_dir, annotations_file, limit=100):
    dataset = []
    
    images_dir = Path(images_dir)
    annotations_file = Path(annotations_file)
    
    logger.info(f"Loading annotations from: {annotations_file}")
    logger.info(f"Looking for images in: {images_dir}")
    
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
            
        images_by_id = {img['id']: img for img in annotations_data['images']}
        annotations_by_image = {}
        
        for ann in annotations_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
            
        processed_count = 0
        
        for image_id, image_info in images_by_id.items():
            if processed_count >= limit:
                break
                
            image_path = images_dir / image_info['file_name']
            logger.info(f"Looking for image: {image_path}")
            
            if image_path.exists():
                # Get image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # Get annotations for this image
                image_annotations = annotations_by_image.get(image_id, [])
                
                # Extract bounding boxes and classes
                boxes = []
                class_labels = []
                
                for ann in image_annotations:
                    # Map category_id to model's expected class index
                    class_label = CATEGORY_ID_MAP.get(ann['category_id'], 0)  # Default to 0 if unknown category
                    if class_label is not None:  # Only add if we have a valid mapping
                        box = convert_coco_bbox_to_normal(ann['bbox'], width, height)
                        boxes.append(box)
                        class_labels.append(class_label)
                
                if boxes and class_labels:  # Only add if we have valid annotations
                    dataset.append({
                        'image_path': str(image_path),
                        'labels': {
                            'boxes': boxes,
                            'class_labels': class_labels
                        }
                    })
                
                    processed_count += 1
                    logger.info(f"Successfully processed image {processed_count}: {image_path}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Total images processed: {processed_count}")
        
        if processed_count == 0:
            raise ValueError("No valid images were found in the dataset")
            
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_function(examples):
    """Preprocess the dataset examples for the TableTransformer model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_data = []
    
    for example in examples:
        try:
            # Load and transform image
            image_path = example['image_path']
            image = Image.open(image_path).convert('RGB')
            tensor_image = transform(image)
            
            # Convert boxes and labels to tensors
            boxes = torch.tensor(example['labels']['boxes'], dtype=torch.float32)
            class_labels = torch.tensor(example['labels']['class_labels'], dtype=torch.long)
            
            # Ensure we have valid boxes and labels
            if len(boxes) > 0 and len(class_labels) > 0:
                processed_data.append({
                    "pixel_values": tensor_image,
                    "labels": {
                        "class_labels": class_labels,
                        "boxes": boxes
                    }
                })
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            continue
    
    return processed_data

def custom_data_collator(features):
    """Custom collate function for batching the processed data."""
    if not features:
        return None
        
    try:
        batch = {}
        
        # Stack all image tensors
        batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
        
        # Collect all labels
        if 'labels' in features[0]:
            batch['labels'] = [f['labels'] for f in features]
        
        return batch
    
    except Exception as e:
        logger.error(f"Error during collation: {str(e)}")
        return None

def main():
    # Define paths for your CISOL dataset
    base_dir = Path(r"C:/Users/Osama/Downloads/archive")
    
    train_images_dir = base_dir / "cisol_TD-TSR/TD-TSR/images/train"
    train_annotations_file = base_dir / "cisol_TD-TSR/TD-TSR/annotations/train.json"
    val_images_dir = base_dir / "cisol_TD-TSR/TD-TSR/images/val"
    val_annotations_file = base_dir / "cisol_TD-TSR/TD-TSR/annotations/val.json"
    
    try:
        # Load datasets
        train_dataset = load_dataset_with_labels(train_images_dir, train_annotations_file, limit=100)
        logger.info(f"Loaded {len(train_dataset)} training samples")
        
        val_dataset = load_dataset_with_labels(val_images_dir, val_annotations_file, limit=100)
        logger.info(f"Loaded {len(val_dataset)} validation samples")
        
        # Initialize model and processor
        model = TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-detection')
        image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        
        # Preprocess datasets
        processed_train_dataset = preprocess_function(train_dataset)
        processed_val_dataset = preprocess_function(val_dataset)
        
        if not processed_train_dataset or not processed_val_dataset:
            raise ValueError("No valid processed samples found in dataset")
        
        # Convert to trainer format
        train_dataset = [{"pixel_values": item["pixel_values"], "labels": item["labels"]} 
                        for item in processed_train_dataset]
        eval_dataset = [{"pixel_values": item["pixel_values"], "labels": item["labels"]} 
                       for item in processed_val_dataset]
        
        # Training arguments
        training_arguments = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            save_steps=100,
            learning_rate=2e-5,
            per_device_train_batch_size=4,  # Reduced batch size
            per_device_eval_batch_size=4,   # Reduced batch size
            num_train_epochs=3,
            logging_dir='./logs',
            logging_steps=10,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=custom_data_collator
        )
        
        # Train model
        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()