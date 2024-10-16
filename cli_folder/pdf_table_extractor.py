# Import libraries
import os
import json
import camelot
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import pandas as pd
import torch
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import cv2
import numpy as np
from google.colab import files

# Upload PDF file
uploaded = files.upload()
pdf_file = next(iter(uploaded))

# Save the uploaded file to a temporary location
with open(pdf_file, 'wb') as f:
    f.write(uploaded[pdf_file])

# Load pre-trained Table Transformer model and image processor
try:
    model = TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-detection')
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Table Transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading the Table Transformer model: {e}")
    model = None
    image_processor = None

def detect_tables(image):
    if model is None or image_processor is None:
        return []

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    return results['boxes'].tolist()

def extract_table_content(image, bbox):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cropped_image = image.crop((x1, y1, x2, y2))

    # Try Camelot first
    temp_image_path = 'temp_table.png'
    cropped_image.save(temp_image_path)

    try:
        tables = camelot.read_pdf(temp_image_path, flavor='stream')
        if tables.n > 0:
            extracted_table = tables[0].df
        else:
            tables = camelot.read_pdf(temp_image_path, flavor='lattice')
            extracted_table = tables[0].df if tables.n > 0 else None

        if extracted_table is not None:
            print("Table extracted using Camelot.")
            return extracted_table
    except Exception as e:
        print(f"Camelot extraction failed: {e}")

    # If Camelot fails, use OCR
    try:
        text = pytesseract.image_to_string(cropped_image)
        lines = text.split('\n')
        table_data = [line.split() for line in lines if line.strip()]
        extracted_table = pd.DataFrame(table_data)
        print("Table extracted using OCR.")
        return extracted_table
    except Exception as e:
        print(f"OCR extraction failed: {e}")

    return None

def extract_pdf_table(pdf_file):
    all_tables = []

    # Convert PDF to images
    try:
        images = convert_from_path(pdf_file, poppler_path="/usr/bin")
        print(f"Successfully converted PDF to {len(images)} images.")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return all_tables

    for i, image in enumerate(images):
        print(f"Processing page {i+1}")
        # Detect tables using Table Transformer
        bboxes = detect_tables(image)

        if bboxes:
            print(f"Detected {len(bboxes)} table(s) on page {i+1}.")
            for j, bbox in enumerate(bboxes):
                table = extract_table_content(image, bbox)
                if table is not None:
                    all_tables.append(table)
                    print(f"Extracted content from table {j+1} on page {i+1}.")
        else:
            print(f"No tables detected on page {i+1}.")

    return all_tables

def table_to_json(table):
    if isinstance(table, pd.DataFrame):
        # Convert DataFrame to list of dictionaries
        return table.to_dict(orient='records')
    else:
        print("Invalid table format for JSON conversion.")
        return []

# Extract tables from PDF file
tables = extract_pdf_table(pdf_file)

# Convert extracted tables to JSON
json_tables = [table_to_json(table) for table in tables]
# Save JSON to file
with open('tables.json', 'w') as f:
    json.dump(json_tables, f, indent=2)

print(f"Extracted {len(json_tables)} tables and saved as JSON.")

# Display first table as an example
if json_tables:
    print("\nFirst extracted table:")
    print(json.dumps(json_tables[0], indent=2))
else:
    print("No tables were extracted.")

# Clean up the temporary file
if os.path.exists(pdf_file):
    os.remove(pdf_file)