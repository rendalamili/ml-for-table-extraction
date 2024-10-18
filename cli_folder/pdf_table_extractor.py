import os
import json
import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import torch
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import numpy as np
import argparse
import cv2
import pytesseract

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

def get_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def detect_and_extract_tables(image):
    if model is None or image_processor is None:
        return []

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    
    tables = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.7:  # You can adjust this threshold
            x1, y1, x2, y2 = box.tolist()
            cropped_image = image.crop((x1, y1, x2, y2))
            table_data = extract_table_from_image(cropped_image)
            tables.append(table_data)
    
    return tables

def extract_table_from_image(image):
    # Convert image to grayscale
    gray = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(gray)
    
    # Simple thresholding to separate text from background
    _, binary = cv2.threshold(img_array, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by y-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    rows = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell = binary[y:y+h, x:x+w]
        text = pytesseract.image_to_string(Image.fromarray(cell)).strip()
        rows.append(text)
    
    return rows

def extract_pdf_table(pdf_path, poppler_path=None):
    all_tables = []

    try:
        if poppler_path:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path)
        print(f"Successfully converted PDF to {len(images)} images.")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return all_tables

    for i, image in enumerate(images):
        print(f"Processing page {i+1}")
        tables = detect_and_extract_tables(image)
        all_tables.extend(tables)
        print(f"Extracted {len(tables)} table(s) from page {i+1}.")

    return all_tables

def main(directory, tesseract_path=None, poppler_path=None):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    pdf_files = get_pdf_files(directory)

    if not pdf_files:
        print(f"No PDF files found in the directory: {directory}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        print(f"Processing {pdf_file}")
        
        tables = extract_pdf_table(pdf_path, poppler_path)

        output_file = os.path.join(directory, f"{os.path.splitext(pdf_file)[0]}_tables.json")
        with open(output_file, 'w') as f:
            json.dump(tables, f, indent=2)

        print(f"Extracted {len(tables)} tables from {pdf_file} and saved as {output_file}.")

        if tables:
            print("\nFirst extracted table:")
            print(json.dumps(tables[0], indent=2))
        else:
            print("No tables were extracted from this PDF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF files in a directory.")
    parser.add_argument("directory", help="Path to the directory containing PDF files")
    parser.add_argument("--tesseract_path", help="Path to Tesseract executable", default=None)
    parser.add_argument("--poppler_path", help="Path to Poppler binaries", default=None)
    args = parser.parse_args()

    main(args.directory, args.tesseract_path, args.poppler_path)