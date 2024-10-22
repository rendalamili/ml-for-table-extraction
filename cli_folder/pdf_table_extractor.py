import os
import json
import camelot
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

def is_pdf_extractable(pdf_path):
    """Check if PDF has extractable text."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfFileReader(file)
            page = pdf.getPage(0)
            text = page.extractText()
            return len(text.strip()) > 0
    except:
        return False

def extract_tables_with_camelot(pdf_path):
    """Extract tables using Camelot."""
    try:
        # Try lattice first
        tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
        
        # If no tables found, try stream
        if len(tables) == 0:
            tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
        
        extracted_tables = []
        for table in tables:
            # Convert to nested list structure and clean data
            table_data = table.df.values.tolist()
            cleaned_table = clean_table_data(table_data)
            if cleaned_table:  # Only add if table has content
                extracted_tables.append({
                    'page': table.page,
                    'data': cleaned_table,
                    'confidence': table.accuracy,
                    'extraction_method': 'camelot'
                })
        
        return extracted_tables
    except Exception as e:
        print(f"Camelot extraction failed: {e}")
        return []

def clean_table_data(table_data):
    """Clean table data by removing empty cells and standardizing format."""
    cleaned_table = []
    for row in table_data:
        cleaned_row = []
        for cell in row:
            # Clean and standardize cell content
            cell = str(cell).strip()
            cell = ' '.join(cell.split())  # Normalize whitespace
            if cell and cell != 'nan':
                cleaned_row.append(cell)
            else:
                cleaned_row.append(None)  # Use None for empty cells
        
        # Only add rows that have at least one non-None value
        if any(cell is not None for cell in cleaned_row):
            cleaned_table.append(cleaned_row)
    
    return cleaned_table if cleaned_table else None

def detect_and_extract_tables(image):
    """Detect tables using Table Transformer and extract text using OCR."""
    if model is None or image_processor is None:
        return []

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    
    tables = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.7:
            x1, y1, x2, y2 = box.tolist()
            cropped_image = image.crop((x1, y1, x2, y2))
            table_data = extract_table_from_image(cropped_image)
            if table_data:  # Only add if table has content
                tables.append({
                    'data': table_data,
                    'confidence': float(score),
                    'extraction_method': 'ocr'
                })
    
    return tables

def extract_table_from_image(image):
    """Extract table from image using OCR with improved structure preservation."""
    # Convert to grayscale and improve image quality
    gray = image.convert('L')
    img_array = np.array(gray)
    
    # Enhance image
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Use Tesseract with table structure recognition
    config = '--psm 6 --oem 3'  # Assume uniform block of text
    data = pytesseract.image_to_data(Image.fromarray(denoised), config=config, output_type=pytesseract.Output.DATAFRAME)
    
    # Group text by lines using y-coordinates
    data = data[data.conf != -1]  # Remove low confidence detections
    lines = {}
    
    for _, row in data.iterrows():
        if not pd.isna(row.text) and str(row.text).strip():
            y_coord = row.top // 10 * 10  # Group similar y-coordinates
            if y_coord not in lines:
                lines[y_coord] = []
            lines[y_coord].append(str(row.text).strip())
    
    # Convert to structured table format
    table_data = [line for _, line in sorted(lines.items())]
    return clean_table_data(table_data)

def extract_pdf_table(pdf_path, poppler_path=None):
    """Main function to extract tables from PDF."""
    all_tables = []
    
    # Try Camelot first if PDF has extractable text
    if is_pdf_extractable(pdf_path):
        print("PDF has extractable text. Using Camelot...")
        all_tables = extract_tables_with_camelot(pdf_path)
    
    # If Camelot failed or found no tables, try OCR approach
    if not all_tables:
        print("Falling back to OCR-based extraction...")
        try:
            if poppler_path:
                images = convert_from_path(pdf_path, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1}")
                tables = detect_and_extract_tables(image)
                for table in tables:
                    table['page'] = i + 1
                all_tables.extend(tables)
                
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
    
    return all_tables

def main(directory, tesseract_path=None, poppler_path=None):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in the directory: {directory}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        print(f"\nProcessing {pdf_file}")
        
        tables = extract_pdf_table(pdf_path, poppler_path)

        # Enhanced output structure
        output = {
            'filename': pdf_file,
            'total_tables': len(tables),
            'tables': tables,
            'metadata': {
                'extraction_date': pd.Timestamp.now().isoformat(),
                'extraction_success': len(tables) > 0
            }
        }

        output_file = os.path.join(directory, f"{os.path.splitext(pdf_file)[0]}_tables.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Extracted {len(tables)} tables from {pdf_file}")
        if tables:
            print("\nFirst extracted table sample:")
            print(json.dumps(tables[0], indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from PDF files in a directory.")
    parser.add_argument("directory", help="Path to the directory containing PDF files")
    parser.add_argument("--tesseract_path", help="Path to Tesseract executable", default=None)
    parser.add_argument("--poppler_path", help="Path to Poppler binaries", default=None)
    args = parser.parse_args()

    main(args.directory, args.tesseract_path, args.poppler_path)