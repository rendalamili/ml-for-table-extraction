import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from ultralyticsplus import YOLO, render_result
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from paddleocr import PaddleOCR
from tqdm import tqdm
import logging
from autocorrect import Speller

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress PaddleOCR debug information
logging.getLogger('ppocr').setLevel(logging.ERROR)

class TableExtractor:
    def __init__(self, poppler_path=None):
        # Initialize YOLO model for table detection
        self.yolo_model = YOLO('keremberke/yolov8m-table-extraction')
        self.yolo_model.overrides['conf'] = 0.25
        self.yolo_model.overrides['iou'] = 0.45
        self.yolo_model.overrides['agnostic_nms'] = False
        self.yolo_model.overrides['max_det'] = 1000

        # Initialize Table Transformer model with try-except block
        try:
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition",
                revision="v1.1-all"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition",
                revision="v1.1-all",
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            logger.error(f"Error loading Table Transformer model: {str(e)}")
            # Fallback to base model if v1.1-all fails
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Initialize spellchecker
        self.spell = Speller(lang='en')

        self.poppler_path = poppler_path
        self.colours = {
            "table": (int(0.000 * 255), int(0.447 * 255), int(0.741 * 255)),  # Blue
            "table row": (int(0.850 * 255), int(0.325 * 255), int(0.098 * 255)),  # Orange
            "table column": (int(0.929 * 255), int(0.694 * 255), int(0.125 * 255)),  # Yellow
            "table cell": (int(0.494 * 255), int(0.184 * 255), int(0.556 * 255)),  # Purple
            "other": (int(0.466 * 255), int(0.674 * 255), int(0.188 * 255))  # Green
        }

    def plot_boxes(self, pil_img, scores, labels, boxes, output_path=None):
        img_copy = pil_img.convert("RGB")
        draw = ImageDraw.Draw(img_copy)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for score, label, (xmin, ymin, xmax, ymax) in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            label_name = self.model.config.id2label.get(label, "other")
            color = self.colours.get(label_name, self.colours["other"])

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            text = f"{label_name}: {score:.2f}"
            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
            draw.rectangle([xmin, ymin - text_bbox[3] + text_bbox[1], xmin + text_bbox[2] - text_bbox[0], ymin], fill="black")
            draw.text((xmin, max(0, ymin - (text_bbox[3] - text_bbox[1]))), text, fill="white", font=font)

        if output_path:
            img_copy.save(output_path)
        return img_copy


    def crop_table_with_dynamic_padding(self, image, box, padding_ratio=0.05):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box.xyxy[0].tolist()]
        width, height = x_max - x_min, y_max - y_min

        padding_x = int(width * padding_ratio)
        padding_y = int(height * padding_ratio)

        x_min, y_min = max(0, x_min - padding_x), max(0, y_min - padding_y)
        x_max, y_max = min(image.width, x_max + padding_x), min(image.height, y_max + padding_y)

        return image.crop((x_min, y_min, x_max, y_max))

    def get_cell_coordinates_by_row(self, table_data):
        rows = sorted(
            [box for box, label in zip(table_data['boxes'], table_data['labels']) 
             if label == self.model.config.label2id['table row']],
            key=lambda x: x[1]
        )
        columns = sorted(
            [box for box, label in zip(table_data['boxes'], table_data['labels']) 
             if label == self.model.config.label2id['table column']],
            key=lambda x: x[0]
        )

        cell_coordinates = [
            {
                'row': row,
                'cells': [{'column': column, 'cell': (column[0], row[1], column[2], row[3])} 
                         for column in columns]
            }
            for row in rows
        ]

        for row_data in cell_coordinates:
            row_data['cells'].sort(key=lambda x: x['cell'][0])

        return cell_coordinates

    def apply_ocr_paddle(self, cell_coordinates, cropped_table_image):
        rows = []
        for row_data in tqdm(cell_coordinates, desc="Applying PaddleOCR"):
            row = []
            for cell in row_data['cells']:
                x_min, y_min, x_max, y_max = [int(coord) for coord in cell['cell']]
                cell_image = cropped_table_image.crop((x_min, y_min, x_max, y_max))
                
                # OCR with PaddleOCR
                ocr_result = self.ocr.ocr(np.array(cell_image), cls=True)
                cell_text = " ".join([line[1][0] for line in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
                row.append(cell_text.strip())
            rows.append(row)
        return rows

    def clean_text(self, text):
        """Clean and correct spelling in text"""
        # Characters to remove
        punct_error = ['|', '(', '[', ']', ')', '{', '}', '_']
        
        # Remove unwanted characters
        for char in punct_error:
            text = text.replace(char, '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Skip spell check for empty strings or numbers
        if not text.strip() or text.replace('.', '').isdigit():
            return text.strip()
            
        # Apply spell check
        return self.spell(text.strip())

    def clean_table_data(self, rows):
        """Clean all cells in the table data"""
        return [[self.clean_text(cell) for cell in row] for row in rows]

    def process_pdf(self, pdf_path, output_dir):
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)

            logger.info(f"Processing PDF: {pdf_path}")
            
            conversion_kwargs = {}
            if self.poppler_path:
                conversion_kwargs['poppler_path'] = self.poppler_path
            
            pages = convert_from_path(pdf_path, dpi=300, **conversion_kwargs)
            
            for page_num, img in enumerate(pages):
                logger.info(f"Processing page {page_num + 1}")
                
                yolo_results = self.yolo_model.predict(img)
                render = render_result(model=self.yolo_model, image=img, result=yolo_results[0])
                render.save(os.path.join(pdf_output_dir, f'page_{page_num + 1}_detected_tables.png'))

                for table_num, box in enumerate(yolo_results[0].boxes):
                    cropped_table = self.crop_table_with_dynamic_padding(img, box)
                    cropped_table.save(os.path.join(pdf_output_dir, f'page_{page_num + 1}_table_{table_num + 1}.png'))

                    encoding = self.processor(images=cropped_table, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model(**encoding)

                    target_sizes = [cropped_table.size[::-1]]
                    results = self.processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

                    structure_img = self.plot_boxes(
                        cropped_table, 
                        results["scores"], 
                        results["labels"], 
                        results["boxes"],
                        os.path.join(pdf_output_dir, f'page_{page_num + 1}_table_{table_num + 1}_structure.png')
                    )

                    cell_coordinates = self.get_cell_coordinates_by_row({
                        'boxes': results["boxes"],
                        'labels': results["labels"]
                    })

                    # Apply OCR and clean the results - now in row-based format
                    table_rows = self.apply_ocr_paddle(cell_coordinates, cropped_table)
                    cleaned_rows = self.clean_table_data(table_rows)

                    # Create DataFrame with first row as headers
                    if cleaned_rows:
                        headers = cleaned_rows[0]
                        data = cleaned_rows[1:] if len(cleaned_rows) > 1 else []
                        df = pd.DataFrame(data, columns=headers)
                        
                        # Modified CSV filename to include PDF name
                        csv_filename = f'{pdf_name}_page_{page_num + 1}_table_{table_num + 1}.csv'
                        csv_path = os.path.join(pdf_output_dir, csv_filename)
                        df.to_csv(csv_path, index=False)
                        
                        logger.info(f"Saved table to {csv_path}")
                        logger.info(f"Headers: {headers}")
                        logger.info(f"Number of data rows: {len(data)}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract tables from PDF documents')
    parser.add_argument('input_path', type=str, help='Path to input PDF file or directory')
    parser.add_argument('--output', type=str, default='output', help='Path to output directory')
    parser.add_argument('--poppler', type=str, help='Path to Poppler binary directory')
    
    args = parser.parse_args()

    # Convert input path to absolute path
    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output)

    # Check if input path exists
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize TableExtractor with error handling
        extractor = TableExtractor(poppler_path=args.poppler)
        
        # Process single PDF file or directory
        if os.path.isfile(input_path):
            if input_path.lower().endswith('.pdf'):
                extractor.process_pdf(input_path, output_dir)
            else:
                logger.error(f"Input file is not a PDF: {input_path}")
        else:
            # Process all PDFs in the directory
            pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith('.pdf')]
            if not pdf_files:
                logger.error(f"No PDF files found in directory: {input_path}")
                sys.exit(1)
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_path, pdf_file)
                extractor.process_pdf(pdf_path, output_dir)
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()