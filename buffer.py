import networkx as nx
from pyvis.network import Network
import fitz  # PyMuPDF
import os
import re
import sys
import json 

class buffer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = self.extract_structured_text_from_pdf(self.pdf_path)

    def extract_structured_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF with formatting information using PyMuPDF.
        Uses geometric analysis to detect underlined text.
        Returns a list of dictionaries with text and its formatting properties.
        """
        try:
            # print(f"Opening PDF file: {pdf_path}")
            doc = fitz.open(pdf_path)
            # print(f"PDF has {len(doc)} pages")
            
            structured_text = []
            
            # First pass: determine maximum font size for reference
            max_font_size = 0
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                max_font_size = max(max_font_size, span["size"])
            # print(f"Maximum font size detected: {max_font_size}")
            
            # Process each page: extract text spans and detect underlines via drawing objects.
            for page_num, page in enumerate(doc):
                # Get drawing objects and filter for those that appear to be underlines.
                drawings = page.get_drawings()
                underline_rects = []
                for d in drawings:
                    # Look for filled rectangles (type 'f') which might be drawn as underlines.
                    if d.get("type") == "f":
                        for item in d.get("items", []):
                            if item[0] == "re":
                                rect = item[1]
                                # Heuristic: if the rectangle is very short in height, consider it an underline.
                                if rect.height < 5:
                                    underline_rects.append(rect)
                
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            is_bold = False
                            is_underlined = False
                            font_size = 0
                            font_used = ""
                            span_bboxes = []
                            
                            # Process each span within the line.
                            for span in line["spans"]:
                                if "bold" in span["font"].lower():
                                    is_bold = True
                                line_text += span["text"] + " "
                                font_size = max(font_size, span["size"])
                                font_used = span["font"]
                                span_bboxes.append(span["bbox"])
                            
                            # Compute the union of the bounding boxes for the whole line.
                            if span_bboxes:
                                x0 = min(b[0] for b in span_bboxes)
                                y0 = min(b[1] for b in span_bboxes)
                                x1 = max(b[2] for b in span_bboxes)
                                y1 = max(b[3] for b in span_bboxes)
                                line_bbox = (x0, y0, x1, y1)
                            else:
                                line_bbox = None
                            
                            # Heuristic: if any underline rectangle overlaps horizontally
                            # and its top is within 5 units of the text bbox bottom, mark as underlined.
                            if line_bbox:
                                for rect in underline_rects:
                                    overlap = min(line_bbox[2], rect.x1) - max(line_bbox[0], rect.x0)
                                    if overlap > 0 and abs(rect.y0 - line_bbox[3]) < 5:
                                        is_underlined = True
                                        break
                            
                            line_text = line_text.strip()
                            if line_text:
                                structured_text.append({
                                    "text": line_text,
                                    "font": font_used,
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                    "is_underlined": is_underlined,
                                    "page": page_num + 1
                                })
            
            # print(f"Extracted {len(structured_text)} text elements")
            return structured_text
        except Exception as e:
            # print(f"Error extracting PDF text: {e}")
            raise e
if __name__ == "__main__":
    # Example usage
    pdf_path = r"C:\Users\bilas\OneDrive\Documents\GENAI\my_web\iitg.pdf"  # Replace with your PDF file path
    buffer_instance = buffer(pdf_path)
    structured_text = buffer_instance.data
    for item in structured_text:
        print(item)